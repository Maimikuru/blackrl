"""Bi-level Reinforcement Learning Algorithm.

This module implements the bi-level optimization problem:
    max_{θ_L} J_L(f_{θ_L}, g^*)
    subject to g^* ∈ argmax_g J_F(f_{θ_L}, g)

where:
    - J_L: Leader's objective (discounted cumulative reward)
    - J_F: Follower's objective (Max-Ent RL with entropy regularization)
    - f_{θ_L}: Leader's policy parameterized by θ_L
    - g^*: Follower's optimal response policy
"""

import importlib
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from blackrl.agents.follower.follower_policy_model import FollowerPolicyModel
from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.follower.soft_q_learning import SoftQLearning
from blackrl.agents.leader.leader_policy import LeaderPolicy
from blackrl.replay_buffer.gamma_replay_buffer import GammaReplayBuffer


def _create_env_from_info(env_class_name, env_module_path, env_init_kwargs):
    """Create environment instance from class name and module path (for parallel processing).

    Args:
        env_class_name: Name of the environment class
        env_module_path: Module path of the environment class
        env_init_kwargs: Initialization arguments for the environment

    Returns:
        Environment instance

    """
    module = importlib.import_module(env_module_path)
    env_class = getattr(module, env_class_name)
    if env_init_kwargs:
        return env_class(**env_init_kwargs)
    return env_class()


def _get_joint_action_from_state(
    obs,
    leader_policy_table,
    follower_q_values,
    env_spec_dict,
    num_follower_actions,
    temperature,
):
    """Get joint action from current state (for parallel processing).

    Args:
        obs: Current observation
        leader_policy_table: Leader policy table (numpy array)
        follower_q_values: Follower Q-values dictionary
        env_spec_dict: Environment specification dictionary
        num_follower_actions: Number of follower actions
        temperature: Temperature parameter for softmax policy

    Returns:
        Tuple of (leader_action, follower_action)

    """
    import numpy as np
    import torch

    # Get leader action from policy table
    state_int = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
    leader_probs = leader_policy_table[state_int]
    leader_act = int(np.random.choice(len(leader_probs), p=leader_probs))

    # Get follower action from Q-values
    # Compute soft value and sample from Max-Ent policy
    q_vals = []
    for b in range(num_follower_actions):
        state_key = (state_int,)
        leader_key = (leader_act,)
        follower_key = (b,)
        q_val = follower_q_values.get(state_key, {}).get(leader_key, {}).get(follower_key, 0.0)
        q_vals.append(q_val)

    # Sample from softmax policy
    q_tensor = torch.tensor(q_vals)
    probs = torch.softmax(q_tensor / temperature, dim=0).numpy()
    follower_act = int(np.random.choice(num_follower_actions, p=probs))

    return leader_act, follower_act


def _collect_single_episode(
    env_class_name,
    env_module_path,
    env_init_kwargs,
    leader_policy_table,
    follower_q_values,
    env_spec_dict,
    num_follower_actions,
    temperature,
):
    """Collect a single episode trajectory (for parallel processing).

    Args:
        env_class_name: Name of the environment class
        env_module_path: Module path of the environment class
        env_init_kwargs: Initialization arguments for the environment
        leader_policy_table: Leader policy table (numpy array)
        follower_q_values: Follower Q-values dictionary
        env_spec_dict: Environment specification dictionary
        num_follower_actions: Number of follower actions
        temperature: Temperature parameter for softmax policy

    Returns:
        Dictionary containing episode transitions

    """
    # Create environment
    env = _create_env_from_info(env_class_name, env_module_path, env_init_kwargs)
    obs, _ = env.reset()
    time_step = 0

    observations = []
    leader_actions = []
    follower_actions = []
    rewards = []
    leader_rewards = []
    next_observations = []
    terminals = []
    last_flags = []
    time_steps = []

    while True:
        # Get joint action
        leader_act, follower_act = _get_joint_action_from_state(
            obs,
            leader_policy_table,
            follower_q_values,
            env_spec_dict,
            num_follower_actions,
            temperature,
        )

        # Step environment
        env_step = env.step(leader_act, follower_act)

        # Get leader reward
        leader_reward = env_step.env_info.get("leader_reward", env_step.reward)

        # Store transition
        observations.append(obs)
        leader_actions.append(leader_act)
        follower_actions.append(follower_act)
        rewards.append(env_step.reward)
        leader_rewards.append(leader_reward)
        next_observations.append(env_step.observation)
        terminals.append(1 if env_step.terminal else 0)
        last_flags.append(1 if env_step.last else 0)
        time_steps.append(time_step)

        obs = env_step.observation
        time_step += 1

        if env_step.last:
            break

    return {
        "observation": observations,
        "leader_action": leader_actions,
        "action": follower_actions,
        "reward": rewards,
        "leader_reward": leader_rewards,
        "next_observation": next_observations,
        "terminal": terminals,
        "last": last_flags,
        "time_step": time_steps,
    }


class BilevelRL:
    """Bi-level Reinforcement Learning Algorithm.

    This algorithm solves the bi-level optimization problem by:
    1. Estimating follower's reward parameters using MDCE IRL
    2. Deriving follower's optimal policy using Soft Q-Learning
    3. Optimizing leader's policy to maximize its objective

    Args:
        env_spec: Global environment specification
        leader_policy: Leader's policy f_θ_L
        follower_policy: Follower's policy g (can be None, will be derived)
        feature_fn: Feature function for MDCE IRL (maps (s, a, b) to feature vector)
        discount_leader: Leader's discount factor γ_L
        discount_follower: Follower's discount factor γ_F
        learning_rate_leader: Learning rate for leader policy updates
        learning_rate_follower: Learning rate for follower Q-learning
        mdce_irl_config: Configuration for MDCE IRL
        soft_q_config: Configuration for Soft Q-Learning

    """

    def __init__(
        self,
        env_spec,
        leader_policy: Callable | None = None,
        follower_policy: Callable | None = None,
        feature_fn: Callable | None = None,
        discount_leader: float = 0.99,
        discount_follower: float = 0.99,
        learning_rate_leader: float = 1e-3,
        learning_rate_leader_critic: float | None = None,
        learning_rate_leader_actor: float | None = None,
        learning_rate_follower: float = 1e-3,
        mdce_irl_config: dict | None = None,
        soft_q_config: dict | None = None,
    ):
        self.env_spec = env_spec
        self.follower_policy = follower_policy
        self.feature_fn = feature_fn
        self.discount_leader = discount_leader
        self.discount_follower = discount_follower

        # Set separate learning rates, falling back to learning_rate_leader if not specified
        self.learning_rate_leader_actor = (
            learning_rate_leader_actor if learning_rate_leader_actor is not None else learning_rate_leader
        )
        self.learning_rate_leader_critic = (
            learning_rate_leader_critic if learning_rate_leader_critic is not None else learning_rate_leader
        )
        self.learning_rate_follower = learning_rate_follower
        self.leader_policy_table: np.ndarray | None = None
        self._use_tabular_policy = False

        # Initialize Leader Policy
        if leader_policy is None:

            def default_leader_policy(observation, deterministic=False):
                """Default uniform leader policy for initial exploration."""
                num_actions = env_spec.leader_action_space.n if hasattr(env_spec.leader_action_space, "n") else 2
                if deterministic:
                    return 0
                return int(np.random.choice(num_actions))

            leader_policy = default_leader_policy

        # Keep leader_policy as Callable for backward compatibility
        self.leader_policy = leader_policy
        self.leader_policy_obj = LeaderPolicy(
            env_spec=env_spec,
            policy=leader_policy,
            use_tabular=False,
        )

        # Initialize MDCE IRL
        mdce_config = mdce_irl_config or {}
        self.mdce_irl = MDCEIRL(
            feature_fn=feature_fn or self._default_feature_fn,
            discount=discount_follower,
            **mdce_config,
        )

        # Initialize Soft Q-Learning
        self.soft_q_learning: SoftQLearning | FollowerPolicyModel | None = None
        self.soft_q_config = soft_q_config or {}

        # === [追加] 真のフォロワーモデルを保持する変数を初期化 ===
        self.true_follower_model = None
        # =====================================================

        # Leader's Q-table
        self.leader_q_table: np.ndarray | None = None

        # Leader's tabular policy
        self.leader_policy_table: np.ndarray | None = None
        self._use_tabular_policy = False
        self.leader_logits_table: np.ndarray | None = None

        # Statistics
        self.stats = defaultdict(list)
        self._cached_true_q_values = None

    def _log_leader_state(self, iteration):
        """Log Leader's Q-values and Policy probabilities."""
        print(f"\n--- Leader State at Iteration {iteration} ---")

        if self.leader_q_table is None:
            return

        num_states = self.leader_q_table.shape[0]
        num_leader_actions = self.leader_q_table.shape[1]
        num_follower_actions = self.leader_q_table.shape[2]

        for s in range(num_states):
            print(f"State {s}:")
            # 1. 方策確率の表示
            if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                probs = self.leader_policy_obj.policy_table[s]
                probs_str = "  ".join([f"P(a={a})={p:.4f}" for a, p in enumerate(probs)])
                print(f"  Policy: {probs_str}")

            # 2. Q値の表示
            for a in range(num_leader_actions):
                # Q_L(s, a, b) の表示
                q_str = "  ".join([f"Q(b={b})={self.leader_q_table[s, a, b]:.4f}" for b in range(num_follower_actions)])

                # (参考) フォロワーの反応を考慮した期待Q値
                follower_probs = self._get_follower_action_probs(s, a)
                expected_q = np.sum(follower_probs * self.leader_q_table[s, a])

                print(f"  Action a={a}: E[Q]={expected_q:.4f} | Details: [{q_str}]")
        print("-" * 60)

    def _default_feature_fn(self, state, leader_action, follower_action):
        """Default feature function (identity).

        Args:
            state: State
            leader_action: Leader action
            follower_action: Follower action

        Returns:
            Feature vector

        """
        # Concatenate state, leader_action, follower_action
        if isinstance(state, np.ndarray):
            state_flat = state.flatten()
        else:
            state_flat = np.array([state])

        if isinstance(leader_action, np.ndarray):
            leader_flat = leader_action.flatten()
        else:
            leader_flat = np.array([leader_action])

        if isinstance(follower_action, np.ndarray):
            follower_flat = follower_action.flatten()
        else:
            follower_flat = np.array([follower_action])

        return np.concatenate([state_flat, leader_flat, follower_flat])

    def _initialize_leader_q_table(self, env):
        """Initialize leader's Q-table Q_L[s, a, b].

        Args:
            env: Environment instance

        """
        obs_space = self.env_spec.observation_space
        leader_action_space = self.env_spec.leader_action_space
        follower_action_space = self.env_spec.action_space

        # Get dimensions
        if hasattr(obs_space, "n"):
            num_states = obs_space.n
        else:
            num_states = 1000  # Fallback

        if hasattr(leader_action_space, "n"):
            num_leader_actions = leader_action_space.n
        else:
            num_leader_actions = 10  # Fallback

        if hasattr(follower_action_space, "n"):
            num_follower_actions = follower_action_space.n
        else:
            num_follower_actions = 10  # Fallback

        # Initialize Q-table: Q_L[state, leader_action, follower_action]
        self.leader_q_table = np.zeros(
            (num_states, num_leader_actions, num_follower_actions),
            dtype=np.float32,
        )

        # Initialize tabular policy in LeaderPolicy
        self.leader_policy_obj._initialize_tabular_policy()

        self.leader_policy_table = self.leader_policy_obj.policy_table
        self._use_tabular_policy = True

        # Update leader_policy to use tabular policy
        def tabular_leader_policy(observation, deterministic=False):
            """Tabular leader policy wrapper."""
            return self.leader_policy_obj.sample_action(observation, deterministic)

        self.leader_policy = tabular_leader_policy

    def _compute_softvi_q_values(self, env, max_iterations: int = 2000, tolerance: float = 1e-5, verbose: bool = True):
        """Compute optimal Q-values for the follower using Soft Value Iteration (SoftVI).

        This computes Q_F^*(s, a, b) using SoftVI, considering the current leader policy.
        Uses Soft Q-Learning Bellman equation (SoftVI):
            Q(s,a,b) = r(s,a,b) + γ * E_{a'~f_θ_L}[V^soft(s',a')]
            V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b') / temperature)

        Args:
            env: Environment instance
            max_iterations: Maximum iterations for value iteration
            tolerance: Convergence tolerance
            verbose: Whether to print progress

        Returns:
            Dictionary mapping (s, a, b) -> Q_F^*(s, a, b)

        """
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
        num_follower_actions = self.env_spec.action_space.n if hasattr(self.env_spec.action_space, "n") else 3

        # Get temperature from soft_q_config (default 1.0)
        temperature = self.soft_q_config.get("temperature", 1.0)

        # Initialize Q-values
        Q = {}
        for s in range(num_states):
            for a in range(num_leader_actions):
                for b in range(num_follower_actions):
                    Q[(s, a, b)] = 0.0

        # Soft Value Iteration
        for iteration in range(max_iterations):
            Q_old = Q.copy()
            max_delta = 0.0

            for s in range(num_states):
                for a in range(num_leader_actions):
                    for b in range(num_follower_actions):
                        # Get reward and next state from environment
                        env.reset(init_state=s)
                        env_step = env.step(a, b)

                        reward = env_step.reward
                        next_state = env_step.observation

                        if env_step.terminal:
                            # True terminal state: no next state value
                            Q[(s, a, b)] = reward
                        else:
                            # SoftVI Bellman update
                            next_state_int = int(next_state.item() if isinstance(next_state, np.ndarray) else next_state)

                            # Compute expected soft value: E_{a'~f_θ_L}[V^soft(s',a')]
                            v_next = 0.0
                            for a_next in range(num_leader_actions):
                                # Get leader policy probability for next state
                                if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                                    leader_prob = self.leader_policy_obj.policy_table[next_state_int, a_next]
                                else:
                                    # For callable policy, estimate probability (simplification)
                                    # In practice, we might need to sample or use a different approach
                                    leader_prob = 1.0 / num_leader_actions  # Uniform approximation

                                # Compute soft value: V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b')/temperature)
                                q_values = [
                                    Q_old.get((next_state_int, a_next, b_next), 0.0) for b_next in range(num_follower_actions)
                                ]
                                # Convert to torch for logsumexp
                                q_tensor = torch.tensor(q_values)
                                soft_v = temperature * torch.logsumexp(q_tensor / temperature, dim=0).item()
                                v_next += leader_prob * soft_v

                            Q[(s, a, b)] = reward + self.discount_follower * v_next

                        # Track convergence
                        delta = abs(Q[(s, a, b)] - Q_old.get((s, a, b), 0.0))
                        max_delta = max(max_delta, delta)

            if verbose and (iteration + 1) % 100 == 0:
                print(f"  SoftVI iteration {iteration + 1}: max_delta = {max_delta:.6f}")

            if max_delta < tolerance:
                if verbose:
                    print(f"\n✓ SoftVI converged in {iteration + 1} iterations (max_delta={max_delta:.6f})")
                break

        if iteration == max_iterations - 1:
            if verbose:
                if max_delta < 1e-4:
                    print(
                        f"\n✓ SoftVI nearly converged after {max_iterations} iterations (max_delta={max_delta:.6f} < 1e-4)",
                    )
                else:
                    print(
                        f"\n⚠ WARNING: SoftVI did NOT converge after {max_iterations} iterations (max_delta={max_delta:.6f})",
                    )

        return Q

    def _compute_true_q_values(self, env, max_iterations: int = 1000, tolerance: float = 1e-6):
        """Compute true Q-values for the follower using value iteration.

        This computes Q_F^*(s, a, b) assuming the follower knows its true reward function.

        Args:
            env: Environment instance
            max_iterations: Maximum iterations for value iteration
            tolerance: Convergence tolerance

        Returns:
            Dictionary mapping (s, a, b) -> Q_F^*(s, a, b)

        """
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
        num_follower_actions = self.env_spec.action_space.n if hasattr(self.env_spec.action_space, "n") else 3

        # Initialize Q-values
        Q_true = {}
        for s in range(num_states):
            for a in range(num_leader_actions):
                for b in range(num_follower_actions):
                    Q_true[(s, a, b)] = 0.0

        # Value iteration
        for iteration in range(max_iterations):
            Q_old = Q_true.copy()
            max_delta = 0.0

            for s in range(num_states):
                for a in range(num_leader_actions):
                    for b in range(num_follower_actions):
                        # Get reward and next state from environment
                        env.reset()
                        env.set_state(s)  # Set state using environment method
                        env_step = env.step(a, b)

                        reward = env_step.reward
                        next_state = env_step.observation

                        if env_step.last:
                            # Terminal state
                            Q_true[(s, a, b)] = reward
                        else:
                            # Bellman update for Soft Q-Learning: Q(s,a,b) = r + γ * E_{a'}[V^soft(s',a')]
                            # V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b') / temperature)
                            next_state_int = int(next_state.item() if isinstance(next_state, np.ndarray) else next_state)

                            # Get temperature from soft_q_config (default 1.0)
                            temperature = self.soft_q_config.get("temperature", 1.0)

                            # Assume uniform leader policy for next state value
                            v_next = 0.0
                            for a_next in range(num_leader_actions):
                                # Compute soft value: V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b')/temperature)
                                q_values = [
                                    Q_old.get((next_state_int, a_next, b_next), 0.0) for b_next in range(num_follower_actions)
                                ]
                                # Convert to torch for logsumexp
                                q_tensor = torch.tensor(q_values)
                                soft_v = temperature * torch.logsumexp(q_tensor / temperature, dim=0).item()
                                v_next += soft_v / num_leader_actions

                            Q_true[(s, a, b)] = reward + self.discount_follower * v_next

                        # Track convergence
                        delta = abs(Q_true[(s, a, b)] - Q_old.get((s, a, b), 0.0))
                        max_delta = max(max_delta, delta)

            if max_delta < tolerance:
                print(f"True Q-values converged in {iteration + 1} iterations (max_delta={max_delta:.6f})")
                break

        if iteration == max_iterations - 1:
            print(f"WARNING: True Q-values did NOT converge after {max_iterations} iterations (max_delta={max_delta:.6f})")

        return Q_true

    def _display_learned_rewards(self, reward_fn):
        """Display learned reward function r_F(s, a, b)."""
        print("\nLearned Reward Function: r_F(s, a, b)")
        print("-" * 80)

        # Get dimensions
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
        num_follower_actions = self.env_spec.action_space.n if hasattr(self.env_spec.action_space, "n") else 3

        for s in range(num_states):
            print(f"\nState {s}:")
            for a in range(num_leader_actions):
                rewards = []
                for b in range(num_follower_actions):
                    r_val = reward_fn(s, a, b)
                    rewards.append(f"b={b}: {r_val:6.3f}")
                print(f"  Leader action {a}: " + "  ".join(rewards))

    def _display_softq_stats(self, soft_q_learning, irl_iteration: int):
        """Display statistics of Q-values from Soft Q-Learning.

        Args:
            soft_q_learning: SoftQLearning instance
            irl_iteration: Current IRL iteration number

        """
        Q = soft_q_learning.Q
        num_states = self.env_spec.observation_space.n
        num_leader_actions = self.env_spec.leader_action_space.n
        num_follower_actions = self.env_spec.action_space.n  # FIXED: follower uses action_space

        # Extract all Q-values
        q_values = []
        for s in range(num_states):
            for a in range(num_leader_actions):
                for b in range(num_follower_actions):
                    q_val = soft_q_learning.get_q_value(s, a, b)
                    q_values.append(q_val)

        q_values = np.array(q_values)

        print(f"\n--- IRL iteration {irl_iteration}: Soft Q-Learning Q-value Statistics ---")
        print(f"  Min:  {np.min(q_values):8.4f}")
        print(f"  Max:  {np.max(q_values):8.4f}")
        print(f"  Mean: {np.mean(q_values):8.4f}")
        print(f"  Std:  {np.std(q_values):8.4f}")
        print(f"  Non-zero entries: {np.count_nonzero(q_values)}/{len(q_values)}")
        print()

    def _display_leader_q_values(self):
        """Display leader's Q-values in a readable format."""
        if self.leader_q_table is None:
            print("Leader Q-table not initialized")
            return

        num_states = self.leader_q_table.shape[0]
        num_leader_actions = self.leader_q_table.shape[1]
        num_follower_actions = self.leader_q_table.shape[2]

        print("\nLeader Q-Function: Q_L(s, a, b)")
        print("-" * 80)

        for s in range(num_states):
            print(f"\nState {s}:")
            for a in range(num_leader_actions):
                q_str = "  ".join([f"Q({s},{a},{b})={self.leader_q_table[s, a, b]:.4f}" for b in range(num_follower_actions)])
                print(f"  Leader action {a}: {q_str}")

        # Statistics
        q_values_flat = self.leader_q_table.flatten()
        print("\nQ-value Statistics (Leader):")
        print(f"  Min: {np.min(q_values_flat):.4f}")
        print(f"  Max: {np.max(q_values_flat):.4f}")
        print(f"  Mean: {np.mean(q_values_flat):.4f}")
        print(f"  Std:  {np.std(q_values_flat):.4f}")
        print(f"  Non-zero entries: {np.count_nonzero(q_values_flat)}/{q_values_flat.size}")

    def _display_follower_q_values(self, env=None, show_true_q: bool = True):
        """Display follower's learned Q-values in a readable format.

        Args:
            env: Environment instance (optional, for computing true Q-values)
            show_true_q: Whether to compute and show true optimal Q-values

        """
        if self.soft_q_learning is None or self.soft_q_learning.Q is None:
            print("Follower Q-values not available (not initialized)")
            return

        # Get dimensions from env_spec
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
        num_follower_actions = self.env_spec.action_space.n if hasattr(self.env_spec.action_space, "n") else 3

        # Compute true Q-values if requested (use cache if available)
        Q_true = None
        if show_true_q and env is not None:
            if self._cached_true_q_values is None:
                print("\nComputing true optimal Q-values (first time only)...")
                self._cached_true_q_values = self._compute_true_q_values(env)
                print()
            Q_true = self._cached_true_q_values

        print("\nFollower Q-Function: Q_F(s, a, b)")
        if Q_true:
            print("Format: Learned | True | Error")
        print("-" * 80)

        for s in range(num_states):
            print(f"\nState {s}:")
            for a in range(num_leader_actions):
                # Access Q-values using correct keys (must match how they were stored)
                if Q_true:
                    # Show learned, true, and error
                    q_comp_strs = []
                    for b in range(num_follower_actions):
                        q_learned = self.soft_q_learning.get_q_value(s, a, b)
                        q_true_val = Q_true[(s, a, b)]
                        error = abs(q_learned - q_true_val)
                        q_comp_strs.append(f"b={b}: {q_learned:6.3f} | {q_true_val:6.3f} | ε={error:6.3f}")
                    print(f"  Leader action {a}:")
                    for comp_str in q_comp_strs:
                        print(f"    {comp_str}")
                else:
                    # Show only learned Q-values
                    q_values_str = "  ".join(
                        [
                            f"Q({s},{a},{b})={self.soft_q_learning.get_q_value(s, a, b):7.4f}"
                            for b in range(num_follower_actions)
                        ],
                    )
                    print(f"  Leader action {a}: {q_values_str}")

                # Also show soft value V_F^soft(s, a)
                v_soft = self.soft_q_learning.compute_soft_value(s, a)
                print(f"    V_F^soft({s},{a}) = {v_soft:7.4f}")

                # Show follower policy probabilities
                probs = []
                for b in range(num_follower_actions):
                    q_val = self.soft_q_learning.get_q_value(s, a, b)
                    prob = np.exp((q_val - v_soft) / self.soft_q_learning.temperature)
                    probs.append(prob)
                probs_str = "  ".join([f"π_F({b}|{s},{a})={p:.4f}" for b, p in enumerate(probs)])
                print(f"    Follower policy: {probs_str}")

        # Show Q-value statistics
        # Extract all Q-values from nested defaultdict structure: Q[s][a][b]
        q_values = []
        q_true_values = []
        errors = []

        for s in range(num_states):
            for a in range(num_leader_actions):
                for b in range(num_follower_actions):
                    q_learned = self.soft_q_learning.get_q_value(s, a, b)
                    q_values.append(q_learned)

                    if Q_true:
                        q_true_val = Q_true[(s, a, b)]
                        q_true_values.append(q_true_val)
                        errors.append(abs(q_learned - q_true_val))

        if q_values:
            print("\nQ-value Statistics (Learned):")
            print(f"  Min: {min(q_values):7.4f}")
            print(f"  Max: {max(q_values):7.4f}")
            print(f"  Mean: {np.mean(q_values):7.4f}")
            print(f"  Std: {np.std(q_values):7.4f}")

        if Q_true and q_true_values:
            print("\nQ-value Statistics (True Optimal):")
            print(f"  Min: {min(q_true_values):7.4f}")
            print(f"  Max: {max(q_true_values):7.4f}")
            print(f"  Mean: {np.mean(q_true_values):7.4f}")
            print(f"  Std: {np.std(q_true_values):7.4f}")

            print("\nApproximation Error:")
            print(f"  Mean Absolute Error: {np.mean(errors):7.4f}")
            print(f"  Max Error: {max(errors):7.4f}")
            print(f"  RMSE: {np.sqrt(np.mean([e**2 for e in errors])):7.4f}")

    def estimate_follower_reward(
        self,
        trajectories: list[dict],
        env,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Estimate follower's reward parameters using MDCE IRL.

        Algorithm Step 2: MDCE IRL (Leader's reward estimation)
        Purpose: リーダーはフォロワーの報酬を観測できないため、
                 収集した軌跡（フォロワーの行動）から報酬関数を逆推定する

        For each IRL iteration:
        1. Reconstruct Follower Q-Function using Soft Q-Learning (with estimated reward w)
        2. Derive current policy g_{w^n} from Q̂_F
        3. Compute current policy FEV φ̄_{g_{w^n}}^{γ_F} (by running episodes)
        4. Compare with expert FEV φ̄_expert^{γ_F}
        5. If converged: Return w^n, g_{w^n}
        6. Update: w^n ← w^n + δ(n)(φ̄_expert^{γ_F} - φ̄_{g_{w^n}}^{γ_F})

        Args:
            trajectories: List of trajectories (observed follower behaviors)
                          Note: Only used for likelihood calculation logging.
            env: Environment instance
            verbose: Whether to print progress

        Returns:
            Estimated reward parameters w (for leader's understanding of follower)

        """
        # === [修正: Apples-to-Apples化] ===
        # エキスパートFEVも、現在の方策FEVと同じ「モンテカルロ法」で計算します。
        # Step 0で学習した「真の最適反応 (self.soft_q_learning)」を使用します。

        # 1. エキスパート方策のラッパー関数を定義
        def expert_policy_fn(state, leader_action, follower_action=None):
            """Expert policy wrapper for compute_policy_fev."""
            if follower_action is None:
                # Step 0で学習済みのモデルからサンプリング
                if self.soft_q_learning is not None:
                    return self.soft_q_learning.sample_action(state, leader_action)
                # フォールバック (通常は発生しない)
                return self.env_spec.action_space.sample()
            return 0.0

        if verbose:
            print("Computing Expert FEV using Monte Carlo simulation (Apples-to-Apples)...")

        # 2. 現在の方策FEVと同じ関数(compute_policy_fev)を使ってExpert FEVを計算
        expert_fev = self.mdce_irl.compute_policy_fev(
            policy=expert_policy_fn,
            leader_policy=self.leader_policy,
            env=env,
        )

        if verbose:
            print(f"Expert FEV: {expert_fev}")

        # Initialize w if not already initialized
        if self.mdce_irl.w is None:
            feature_dim = expert_fev.shape[0]
            self.mdce_irl.w = torch.randn(feature_dim, requires_grad=True) * 0.01

        # MDCE IRL iterations
        for irl_iteration in range(self.mdce_irl.max_iterations):
            # Step 2.1: Reconstruct Follower Q-Function using Soft Q-Learning
            # Create reward function from current w
            current_w = self.mdce_irl.w.detach().clone()

            def make_reward_fn(w_val):
                """Create reward function with captured w value."""

                def reward_fn(state, leader_action, follower_action):
                    """Compute reward r_F(s, a, b) = w^T φ(s, a, b)."""
                    phi = self.mdce_irl.feature_fn(state, leader_action, follower_action)
                    if isinstance(phi, np.ndarray):
                        phi = torch.from_numpy(phi).float()
                    elif not isinstance(phi, torch.Tensor):
                        phi = torch.tensor(phi, dtype=torch.float32)
                    return torch.dot(w_val, phi).item()

                return reward_fn

            reward_fn = make_reward_fn(current_w)

            # Initialize/Reset Soft Q-Learning with current reward function
            soft_q_config = self.soft_q_config.copy()
            soft_q_config.pop("learning_rate", None)

            # Create leader policy function that returns probability distribution
            def leader_policy_probs(state):
                """Get leader policy probability distribution."""
                if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                    state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
                    return self.leader_policy_obj.policy_table[state_int].tolist()
                # For callable policy, return uniform distribution as approximation
                num_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
                return [1.0 / num_actions] * num_actions

            temp_soft_q_learning = SoftQLearning(
                env_spec=self.env_spec,
                reward_fn=reward_fn,
                leader_policy=leader_policy_probs,
                discount=self.discount_follower,
                learning_rate=self.learning_rate_follower,
                **soft_q_config,
            )

            # Train Q-function using Soft Q-Learning
            n_soft_q_iterations = self.mdce_irl.n_soft_q_iterations
            soft_q_start_time = time.time()

            # Save initial Q-values to track convergence
            initial_q_values = {}
            for s_key in temp_soft_q_learning.Q:
                initial_q_values[s_key] = {}
                for a_key in temp_soft_q_learning.Q[s_key]:
                    initial_q_values[s_key][a_key] = {}
                    for b_key in temp_soft_q_learning.Q[s_key][a_key]:
                        initial_q_values[s_key][a_key][b_key] = temp_soft_q_learning.Q[s_key][a_key][b_key]

            # Track Q-value updates
            q_change_history = []

            for episode_idx in range(n_soft_q_iterations):
                obs, _ = env.reset()
                while True:
                    # Sample leader action
                    leader_act = self.leader_policy_obj.sample_action(obs, deterministic=False)

                    # Sample follower action
                    follower_act = temp_soft_q_learning.sample_action(obs, leader_act)

                    # Step environment
                    env_step = env.step(leader_act, follower_act)

                    # CRITICAL: Use estimated reward function (not environment reward!)
                    reward = reward_fn(obs, leader_act, follower_act)

                    # Update Q-function with estimated reward
                    temp_soft_q_learning.update(
                        obs,
                        leader_act,
                        follower_act,
                        reward,
                        env_step.observation,
                        env_step.last,
                    )

                    obs = env_step.observation
                    if env_step.last:
                        break

                # Track Q-value changes every 10 episodes
                if (episode_idx + 1) % 10 == 0:
                    max_q_change = 0.0
                    for s_key in temp_soft_q_learning.Q:
                        for a_key in temp_soft_q_learning.Q[s_key]:
                            for b_key in temp_soft_q_learning.Q[s_key][a_key]:
                                q_current = temp_soft_q_learning.Q[s_key][a_key][b_key]
                                q_initial = initial_q_values.get(s_key, {}).get(a_key, {}).get(b_key, 0.0)
                                q_change = abs(q_current - q_initial)
                                max_q_change = max(max_q_change, q_change)
                    q_change_history.append((episode_idx + 1, max_q_change))

            soft_q_time = time.time() - soft_q_start_time

            # Compute final max Q-value change
            final_max_q_change = 0.0
            for s_key in temp_soft_q_learning.Q:
                for a_key in temp_soft_q_learning.Q[s_key]:
                    for b_key in temp_soft_q_learning.Q[s_key][a_key]:
                        q_current = temp_soft_q_learning.Q[s_key][a_key][b_key]
                        q_initial = initial_q_values.get(s_key, {}).get(a_key, {}).get(b_key, 0.0)
                        q_change = abs(q_current - q_initial)
                        final_max_q_change = max(final_max_q_change, q_change)

            # Step 2.2: Derive current policy g_{w^n} from Q̂_F
            def make_policy_fn(sq_instance):
                """Create policy function with captured Soft Q-Learning instance."""

                def policy_fn(state, leader_action, follower_action=None):
                    if follower_action is None:
                        # Sample action (for compute_policy_fev)
                        return sq_instance.sample_action(state, leader_action)
                    # Return log probability
                    q_val = sq_instance.get_q_value(state, leader_action, follower_action)
                    soft_value = sq_instance.compute_soft_value(state, leader_action)
                    log_prob = (q_val - soft_value) / sq_instance.temperature
                    return log_prob

                return policy_fn

            policy_fn = make_policy_fn(temp_soft_q_learning)

            # Display debug info
            if verbose and (irl_iteration % 10 == 0 or irl_iteration < 5):
                self._display_softq_stats(temp_soft_q_learning, irl_iteration)
                # ... (rewards display omitted for brevity) ...

            # Step 2.3: Compute current policy FEV φ̄_{g_{w^n}}^{γ_F}
            fev_start_time = time.time()
            policy_fev = self.mdce_irl.compute_policy_fev(
                policy_fn,
                self.leader_policy,
                env,
            )
            fev_time = time.time() - fev_start_time

            # Step 2.4: Compare with expert FEV and update w
            # Compute gradient: ∇L(w) ∝ φ̄_expert^γ - φ̄_{g_w}^γ
            gradient = expert_fev - policy_fev

            if verbose and irl_iteration % 10 == 0:
                print(f"\n--- IRL iteration {irl_iteration}: Timing ---")
                print(f"  Soft Q-Learning: {soft_q_time:.2f}s ({n_soft_q_iterations} episodes)")
                print(f"  Policy FEV computation: {fev_time:.2f}s ({self.mdce_irl.n_monte_carlo_samples} samples)")
                print(f"  Total per iteration: {soft_q_time + fev_time:.2f}s")

            # Update w with learning rate schedule
            learning_rate_schedule = 0.1 / (1.0 + 0.01 * irl_iteration)
            self.mdce_irl.w = self.mdce_irl.w + learning_rate_schedule * gradient

            # Step 2.5: Check convergence using FEM constraint
            epsilon = 1e-8
            mask = torch.abs(expert_fev) > epsilon  # True for non-zero expert features
            relative_errors = torch.abs(gradient) / (torch.abs(expert_fev) + epsilon)
            absolute_errors = torch.abs(gradient)
            errors = torch.where(mask, relative_errors, absolute_errors)
            delta_fem = torch.max(errors).item()
            max_error_idx = torch.argmax(errors).item()

            # CRITICAL: Update self.soft_q_learning with estimated reward-based policy
            self.soft_q_learning = temp_soft_q_learning

            if delta_fem < self.mdce_irl.tolerance:
                if verbose:
                    print(f"MDCE IRL converged at iteration {irl_iteration}: δ_FEM={delta_fem:.6f}")

                    # === [追加] 収束時の詳細ログ出力 ===
                    print("\n" + "=" * 80)
                    print("MDCE IRL CONVERGED - FINAL STATISTICS")
                    print("=" * 80)

                    # 1. 報酬関数の表示 (現在学習に使った reward_fn を使用)
                    self._display_learned_rewards(reward_fn)

                    # 2. Q値と方策確率の表示 (既存メソッドを再利用)
                    # show_true_q=False にして、学習された値のみを表示
                    self._display_follower_q_values(env=env, show_true_q=False)

                    print("=" * 80 + "\n")
                    # ====================================

                break

            # Log IRL metrics
            gradient_norm = torch.norm(gradient).item()
            self.stats["irl_gradient_norm"].append(gradient_norm)
            self.stats.setdefault("irl_delta_fem", []).append(delta_fem)

            if verbose and irl_iteration % 10 == 0:
                # Likelihood calculation (for logging only)
                likelihood = self.mdce_irl.compute_likelihood(trajectories, policy_fn)
                self.stats["irl_likelihood"].append((irl_iteration, likelihood))

                print(
                    f"IRL iteration {irl_iteration}: δ_FEM={delta_fem:.6f}, ||gradient||={gradient_norm:.6f}, likelihood={likelihood:.6f}",
                )
                print(f"  Expert FEV:  {expert_fev.detach().cpu().numpy()}")
                print(f"  Policy FEV:  {policy_fev.detach().cpu().numpy()}")

                w_norm = torch.norm(self.mdce_irl.w).item()
                w_np = self.mdce_irl.w.detach().cpu().numpy()
                print(f"  Reward params w (||w||={w_norm:.4f}):")
                print(f"    w = {w_np}")

        return self.mdce_irl.w

    def _generate_expert_trajectories(self, env, n_trajectories: int = 50, verbose: bool = False):
        """Generate expert demonstration trajectories using current leader policy.

        Collects trajectories where:
        - Leader uses current policy (self.leader_policy_obj)
        - Follower uses optimal response (env.get_opt_follower_act_array())

        Args:
            env: Environment instance
            n_trajectories: Number of trajectories to generate
            verbose: Whether to print progress

        Returns:
            List of trajectory dictionaries for MDCE IRL

        """
        trajectories = []

        for traj_idx in range(n_trajectories):
            obs, _ = env.reset()
            traj = {
                "observations": [],
                "leader_actions": [],
                "follower_actions": [],
                "rewards": [],
            }

            while True:
                # Leader uses current policy
                leader_act = self.leader_policy_obj.sample_action(obs, deterministic=False)

                # Follower uses optimal response
                state = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
                follower_act = env.get_opt_follower_act_array()[leader_act, state]

                traj["observations"].append(obs)
                traj["leader_actions"].append(leader_act)
                traj["follower_actions"].append(follower_act)

                env_step = env.step(leader_act, follower_act)
                traj["rewards"].append(env_step.reward)

                obs = env_step.observation

                if env_step.last:
                    break

            trajectories.append(traj)

            if verbose and (traj_idx + 1) % 10000 == 0:
                print(f"  Generated {traj_idx + 1}/{n_trajectories} expert trajectories")

        return trajectories

    def _get_follower_actions(self):
        """Get all possible follower actions."""
        action_space = self.env_spec.follower_policy_env_spec.action_space
        if hasattr(action_space, "n"):
            return list(range(action_space.n))
            return [action_space.sample() for _ in range(10)]

    def _get_follower_action_probs(self, state, leader_action):
        """Get follower action probabilities g_{w^n}(b|s, a) for given state and leader action.

        Args:
            state: Current state
            leader_action: Leader action

        Returns:
            Array of probabilities for each follower action

        """
        if self.soft_q_learning is None:
            # If Soft Q-Learning not initialized, return uniform distribution
            num_follower_actions = self.leader_q_table.shape[2]
            return np.ones(num_follower_actions) / num_follower_actions

        # Get Q-values for all follower actions
        num_follower_actions = self.leader_q_table.shape[2]
        q_values = np.zeros(num_follower_actions)

        for follower_action in range(num_follower_actions):
            q_val = self.soft_q_learning.get_q_value(state, leader_action, follower_action)
            q_values[follower_action] = q_val

        # Compute soft value: V_F^soft(s, a) = log Σ_b exp(Q_F^soft(s, a, b))
        soft_value = self.soft_q_learning.compute_soft_value(state, leader_action)

        # Compute probabilities: g(b|s, a) = exp(Q_F^soft(s, a, b) - V_F^soft(s, a))
        # Using temperature from Soft Q-Learning
        temperature = self.soft_q_learning.temperature
        log_probs = (q_values - soft_value) / temperature
        probs = np.exp(log_probs)

        # Normalize to ensure valid probability distribution
        probs = probs / np.sum(probs)

        return probs

    def derive_follower_policy(
        self,
        env,
        n_iterations: int = 1000,
        verbose: bool = True,
    ):
        """Derive follower's optimal policy using Soft Q-Learning.

        Args:
            env: Environment instance
            n_iterations: Number of Q-learning iterations
            verbose: Whether to print progress

        """
        # Create reward function from estimated parameters
        w = self.mdce_irl.get_reward_params()

        def reward_fn(state, leader_action, follower_action):
            """Compute reward r_F(s, a, b) = w^T φ(s, a, b)."""
            phi = self.mdce_irl.feature_fn(state, leader_action, follower_action)
            if isinstance(phi, np.ndarray):
                phi = torch.from_numpy(phi).float()
            return torch.dot(w, phi).item()

        # Initialize Soft Q-Learning
        # Remove learning_rate from soft_q_config if present to avoid duplicate
        soft_q_config = self.soft_q_config.copy()
        soft_q_config.pop("learning_rate", None)

        # Create leader policy function that returns probability distribution
        def leader_policy_probs(state):
            """Get leader policy probability distribution."""
            if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
                return self.leader_policy_obj.policy_table[state_int].tolist()
            # For callable policy, return uniform distribution as approximation
            num_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
            return [1.0 / num_actions] * num_actions

        self.soft_q_learning = SoftQLearning(
            env_spec=self.env_spec,
            reward_fn=reward_fn,
            leader_policy=leader_policy_probs,
            discount=self.discount_follower,
            learning_rate=self.learning_rate_follower,
            **soft_q_config,
        )

        # Train Q-function
        # Option 1: Fixed number of iterations (current implementation)
        for iteration in range(n_iterations):
            obs, _ = env.reset()
            total_reward = 0.0

            while True:
                # Sample leader action
                leader_act = self.leader_policy_obj.sample_action(obs, deterministic=False)

                # Sample follower action (exploration)
                follower_act = self.soft_q_learning.sample_action(obs, leader_act)

                # Step environment
                env_step = env.step(leader_act, follower_act)
                reward = env_step.reward

                # Update Q-function
                self.soft_q_learning.update(
                    obs,
                    leader_act,
                    follower_act,
                    reward,
                    env_step.observation,
                    env_step.last,
                )

                total_reward += reward
                obs = env_step.observation

                if env_step.last:
                    break

            if verbose and iteration % 100 == 0:
                print(f"Soft Q-Learning iteration {iteration}: reward={total_reward:.4f}")

        # Create follower policy from learned Q-function
        # If deterministic, choose action with maximum Q-value
        # Otherwise, sample from Max-Ent policy
        def follower_policy_fn(obs, leader_act, deterministic=False):
            if deterministic:
                # Choose action with maximum Q-value
                num_follower_actions = self.env_spec.action_space.n
                best_action = 0
                best_q = float("-inf")
                for b in range(num_follower_actions):
                    q_val = self.soft_q_learning.get_q_value(obs, leader_act, b)
                    if q_val > best_q:
                        best_q = q_val
                        best_action = b
                return np.array(best_action, dtype=np.int32)
            # Sample from Max-Ent policy
            return self.soft_q_learning.sample_action(obs, leader_act)

        self.follower_policy = follower_policy_fn

        # [修正] 既存の compute_leader_objective を置き換え

    def evaluate_leader(
        self,
        env,
        follower_policy_fn: Callable,  # 対戦相手（フォロワー）の方策関数
        n_episodes: int = 10,
    ) -> dict[str, float]:
        """Evaluate leader's performance against a specific follower policy.

        Args:
            env: Environment instance
            follower_policy_fn: Policy function to use for the follower
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with mean and std of returns

        """
        # 一時的にクラスのフォロワー方策を、指定されたもの（True or Learned）に差し替える
        # get_joint_action が self.follower_policy を参照するため
        original_policy = self.follower_policy
        self.follower_policy = follower_policy_fn

        total_returns = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_return = 0.0

            while True:
                # 指定されたフォロワーを使って行動決定
                # ※ deterministic=False (確率的) で評価するのが基本
                leader_act, follower_act = self.get_joint_action(obs, deterministic=False)

                env_step = env.step(leader_act, follower_act)

                # リーダーの累積報酬 (Raw Score) を集計
                leader_reward = env_step.env_info.get("leader_reward", env_step.reward)
                episode_return += leader_reward

                obs = env_step.observation
                if env_step.last:
                    break

            total_returns.append(episode_return)

        # 元に戻す
        self.follower_policy = original_policy

        return {
            "mean": np.mean(total_returns),
            "std": np.std(total_returns),
        }

    def compute_leader_objective(
        self,
        env,
        n_episodes: int = 100,
    ) -> float:
        """Compute leader's objective J_L(f_{θ_L}, g^*).

        J_L = E^{f_{θ_L}, g^*} [Σ_t γ_L^t r_L(s_t, a_t, b_t)]

        Args:
            env: Environment instance
            n_episodes: Number of episodes to evaluate

        Returns:
            Average discounted return

        """
        total_returns = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            discount_factor = 1.0

            while True:
                # Get joint action
                leader_act, follower_act = self.get_joint_action(obs)

                # Step environment
                env_step = env.step(leader_act, follower_act)

                # Get leader reward (leader_reward from env_info)
                leader_reward = env_step.env_info.get("leader_reward", env_step.reward)

                episode_return += discount_factor * leader_reward
                discount_factor *= self.discount_leader

                obs = env_step.observation

                if env_step.last:
                    break

            total_returns.append(episode_return)

        return np.mean(total_returns)

    def get_joint_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get joint action from joint policy.

        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policies

        Returns:
            Tuple of (leader_action, follower_action)

        """
        if self.follower_policy is None:
            raise ValueError("Follower policy not derived. Call derive_follower_policy() first.")

        # Get leader action
        leader_act = self.leader_policy_obj.sample_action(observation, deterministic=deterministic)

        # Convert to numpy arrays for get_inputs_for
        obs_array = np.array([observation]) if not isinstance(observation, np.ndarray) else np.array([observation])
        leader_act_array = np.array([leader_act]) if not isinstance(leader_act, np.ndarray) else np.array([leader_act])

        # Get follower action
        follower_obs = self.env_spec.get_inputs_for(
            "follower",
            "policy",
            obs=obs_array,
            leader_act=leader_act_array,
        )

        if isinstance(follower_obs, torch.Tensor):
            follower_obs_np = follower_obs[0].cpu().numpy()
        else:
            follower_obs_np = follower_obs[0] if isinstance(follower_obs, list) else follower_obs

        follower_act = self.follower_policy(
            follower_obs_np,
            leader_act,
            deterministic=deterministic,
        )

        return leader_act, follower_act

    def _update_leader_critic(self, replay_buffer, n_updates: int = 100):
        """Update leader's Q-table using Q-Learning.

        Args:
            replay_buffer: Replay buffer containing transitions
            n_updates: Number of Q-learning updates

        """
        if self.leader_q_table is None:
            raise ValueError("Leader Q-table not initialized. Call _initialize_leader_q_table() first.")

        for _ in range(n_updates):
            # Sample batch from replay buffer
            samples = replay_buffer.sample_transitions(
                batch_size=64,
                replace=True,
                discount=False,
                with_subsequence=False,
            )

            obs = samples["observation"]
            leader_acts = samples["leader_action"]
            follower_acts = samples["action"]
            rewards = samples["leader_reward"]  # Leader's reward
            next_obs = samples["next_observation"]
            terminals = samples["terminal"]

            # Convert to numpy arrays if needed
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            if isinstance(leader_acts, torch.Tensor):
                leader_acts = leader_acts.cpu().numpy()
            if isinstance(follower_acts, torch.Tensor):
                follower_acts = follower_acts.cpu().numpy()
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.cpu().numpy()
            if isinstance(terminals, torch.Tensor):
                terminals = terminals.cpu().numpy()

            # Flatten arrays
            obs = obs.flatten().astype(int)
            leader_acts = leader_acts.flatten().astype(int)
            follower_acts = follower_acts.flatten().astype(int)
            rewards = rewards.flatten()
            next_obs = next_obs.flatten().astype(int)
            terminals = terminals.flatten().astype(bool)

            # Compute target Q-values: V_L(s_{t+1}) = E_{a'~f_θ_L, b'~g_w}[Q_L[s_{t+1}, a', b']]
            # Algorithm requires exact expectation, not Monte Carlo sampling
            target_q_values = np.zeros(len(obs))
            for i in range(len(obs)):
                if terminals[i]:
                    target_q_values[i] = rewards[i]
                else:
                    # Compute V_L(s_{t+1}) = E_{a'~f_θ_L^n, b'~g_{w^n}}[Q_L[s_{t+1}, a', b']]
                    next_state = next_obs[i]
                    v_next = 0.0

                    # Get action spaces
                    num_leader_actions = self.leader_q_table.shape[1]
                    num_follower_actions = self.leader_q_table.shape[2]

                    # Compute expectation over all (a', b') pairs
                    for leader_act_next in range(num_leader_actions):
                        # Probability of leader action: f_θ_L^n(a'|s_{t+1})
                        if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                            leader_prob = self.leader_policy_obj.policy_table[next_state, leader_act_next]
                        else:
                            # For callable policy, we need to estimate probability
                            # This is a simplification - in practice, we might need to sample
                            leader_prob = 1.0 / num_leader_actions  # Uniform approximation

                        # Get follower action probabilities: g_{w^n}(b'|s_{t+1}, a')
                        # We need to get follower policy probabilities for (next_state, leader_act_next)
                        follower_probs = self._get_follower_action_probs(next_state, leader_act_next)

                        for follower_act_next in range(num_follower_actions):
                            follower_prob = follower_probs[follower_act_next]
                            q_val = self.leader_q_table[next_state, leader_act_next, follower_act_next]
                            v_next += leader_prob * follower_prob * q_val

                    target_q_values[i] = rewards[i] + self.discount_leader * v_next

                # Q-Learning update
                state = obs[i]
                leader_act = leader_acts[i]
                follower_act = follower_acts[i]

                current_q = self.leader_q_table[state, leader_act, follower_act]
                self.leader_q_table[state, leader_act, follower_act] = current_q + self.learning_rate_leader_critic * (
                    target_q_values[i] - current_q
                )

    def _estimate_leader_gradient(self, replay_buffer, batch_size: int = 64):
        """Estimate leader's policy gradient using Equation 5.20.

        Implements the full gradient formula (Equation 5.20):
        ∇_{θ_L} J_L(θ_L) = [1/(1-γ_L)] E_{d_{γ_L}} [
            ∇_{θ_L} log f_{θ_L}(a|s) Q_L(s, a, b)
            + [1/(β_F(1-γ_F))] (Q_L(s,a,b) - E_{b~g}[Q_L(s,a,b)])
            * E_{d_{γ_F}} [ ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) V_F(ṡ,ȧ) | s,a,b ]
        ]

        For tabular policy π_L[s, a], we approximate this as:
        ∇_{π_L[s, a]} J_L ≈ E[Q_L(s, a, b) | s, a] / (1 - γ_L)
        + (Follower influence term) / (β_F(1-γ_F))

        Args:
            replay_buffer: Replay buffer containing transitions
            batch_size: Batch size for gradient estimation

        Returns:
            Dictionary with gradient information and policy gradients

        """
        if (
            self.leader_q_table is None
            or not self.leader_policy_obj.use_tabular
            or self.leader_policy_obj.policy_table is None
        ):
            return {"gradient_norm": 0.0, "estimated_gradient": None, "policy_gradients": None}

        # Sample batch with subsequences for computing conditional expectation
        samples = replay_buffer.sample_transitions(
            batch_size=batch_size,
            replace=True,
            discount=False,
            with_subsequence=True,  # Need subsequences for Equation 5.20
        )

        obs = samples["observation"]
        leader_acts = samples["leader_action"]
        follower_acts = samples["action"]
        subsequences = samples["subsequence"]

        # Convert to numpy arrays if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(leader_acts, torch.Tensor):
            leader_acts = leader_acts.cpu().numpy()
        if isinstance(follower_acts, torch.Tensor):
            follower_acts = follower_acts.cpu().numpy()

        # Flatten arrays
        obs = obs.flatten().astype(int)
        leader_acts = leader_acts.flatten().astype(int)
        follower_acts = follower_acts.flatten().astype(int)

        # Get Q-values from Q-table: Q_L(s, a, b)
        q_values = np.array([self.leader_q_table[obs[i], leader_acts[i], follower_acts[i]] for i in range(len(obs))])

        # First term: Standard policy gradient
        # ∇_{θ_L} log f_{θ_L}(a|s) Q_L(s, a, b)
        # For tabular policy: ∇_{π_L[s, a]} log π_L(a|s) = 1/π_L(a|s) if action matches, else 0
        first_term_gradients = np.zeros_like(self.leader_policy_obj.policy_table)

        # Second term: Follower influence term
        # [1/(β_F(1-γ_F))] * (Q_L(s,a,b) - E_{b~g}[Q_L(s,a,b)])
        # * E_{d_{γ_F}} [ ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) V_F(ṡ,ȧ) | s,a,b ]
        second_term_gradients = np.zeros_like(self.leader_policy_obj.policy_table)

        # Get β_F (temperature parameter from Soft Q-Learning)
        beta_F = 1.0  # Default temperature
        if self.soft_q_learning is not None:
            beta_F = self.soft_q_learning.temperature

        # Accumulate gradients per (state, action) pair
        for i in range(len(obs)):
            state = obs[i]
            action = leader_acts[i]
            q_val = q_values[i]

            # First term: Standard policy gradient
            # For tabular policy, gradient w.r.t. π_L[s, a] is Q-value
            first_term_gradients[state, action] += q_val

            # Second term: Follower influence term
            # Compute E_{b \sim g_{θ_L}^*(\cdot|s,a)}[Q_L(s, a, b)]
            follower_probs = self._get_follower_action_probs(state, action)
            num_follower_actions = len(follower_probs)
            expected_q_follower = 0.0
            for b in range(num_follower_actions):
                expected_q_follower += follower_probs[b] * self.leader_q_table[state, action, b]

            # Benefit: Q_L(s, a, b) - E_{b \sim g_{θ_L}^*(\cdot|s,a)}[Q_L(s, a, b)]
            benefit = q_val - expected_q_follower

            # Influence: E_{d_{γ_F}} [ ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) V_F(ṡ, ȧ) | s, a, b ]
            # This is computed using the subsequence starting from (s, a, b)
            influence_gradients = np.zeros_like(self.leader_policy_obj.policy_table)

            if self.soft_q_learning is not None and subsequences is not None:
                # Get subsequence for this sample
                subseq_obs = subsequences["observation"][i]
                subseq_leader_acts = subsequences["leader_action"][i]

                # Convert to numpy arrays if needed
                if isinstance(subseq_obs, torch.Tensor):
                    subseq_obs = subseq_obs.cpu().numpy()
                if isinstance(subseq_leader_acts, torch.Tensor):
                    subseq_leader_acts = subseq_leader_acts.cpu().numpy()

                subseq_obs = subseq_obs.flatten().astype(int)
                subseq_leader_acts = subseq_leader_acts.flatten().astype(int)

                # Compute conditional expectation over subsequence
                # E_{d_{γ_F}} [ ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) V_F^soft(ṡ, ȧ) | s, a, b ]
                for t, (s_dot, a_dot) in enumerate(zip(subseq_obs, subseq_leader_acts, strict=True)):
                    # Compute V_F^soft(ṡ, ȧ)
                    v_f_soft = self.soft_q_learning.compute_soft_value(s_dot, a_dot)

                    # For tabular policy: ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) is 1/π_L(ȧ|ṡ)
                    # So gradient w.r.t. π_L[ṡ, ȧ] is: V_F^soft(ṡ, ȧ) / π_L(ȧ|ṡ)
                    # We accumulate V_F^soft(ṡ, ȧ) and normalize later

                    # Discount by γ_F^t
                    discount_factor = self.discount_follower**t
                    influence_gradients[s_dot, a_dot] += discount_factor * v_f_soft

                # Normalize by (1 - γ_F) for discounted state distribution expectation
                # This is part of the d_{γ_F} normalization
                if len(subseq_obs) > 0:
                    influence_gradients = influence_gradients / (1.0 - self.discount_follower)

            # Second term contribution
            # benefit * influence_gradients / β_F
            # Note: We already divided by (1-γ_F) in influence_gradients computation
            second_term_gradients += benefit * influence_gradients / beta_F

        # Normalize by number of samples per (state, action)
        state_action_counts = np.zeros_like(self.leader_policy_obj.policy_table)
        for i in range(len(obs)):
            state = obs[i]
            action = leader_acts[i]
            state_action_counts[state, action] += 1

        # Avoid division by zero
        state_action_counts = np.maximum(state_action_counts, 1.0)
        first_term_gradients = first_term_gradients / state_action_counts
        second_term_gradients = second_term_gradients / state_action_counts

        # Normalize each term by (1 - γ_L) separately for analysis
        first_term_normalized = first_term_gradients / (1.0 - self.discount_leader)
        second_term_normalized = second_term_gradients / (1.0 - self.discount_leader)

        # Combine terms
        policy_gradients = first_term_normalized + second_term_normalized

        # Compute norms for each term
        first_term_norm = np.linalg.norm(first_term_normalized)
        second_term_norm = np.linalg.norm(second_term_normalized)
        gradient_norm = np.linalg.norm(policy_gradients)

        # Compute advantages for statistics
        advantages = q_values.copy()
        for i in range(len(obs)):
            state = obs[i]
            follower_probs = self._get_follower_action_probs(state, leader_acts[i])
            baseline = np.sum(follower_probs * self.leader_q_table[state, leader_acts[i], :])
            advantages[i] = q_values[i] - baseline

        return {
            "gradient_norm": gradient_norm,
            "first_term_norm": first_term_norm,
            "second_term_norm": second_term_norm,
            "estimated_gradient": np.mean(advantages),
            "mean_q_value": np.mean(q_values),
            "mean_advantage": np.mean(advantages),
            "policy_gradients": policy_gradients,
        }

    def _update_leader_actor(self, gradient_info):
        """Update leader's policy using estimated gradient."""
        if gradient_info is None:
            return

        # Log gradient information (既存のログ処理)
        self.stats.setdefault("gradient_norm", []).append(
            gradient_info.get("gradient_norm", 0.0),
        )
        self.stats.setdefault("mean_q_value", []).append(
            gradient_info.get("mean_q_value", 0.0),
        )
        self.stats.setdefault("mean_advantage", []).append(
            gradient_info.get("mean_advantage", 0.0),
        )

        # Update tabular policy ivf available
        if self._use_tabular_policy and self.leader_policy_table is not None:
            policy_gradients = gradient_info.get("policy_gradients")

            if policy_gradients is not None:
                # ※ ここは元のまま（確率の直接更新）
                self.leader_policy_table = self.leader_policy_table + self.learning_rate_leader_actor * policy_gradients

                # Normalize to ensure valid probability distribution
                # Add small epsilon to avoid numerical issues
                self.leader_policy_table = np.maximum(self.leader_policy_table, 1e-8)

                # Normalize per state
                for state in range(self.leader_policy_table.shape[0]):
                    state_sum = np.sum(self.leader_policy_table[state])
                    if state_sum > 0:
                        self.leader_policy_table[state] = self.leader_policy_table[state] / state_sum
                    else:
                        # If all probabilities are zero or negative, reset to uniform
                        num_actions = self.leader_policy_table.shape[1]
                        self.leader_policy_table[state] = np.ones(num_actions) / num_actions

                self.leader_policy_obj.policy_table = self.leader_policy_table

    def train(
        self,
        env,
        n_leader_iterations: int = 1000,
        n_follower_iterations: int = 1000,
        n_episodes_per_iteration: int = 50,
        n_critic_updates: int = 100,
        replay_buffer_size: int = 100000000,
        oracle_mode: str = "none",  # "none", "softql", "softvi"
        mdce_irl_frequency: int = 10,
        verbose: bool = True,
    ):
        """Train the bi-level RL algorithm."""
        # Initialize leader's Q-table
        self._initialize_leader_q_table(env)

        # Initialize replay buffer
        replay_buffer = GammaReplayBuffer(size=replay_buffer_size, gamma=self.discount_leader)

        # Main training loop
        for iteration in range(n_leader_iterations):
            if verbose:
                print(f"\n=== Leader Iteration {iteration} (Mode: {oracle_mode}) ===")

            # ===============================================================
            # Phase A: フォロワーの方策決定 (モード分岐)
            # ===============================================================

            if oracle_mode == "softvi":
                # --- [Oracle 1] Soft Value Iteration (Model-Based) ---
                if verbose:
                    print("Using Oracle (SoftVI): Computing optimal policy...")

                Q_optimal = self._compute_softvi_q_values(
                    env,
                    max_iterations=2000,
                    tolerance=1e-15,
                    verbose=(iteration == 0),
                )

                # Initialize/Update FollowerPolicyModel
                temperature = self.soft_q_config.get("temperature", 1.0)
                self.soft_q_learning = FollowerPolicyModel(self.env_spec, temperature)
                self.soft_q_learning.set_q_values(Q_optimal)
                self.true_follower_model = self.soft_q_learning

                # Create follower policy
                def follower_policy_fn(obs, leader_act, deterministic=False):
                    if deterministic:
                        num_follower_actions = self.env_spec.action_space.n
                        best_action = 0
                        best_q = float("-inf")
                        for b in range(num_follower_actions):
                            q_val = self.soft_q_learning.get_q_value(obs, leader_act, b)
                            if q_val > best_q:
                                best_q, best_action = q_val, b
                        return np.array(best_action, dtype=np.int32)
                    return self.soft_q_learning.sample_action(obs, leader_act)

                self.follower_policy = follower_policy_fn

            elif oracle_mode == "softql":
                # --- [Oracle 2] Soft Q-Learning (Model-Free, True Reward) ---
                if verbose:
                    print("Using Oracle (SoftQL): Learning with TRUE reward...")

                # Initialize SoftQLearning if needed
                if self.soft_q_learning is None or not isinstance(self.soft_q_learning, SoftQLearning):

                    def dummy_reward_fn(s, a, b):
                        return 0.0

                    def leader_policy_probs(state):
                        if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                            s_int = int(state.item() if hasattr(state, "item") else state)
                            return self.leader_policy_obj.policy_table[s_int].tolist()
                        return [1.0 / self.env_spec.leader_action_space.n] * self.env_spec.leader_action_space.n

                    soft_q_config = self.soft_q_config.copy()
                    soft_q_config.pop("learning_rate", None)

                    self.soft_q_learning = SoftQLearning(
                        env_spec=self.env_spec,
                        reward_fn=dummy_reward_fn,
                        leader_policy=leader_policy_probs,
                        discount=self.discount_follower,
                        learning_rate=self.learning_rate_follower,
                        **soft_q_config,
                    )

                # Train Follower using True Reward
                for _ in range(n_follower_iterations):
                    obs, _ = env.reset()
                    while True:
                        leader_act = self.leader_policy_obj.sample_action(obs)
                        follower_act = self.soft_q_learning.sample_action(obs, leader_act)
                        env_step = env.step(leader_act, follower_act)

                        # Update with TRUE reward
                        self.soft_q_learning.update(
                            obs,
                            leader_act,
                            follower_act,
                            env_step.reward,
                            env_step.observation,
                            env_step.last,
                        )
                        obs = env_step.observation
                        if env_step.last:
                            break

                # Create follower policy
                def follower_policy_fn(obs, leader_act, deterministic=False):
                    if deterministic:
                        num_follower_actions = self.env_spec.action_space.n
                        best_action = 0
                        best_q = float("-inf")
                        for b in range(num_follower_actions):
                            q_val = self.soft_q_learning.get_q_value(obs, leader_act, b)
                            if q_val > best_q:
                                best_q, best_action = q_val, b
                        return np.array(best_action, dtype=np.int32)
                    return self.soft_q_learning.sample_action(obs, leader_act)

                self.follower_policy = follower_policy_fn

                # Compute True Q for evaluation reference (once)
                if self.true_follower_model is None:
                    Q_ref = self._compute_softvi_q_values(env, max_iterations=2000, verbose=False)
                    self.true_follower_model = FollowerPolicyModel(self.env_spec, self.soft_q_config.get("temperature", 1.0))
                    self.true_follower_model.set_q_values(Q_ref)

            # --- [Proposed] MDCE IRL (Reward Estimation) ---

            # Step 0: Initial Follower Learning (First time only)
            elif iteration == 0:
                if verbose:
                    print("Step 0 (First iteration): Initializing models...")
                Q_softvi = self._compute_softvi_q_values(env, max_iterations=2000, tolerance=1e-15, verbose=verbose)

                # Set True Model for evaluation
                temperature = self.soft_q_config.get("temperature", 1.0)
                self.true_follower_model = FollowerPolicyModel(self.env_spec, temperature)
                self.true_follower_model.set_q_values(Q_softvi)

                # Initialize Learner Model (starts with true Q, will be updated by IRL)
                self.soft_q_learning = FollowerPolicyModel(self.env_spec, temperature)
                self.soft_q_learning.set_q_values(Q_softvi)

                def follower_policy_fn(obs, leader_act, deterministic=False):
                    # (Same logic as above)
                    if deterministic:
                        num_follower_actions = self.env_spec.action_space.n
                        best_action = 0
                        best_q = float("-inf")
                        for b in range(num_follower_actions):
                            q_val = self.soft_q_learning.get_q_value(obs, leader_act, b)
                            if q_val > best_q:
                                best_q, best_action = q_val, b
                        return np.array(best_action, dtype=np.int32)
                    return self.soft_q_learning.sample_action(obs, leader_act)

                self.follower_policy = follower_policy_fn

            # ===============================================================
            # Phase B: データ収集 (Step 1) - ★全モード・全イテレーションで実行★
            # ===============================================================
            # 以前はここが if should_run_mdce_irl: の中に入っていたためバグでした

            if verbose:
                print(f"Step 1: Collecting {n_episodes_per_iteration} episodes (parallelized)...")

            env_class = type(env)
            env_class_name = env_class.__name__
            env_module_path = env_class.__module__

            # 修正: _init_kwargs が無ければ、余計なものは入れずに空辞書 {} を使う
            env_init_kwargs = getattr(env, "_init_kwargs", {})
            # Clean up dict to avoid pickling issues if needed

            if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                leader_policy_table = self.leader_policy_obj.policy_table.copy()
            else:
                # Fallback
                n_s = self.env_spec.observation_space.n
                n_la = self.env_spec.leader_action_space.n
                leader_policy_table = np.ones((n_s, n_la)) / n_la

            follower_q_values = {}
            if self.soft_q_learning is not None and hasattr(self.soft_q_learning, "Q"):
                for sk, ld in self.soft_q_learning.Q.items():
                    follower_q_values[sk] = {}
                    for lk, fd in ld.items():
                        follower_q_values[sk][lk] = dict(fd)

            temp_soft_q_temp = self.soft_q_config.get("temperature", 1.0)
            n_fa = self.env_spec.action_space.n

            # Run parallel collection
            max_workers = min(n_episodes_per_iteration, 8)
            observations_list = []
            leader_actions_list = []
            follower_actions_list = []
            rewards_list = []
            leader_rewards_list = []
            next_observations_list = []
            terminals_list = []
            last_flags_list = []
            time_steps_list = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _collect_single_episode,
                        env_class_name,
                        env_module_path,
                        env_init_kwargs,
                        leader_policy_table,
                        follower_q_values,
                        {},
                        n_fa,
                        temp_soft_q_temp,
                    )
                    for _ in range(n_episodes_per_iteration)
                ]
                for future in as_completed(futures):
                    data = future.result()
                    observations_list.extend(data["observation"])
                    leader_actions_list.extend(data["leader_action"])
                    follower_actions_list.extend(data["action"])
                    rewards_list.extend(data["reward"])
                    leader_rewards_list.extend(data["leader_reward"])
                    next_observations_list.extend(data["next_observation"])
                    terminals_list.extend(data["terminal"])
                    last_flags_list.extend(data["last"])
                    time_steps_list.extend(data["time_step"])

            if observations_list:
                replay_buffer.add_transitions(
                    observation=np.array(observations_list),
                    leader_action=np.array(leader_actions_list),
                    action=np.array(follower_actions_list),
                    reward=np.array(rewards_list),
                    leader_reward=np.array(leader_rewards_list),
                    next_observation=np.array(next_observations_list),
                    terminal=np.array(terminals_list),
                    last=np.array(last_flags_list),
                    time_step=np.array(time_steps_list),
                )

            # ===============================================================
            # Phase C: IRL実行 (Step 2) - Proposedモードのみ
            # ===============================================================

            # IRLを実行するタイミングかどうか
            should_run_mdce_irl = iteration == 0

            if oracle_mode == "none" and should_run_mdce_irl:
                if verbose:
                    print("Step 2: Generating expert trajectories & Running MDCE IRL...")

                # Generate expert trajectories (Current Leader vs Optimal Follower)
                current_expert_trajectories = self._generate_expert_trajectories(
                    env,
                    n_trajectories=n_episodes_per_iteration,
                    verbose=verbose,
                )

            # ===============================================================
            # Phase C: IRL実行 (Step 2) - Proposedモードのみ
            # ===============================================================

            # IRLを実行するタイミングかどうか
            should_run_mdce_irl = iteration == 0

            if oracle_mode == "none" and should_run_mdce_irl:
                if verbose:
                    print("Step 2: Generating expert trajectories & Running MDCE IRL...")

                # Generate expert trajectories (Current Leader vs Optimal Follower)
                current_expert_trajectories = self._generate_expert_trajectories(
                    env,
                    n_trajectories=n_episodes_per_iteration,
                    verbose=verbose,
                )

                # Run MDCE IRL
                self.estimate_follower_reward(current_expert_trajectories, env, verbose=verbose)

                # Debug Display
                if verbose:
                    self._display_follower_q_values(env=env, show_true_q=False)

            elif oracle_mode == "none" and verbose:
                print(f"Step 2: Skipping MDCE IRL (Next: {((iteration // mdce_irl_frequency) + 1) * mdce_irl_frequency})")

            # ===============================================================
            # Phase D: リーダーの学習 (Step 3-5) - 共通
            # ===============================================================

            # Step 3: Update leader's Critic
            if replay_buffer._current_size > 0:
                if verbose:
                    print(f"Step 3: Updating leader's Critic ({n_critic_updates} updates)...")
                self._update_leader_critic(replay_buffer, n_updates=n_critic_updates)

                if verbose:
                    self._display_leader_q_values()

            # Step 4: Estimate leader's gradient
            gradient_info = None
            if replay_buffer._current_size > 0:
                if verbose:
                    print("Step 4: Estimating leader's gradient...")
                gradient_info = self._estimate_leader_gradient(replay_buffer)

                # Log metrics...
                if gradient_info:
                    self.stats["leader_gradient_norm"].append(gradient_info.get("gradient_norm", 0.0))
                    # ... (ログ出力省略) ...

            # Step 5: Update leader's Actor
            if verbose:
                print("Step 5: Updating leader's Actor...")
            self._update_leader_actor(gradient_info)

            # === 評価 (Evaluation) ===
            if verbose:
                self._log_leader_state(iteration)

            # Evaluate against Learned Follower
            eval_learned = self.evaluate_leader(env, self.follower_policy, n_episodes=10)
            self.stats.setdefault("leader_return_learned", []).append(eval_learned["mean"])
            self.stats.setdefault("leader_return_learned_std", []).append(eval_learned["std"])

            # Evaluate against True Follower
            # Evaluate against True Follower
            if self.true_follower_model:
                # 修正: 引数名を 'deterministic' に合わせる
                def true_f_pol(obs, leader_act, deterministic=False):
                    return self.true_follower_model.sample_action(obs, leader_act)

                eval_true = self.evaluate_leader(env, true_f_pol, n_episodes=10)

                self.stats.setdefault("leader_return_true", []).append(eval_true["mean"])
                self.stats.setdefault("leader_return_true_std", []).append(eval_true["std"])
                self.stats.setdefault("leader_return", []).append(eval_true["mean"])  # Main metric
                self.stats.setdefault("leader_return_std", []).append(eval_true["std"])
            else:
                # Fallback
                self.stats.setdefault("leader_return", []).append(eval_learned["mean"])
                eval_true = eval_learned

            if verbose:
                print("Leader Performance (N=10):")
                print(f"  vs True:    {eval_true['mean']:.4f}")
                print(f"  vs Learned: {eval_learned['mean']:.4f}")
                print(f"  Gap:        {eval_true['mean'] - eval_learned['mean']:.4f}")

        return self.stats
