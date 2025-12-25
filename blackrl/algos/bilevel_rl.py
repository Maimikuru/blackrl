"""Bi-level Reinforcement Learning Algorithm."""

import importlib
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from blackrl.agents.follower.follower_policy_model import FollowerPolicyModel
from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.follower.sf_learning import SuccessorFeatureLearning
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
    num_follower_actions,
    temperature,
):
    """Get joint action from current state (for parallel processing).

    Args:
        obs: Current observation
        leader_policy_table: Leader policy table (numpy array)
        follower_q_values: Follower Q-values dictionary
        num_follower_actions: Number of follower actions
        temperature: Temperature parameter for softmax policy

    Returns:
        Tuple of (leader_action, follower_action)

    """
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
        self.temperature = soft_q_config.get("temperature", 1.0)

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

            def default_leader_policy():
                num_actions = env_spec.leader_action_space.n if hasattr(env_spec.leader_action_space, "n") else 2
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
        mdce_config = mdce_irl_config
        self.mdce_irl = MDCEIRL(
            feature_fn=feature_fn,
            discount=discount_follower,
            **mdce_config,
        )

        # Initialize Soft Q-Learning
        self.soft_q_learning: SoftQLearning | FollowerPolicyModel | None = None
        self.soft_q_config = soft_q_config

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

    def _initialize_leader_q_table(self):
        """Initialize leader's Q-table Q_L[s, a, b]."""
        obs_space = self.env_spec.observation_space
        leader_action_space = self.env_spec.leader_action_space
        follower_action_space = self.env_spec.action_space

        # Get dimensions
        num_states, num_leader_actions, num_follower_actions = obs_space.n, leader_action_space.n, follower_action_space.n

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
        def tabular_leader_policy(observation):
            """Tabular leader policy wrapper."""
            return self.leader_policy_obj.sample_action(observation)

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
                                soft_v = self.temperature * torch.logsumexp(q_tensor / self.temperature, dim=0).item()
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

                            # Assume uniform leader policy for next state value
                            v_next = 0.0
                            for a_next in range(num_leader_actions):
                                # Compute soft value: V^soft(s',a') = temperature × log Σ_{b'} exp(Q(s',a',b')/temperature)
                                q_values = [
                                    Q_old.get((next_state_int, a_next, b_next), 0.0) for b_next in range(num_follower_actions)
                                ]
                                # Convert to torch for logsumexp
                                q_tensor = torch.tensor(q_values)
                                soft_v = self.temperature * torch.logsumexp(q_tensor / self.temperature, dim=0).item()
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
        """Estimate follower's reward parameters using Hybrid (SoftQ + SF) MDCE IRL."""
        # 1. Expert FEV 計算 (Discounted Sum として計算される)
        expert_fev = self.mdce_irl.compute_expert_fev(trajectories)

        # Initialize w
        if self.mdce_irl.w is None:
            feature_dim = expert_fev.shape[0]
            self.mdce_irl.w = torch.randn(feature_dim, requires_grad=True) * 0.01
        else:
            feature_dim = self.mdce_irl.w.shape[0]

        # Config
        sf_lr = 0.05
        q_lr = self.learning_rate_follower

        # Helper functions
        def get_leader_probs(state):
            if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                s_int = int(state.item() if hasattr(state, "item") else state)
                return self.leader_policy_obj.policy_table[s_int].tolist()
            num_actions = self.env_spec.leader_action_space.n
            return [1.0 / num_actions] * num_actions

        def dynamic_reward_fn(s, la, fa):
            phi = self.mdce_irl.feature_fn(s, la, fa)
            if isinstance(phi, np.ndarray):
                phi = torch.from_numpy(phi).float()
            return torch.dot(self.mdce_irl.w, phi).item()

        # === Learner 1: Soft Q-Learning (Warm Start) ===
        if self.soft_q_learning is None or not isinstance(self.soft_q_learning, SoftQLearning):
            self.soft_q_learning = SoftQLearning(
                env_spec=self.env_spec,
                reward_fn=dynamic_reward_fn,
                leader_policy=get_leader_probs,
                discount=self.discount_follower,
                learning_rate=q_lr,
                temperature=self.temperature,
            )
        else:
            self.soft_q_learning.reward_fn = dynamic_reward_fn
            self.soft_q_learning.leader_policy = get_leader_probs
            self.soft_q_learning.learning_rate = q_lr

        soft_q_learner = self.soft_q_learning

        # === Learner 2: Successor Features (Warm Start) ===
        # ★ここ重要: 以前の学習結果を引き継ぐためにクラスメンバとして保持するか確認
        if not hasattr(self, "sf_learning_model") or self.sf_learning_model is None:
            self.sf_learning_model = SuccessorFeatureLearning(
                env_spec=self.env_spec,
                feature_dim=feature_dim,
                feature_fn=self.mdce_irl.feature_fn,
                discount=self.discount_follower,
                learning_rate=sf_lr,
                temperature=self.temperature,
            )

        sf_learner = self.sf_learning_model

        # Parallel Setup
        env_class = type(env)
        env_class_name = env_class.__name__
        env_module_path = env_class.__module__
        env_init_kwargs = getattr(env, "_init_kwargs", {})

        if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
            leader_policy_table = self.leader_policy_obj.policy_table.copy()
        else:
            n_s = self.env_spec.observation_space.n
            n_la = self.env_spec.leader_action_space.n
            leader_policy_table = np.ones((n_s, n_la)) / n_la

        n_fa = self.env_spec.action_space.n

        # === IRL Loop ===
        for irl_iteration in range(self.mdce_irl.max_iterations):
            # 1. Parallel Data Collection (On-Policy)
            current_follower_q = {}
            if hasattr(soft_q_learner, "Q"):
                for sk, ld in soft_q_learner.Q.items():
                    current_follower_q[sk] = {}
                    for lk, fd in ld.items():
                        current_follower_q[sk][lk] = dict(fd)

            n_episodes_irl = 50
            max_workers = min(n_episodes_irl, 8)
            collected_transitions = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _collect_single_episode,
                        env_class_name,
                        env_module_path,
                        env_init_kwargs,
                        leader_policy_table,
                        current_follower_q,
                        n_fa,
                        self.temperature,
                    )
                    for _ in range(n_episodes_irl)
                ]
                for future in as_completed(futures):
                    data = future.result()
                    steps = len(data["observation"])
                    for i in range(steps):
                        transition = {
                            "obs": data["observation"][i],
                            "leader_act": data["leader_action"][i],
                            "follower_act": data["action"][i],
                            "next_obs": data["next_observation"][i],
                            "done": data["last"][i],
                        }
                        collected_transitions.append(transition)

            # 2. Update Learners
            for t in collected_transitions:
                obs, la, fa = t["obs"], t["leader_act"], t["follower_act"]
                next_obs, done = t["next_obs"], t["done"]

                phi = self.mdce_irl.feature_fn(obs, la, fa)
                phi_tensor = torch.from_numpy(phi).float()
                current_reward = torch.dot(self.mdce_irl.w, phi_tensor).item()

                # A. Soft Q Update
                soft_q_learner.update(obs, la, fa, current_reward, next_obs, done)

                # B. SF Update (Target: Discounted Sum)
                if not done:
                    leader_probs_next = get_leader_probs(next_obs)
                    expected_next_sf = torch.zeros(feature_dim)
                    for la_next, la_prob in enumerate(leader_probs_next):
                        if la_prob <= 1e-8:
                            continue
                        q_vals = [soft_q_learner.get_q_value(next_obs, la_next, b) for b in range(n_fa)]
                        fa_probs = torch.softmax(torch.tensor(q_vals) / self.temperature, dim=0).numpy()
                        for fa_next, fa_prob in enumerate(fa_probs):
                            if fa_prob <= 1e-8:
                                continue
                            psi_next = sf_learner.sf_table[int(next_obs), la_next, fa_next]
                            expected_next_sf += la_prob * fa_prob * torch.from_numpy(psi_next).float()
                    target_sf = phi_tensor + self.discount_follower * expected_next_sf
                else:
                    target_sf = phi_tensor

                current_sf = sf_learner.sf_table[int(obs), int(la), int(fa)]
                sf_learner.sf_table[int(obs), int(la), int(fa)] += sf_lr * (target_sf.numpy() - current_sf)

            # 3. Update Reward Parameters w
            # Policy FEV (From SF)
            policy_fev = torch.zeros(feature_dim)
            s0 = 0
            leader_probs_0 = get_leader_probs(s0)

            for la_0, la_prob in enumerate(leader_probs_0):
                q_vals = [soft_q_learner.get_q_value(s0, la_0, b) for b in range(n_fa)]
                fa_probs = torch.softmax(torch.tensor(q_vals) / self.temperature, dim=0).numpy()
                for fa_0, fa_prob in enumerate(fa_probs):
                    psi_0 = sf_learner.sf_table[s0, la_0, fa_0]
                    policy_fev += la_prob * fa_prob * torch.from_numpy(psi_0).float()

            # ★修正: スケーリングを削除 (SFは総和、Expert FEVも総和に修正済みなので、そのまま比較)
            # policy_fev *= (1.0 - self.discount_follower)  # <--- 削除！！

            gradient = expert_fev - policy_fev

            # Update w
            lr = 0.001 / (1.0 + 0.01 * irl_iteration)
            self.mdce_irl.w = self.mdce_irl.w + lr * gradient

            # Logging
            delta_fem = torch.max(torch.abs(gradient)).item()
            gradient_norm = torch.norm(gradient).item()

            # Q-Stats for logging
            all_q_values = []
            for s in range(self.env_spec.observation_space.n):
                for la in range(self.env_spec.leader_action_space.n):
                    for fa in range(self.env_spec.action_space.n):
                        all_q_values.append(soft_q_learner.get_q_value(s, la, fa))
            all_q_values = np.array(all_q_values)

            self.stats["irl_delta_fem"].append(delta_fem)
            self.stats["irl_gradient_norm"].append(gradient_norm)
            self.stats["irl_q_value_mean"].append(float(np.mean(all_q_values)))

            if verbose and (irl_iteration % 10 == 0):
                print(f"\n--- IRL iteration {irl_iteration}: Hybrid (SoftQ+SF) Statistics ---")
                print(f"  Grad Norm: {gradient_norm:.6f}, Delta FEM: {delta_fem:.6f}")
                print(f"  Q Mean: {np.mean(all_q_values):.4f}")

                # 報酬パラメータwの表示
                w_np = self.mdce_irl.w.detach().cpu().numpy()
                w_norm = torch.norm(self.mdce_irl.w).item()
                print(f"  w norm: {w_norm:.4f}")
                # print(f"  w = {w_np}") # 詳細が必要ならコメントアウト解除

            if delta_fem < self.mdce_irl.tolerance:
                if verbose:
                    print(f"Converged at iter {irl_iteration}")
                    self._display_learned_rewards(dynamic_reward_fn)
                break

        if verbose:
            print("Follower policy updated (Hybrid SoftQ+SF).")
            print("\n" + "=" * 80)
            print("FINAL RECOVERED MODEL DETAILS")
            print("=" * 80)

            # 1. 回復した報酬関数の全表示
            # r(s, a, b) = w^T φ(s, a, b)
            print("\n[1] Recovered Reward Function: r(s, a, b)")
            self._display_learned_rewards(dynamic_reward_fn)

            # 2. 方策の全状態行動確率の表示
            # π(b|s, a) ∝ exp(Q(s, a, b) / temperature)
            print("\n[2] Recovered Follower Policy Probabilities: π(b | s, a)")
            print("-" * 80)

            num_states = self.env_spec.observation_space.n
            num_leader_actions = self.env_spec.leader_action_space.n
            num_follower_actions = self.env_spec.action_space.n

            for s in range(num_states):
                print(f"\nState {s}:")
                for la in range(num_leader_actions):
                    # Q値を取得
                    q_vals = [soft_q_learner.get_q_value(s, la, fa) for fa in range(num_follower_actions)]

                    # Softmax確率の計算 (ボルツマン分布)
                    q_tensor = torch.tensor(q_vals)
                    probs = torch.softmax(q_tensor / self.temperature, dim=0).numpy()

                    # 整形して表示
                    probs_str = "  ".join([f"P(b={b})={p:.4f}" for b, p in enumerate(probs)])
                    print(f"  Leader Action a={la}:  {probs_str}")

                    # (参考) Q値も併記（確率の根拠となる値）
                    q_str = "  ".join([f"Q={q:6.3f}" for q in q_vals])
                    print(f"                      (Q-vals: {q_str})")

            print("=" * 80)
        return self.mdce_irl.w

    def _generate_expert_trajectories(self, env, n_trajectories: int = 50, verbose: bool = False):
        """Generate expert demonstration trajectories using current leader policy (Parallelized)."""
        # 必要な情報の準備
        env_class = type(env)
        env_class_name = env_class.__name__
        env_module_path = env_class.__module__
        env_init_kwargs = getattr(env, "_init_kwargs", {})

        # リーダーの方策テーブル
        if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
            leader_policy_table = self.leader_policy_obj.policy_table.copy()
        else:
            # Fallback
            n_s = self.env_spec.observation_space.n
            n_la = self.env_spec.leader_action_space.n
            leader_policy_table = np.ones((n_s, n_la)) / n_la

        # ★ここがポイント: フォロワーのQ値として「真のモデル(Expert)」のQ値を渡す
        if self.true_follower_model is not None:
            # FollowerPolicyModel.Q は defaultdict なので dict に変換
            expert_q_values = {}
            for sk, ld in self.true_follower_model.Q.items():
                expert_q_values[sk] = {}
                for lk, fd in ld.items():
                    expert_q_values[sk][lk] = dict(fd)
        else:
            expert_q_values = {}

        n_fa = self.env_spec.action_space.n

        # 並列実行
        trajectories = []
        max_workers = min(n_trajectories, 8)

        if verbose:
            print(f"  Generating {n_trajectories} expert trajectories with {max_workers} workers...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _collect_single_episode,
                    env_class_name,
                    env_module_path,
                    env_init_kwargs,
                    leader_policy_table,
                    expert_q_values,  # ★ExpertのQ値を使う
                    n_fa,
                    self.temperature,
                )
                for _ in range(n_trajectories)
            ]

            for future in as_completed(futures):
                raw_traj = future.result()

                formatted_traj = {
                    "observations": raw_traj["observation"],
                    "leader_actions": raw_traj["leader_action"],
                    "follower_actions": raw_traj["action"],
                    "rewards": raw_traj["reward"],
                }
                trajectories.append(formatted_traj)

        return trajectories

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
        log_probs = (q_values - soft_value) / self.temperature
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
                leader_act = self.leader_policy_obj.sample_action(obs)

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
        # Always sample from Max-Ent policy
        def follower_policy_fn(obs, leader_act):
            # Max-Ent RL always uses stochastic policy
            return self.soft_q_learning.sample_action(obs, leader_act)

        self.follower_policy = follower_policy_fn

    def evaluate_leader(
        self,
        env,
        follower_policy_fn: Callable,
        n_episodes: int = 10,
        return_trajectories: bool = False,
    ) -> dict:
        """Evaluate leader's performance and optionally return trajectories."""
        original_policy = self.follower_policy
        self.follower_policy = follower_policy_fn

        total_returns = []
        all_trajectories = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_return = 0.0

            # 1エピソード分のログ
            trajectory = {
                "observations": [],
                "leader_actions": [],
                "follower_actions": [],
                "rewards": [],
                "leader_rewards": [],
            }

            while True:
                leader_act, follower_act = self.get_joint_action(obs)

                env_step = env.step(leader_act, follower_act)
                leader_reward = env_step.env_info.get("leader_reward", env_step.reward)
                # === デバッグ用ログ追加 ===
                # フォロワーが確率的方策モデルを持っている場合、その確率分布を表示
                # === デバッグ用ログ追加 (修正版) ===
                if hasattr(self, "true_follower_model") and self.true_follower_model is not None:
                    # 1. 行動選択時と同じ「フォロワー用観測」を取得する
                    #    (get_joint_action 内のロジックを再現、または変数として取得しておくのがベスト)
                    if isinstance(obs, np.ndarray):
                        obs_array = np.array([obs], dtype=np.int32)
                    else:
                        obs_array = np.array([int(obs)], dtype=np.int32)

                    if isinstance(leader_act, np.ndarray):
                        leader_act_array = np.array([leader_act], dtype=np.int32)
                    else:
                        leader_act_array = np.array([int(leader_act)], dtype=np.int32)

                    # デバッグ用：元のobsを直接使う（get_inputs_forは結合されたベクトルを返すため）
                    # true_f_pol_evalと同じロジックでobsを変換
                    if isinstance(obs, np.ndarray):
                        flat = obs.flatten()
                        if flat.size == 1:
                            s_val = int(flat[0])
                        else:
                            s_val = int(np.argmax(flat))
                    else:
                        s_val = int(obs)

                    # leader_actも同様に確実にint化
                    if isinstance(leader_act, np.ndarray):
                        flat = leader_act.flatten()
                        l_val = int(flat[0]) if flat.size == 1 else int(np.argmax(flat))
                    else:
                        l_val = int(leader_act)

                    # 3. 整合性の取れた状態で確率を取得
                    debug_probs = self.true_follower_model.get_policy(s_val, l_val)

                    # 表示
                    probs_str = ", ".join([f"b{b}: {p:.4f}" for b, p in debug_probs.items()])
                    q_vals_str = ", ".join(
                        [f"Q(b{b})={self.true_follower_model.get_q_value(s_val, l_val, b):.2f}" for b in debug_probs.keys()],
                    )

                    # print(f"  [Debug] S(raw)={obs}, S(key)={s_val}, L_Act={l_val}")
                    # print(f"          Probs=[{probs_str}]")
                    # print(f"          Qs   =[{q_vals_str}]")
                    # print(f"          Selected F_Act={follower_act}")
                # ==========================

                # ログ保存
                trajectory["observations"].append(int(obs))
                trajectory["leader_actions"].append(int(leader_act))
                trajectory["follower_actions"].append(int(follower_act))
                trajectory["rewards"].append(float(env_step.reward))
                trajectory["leader_rewards"].append(float(leader_reward))

                episode_return += leader_reward
                obs = env_step.observation

                if env_step.last:
                    break

            total_returns.append(episode_return)
            all_trajectories.append(trajectory)

        self.follower_policy = original_policy

        result = {
            "mean": np.mean(total_returns),
            "std": np.std(total_returns),
        }

        if return_trajectories:
            result["trajectories"] = all_trajectories

        return result

    def get_joint_action(
        self,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get joint action from joint policy.

        Args:
            observation: Current observation

        Returns:
            Tuple of (leader_action, follower_action)

        """
        if self.follower_policy is None:
            raise ValueError("Follower policy not derived. Call derive_follower_policy() first.")

        # Get leader action

        leader_act = self.leader_policy_obj.sample_action(observation)

        # Get follower action
        # Note: follower_policy expects (obs, leader_act) as separate arguments,
        # not the concatenated vector from get_inputs_for
        # So we pass the original obs and leader_act directly
        if isinstance(observation, np.ndarray):
            obs_for_follower = observation.astype(np.int32) if observation.dtype != np.int32 else observation
        else:
            obs_for_follower = np.array(int(observation), dtype=np.int32)

        if isinstance(leader_act, np.ndarray):
            leader_act_for_follower = leader_act.astype(np.int32) if leader_act.dtype != np.int32 else leader_act
        else:
            leader_act_for_follower = np.array(int(leader_act), dtype=np.int32)

        follower_act = self.follower_policy(
            obs_for_follower,
            leader_act_for_follower,
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
            # Check if leader_reward exists in samples, otherwise use reward
            if "leader_reward" in samples:
                rewards = samples["leader_reward"]  # Leader's reward
            else:
                rewards = samples.get("reward", np.zeros_like(obs))  # Fallback to follower reward
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

    def _estimate_leader_gradient(self, replay_buffer, batch_size: int = 64, use_second_term: bool = True):
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
            use_second_term: If False, only compute first term (standard policy gradient)

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

        # Accumulate gradients per (state, action) pair
        for i in range(len(obs)):
            state = obs[i]
            action = leader_acts[i]
            q_val = q_values[i]

            # First term: Standard policy gradient
            # For tabular policy, gradient w.r.t. π_L[s, a] is Q-value
            current_prob = self.leader_policy_table[state, action]
            first_term_gradients[state, action] += q_val / max(current_prob, 1e-6)

            # Second term: Follower influence term (only if use_second_term=True)
            if use_second_term:
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
                    for t, (s_dot, a_dot) in enumerate(zip(subseq_obs[1:], subseq_leader_acts[1:], strict=False)):
                        # Compute V_F^soft(ṡ, ȧ)
                        v_f_soft = self.soft_q_learning.compute_soft_value(s_dot, a_dot)

                        # For tabular policy: ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) is 1/π_L(ȧ|ṡ)
                        # So gradient w.r.t. π_L[ṡ, ȧ] is: V_F^soft(ṡ, ȧ) / π_L(ȧ|ṡ)
                        # We accumulate V_F^soft(ṡ, ȧ) and normalize later

                        # Discount by γ_F^t
                        # [修正後]
                        discount_factor = self.discount_follower ** (t + 1)
                        # リーダーの方策確率を取得
                        current_prob = self.leader_policy_table[s_dot, a_dot]
                        # 0除算防止（微小値を足すなど）
                        current_prob = max(current_prob, 1e-6)

                        # log f の勾配は 1/f なので、V_F / f を加算する必要がある
                        influence_gradients[s_dot, a_dot] += discount_factor * (v_f_soft / current_prob)

                # Second term contribution
                # benefit * influence_gradients / β_F
                # Note: We already divided by (1-γ_F) in influence_gradients computation
                second_term_gradients += benefit * influence_gradients / self.temperature

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
        second_term_normalized = (
            second_term_gradients / (1.0 - self.discount_leader) if use_second_term else np.zeros_like(first_term_normalized)
        )

        # Combine terms (only first term if use_second_term=False)
        policy_gradients = first_term_normalized + (second_term_normalized if use_second_term else 0)

        # Compute norms for each term
        first_term_norm = np.linalg.norm(first_term_normalized)
        second_term_norm = np.linalg.norm(second_term_normalized) if use_second_term else 0.0
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
        self.stats.setdefault("first_term_norm", []).append(
            gradient_info.get("first_term_norm", 0.0),
        )
        self.stats.setdefault("second_term_norm", []).append(
            gradient_info.get("second_term_norm", 0.0),
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
        n_episodes_per_iteration: int = 50,
        n_critic_updates: int = 100,
        replay_buffer_size: int = 100000000,
        oracle_mode: str = "none",  # "none", "softql", "softvi"
        mdce_irl_frequency: int = 10,
        use_second_term: bool = True,
        verbose: bool = True,
    ):
        """Train the bi-level RL algorithm."""
        # Initialize leader's Q-table
        self._initialize_leader_q_table()

        # Initialize replay buffer
        replay_buffer = GammaReplayBuffer(size=replay_buffer_size, gamma=self.discount_leader)

        # ★修正: 最初のイテレーションの前に初期状態を評価
        if verbose:
            print("\n=== Initial Evaluation (Before Training) ===")

        # 初期リーダー方策に対する最適フォロワーを計算
        Q_optimal_initial = self._compute_softvi_q_values(
            env,
            max_iterations=2000,
            tolerance=1e-10,
            verbose=False,
        )
        self.true_follower_model = FollowerPolicyModel(self.env_spec, self.temperature)
        self.true_follower_model.set_q_values(Q_optimal_initial)

        def true_f_pol_eval_initial(obs, leader_act, deterministic=False):
            if hasattr(obs, "cpu"):
                obs = obs.cpu().numpy()
            obs_val = 0
            if isinstance(obs, np.ndarray):
                flat = obs.flatten()
                if flat.size == 1:
                    obs_val = int(flat[0])
                else:
                    obs_val = int(np.argmax(flat))
            else:
                obs_val = int(obs)

            if hasattr(leader_act, "cpu"):
                leader_act = leader_act.cpu().numpy()
            leader_act_val = 0
            if isinstance(leader_act, np.ndarray):
                flat = leader_act.flatten()
                if flat.size == 1:
                    leader_act_val = int(flat[0])
                else:
                    leader_act_val = int(np.argmax(flat))
            else:
                leader_act_val = int(leader_act)

            return self.true_follower_model.sample_action(obs_val, leader_act_val)

        eval_initial = self.evaluate_leader(env, true_f_pol_eval_initial, n_episodes=10, return_trajectories=True)
        self.stats.setdefault("leader_return_true", []).append(eval_initial["mean"])
        self.stats.setdefault("leader_return_true_std", []).append(eval_initial["std"])
        self.stats.setdefault("leader_return", []).append(eval_initial["mean"])
        if verbose:
            print(f"Leader Performance (vs Best Response): {eval_initial['mean']:.4f} ± {eval_initial['std']:.4f}")

        # Main training loop
        for iteration in range(n_leader_iterations):
            if verbose:
                print(f"\n=== Leader Iteration {iteration} (Mode: {oracle_mode}) ===")

            # ===============================================================
            # Phase A: フォロワーの更新 (Oracle計算 & 方策決定)
            # ===============================================================

            # 1. 評価用(兼 softviモード用)の最適解を計算 [重複排除: ここで1回だけ実行]
            if verbose:
                print("Computing Best Response (SoftVI) for current leader...")

            # 現在のリーダーに対する最適Q値を計算
            Q_optimal = self._compute_softvi_q_values(
                env,
                max_iterations=2000,
                tolerance=1e-10,
                verbose=False,  # 毎回詳細ログは出さない
            )

            # 評価用の「真のフォロワーモデル」を更新
            self.true_follower_model = FollowerPolicyModel(self.env_spec, self.temperature)
            self.true_follower_model.set_q_values(Q_optimal)

            # 2. モード別のフォロワー設定
            if oracle_mode == "softvi":
                # --- [Oracle 1] Soft Value Iteration ---
                if verbose:
                    print("Using Oracle (SoftVI): Using computed optimal policy.")

                # 計算済みのQ_optimalをそのまま使う
                self.soft_q_learning = self.true_follower_model

                # 方策関数を作成
                # Max-Ent RL always uses stochastic policy
                def follower_policy_fn(obs, leader_act):
                    return self.soft_q_learning.sample_action(obs, leader_act)

                self.follower_policy = follower_policy_fn

            elif oracle_mode == "softql":
                # --- [Oracle 2] Soft Q-Learning (Model-Free) ---
                # (既存のコードを維持しますが、今回は使わない想定なら省略可能)
                if verbose:
                    print("Using Oracle (SoftQL): Learning with TRUE reward...")
                # ... (SoftQLの初期化と学習ループは元のまま) ...
                # (実装が長くなるので元のコードのままでOKですが、必要なら記述します)

            elif oracle_mode == "none":
                # --- [Proposed] MDCE IRL + Adaptation ---

                if iteration == 0:
                    # 初回初期化: ダミーで作成
                    if verbose:
                        print("Step 0: Initializing Learner Model...")

                    def dummy_reward_fn(s, a, b):
                        return 0.0

                    def dummy_leader_probs(state):
                        return [1.0 / self.env_spec.leader_action_space.n] * self.env_spec.leader_action_space.n

                    soft_q_config = self.soft_q_config.copy()
                    soft_q_config.pop("learning_rate", None)

                    self.soft_q_learning = SoftQLearning(
                        env_spec=self.env_spec,
                        reward_fn=dummy_reward_fn,
                        leader_policy=dummy_leader_probs,
                        discount=self.discount_follower,
                        learning_rate=self.learning_rate_follower,
                        **soft_q_config,
                    )

                    # 方策関数
                    def follower_policy_fn(obs, leader_act):
                        return self.soft_q_learning.sample_action(obs, leader_act)

                    self.follower_policy = follower_policy_fn

                else:
                    # 2回目以降: 適応 (Adaptation)
                    if verbose:
                        print("Step 0: Adapting Learner to current leader...")

                    # IRLで学習した w を使用
                    w = self.mdce_irl.get_reward_params()

                    def adaptation_reward_fn(s, a, b):
                        phi = self.mdce_irl.feature_fn(s, a, b)
                        if isinstance(phi, np.ndarray):
                            phi = torch.from_numpy(phi).float()
                        return torch.dot(w, phi).item()

                    # 現在のリーダー方策
                    def current_leader_probs(state):
                        if self.leader_policy_obj.use_tabular and self.leader_policy_obj.policy_table is not None:
                            s_int = int(state.item() if hasattr(state, "item") else state)
                            return self.leader_policy_obj.policy_table[s_int].tolist()
                        return [0.5, 0.5]  # fallback

                    self.soft_q_learning.reward_fn = adaptation_reward_fn
                    self.soft_q_learning.leader_policy = current_leader_probs

                    # 適応ループ
                    n_adaptation_episodes = 200  # 多めに確保
                    for _ in range(n_adaptation_episodes):
                        obs, _ = env.reset()
                        while True:
                            leader_act = self.leader_policy_obj.sample_action(obs)
                            follower_act = self.soft_q_learning.sample_action(obs, leader_act)
                            env_step = env.step(leader_act, follower_act)
                            # 更新
                            r_val = adaptation_reward_fn(obs, leader_act, follower_act)
                            self.soft_q_learning.update(
                                obs,
                                leader_act,
                                follower_act,
                                r_val,
                                env_step.observation,
                                env_step.last,
                            )
                            obs = env_step.observation
                            if env_step.last:
                                break

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
                        n_fa,
                        self.temperature,
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
            # ★修正: mdce_irl_frequencyに従って実行
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

                # Run MDCE IRL (内部でフォロワーモデルに反映される)
                self.estimate_follower_reward(current_expert_trajectories, env, verbose=verbose)

                # Debug Display
                if verbose:
                    self._display_follower_q_values(env=env, show_true_q=False)

            elif oracle_mode == "none" and verbose:
                next_irl_iter = ((iteration // mdce_irl_frequency) + 1) * mdce_irl_frequency
                print(f"Step 2: Skipping MDCE IRL (Next: iteration {next_irl_iter})")

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
                gradient_info = self._estimate_leader_gradient(replay_buffer, use_second_term=use_second_term)

                # Log metrics and print gradient details
                if gradient_info:
                    self.stats["leader_gradient_norm"].append(gradient_info.get("gradient_norm", 0.0))

                    if verbose:
                        first_term_norm = gradient_info.get("first_term_norm", 0.0)
                        second_term_norm = gradient_info.get("second_term_norm", 0.0)
                        total_norm = gradient_info.get("gradient_norm", 0.0)
                        mean_q = gradient_info.get("mean_q_value", 0.0)
                        mean_adv = gradient_info.get("mean_advantage", 0.0)

                        print(f"  Gradient Norm: {total_norm:.6f}")
                        print(
                            f"    Term 1 (policy gradient): {first_term_norm:.6f} ({100 * first_term_norm / total_norm if total_norm > 0 else 0:.1f}%)",
                        )
                        print(
                            f"    Term 2 (follower influence): {second_term_norm:.6f} ({100 * second_term_norm / total_norm if total_norm > 0 else 0:.1f}%)",
                        )
                        print(f"  Mean Q-value: {mean_q:.6f}")
                        print(f"  Mean Advantage: {mean_adv:.6f}")

                        # Print policy gradients for each state-action pair
                        policy_gradients = gradient_info.get("policy_gradients")
                        if policy_gradients is not None and self._use_tabular_policy and self.leader_policy_table is not None:
                            print("  Policy Gradients (per state-action):")
                            num_states = policy_gradients.shape[0]
                            num_actions = policy_gradients.shape[1]
                            for s in range(num_states):
                                print(f"    State {s}:")
                                for a in range(num_actions):
                                    grad_val = policy_gradients[s, a]
                                    current_prob = self.leader_policy_table[s, a]
                                    # Calculate update amount
                                    update_amount = self.learning_rate_leader_actor * grad_val
                                    new_prob = current_prob + update_amount
                                    print(
                                        f"      Action {a}: grad={grad_val:8.4f}, current_π={current_prob:.4f}, update={update_amount:+.4f}, new_π≈{new_prob:.4f}",
                                    )

            # Step 5: Update leader's Actor
            if verbose:
                print("Step 5: Updating leader's Actor...")
            self._update_leader_actor(gradient_info)

            # === 評価 (Evaluation) ===
            if verbose:
                self._log_leader_state(iteration)

            # ★修正: 評価の直前に、更新後のリーダー方策に基づいてtrue_follower_modelを再計算
            if verbose:
                print("Recomputing Best Response (SoftVI) for updated leader policy...")
            Q_optimal_updated = self._compute_softvi_q_values(
                env,
                max_iterations=2000,
                tolerance=1e-10,
                verbose=False,
            )
            self.true_follower_model = FollowerPolicyModel(self.env_spec, self.temperature)
            self.true_follower_model.set_q_values(Q_optimal_updated)

            # 正解フォロワー（その反復の最適反応）を用いてリーダーを評価
            def true_f_pol_eval(obs, leader_act, deterministic=False):
                # --- 修正: obs の型変換をあらゆるケースに対応させる ---

                # 1. Tensorならnumpyへ
                if hasattr(obs, "cpu"):
                    obs = obs.cpu().numpy()

                # 2. 状態の整数化 (確実な変換)
                obs_val = 0
                if isinstance(obs, np.ndarray):
                    # flattenして確認
                    flat = obs.flatten()
                    if flat.size == 1:
                        # スカラ配列 (例: array([2]))
                        obs_val = int(flat[0])
                    else:
                        # ベクトル (例: one-hot [0, 0, 1] -> 2, または [0.1, 0.9] -> 1)
                        # one-hot環境なら argmax を使うのが一般的
                        obs_val = int(np.argmax(flat))
                else:
                    # int/floatの場合
                    obs_val = int(obs)

                # 3. leader_act も同様に確実に処理
                if hasattr(leader_act, "cpu"):
                    leader_act = leader_act.cpu().numpy()

                leader_act_val = 0
                if isinstance(leader_act, np.ndarray):
                    flat = leader_act.flatten()
                    if flat.size == 1:
                        leader_act_val = int(flat[0])
                    else:
                        leader_act_val = int(np.argmax(flat))
                else:
                    leader_act_val = int(leader_act)

                # 整数化した値でモデルを呼び出す
                return self.true_follower_model.sample_action(obs_val, leader_act_val)

            eval_true = self.evaluate_leader(env, true_f_pol_eval, n_episodes=10, return_trajectories=True)
            # traj = eval_true["trajectories"][0]
            # print("--- Sample Trajectory ---")
            # for t in range(len(traj["observations"])):
            #     s = traj["observations"][t]
            #     la = traj["leader_actions"][t]
            #     fa = traj["follower_actions"][t]
            #     lr = traj["leader_rewards"][t]
            #     print(f"Step {t}: S={s}, L_Act={la}, F_Act={fa} -> L_Reward={lr}")

            # 統計記録
            self.stats.setdefault("leader_return_true", []).append(eval_true["mean"])
            self.stats.setdefault("leader_return_true_std", []).append(eval_true["std"])
            self.stats.setdefault("leader_return", []).append(eval_true["mean"])  # Main metric

            if verbose:
                print(f"Leader Performance (vs Best Response): {eval_true['mean']:.4f} ± {eval_true['std']:.4f}")

        return self.stats
