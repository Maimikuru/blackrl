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

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch

from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.follower.soft_q_learning import SoftQLearning
from blackrl.replay_buffer.gamma_replay_buffer import GammaReplayBuffer


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
        reward_fn: Follower's reward function (or feature function for IRL)
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
        leader_policy: Callable,
        follower_policy: Callable | None = None,
        reward_fn: Callable | None = None,
        discount_leader: float = 0.99,
        discount_follower: float = 0.99,
        learning_rate_leader: float = 1e-3,
        learning_rate_follower: float = 1e-3,
        mdce_irl_config: dict | None = None,
        soft_q_config: dict | None = None,
    ):
        self.env_spec = env_spec
        self.leader_policy = leader_policy
        self.follower_policy = follower_policy
        self.reward_fn = reward_fn
        self.discount_leader = discount_leader
        self.discount_follower = discount_follower
        self.learning_rate_leader = learning_rate_leader
        self.learning_rate_follower = learning_rate_follower

        # Initialize MDCE IRL
        mdce_config = mdce_irl_config or {}
        self.mdce_irl = MDCEIRL(
            feature_fn=reward_fn or self._default_feature_fn,
            discount=discount_follower,
            **mdce_config,
        )

        # Initialize Soft Q-Learning (will be set up after reward estimation)
        self.soft_q_learning: SoftQLearning | None = None
        self.soft_q_config = soft_q_config or {}

        # Leader's Q-table (Q_L[s, a, b])
        self.leader_q_table: np.ndarray | None = None

        # Leader's tabular policy π_L[s, a] (will be initialized in train)
        self.leader_policy_table: np.ndarray | None = None
        self._use_tabular_policy = False  # Flag to use tabular policy instead of Callable

        # Statistics
        self.stats = defaultdict(list)

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

        # Initialize tabular policy: π_L[state, leader_action]
        # Initialize with uniform distribution
        self.leader_policy_table = (
            np.ones(
                (num_states, num_leader_actions),
                dtype=np.float32,
            )
            / num_leader_actions
        )

        # Enable tabular policy mode
        self._use_tabular_policy = True

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
                            # Bellman update: Q(s,a,b) = r + γ * V(s')
                            # V(s') = max_b Q(s', a', b) for any a' (we'll average over leader actions)
                            next_state_int = int(next_state.item() if isinstance(next_state, np.ndarray) else next_state)

                            # For simplicity, assume uniform leader policy for next state value
                            v_next = 0.0
                            for a_next in range(num_leader_actions):
                                max_q_next = max(
                                    [
                                        Q_old.get((next_state_int, a_next, b_next), 0.0)
                                        for b_next in range(num_follower_actions)
                                    ],
                                )
                                v_next += max_q_next / num_leader_actions

                            Q_true[(s, a, b)] = reward + self.discount_follower * v_next

                        # Track convergence
                        delta = abs(Q_true[(s, a, b)] - Q_old.get((s, a, b), 0.0))
                        max_delta = max(max_delta, delta)

            if max_delta < tolerance:
                print(f"True Q-values converged in {iteration + 1} iterations")
                break

        return Q_true

    def _display_follower_q_values(self, env=None, show_true_q: bool = True):
        """Display follower's learned Q-values in a readable format.

        Args:
            env: Environment instance (optional, for computing true Q-values)
            show_true_q: Whether to compute and show true optimal Q-values

        """
        if self.soft_q_learning is None or self.soft_q_learning.Q is None:
            print("Follower Q-values not available (not initialized)")
            return

        Q = self.soft_q_learning.Q

        # Get dimensions from env_spec
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
        num_follower_actions = self.env_spec.action_space.n if hasattr(self.env_spec.action_space, "n") else 3

        # Compute true Q-values if requested
        Q_true = None
        if show_true_q and env is not None:
            print("\nComputing true optimal Q-values...")
            Q_true = self._compute_true_q_values(env)
            print()

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
            env: Environment instance
            verbose: Whether to print progress

        Returns:
            Estimated reward parameters w (for leader's understanding of follower)

        """
        # Compute expert FEV from demonstration trajectories
        expert_fev = self.mdce_irl.compute_expert_fev(trajectories)

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

            temp_soft_q_learning = SoftQLearning(
                env_spec=self.env_spec,
                reward_fn=reward_fn,
                leader_policy=self.leader_policy,
                discount=self.discount_follower,
                learning_rate=self.learning_rate_follower,
                **soft_q_config,
            )

            # Train Q-function using Soft Q-Learning
            # Use fewer iterations for each IRL step to speed up
            # n_soft_q_iterations = int(min(100, max(1, self.learning_rate_follower * 1000)))
            n_soft_q_iterations = self.mdce_irl.n_soft_q_iterations
            # Adjust as needed
            for _ in range(n_soft_q_iterations):
                obs, _ = env.reset()
                while True:
                    # Sample leader action
                    if self._use_tabular_policy and self.leader_policy_table is not None:
                        state = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
                        probs = self.leader_policy_table[state]
                        leader_act = int(np.random.choice(len(probs), p=probs))
                    else:
                        leader_act = self.leader_policy(obs)

                    # Sample follower action
                    follower_act = temp_soft_q_learning.sample_action(obs, leader_act)

                    # Step environment
                    env_step = env.step(leader_act, follower_act)

                    # CRITICAL: Use estimated reward function (not environment reward!)
                    # In MDCE IRL, we reconstruct Q_F using r_F(s, a, b) = w^T φ(s, a, b)
                    reward = reward_fn(obs, leader_act, follower_act)

                    # Update Q-function with estimated reward
                    temp_soft_q_learning.update(
                        obs,
                        leader_act,
                        follower_act,
                        reward,  # Estimated reward from w^T φ
                        env_step.observation,
                        env_step.last,
                    )

                    obs = env_step.observation
                    if env_step.last:
                        break

            # Step 2.2: Derive current policy g_{w^n} from Q̂_F
            def make_policy_fn(sq_instance):
                """Create policy function with captured Soft Q-Learning instance."""

                def policy_fn(state, leader_action, follower_action=None):
                    """Follower policy function from Soft Q-Learning.

                    If follower_action is None, sample an action (for compute_policy_fev).
                    Otherwise, return log probability (for compute_discounted_causal_likelihood).
                    """
                    if follower_action is None:
                        # Sample action (for compute_policy_fev)
                        return sq_instance.sample_action(state, leader_action)
                    # Return log probability (for compute_discounted_causal_likelihood)
                    # Compute soft value and Q-value
                    q_val = sq_instance.get_q_value(state, leader_action, follower_action)
                    soft_value = sq_instance.compute_soft_value(state, leader_action)
                    # Log probability: log g(b|s,a) = (Q(s,a,b) - V^soft(s,a)) / temperature
                    log_prob = (q_val - soft_value) / sq_instance.temperature
                    return log_prob

                return policy_fn

            policy_fn = make_policy_fn(temp_soft_q_learning)

            # Step 2.3: Compute current policy FEV φ̄_{g_{w^n}}^{γ_F}
            policy_fev = self.mdce_irl.compute_policy_fev(
                policy_fn,
                self.leader_policy,
                env,
            )

            # Step 2.4: Compare with expert FEV and update w
            # Compute gradient: ∇L(w) ∝ φ̄_expert^γ - φ̄_{g_w}^γ
            gradient = expert_fev - policy_fev

            # Update w: w^n ← w^n + α(n)(φ̄_expert^{γ_F} - φ̄_{g_{w^n}}^{γ_F})
            # Learning rate schedule: α(n) = 0.1 / (1.0 + n)
            # More conservative: n=0→0.1, n=9→0.01, n=99→0.001
            learning_rate_schedule = 0.1 / (1.0 + irl_iteration)
            self.mdce_irl.w = self.mdce_irl.w + learning_rate_schedule * gradient

            # Step 2.5: Check convergence
            if torch.norm(gradient) < self.mdce_irl.tolerance:
                if verbose:
                    print(f"MDCE IRL converged at iteration {irl_iteration}")
                break

            if verbose and irl_iteration % 10 == 0:
                likelihood = self.mdce_irl.compute_likelihood(trajectories, policy_fn)
                print(
                    f"IRL iteration {irl_iteration}: ||gradient||={torch.norm(gradient):.6f}, likelihood={likelihood:.6f}",
                )

        return self.mdce_irl.w

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

        self.soft_q_learning = SoftQLearning(
            env_spec=self.env_spec,
            reward_fn=reward_fn,
            leader_policy=self.leader_policy,
            discount=self.discount_follower,
            learning_rate=self.learning_rate_follower,
            **soft_q_config,
        )

        # Train Q-function
        # Option 1: Fixed number of iterations (current implementation)
        for iteration in range(n_iterations):
            # Sample trajectory

            # Option 2: Train until convergence (commented out - uncomment to enable)
            # Step 0: While follower's policy g has not converged
            # prev_q_table = None
            # convergence_threshold = 1e-4  # Threshold for Q-table convergence
            # max_iterations = n_iterations  # Maximum iterations as fallback
            #
            # for iteration in range(max_iterations):
            #     # Sample trajectory
            #     obs, _ = env.reset()
            #     total_reward = 0.0
            #
            #     while True:
            #         # Sample leader action
            #         if self._use_tabular_policy and self.leader_policy_table is not None:
            #             # Use tabular policy
            #             state = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
            #             probs = self.leader_policy_table[state]
            #             leader_act = int(np.random.choice(len(probs), p=probs))
            #         else:
            #             # Use Callable policy
            #             leader_act = self.leader_policy(obs)
            #
            #         # Sample follower action (exploration)
            #         follower_act = self.soft_q_learning.sample_action(obs, leader_act)
            #
            #         # Step environment
            #         env_step = env.step(leader_act, follower_act)
            #         reward = env_step.reward
            #
            #         # Update Q-function
            #         self.soft_q_learning.update(
            #             obs,
            #             leader_act,
            #             follower_act,
            #             reward,
            #             env_step.observation,
            #             env_step.last,
            #         )
            #
            #         total_reward += reward
            #         obs = env_step.observation
            #
            #         if env_step.last:
            #             break
            #
            #     # Check convergence: compare Q-table changes
            #     # Step 0: While follower's policy g has not converged
            #     if prev_q_table is not None and hasattr(self.soft_q_learning, "Q"):
            #         current_q_table = self.soft_q_learning.Q
            #         if isinstance(current_q_table, dict):
            #             # For dict-based Q-table (nested dict: Q[state][leader_action][follower_action])
            #             max_change = 0.0
            #
            #             # Collect all keys from both tables
            #             for state_key in set(list(prev_q_table.keys()) + list(current_q_table.keys())):
            #                 prev_state = prev_q_table.get(state_key, {})
            #                 curr_state = current_q_table.get(state_key, {})
            #                 for leader_key in set(list(prev_state.keys()) + list(curr_state.keys())):
            #                     prev_leader = prev_state.get(leader_key, {})
            #                     curr_leader = curr_state.get(leader_key, {})
            #                     for follower_key in set(list(prev_leader.keys()) + list(curr_leader.keys())):
            #                         prev_val = prev_leader.get(follower_key, 0.0)
            #                         curr_val = curr_leader.get(follower_key, 0.0)
            #                         max_change = max(max_change, abs(curr_val - prev_val))
            #
            #             if max_change < convergence_threshold:
            #                 if verbose:
            #                     print(f"Follower policy converged at iteration {iteration} (max Q-change: {max_change:.6f})")
            #                 break
            #
            #             # Deep copy for next iteration
            #             prev_q_table = copy.deepcopy(current_q_table)
            #         # For numpy array Q-table
            #         elif isinstance(current_q_table, np.ndarray) and isinstance(prev_q_table, np.ndarray):
            #             max_change = np.max(np.abs(current_q_table - prev_q_table))
            #             if max_change < convergence_threshold:
            #                 if verbose:
            #                     print(f"Follower policy converged at iteration {iteration} (max Q-change: {max_change:.6f})")
            #                 break
            #             prev_q_table = current_q_table.copy()
            #     # Initialize prev_q_table for next iteration
            #     elif hasattr(self.soft_q_learning, "Q"):
            #         prev_q_table = copy.deepcopy(self.soft_q_learning.Q)
            #
            #     if verbose and iteration % 100 == 0:
            #         print(f"Soft Q-Learning iteration {iteration}: reward={total_reward:.4f}")
            obs, _ = env.reset()
            total_reward = 0.0

            while True:
                # Sample leader action
                if self._use_tabular_policy and self.leader_policy_table is not None:
                    # Use tabular policy
                    state = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
                    probs = self.leader_policy_table[state]
                    leader_act = int(np.random.choice(len(probs), p=probs))
                else:
                    # Use Callable policy
                    leader_act = self.leader_policy(obs)

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

            # Option 2: Check convergence (commented out - uncomment to enable convergence check)
            # Step 0: While follower's policy g has not converged
            # if prev_q_table is not None and hasattr(self.soft_q_learning, "Q"):
            #     current_q_table = self.soft_q_learning.Q
            #     if isinstance(current_q_table, dict):
            #         # For dict-based Q-table (nested dict: Q[state][leader_action][follower_action])
            #         max_change = 0.0
            #
            #         # Collect all keys from both tables
            #         for state_key in set(list(prev_q_table.keys()) + list(current_q_table.keys())):
            #             prev_state = prev_q_table.get(state_key, {})
            #             curr_state = current_q_table.get(state_key, {})
            #             for leader_key in set(list(prev_state.keys()) + list(curr_state.keys())):
            #                 prev_leader = prev_state.get(leader_key, {})
            #                 curr_leader = curr_state.get(leader_key, {})
            #                 for follower_key in set(list(prev_leader.keys()) + list(curr_leader.keys())):
            #                     prev_val = prev_leader.get(follower_key, 0.0)
            #                     curr_val = curr_leader.get(follower_key, 0.0)
            #                     max_change = max(max_change, abs(curr_val - prev_val))
            #
            #         if max_change < convergence_threshold:
            #             if verbose:
            #                 print(f"Follower policy converged at iteration {iteration} (max Q-change: {max_change:.6f})")
            #             break
            #
            #         # Deep copy for next iteration
            #         prev_q_table = copy.deepcopy(current_q_table)
            #     # For numpy array Q-table
            #     elif isinstance(current_q_table, np.ndarray) and isinstance(prev_q_table, np.ndarray):
            #         max_change = np.max(np.abs(current_q_table - prev_q_table))
            #         if max_change < convergence_threshold:
            #             if verbose:
            #                 print(f"Follower policy converged at iteration {iteration} (max Q-change: {max_change:.6f})")
            #             break
            #         prev_q_table = current_q_table.copy()
            # # Initialize prev_q_table for next iteration
            # elif hasattr(self.soft_q_learning, "Q"):
            #     prev_q_table = copy.deepcopy(self.soft_q_learning.Q)

            if verbose and iteration % 100 == 0:
                print(f"Soft Q-Learning iteration {iteration}: reward={total_reward:.4f}")

        # Create follower policy from learned Q-function
        self.follower_policy = lambda obs, leader_act, deterministic=False: (
            self.soft_q_learning.sample_action(obs, leader_act)
        )

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

                # Get leader reward (target_reward from env_info)
                leader_reward = env_step.env_info.get("target_reward", env_step.reward)

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
        if self._use_tabular_policy and self.leader_policy_table is not None:
            # Use tabular policy
            state = int(observation.item() if isinstance(observation, np.ndarray) and observation.size == 1 else observation)
            if deterministic:
                leader_act = int(np.argmax(self.leader_policy_table[state]))
            else:
                probs = self.leader_policy_table[state]
                leader_act = int(np.random.choice(len(probs), p=probs))
        else:
            # Use Callable policy
            leader_act = self.leader_policy(observation, deterministic=deterministic)

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
            rewards = samples["target_reward"]  # Leader's reward
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
                        if self._use_tabular_policy and self.leader_policy_table is not None:
                            leader_prob = self.leader_policy_table[next_state, leader_act_next]
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
                self.leader_q_table[state, leader_act, follower_act] = current_q + self.learning_rate_leader * (
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
        if self.leader_q_table is None or not self._use_tabular_policy or self.leader_policy_table is None:
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
        first_term_gradients = np.zeros_like(self.leader_policy_table)

        # Second term: Follower influence term
        # [1/(β_F(1-γ_F))] * (Q_L(s,a,b) - E_{b~g}[Q_L(s,a,b)])
        # * E_{d_{γ_F}} [ ∇_{θ_L} log f_{θ_L}(ȧ|ṡ) V_F(ṡ,ȧ) | s,a,b ]
        second_term_gradients = np.zeros_like(self.leader_policy_table)

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
            influence_gradients = np.zeros_like(self.leader_policy_table)

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
        state_action_counts = np.zeros_like(self.leader_policy_table)
        for i in range(len(obs)):
            state = obs[i]
            action = leader_acts[i]
            state_action_counts[state, action] += 1

        # Avoid division by zero
        state_action_counts = np.maximum(state_action_counts, 1.0)
        first_term_gradients = first_term_gradients / state_action_counts
        second_term_gradients = second_term_gradients / state_action_counts

        # Combine terms and normalize by (1 - γ_L)
        policy_gradients = (first_term_gradients + second_term_gradients) / (1.0 - self.discount_leader)

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
            "estimated_gradient": np.mean(advantages),
            "mean_q_value": np.mean(q_values),
            "mean_advantage": np.mean(advantages),
            "policy_gradients": policy_gradients,
        }

    def _update_leader_actor(self, gradient_info):
        """Update leader's policy using estimated gradient.

        For tabular policy:
        π_L^{n+1}[s, a] ← π_L^n[s, a] + α_L * ∇_{π_L[s, a]} J_L(θ_L^n)

        Then normalize to ensure it's a valid probability distribution.

        Args:
            gradient_info: Dictionary with gradient information from _estimate_leader_gradient

        """
        if gradient_info is None:
            return

        # Log gradient information
        self.stats.setdefault("gradient_norm", []).append(
            gradient_info.get("gradient_norm", 0.0),
        )
        self.stats.setdefault("mean_q_value", []).append(
            gradient_info.get("mean_q_value", 0.0),
        )
        self.stats.setdefault("mean_advantage", []).append(
            gradient_info.get("mean_advantage", 0.0),
        )

        # Update tabular policy if available
        if self._use_tabular_policy and self.leader_policy_table is not None:
            policy_gradients = gradient_info.get("policy_gradients")
            if policy_gradients is not None:
                # Update policy: π_L^{n+1} ← π_L^n + α_L * ∇_{π_L} J_L
                self.leader_policy_table = self.leader_policy_table + self.learning_rate_leader * policy_gradients

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

    def train(
        self,
        env,
        expert_trajectories: list[dict],
        n_leader_iterations: int = 1000,
        n_follower_iterations: int = 1000,
        n_episodes_per_iteration: int = 10,
        n_critic_updates: int = 100,
        replay_buffer_size: int = 10000,
        verbose: bool = True,
    ):
        """Train the bi-level RL algorithm.

        This method implements the full SAC-IRLF algorithm:
        1. Estimates follower's reward parameters using MDCE IRL
        2. Derives follower's optimal policy using Soft Q-Learning
        3. Collects trajectories and stores in replay buffer
        4. Updates leader's Q-table (Critic)
        5. Estimates leader's policy gradient
        6. Updates leader's policy (Actor)

        Args:
            env: Environment instance
            expert_trajectories: Expert trajectories for IRL
            n_leader_iterations: Number of leader policy update iterations
            n_follower_iterations: Number of follower Q-learning iterations
            n_episodes_per_iteration: Number of episodes to collect per iteration
            n_critic_updates: Number of Q-learning updates per iteration
            replay_buffer_size: Size of replay buffer
            verbose: Whether to print progress

        """
        # Initialize leader's Q-table
        self._initialize_leader_q_table(env)

        # Initialize replay buffer
        replay_buffer = GammaReplayBuffer(size=replay_buffer_size, gamma=self.discount_leader)

        # Main training loop
        # Algorithm order: Step 0 (Follower learning) → Step 1 (Collect trajectories) → Step 2 (MDCE IRL) → Steps 3-5
        for iteration in range(n_leader_iterations):
            if verbose:
                print(f"\n=== Leader Iteration {iteration} ===")

            # Step 0: フォロワー方策の学習
            # Follower interacts with environment (under f_{θ_L^n}) and updates its own Q_F
            # using its own (observable) reward r_F from the environment
            # Note: フォロワーは環境から報酬を直接獲得できるため、真の報酬でQ_Fを学習する
            # （MDCE IRLはリーダーがフォロワーの報酬を推定するために使用し、ここでは使わない）
            if iteration == 0:
                # First iteration: Follower learns its policy using true rewards from environment
                if verbose:
                    print("Step 0 (First iteration): Deriving initial follower policy using true rewards...")

                # Follower learns Q_F using Soft Q-Learning with true rewards from environment
                # The actual rewards are obtained from env.step() and passed directly to update()
                # No reward_fn needed here

                # Initialize Soft Q-Learning
                # The actual learning happens through environment interactions
                soft_q_config = self.soft_q_config.copy()
                soft_q_config.pop("learning_rate", None)

                self.soft_q_learning = SoftQLearning(
                    env_spec=self.env_spec,
                    reward_fn=None,  # Not needed: rewards come from env.step()
                    leader_policy=self.leader_policy,
                    discount=self.discount_follower,
                    learning_rate=self.learning_rate_follower,
                    **soft_q_config,
                )

                # Create follower policy from learned Q-function
                self.follower_policy = lambda obs, leader_act, deterministic=False: (
                    self.soft_q_learning.sample_action(obs, leader_act)
                )

                if verbose:
                    print("Step 0: Learning follower policy with true rewards from environment...")
                # Follower learns by interacting with environment
                # The rewards are obtained directly from env.step() calls
                total_steps = 0
                total_reward = 0.0
                for episode_idx in range(n_follower_iterations):
                    obs, _ = env.reset()
                    episode_steps = 0
                    episode_reward = 0.0
                    while True:
                        # Sample leader action
                        if self._use_tabular_policy and self.leader_policy_table is not None:
                            state = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
                            probs = self.leader_policy_table[state]
                            leader_act = int(np.random.choice(len(probs), p=probs))
                        else:
                            leader_act = self.leader_policy(obs)

                        # Sample follower action
                        follower_act = self.soft_q_learning.sample_action(obs, leader_act)

                        # Step environment and get TRUE follower reward
                        env_step = env.step(leader_act, follower_act)
                        true_follower_reward = env_step.reward  # This is the true follower reward

                        # Update Q-function with TRUE reward
                        self.soft_q_learning.update(
                            obs,
                            leader_act,
                            follower_act,
                            true_follower_reward,  # Use true reward, not estimated
                            env_step.observation,
                            env_step.last,
                        )

                        episode_steps += 1
                        episode_reward += true_follower_reward
                        obs = env_step.observation
                        if env_step.last:
                            break

                    total_steps += episode_steps
                    total_reward += episode_reward
                    if verbose and episode_idx < 3:  # Print first 3 episodes
                        print(f"  Episode {episode_idx}: {episode_steps} steps, total reward = {episode_reward:.4f}")

                if verbose:
                    print("\nStep 0 Summary:")
                    print(f"  Total episodes: {n_follower_iterations}")
                    print(f"  Total steps: {total_steps}")
                    print(f"  Average steps per episode: {total_steps / n_follower_iterations:.2f}")
                    print(f"  Total reward: {total_reward:.4f}")
                    print(f"  Average reward per episode: {total_reward / n_follower_iterations:.4f}\n")

                # Display learned Q-values for verification
                if verbose:
                    print("\n" + "=" * 80)
                    print("FOLLOWER Q-VALUES AFTER STEP 0 (First Iteration)")
                    print("=" * 80)
                    self._display_follower_q_values(env=env, show_true_q=True)
                    print("=" * 80 + "\n")

            else:
                # Subsequent iterations: Follower re-learns its policy using true rewards
                # Note: リーダーの方策が更新されたため、フォロワーも再学習が必要
                if verbose:
                    print(f"Step 0 (Iteration {iteration}): Re-learning follower policy with true rewards...")
                # Follower learns by interacting with environment using TRUE rewards
                for _ in range(n_follower_iterations):
                    obs, _ = env.reset()
                    while True:
                        # Sample leader action from updated leader policy
                        if self._use_tabular_policy and self.leader_policy_table is not None:
                            state = int(obs.item() if isinstance(obs, np.ndarray) and obs.size == 1 else obs)
                            probs = self.leader_policy_table[state]
                            leader_act = int(np.random.choice(len(probs), p=probs))
                        else:
                            leader_act = self.leader_policy(obs)

                        # Sample follower action
                        follower_act = self.soft_q_learning.sample_action(obs, leader_act)

                        # Step environment and get TRUE follower reward
                        env_step = env.step(leader_act, follower_act)
                        true_follower_reward = env_step.reward  # This is the true follower reward

                        # Update Q-function with TRUE reward
                        self.soft_q_learning.update(
                            obs,
                            leader_act,
                            follower_act,
                            true_follower_reward,  # Use true reward, not estimated
                            env_step.observation,
                            env_step.last,
                        )

                        obs = env_step.observation
                        if env_step.last:
                            break

                # Display learned Q-values for verification
                if verbose:
                    print("\n" + "=" * 80)
                    print(f"FOLLOWER Q-VALUES AFTER STEP 0 (Iteration {iteration})")
                    print("=" * 80)
                    self._display_follower_q_values(env=env, show_true_q=True)
                    print("=" * 80 + "\n")

            # Step 1: Collect trajectories
            if verbose:
                print(f"Step 1: Collecting {n_episodes_per_iteration} episodes...")
            observations_list = []
            leader_actions_list = []
            follower_actions_list = []
            rewards_list = []
            target_rewards_list = []
            next_observations_list = []
            terminals_list = []
            last_flags_list = []
            time_steps_list = []

            # Store trajectories for MDCE IRL (episode format)
            trajectories_for_irl = []

            for _ in range(n_episodes_per_iteration):
                obs, _ = env.reset()
                time_step = 0

                # Episode trajectory for MDCE IRL
                episode_obs = []
                episode_leader_acts = []
                episode_follower_acts = []
                episode_rewards = []

                while True:
                    # Get joint action
                    leader_act, follower_act = self.get_joint_action(obs)

                    # Step environment
                    env_step = env.step(leader_act, follower_act)

                    # Get leader reward (target_reward)
                    leader_reward = env_step.env_info.get("target_reward", env_step.reward)

                    # Store transition for replay buffer
                    observations_list.append(obs)
                    leader_actions_list.append(leader_act)
                    follower_actions_list.append(follower_act)
                    rewards_list.append(env_step.reward)  # Follower reward
                    target_rewards_list.append(leader_reward)  # Leader reward
                    next_observations_list.append(env_step.observation)
                    terminals_list.append(1 if env_step.terminal else 0)
                    last_flags_list.append(1 if env_step.last else 0)
                    time_steps_list.append(time_step)

                    # Store for MDCE IRL trajectory
                    episode_obs.append(obs)
                    episode_leader_acts.append(leader_act)
                    episode_follower_acts.append(follower_act)
                    episode_rewards.append(env_step.reward)

                    obs = env_step.observation
                    time_step += 1

                    if env_step.last:
                        break

                # Store episode trajectory for MDCE IRL (for next iteration)
                trajectories_for_irl.append(
                    {
                        "observations": episode_obs,
                        "leader_actions": episode_leader_acts,
                        "follower_actions": episode_follower_acts,
                        "rewards": episode_rewards,
                    },
                )

            # Add transitions to replay buffer
            if observations_list:
                replay_buffer.add_transitions(
                    observation=np.array(observations_list),
                    leader_action=np.array(leader_actions_list),
                    action=np.array(follower_actions_list),
                    reward=np.array(rewards_list),
                    target_reward=np.array(target_rewards_list),
                    next_observation=np.array(next_observations_list),
                    terminal=np.array(terminals_list),
                    last=np.array(last_flags_list),
                    time_step=np.array(time_steps_list),
                )

            # Step 2: MDCE IRL - Leader estimates follower's reward function
            # Algorithm Step 2: MDCE IRL
            # Purpose: リーダーはフォロワーの報酬を観測できないため、
            #          収集した軌跡（フォロワーの行動）から報酬関数を逆推定する
            # - Compute expert FEV φ̄_expert^{γ_F} from D (collected trajectories)
            # - For IRL update step i = 1 to I:
            #   - Reconstruct Follower Q-Function (Soft Q-Learning with estimated reward)
            #   - Derive current policy g_{w^n} from Q̂_F
            #   - Compute current policy FEV φ̄_{g_{w^n}}^{γ_F}
            #   - If converged: Return w^n, g_{w^n}
            #   - Update: w^n ← w^n + δ(n)(φ̄_expert^{γ_F} - φ̄_{g_{w^n}}^{γ_F})
            if verbose:
                print("Step 2: Leader estimates follower's reward function using MDCE IRL...")

            # Use collected trajectories for MDCE IRL
            if trajectories_for_irl:
                w = self.estimate_follower_reward(trajectories_for_irl, env, verbose=verbose)
            else:
                # Fallback: use expert trajectories if no trajectories collected
                if verbose:
                    print("No trajectories collected, using expert trajectories for MDCE IRL...")
                w = self.estimate_follower_reward(expert_trajectories, env, verbose=verbose)

            # Note: MDCE IRLで推定された報酬wは、リーダーの勾配計算（Step 4）で使用される
            # フォロワー自身は次のイテレーションのStep 0で真の報酬を使って再学習する

            # Step 3: Update leader's Critic (Q-table)
            if replay_buffer._current_size > 0:
                if verbose:
                    print(f"Step 3: Updating leader's Critic ({n_critic_updates} updates)...")
                self._update_leader_critic(replay_buffer, n_updates=n_critic_updates)

            # Step 4: Estimate leader's gradient
            gradient_info = None
            if replay_buffer._current_size > 0:
                if verbose:
                    print("Step 4: Estimating leader's gradient...")
                gradient_info = self._estimate_leader_gradient(replay_buffer)
                if verbose and gradient_info:
                    print(f"  Gradient norm: {gradient_info.get('gradient_norm', 0.0):.4f}")
                    print(f"  Mean Q-value: {gradient_info.get('mean_q_value', 0.0):.4f}")

            # Step 5: Update leader's Actor
            if verbose:
                print("Step 5: Updating leader's Actor...")
            self._update_leader_actor(gradient_info)

            # Compute and log objective
            objective = self.compute_leader_objective(env, n_episodes=10)
            self.stats["leader_objective"].append(objective)

            if verbose:
                print(f"Leader objective: {objective:.4f}")

        return self.stats
