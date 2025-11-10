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
            n_soft_q_iterations = min(100, self.learning_rate_follower * 1000)  # Adjust as needed
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
                    reward = env_step.reward

                    # Update Q-function
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
                n_samples=100,  # Can be adjusted
            )

            # Step 2.4: Compare with expert FEV and update w
            # Compute gradient: ∇L(w) ∝ φ̄_expert^γ - φ̄_{g_w}^γ
            gradient = expert_fev - policy_fev

            # Update w: w^n ← w^n + δ(n)(φ̄_expert^{γ_F} - φ̄_{g_{w^n}}^{γ_F})
            self.mdce_irl.w = self.mdce_irl.w + self.mdce_irl.learning_rate * gradient

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

        Implements the full gradient formula:
        ∇_{θ_L} J_L(θ_L) = \frac{1}{1-γ_L} \\mathbb{E}_{d_{γ_L}}^{f_{θ_L}, g_{θ_L}^*} \\Biggl[
            & \nabla_{θ_L} \\log f_{θ_L}(a|s) Q_L^{f_{θ_L}, g_{θ_L}^*}(s, a, b) \\
            & + \frac{1}{\beta_F(1-\\gamma_F)} \\left( Q_L^{f_{θ_L}, g_{θ_L}^*}(s, a, b) - \\mathbb{E}_{b \\sim g_{θ_L}^*(\\cdot|s,a)}[Q_L^{f_{θ_L}, g_{θ_L}^*}(s, a, b)] \right) \\
            & \\cdot \\mathbb{E}_{d_{γ_F}}^{f_{θ_L}, g_{θ_L}^*} \\left[ \nabla_{θ_L} \\log f_{θ_L}(\\dot{a}|\\dot{s}) V_F^{f_{θ_L}\\dag}(\\dot{s}, \\dot{a}) \\Big| s, a, b \right]
        \\Biggr]

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

        # Sample batch
        samples = replay_buffer.sample_transitions(
            batch_size=batch_size,
            replace=True,
            discount=False,
            with_subsequence=False,
        )

        obs = samples["observation"]
        leader_acts = samples["leader_action"]
        follower_acts = samples["action"]

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
        # \frac{1}{\beta_F(1-\gamma_F)} \left( Q_L(s, a, b) - \mathbb{E}_{b \sim g_{θ_L}^*(\cdot|s,a)}[Q_L(s, a, b)] \right)
        # \cdot \mathbb{E}_{d_{γ_F}}^{f_{θ_L}, g_{θ_L}^*} \left[ \nabla_{θ_L} \log f_{θ_L}(\dot{a}|\dot{s}) V_F^{f_{θ_L}\dag}(\dot{s}, \dot{a}) \Big| s, a, b \right]
        second_term_gradients = np.zeros_like(self.leader_policy_table)

        # Get β_F (temperature parameter from Soft Q-Learning)
        beta_F = 1.0  # Default temperature
        if self.soft_q_learning is not None:
            beta_F = self.soft_q_learning.temperature

        # Accumulate gradients per (state, action) pair
        for i in range(len(obs)):
            state = obs[i]
            action = leader_acts[i]
            follower_action = follower_acts[i]
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

            # Influence: E_{d_{γ_F}}^{f_{θ_L}, g_{θ_L}^*} \left[ \nabla_{θ_L} \log f_{θ_L}(\dot{a}|\dot{s}) V_F^{f_{θ_L}\dag}(\dot{s}, \dot{a}) \Big| s, a, b \right]
            # This is approximated using the follower's soft value function V_F^soft
            # For simplicity, we use V_F^soft(s, a) from Soft Q-Learning
            if self.soft_q_learning is not None:
                v_f_soft = self.soft_q_learning.compute_soft_value(state, action)
                # Influence term: ∇_{θ_L} log f_{θ_L}(a|s) * V_F^soft(s, a)
                # For tabular policy, this is V_F^soft(s, a) / π_L(a|s)
                influence = v_f_soft
            else:
                influence = 0.0

            # Second term contribution
            second_term_contribution = benefit * influence / (beta_F * (1.0 - self.discount_follower))
            second_term_gradients[state, action] += second_term_contribution

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
                # We need to set up a reward function that returns the true follower reward
                def true_follower_reward_fn(state, leader_action, follower_action):
                    """Return true follower reward from environment (not estimated)."""
                    # For the toy environment, we can access the true reward
                    # In practice, the follower observes r_F directly from environment interactions
                    # This is a placeholder - the actual reward is obtained during environment steps
                    return 0.0  # Placeholder, actual rewards come from env.step()

                # Initialize Soft Q-Learning with placeholder reward function
                # The actual learning happens through environment interactions in derive_follower_policy
                soft_q_config = self.soft_q_config.copy()
                soft_q_config.pop("learning_rate", None)

                self.soft_q_learning = SoftQLearning(
                    env_spec=self.env_spec,
                    reward_fn=true_follower_reward_fn,  # Placeholder
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
                for _ in range(n_follower_iterations):
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

            for episode in range(n_episodes_per_iteration):
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
                replay_buffer.add_batch(
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
