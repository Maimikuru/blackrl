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
import numpy as np
import torch
from typing import Callable, Optional, List, Dict
from collections import defaultdict

from blackrl.src.agents.follower.mdce_irl import MDCEIRL
from blackrl.src.agents.follower.soft_q_learning import SoftQLearning
from blackrl.src.policies.joint_policy import JointPolicy


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
        follower_policy: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        discount_leader: float = 0.99,
        discount_follower: float = 0.99,
        learning_rate_leader: float = 1e-3,
        learning_rate_follower: float = 1e-3,
        mdce_irl_config: Optional[Dict] = None,
        soft_q_config: Optional[Dict] = None,
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
        self.soft_q_learning: Optional[SoftQLearning] = None
        self.soft_q_config = soft_q_config or {}

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

    def estimate_follower_reward(
        self,
        trajectories: List[Dict],
        env,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Estimate follower's reward parameters using MDCE IRL.

        Args:
            trajectories: List of expert trajectories
            env: Environment instance
            verbose: Whether to print progress

        Returns:
            Estimated reward parameters w
        """
        # Create policy factory for MDCE IRL
        def policy_fn_factory(w):
            """Create follower policy from reward parameters w."""

            def policy_fn(state, leader_action, follower_action):
                """Compute log probability of follower action."""
                # This is a placeholder - in practice, this should use
                # Soft Q-Learning to compute the policy
                # For now, return uniform log probability
                return -np.log(len(self._get_follower_actions()))

            return policy_fn

        # Fit MDCE IRL
        w = self.mdce_irl.fit(
            trajectories,
            policy_fn_factory,
            self.leader_policy,
            env,
            verbose=verbose,
        )

        return w

    def _get_follower_actions(self):
        """Get all possible follower actions."""
        action_space = self.env_spec.follower_policy_env_spec.action_space
        if hasattr(action_space, 'n'):
            return list(range(action_space.n))
        else:
            return [action_space.sample() for _ in range(10)]

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
        self.soft_q_learning = SoftQLearning(
            env_spec=self.env_spec,
            reward_fn=reward_fn,
            leader_policy=self.leader_policy,
            discount=self.discount_follower,
            learning_rate=self.learning_rate_follower,
            **self.soft_q_config,
        )

        # Train Q-function
        for iteration in range(n_iterations):
            # Sample trajectory
            obs, _ = env.reset()
            total_reward = 0.0

            while True:
                # Sample leader action
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

                # Get leader reward (assuming it's in env_info)
                leader_reward = env_step.env_info.get('leader_reward', env_step.reward)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        leader_act = self.leader_policy(observation, deterministic=deterministic)

        # Get follower action
        follower_obs = self.env_spec.get_inputs_for(
            'follower',
            'policy',
            obs=[observation],
            leader_act=[leader_act],
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

    def train(
        self,
        env,
        expert_trajectories: List[Dict],
        n_leader_iterations: int = 1000,
        n_follower_iterations: int = 1000,
        verbose: bool = True,
    ):
        """Train the bi-level RL algorithm.

        This method:
        1. Estimates follower's reward parameters using MDCE IRL
        2. Derives follower's optimal policy using Soft Q-Learning
        3. Optimizes leader's policy

        Args:
            env: Environment instance
            expert_trajectories: Expert trajectories for IRL
            n_leader_iterations: Number of leader policy update iterations
            n_follower_iterations: Number of follower Q-learning iterations
            verbose: Whether to print progress
        """
        # Step 1: Estimate follower reward parameters
        if verbose:
            print("Step 1: Estimating follower reward parameters using MDCE IRL...")
        w = self.estimate_follower_reward(expert_trajectories, env, verbose=verbose)

        # Step 2: Derive follower policy
        if verbose:
            print("Step 2: Deriving follower optimal policy using Soft Q-Learning...")
        self.derive_follower_policy(env, n_iterations=n_follower_iterations, verbose=verbose)

        # Step 3: Optimize leader policy
        if verbose:
            print("Step 3: Optimizing leader policy...")
        # This is a placeholder - in practice, you would implement
        # leader policy gradient updates here
        for iteration in range(n_leader_iterations):
            objective = self.compute_leader_objective(env, n_episodes=10)
            self.stats['leader_objective'].append(objective)

            if verbose and iteration % 100 == 0:
                print(f"Leader iteration {iteration}: objective={objective:.4f}")

        return self.stats

