"""Maximum Discounted Causal Entropy Inverse Reinforcement Learning (MDCE IRL).

This module implements MDCE IRL for recovering follower's reward parameters
from observed trajectory data.
"""

from collections.abc import Callable

import numpy as np
import torch


class MDCEIRL:
    """MDCE IRL for recovering follower's reward parameters.

    MDCE IRL solves the following optimization problem:
        max_g H^γ(g)  subject to  φ̄_g^γ = φ̄_expert^γ

    The dual problem is:
        max_w L(w; D) = E_τ~D [Σ_t γ^t log g_w(b_t|s_t, a_t)]

    where g_w is the Soft Bellman policy induced by reward r_F = w^T φ.

    Args:
        feature_fn: Feature mapping φ: (s, a, b) -> R^K
        discount: Discount factor γ_F
        learning_rate: Learning rate for gradient ascent
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    """

    def __init__(
        self,
        feature_fn: Callable,
        discount: float = 0.99,
        max_iterations: int = 1000,
        tolerance: float = 0.025,
    ):
        self.feature_fn = feature_fn
        self.discount = discount
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Reward parameter (to be learned)
        self.w: torch.Tensor | None = None

    def compute_expert_fev(
        self,
        trajectories: list[dict],
    ) -> torch.Tensor:
        """Compute expert's discounted feature expectation value (FEV).

        φ̄_expert^γ = E_τ~D [Σ_t γ^t φ(s_t, a_t, b_t)]

        Args:
            trajectories: List of expert trajectories, each containing:
                - observations: List of states
                - leader_actions: List of leader actions
                - follower_actions: List of follower actions

        Returns:
            Expert FEV vector of shape (K,)

        """
        fev = None
        total_weight = 0.0

        for traj in trajectories:
            obs = traj.get("observations", [])
            leader_acts = traj.get("leader_actions", [])
            follower_acts = traj.get("follower_actions", [])

            if len(obs) == 0:
                continue

            # Compute discounted feature sum for this trajectory
            traj_fev = None
            weight = 0.0

            for t in range(len(obs)):
                # Extract state, leader action, follower action
                s = obs[t]
                a = leader_acts[t] if t < len(leader_acts) else None
                b = follower_acts[t] if t < len(follower_acts) else None

                if a is None or b is None:
                    continue

                # Compute feature vector
                phi_t = self.feature_fn(s, a, b)
                if isinstance(phi_t, np.ndarray):
                    phi_t = torch.from_numpy(phi_t).float()
                elif not isinstance(phi_t, torch.Tensor):
                    phi_t = torch.tensor(phi_t, dtype=torch.float32)

                # Discounted feature
                discounted_phi = (self.discount**t) * phi_t

                if traj_fev is None:
                    traj_fev = discounted_phi
                else:
                    traj_fev = traj_fev + discounted_phi

                weight += self.discount**t

            if traj_fev is not None:
                if fev is None:
                    fev = traj_fev
                else:
                    fev = fev + traj_fev
                total_weight += weight

        # Normalize by total weight
        if fev is not None and total_weight > 0:
            fev = fev / total_weight

        return fev if fev is not None else torch.zeros(self.feature_fn.dim if hasattr(self.feature_fn, "dim") else 1)

    def compute_policy_fev(
        self,
        policy: Callable,
        leader_policy: Callable,
        env,
        n_samples: int = 1000,
    ) -> torch.Tensor:
        """Compute policy's discounted feature expectation value (FEV).

        φ̄_g^γ = E^{f_θ_L, g} [Σ_t γ^t φ(s_t, a_t, b_t)]

        Args:
            policy: Follower policy g(b|s, a)
            leader_policy: Leader policy f_θ_L(a|s)
            env: Environment instance
            n_samples: Number of trajectories to sample

        Returns:
            Policy FEV vector of shape (K,)

        """
        fev = None
        total_weight = 0.0

        for _ in range(n_samples):
            obs, _ = env.reset()
            traj_fev = None
            weight = 0.0
            t = 0

            while True:
                # Sample leader action
                leader_act = leader_policy(obs)

                # Sample follower action from policy
                follower_act = policy(obs, leader_act)

                # Compute feature vector
                phi_t = self.feature_fn(obs, leader_act, follower_act)
                if isinstance(phi_t, np.ndarray):
                    phi_t = torch.from_numpy(phi_t).float()
                elif not isinstance(phi_t, torch.Tensor):
                    phi_t = torch.tensor(phi_t, dtype=torch.float32)

                # Discounted feature
                discounted_phi = (self.discount**t) * phi_t

                if traj_fev is None:
                    traj_fev = discounted_phi
                else:
                    traj_fev = traj_fev + discounted_phi

                weight += self.discount**t

                # Step environment
                env_step = env.step(leader_act, follower_act)
                obs = env_step.observation

                t += 1

                if env_step.last or t >= env.spec.max_episode_length:
                    break

            if traj_fev is not None:
                if fev is None:
                    fev = traj_fev
                else:
                    fev = fev + traj_fev
                total_weight += weight

        # Normalize by total weight
        if fev is not None and total_weight > 0:
            fev = fev / total_weight

        return fev if fev is not None else torch.zeros(self.feature_fn.dim if hasattr(self.feature_fn, "dim") else 1)

    def compute_likelihood(
        self,
        trajectories: list[dict],
        policy_fn: Callable,
    ) -> float:
        """Compute discounted causal likelihood.

        L(w; D) = E_τ~D [Σ_t γ^t log g_w(b_t|s_t, a_t)]

        Args:
            trajectories: List of expert trajectories
            policy_fn: Policy function g_w(b|s, a) that returns log probabilities

        Returns:
            Likelihood value

        """
        total_likelihood = 0.0
        total_weight = 0.0

        for traj in trajectories:
            obs = traj.get("observations", [])
            leader_acts = traj.get("leader_actions", [])
            follower_acts = traj.get("follower_actions", [])

            traj_likelihood = 0.0
            weight = 0.0

            for t in range(len(obs)):
                if t >= len(leader_acts) or t >= len(follower_acts):
                    break

                s = obs[t]
                a = leader_acts[t]
                b = follower_acts[t]

                # Get log probability
                log_prob = policy_fn(s, a, b)

                traj_likelihood += (self.discount**t) * log_prob
                weight += self.discount**t

            total_likelihood += traj_likelihood
            total_weight += weight

        return total_likelihood / total_weight if total_weight > 0 else 0.0

    def get_reward_params(self) -> torch.Tensor:
        """Get learned reward parameters.

        Returns:
            Reward parameter vector w

        """
        if self.w is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.w
