"""Maximum Discounted Causal Entropy Inverse Reinforcement Learning (MDCE IRL).

This module implements MDCE IRL for recovering follower's reward parameters
from observed trajectory data.
"""
import numpy as np
import torch
from typing import List, Dict, Optional, Callable
from collections import defaultdict


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
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.feature_fn = feature_fn
        self.discount = discount
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Reward parameter (to be learned)
        self.w: Optional[torch.Tensor] = None

    def compute_expert_fev(
        self,
        trajectories: List[Dict],
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
            obs = traj.get('observations', [])
            leader_acts = traj.get('leader_actions', [])
            follower_acts = traj.get('follower_actions', [])
            rewards = traj.get('rewards', [])

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
                discounted_phi = (self.discount ** t) * phi_t

                if traj_fev is None:
                    traj_fev = discounted_phi
                else:
                    traj_fev = traj_fev + discounted_phi

                weight += self.discount ** t

            if traj_fev is not None:
                if fev is None:
                    fev = traj_fev
                else:
                    fev = fev + traj_fev
                total_weight += weight

        # Normalize by total weight
        if fev is not None and total_weight > 0:
            fev = fev / total_weight

        return fev if fev is not None else torch.zeros(self.feature_fn.dim if hasattr(self.feature_fn, 'dim') else 1)

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
                discounted_phi = (self.discount ** t) * phi_t

                if traj_fev is None:
                    traj_fev = discounted_phi
                else:
                    traj_fev = traj_fev + discounted_phi

                weight += self.discount ** t

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

        return fev if fev is not None else torch.zeros(self.feature_fn.dim if hasattr(self.feature_fn, 'dim') else 1)

    def compute_likelihood(
        self,
        trajectories: List[Dict],
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
            obs = traj.get('observations', [])
            leader_acts = traj.get('leader_actions', [])
            follower_acts = traj.get('follower_actions', [])

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

                traj_likelihood += (self.discount ** t) * log_prob
                weight += self.discount ** t

            total_likelihood += traj_likelihood
            total_weight += weight

        return total_likelihood / total_weight if total_weight > 0 else 0.0

    def fit(
        self,
        trajectories: List[Dict],
        policy_fn_factory: Callable,
        leader_policy: Callable,
        env,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Fit MDCE IRL to recover reward parameters.

        Updates w using gradient ascent:
            w^{k+1} <- w^k + α(k) (φ̄_expert^γ - φ̄_{g_w^k}^γ)

        Args:
            trajectories: List of expert trajectories
            policy_fn_factory: Function that creates policy g_w from w
            leader_policy: Leader policy f_θ_L
            env: Environment instance
            verbose: Whether to print progress

        Returns:
            Learned reward parameter w
        """
        # Initialize w
        if self.w is None:
            # Initialize with small random values
            feature_dim = self.feature_fn.dim if hasattr(self.feature_fn, 'dim') else 1
            self.w = torch.randn(feature_dim, requires_grad=True) * 0.01

        # Compute expert FEV
        expert_fev = self.compute_expert_fev(trajectories)

        if verbose:
            print(f"Expert FEV: {expert_fev}")

        # Gradient ascent
        for iteration in range(self.max_iterations):
            # Create policy from current w
            policy_fn = policy_fn_factory(self.w)

            # Compute policy FEV
            policy_fev = self.compute_policy_fev(
                policy_fn,
                leader_policy,
                env,
                n_samples=100,  # Can be adjusted
            )

            # Compute gradient: ∇L(w) ∝ φ̄_expert^γ - φ̄_{g_w}^γ
            gradient = expert_fev - policy_fev

            # Update w
            self.w = self.w + self.learning_rate * gradient

            # Check convergence
            if torch.norm(gradient) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

            if verbose and iteration % 100 == 0:
                likelihood = self.compute_likelihood(trajectories, policy_fn)
                print(
                    f"Iteration {iteration}: ||gradient||={torch.norm(gradient):.6f}, "
                    f"likelihood={likelihood:.6f}"
                )

        return self.w

    def get_reward_params(self) -> torch.Tensor:
        """Get learned reward parameters.

        Returns:
            Reward parameter vector w
        """
        if self.w is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.w

