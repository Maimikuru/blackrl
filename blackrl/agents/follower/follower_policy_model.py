"""Follower policy model based on Q-values.

This module implements a follower policy model that computes and samples
from a Max-Ent policy given Q-values and temperature.
"""

from collections import defaultdict

import numpy as np
import torch


class FollowerPolicyModel:
    """Follower policy model that computes policy from Q-values.

    This class stores Q-values and computes the follower's Max-Ent policy
    g^*(b|s, a) from them. The soft value function is:
        V_F^soft(s, a) = log Σ_{b'} exp(Q_F^soft(s, a, b') / temperature)

    The optimal policy is:
        g^*(b|s, a) = exp((Q_F^soft(s, a, b) - V_F^soft(s, a)) / temperature)

    Args:
        env_spec: Environment specification
        temperature: Temperature parameter (default: 1.0 for Max-Ent)
        device: PyTorch device

    """

    def __init__(
        self,
        env_spec,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        self.env_spec = env_spec
        self.temperature = temperature
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-function: Q[state][leader_action][follower_action]
        self.Q: dict | None = None
        self._initialize_q_function()

    def _initialize_q_function(self):
        """Initialize Q-function as empty dictionary."""
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def set_q_values(self, q_values: dict):
        """Set Q-values from dictionary.

        Args:
            q_values: Dictionary mapping (s, a, b) -> Q-value

        """
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for (s, a, b), q_val in q_values.items():
            self.Q[s][a][b] = q_val

    def get_q_value(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
        follower_action: np.ndarray,
    ) -> float:
        """Get Q-value Q_F^soft(s, a, b).

        Args:
            state: State s
            leader_action: Leader action a
            follower_action: Follower action b

        Returns:
            Q-value

        """
        # Convert to hashable key
        state_key = self._state_to_key(state)
        leader_key = self._action_to_key(leader_action)
        follower_key = self._action_to_key(follower_action)

        return self.Q[state_key][leader_key][follower_key]

    def compute_soft_value(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
    ) -> float:
        """Compute soft value function V_F^soft(s, a).

        V_F^soft(s, a) = temperature * log Σ_{b'} exp(Q_F^soft(s, a, b') / temperature)

        Args:
            state: State s
            leader_action: Leader action a

        Returns:
            Soft value V_F^soft(s, a)

        """
        # Get Q-values for all follower actions
        num_follower_actions = self.env_spec.action_space.n
        q_values = []

        for b in range(num_follower_actions):
            q_val = self.get_q_value(state, leader_action, b)
            q_values.append(q_val / self.temperature)

        if len(q_values) == 0:
            return 0.0

        # Softmax: log Σ exp(q / temp)
        q_values_tensor = torch.tensor(q_values)
        soft_value = self.temperature * torch.logsumexp(q_values_tensor, dim=0).item()

        return soft_value

    def get_policy(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
    ) -> dict:
        """Get optimal Max-Ent policy g^*(b|s, a).

        g^*(b|s, a) = exp((Q_F^soft(s, a, b) - V_F^soft(s, a)) / temperature)

        Args:
            state: State s
            leader_action: Leader action a

        Returns:
            Dictionary mapping follower actions to probabilities

        """
        # Compute soft value
        soft_value = self.compute_soft_value(state, leader_action)

        # Compute probabilities for all follower actions
        num_follower_actions = self.env_spec.action_space.n
        probs = {}

        for b in range(num_follower_actions):
            q_val = self.get_q_value(state, leader_action, b)
            log_prob = (q_val - soft_value) / self.temperature
            probs[b] = np.exp(log_prob)

        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            for key in probs:
                probs[key] /= total_prob

        return probs

    def sample_action(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
    ) -> np.ndarray:
        """Sample action from optimal Max-Ent policy.

        Args:
            state: State s
            leader_action: Leader action a

        Returns:
            Sampled follower action b

        """
        policy = self.get_policy(state, leader_action)

        # Sample from policy
        actions = list(policy.keys())
        probs = list(policy.values())

        sampled_action = int(np.random.choice(actions, p=probs))
        return np.array(sampled_action, dtype=np.int32)

    def _state_to_key(self, state: np.ndarray) -> tuple:
        """Convert state to hashable key."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        # Handle scalar values (int, float, etc.)
        if isinstance(state, (int, float, np.integer, np.floating)):
            return (state,)
        return tuple(state)

    def _action_to_key(self, action: np.ndarray) -> tuple:
        """Convert action to hashable key."""
        if isinstance(action, np.ndarray):
            return tuple(action.flatten())
        # Handle scalar values (int, float, etc.)
        if isinstance(action, (int, float, np.integer, np.floating)):
            return (action,)
        return tuple(action)
