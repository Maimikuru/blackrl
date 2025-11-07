"""Soft Q-Learning for follower policy derivation.

This module implements Soft Q-Learning to derive the follower's optimal
Max-Ent policy given a fixed leader policy and estimated reward function.
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np
import torch


class SoftQLearning:
    """Soft Q-Learning for follower policy in bilevel RL.

    The follower learns a Soft Q-function Q_F^soft(s, a, b) under a fixed
    leader policy f_θ_L. The soft value function is:
        V_F^soft(s, a) = log Σ_{b'} exp(Q_F^soft(s, a, b'))

    The optimal policy is:
        g^*(b|s, a) = exp(Q_F^soft(s, a, b) - V_F^soft(s, a))

    Args:
        env_spec: Environment specification
        reward_fn: Reward function r_F(s, a, b) = w^T φ(s, a, b)
        leader_policy: Fixed leader policy f_θ_L(a|s)
        discount: Discount factor γ_F
        learning_rate: Learning rate for Q-function updates
        temperature: Temperature parameter (default: 1.0 for Max-Ent)
        device: PyTorch device

    """

    def __init__(
        self,
        env_spec,
        reward_fn: Callable,
        leader_policy: Callable,
        discount: float = 0.99,
        learning_rate: float = 1e-3,
        temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        self.env_spec = env_spec
        self.reward_fn = reward_fn
        self.leader_policy = leader_policy
        self.discount = discount
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-function network (can be replaced with tabular or neural network)
        self.Q: dict | None = None
        self._initialize_q_function()

    def _initialize_q_function(self):
        """Initialize Q-function.

        This is a placeholder. In practice, you might use:
        - Tabular Q-function for discrete state/action spaces
        - Neural network for continuous/large state spaces
        """
        # For discrete spaces, use dictionary
        # For continuous spaces, use neural network
        self.Q = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def compute_soft_value(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
    ) -> float:
        """Compute soft value function V_F^soft(s, a).

        V_F^soft(s, a) = log Σ_{b'} exp(Q_F^soft(s, a, b') / temperature)

        Args:
            state: State s
            leader_action: Leader action a

        Returns:
            Soft value V_F^soft(s, a)

        """
        # Get Q-values for all follower actions
        q_values = []
        for b in self._get_follower_actions():
            q_val = self.get_q_value(state, leader_action, b)
            q_values.append(q_val / self.temperature)

        if len(q_values) == 0:
            return 0.0

        # Softmax: log Σ exp(q / temp)
        q_values_tensor = torch.tensor(q_values)
        soft_value = self.temperature * torch.logsumexp(q_values_tensor, dim=0).item()

        return soft_value

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
        # Convert to hashable key (simplified)
        state_key = self._state_to_key(state)
        leader_key = self._action_to_key(leader_action)
        follower_key = self._action_to_key(follower_action)

        return self.Q[state_key][leader_key][follower_key]

    def set_q_value(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
        follower_action: np.ndarray,
        value: float,
    ):
        """Set Q-value Q_F^soft(s, a, b).

        Args:
            state: State s
            leader_action: Leader action a
            follower_action: Follower action b
            value: Q-value to set

        """
        state_key = self._state_to_key(state)
        leader_key = self._action_to_key(leader_action)
        follower_key = self._action_to_key(follower_action)

        self.Q[state_key][leader_key][follower_key] = value

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

    def _get_follower_actions(self):
        """Get all possible follower actions.

        Returns:
            List of follower actions

        """
        # This should sample from follower action space
        # For discrete: return all actions
        # For continuous: return sampled actions
        action_space = self.env_spec.follower_policy_env_spec.action_space
        if hasattr(action_space, "n"):
            # Discrete
            return list(range(action_space.n))
        # Continuous - sample actions
        n_samples = 10
        return [action_space.sample() for _ in range(n_samples)]

    def update(
        self,
        state: np.ndarray,
        leader_action: np.ndarray,
        follower_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        learning_rate: float | None = None,
    ):
        """Update Q-function using Soft Q-Learning.

        Q_F^soft(s_t, a_t, b_t) <- Q_F^soft(s_t, a_t, b_t) + η(t) [
            r_F(s_t, a_t, b_t) + γ_F E_{a'~f_θ_L(·|s_{t+1})}[V_F^soft(s_{t+1}, a')]
            - Q_F^soft(s_t, a_t, b_t)
        ]

        Args:
            state: Current state s_t
            leader_action: Leader action a_t
            follower_action: Follower action b_t
            reward: Reward r_F(s_t, a_t, b_t)
            next_state: Next state s_{t+1}
            done: Whether episode terminated
            learning_rate: Optional learning rate override

        """
        lr = learning_rate or self.learning_rate

        # Current Q-value
        current_q = self.get_q_value(state, leader_action, follower_action)

        if done:
            # Terminal state: Q = reward
            target_q = reward
        else:
            # Compute expected soft value at next state
            # E_{a'~f_θ_L(·|s')}[V_F^soft(s', a')]
            next_leader_actions = self._sample_leader_actions(next_state)
            expected_soft_value = 0.0

            for next_leader_act in next_leader_actions:
                soft_value = self.compute_soft_value(next_state, next_leader_act)
                # Assume uniform distribution for simplicity
                # In practice, use leader policy probabilities
                expected_soft_value += soft_value / len(next_leader_actions)

            target_q = reward + self.discount * expected_soft_value

        # Update Q-value
        new_q = current_q + lr * (target_q - current_q)
        self.set_q_value(state, leader_action, follower_action, new_q)

    def _sample_leader_actions(self, state: np.ndarray) -> list:
        """Sample leader actions from leader policy.

        Args:
            state: State to sample actions for

        Returns:
            List of leader actions

        """
        # Sample from leader policy
        # For discrete: return all actions with probabilities
        # For continuous: sample multiple actions
        n_samples = 5
        return [self.leader_policy(state) for _ in range(n_samples)]

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
        follower_actions = self._get_follower_actions()
        log_probs = []
        probs = {}

        for b in follower_actions:
            q_val = self.get_q_value(state, leader_action, b)
            log_prob = (q_val - soft_value) / self.temperature
            log_probs.append(log_prob)
            probs[tuple(b) if isinstance(b, np.ndarray) else b] = np.exp(log_prob)

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
        actions = list(policy.keys())
        probs = list(policy.values())

        # Sample according to probabilities
        sampled_idx = np.random.choice(len(actions), p=probs)
        sampled_action = actions[sampled_idx]

        # Convert back to numpy array if needed
        if isinstance(sampled_action, tuple):
            return np.array(sampled_action)
        return sampled_action
