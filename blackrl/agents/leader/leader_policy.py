"""Leader policy implementation for bilevel RL."""

from collections.abc import Callable

import numpy as np


class LeaderPolicy:
    """Leader policy that can be either tabular or callable.

    This class manages the leader's policy π_L(a|s), which can be:
    - Tabular policy: π_L[state, action] (probability table)
    - Callable policy: function(state) -> action or probabilities

    Args:
        env_spec: Environment specification
        policy: Callable policy function (optional)
        use_tabular: Whether to use tabular policy

    """

    def __init__(
        self,
        env_spec,
        policy: Callable | None = None,
        use_tabular: bool = False,
    ):
        self.env_spec = env_spec
        self.policy = policy
        self.use_tabular = use_tabular

        # Tabular policy: π_L[state, leader_action]
        self.policy_table: np.ndarray | None = None

        if use_tabular:
            self._initialize_tabular_policy()

    def _initialize_tabular_policy(self):
        """Initialize tabular policy with uniform distribution."""
        num_states = self.env_spec.observation_space.n if hasattr(self.env_spec.observation_space, "n") else 3
        num_leader_actions = (
            self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
        )

        # Initialize with uniform distribution
        self.policy_table = np.ones((num_states, num_leader_actions), dtype=np.float32) / num_leader_actions
        self.use_tabular = True

    def sample_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Sample action from policy.

        Args:
            state: Current state
            deterministic: Whether to use deterministic policy (argmax)

        Returns:
            Leader action

        """
        if self.use_tabular and self.policy_table is not None:
            state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
            if deterministic:
                return int(np.argmax(self.policy_table[state_int]))
            else:
                probs = self.policy_table[state_int]
                return int(np.random.choice(len(probs), p=probs))
        else:
            # Use Callable policy
            if self.policy is None:
                raise ValueError("Policy not set. Either use tabular policy or provide callable policy.")
            return self.policy(state, deterministic=deterministic)

    def get_probability(self, state: np.ndarray, action: int) -> float:
        """Get probability of action in given state.

        Args:
            state: Current state
            action: Leader action

        Returns:
            Probability of action

        """
        if self.use_tabular and self.policy_table is not None:
            state_int = int(state.item() if isinstance(state, np.ndarray) and state.size == 1 else state)
            return float(self.policy_table[state_int, action])
        else:
            # For callable policy, we can't get exact probability
            # Return uniform as approximation
            num_actions = (
                self.env_spec.leader_action_space.n if hasattr(self.env_spec.leader_action_space, "n") else 2
            )
            return 1.0 / num_actions

    def update(self, gradients: np.ndarray, learning_rate: float):
        """Update tabular policy using gradients.

        Args:
            gradients: Policy gradients ∇_{π_L} J_L
            learning_rate: Learning rate α_L

        """
        if not self.use_tabular or self.policy_table is None:
            raise ValueError("Can only update tabular policy. Initialize with use_tabular=True.")

        # Update policy: π_L^{n+1} ← π_L^n + α_L * ∇_{π_L} J_L
        self.policy_table = self.policy_table + learning_rate * gradients

        # Normalize to ensure valid probability distribution
        # Add small epsilon to avoid numerical issues
        self.policy_table = np.maximum(self.policy_table, 1e-8)

        # Normalize per state
        for state in range(self.policy_table.shape[0]):
            state_sum = np.sum(self.policy_table[state])
            if state_sum > 0:
                self.policy_table[state] = self.policy_table[state] / state_sum
            else:
                # If all probabilities are zero or negative, reset to uniform
                num_actions = self.policy_table.shape[1]
                self.policy_table[state] = np.ones(num_actions) / num_actions

