"""
Base agent interface for reinforcement learning.
強化学習のための基本エージェントインターフェース
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for RL agents."""

    def __init__(self, observation_space: Any, action_space: Any):
        """
        Initialize the agent.

        Args:
            observation_space: The observation space of the environment
            action_space: The action space of the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def select_action(self, observation: Any) -> Any:
        """
        Select an action given an observation.

        Args:
            observation: Current observation from the environment

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(
        self, observation: Any, action: Any, reward: float, next_observation: Any, done: bool
    ) -> dict:
        """
        Update the agent with a transition.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether the episode is done

        Returns:
            Dictionary with update information (e.g., loss values)
        """
        pass

    def reset(self):
        """Reset the agent's internal state (if any)."""
        pass
