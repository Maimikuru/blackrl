"""
Random agent implementation.
ランダムエージェントの実装
"""

from typing import Any

from .agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that takes random actions."""

    def select_action(self, observation: Any) -> Any:
        """
        Select a random action.

        Args:
            observation: Current observation (unused)

        Returns:
            Random action from the action space
        """
        return self.action_space.sample()

    def update(
        self, observation: Any, action: Any, reward: float, next_observation: Any, done: bool
    ) -> dict:
        """
        Random agent doesn't learn, so this is a no-op.

        Returns:
            Empty dictionary
        """
        return {}
