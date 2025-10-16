"""
BlackRL - Reinforcement Learning Experiments
卒研の実験リポジトリ
"""

__version__ = "0.1.0"

from .agent import BaseAgent
from .random_agent import RandomAgent

__all__ = ["BaseAgent", "RandomAgent", "__version__"]
