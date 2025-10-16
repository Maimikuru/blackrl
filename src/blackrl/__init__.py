"""
BlackRL - Reinforcement Learning Experiments
卒研の実験リポジトリ
"""

__version__ = "0.1.0"

from .agent import BaseAgent
from .random_agent import RandomAgent
from .training import evaluate_agent, train_agent

__all__ = ["BaseAgent", "RandomAgent", "train_agent", "evaluate_agent", "__version__"]
