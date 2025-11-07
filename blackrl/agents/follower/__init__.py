"""Follower agent modules."""

from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.follower.soft_q_learning import SoftQLearning

__all__ = [
    "MDCEIRL",
    "SoftQLearning",
]
