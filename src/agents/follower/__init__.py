"""Follower agent modules."""
from blackrl.src.agents.follower.mdce_irl import MDCEIRL
from blackrl.src.agents.follower.soft_q_learning import SoftQLearning

__all__ = [
    'MDCEIRL',
    'SoftQLearning',
]

