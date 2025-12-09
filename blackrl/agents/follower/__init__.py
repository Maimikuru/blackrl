"""Follower agent modules."""

from blackrl.agents.follower.follower_policy_model import FollowerPolicyModel
from blackrl.agents.follower.mdce_irl import MDCEIRL
from blackrl.agents.follower.sf_learning import SuccessorFeatureLearning
from blackrl.agents.follower.soft_q_learning import SoftQLearning

__all__ = [
    "MDCEIRL",
    "FollowerPolicyModel",
    "SoftQLearning",
    "SuccessorFeatureLearning",
]
