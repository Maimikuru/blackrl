"""Replay buffer modules."""
from blackrl.replay_buffer.base import ReplayBufferBase
from blackrl.replay_buffer.gamma_replay_buffer import GammaReplayBuffer

__all__ = [
    'ReplayBufferBase',
    'GammaReplayBuffer',
]

