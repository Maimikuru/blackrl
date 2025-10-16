"""
Tests for agent classes.
"""

import gymnasium as gym
import pytest

from blackrl import BaseAgent, RandomAgent


def test_random_agent_creation():
    """Test RandomAgent can be created."""
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.observation_space, env.action_space)
    assert agent is not None
    env.close()


def test_random_agent_select_action():
    """Test RandomAgent can select actions."""
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.observation_space, env.action_space)

    observation, _ = env.reset()
    action = agent.select_action(observation)

    assert action in [0, 1]  # CartPole has 2 actions
    env.close()


def test_random_agent_update():
    """Test RandomAgent update method."""
    env = gym.make("CartPole-v1")
    agent = RandomAgent(env.observation_space, env.action_space)

    observation, _ = env.reset()
    action = agent.select_action(observation)
    next_observation, reward, terminated, truncated, info = env.step(action)

    result = agent.update(observation, action, reward, next_observation, terminated)

    assert isinstance(result, dict)
    env.close()


def test_base_agent_is_abstract():
    """Test that BaseAgent cannot be instantiated directly."""
    env = gym.make("CartPole-v1")

    with pytest.raises(TypeError):
        BaseAgent(env.observation_space, env.action_space)

    env.close()
