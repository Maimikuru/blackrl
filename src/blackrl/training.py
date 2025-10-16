"""
Training utilities for reinforcement learning experiments.
強化学習実験のためのトレーニングユーティリティ
"""

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
from tqdm import tqdm


def train_agent(
    env: gym.Env,
    agent: Any,
    num_episodes: int,
    max_steps_per_episode: Optional[int] = None,
    callback: Optional[Callable[[int, float, Dict], None]] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train an agent in an environment.

    Args:
        env: Gymnasium environment
        agent: Agent with select_action and update methods
        num_episodes: Number of episodes to train for
        max_steps_per_episode: Maximum steps per episode (None for env default)
        callback: Optional callback function called after each episode
                 with (episode_num, episode_reward, info)
        verbose: Whether to show progress bar

    Returns:
        Dictionary containing training history (rewards, losses, etc.)
    """
    episode_rewards = []
    episode_lengths = []

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Training")

    for episode in iterator:
        observation, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        agent.reset()

        while not done:
            if max_steps_per_episode and episode_length >= max_steps_per_episode:
                break

            # Select action
            action = agent.select_action(observation)

            # Take step
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update agent
            agent.update(observation, action, reward, next_observation, done)

            episode_reward += reward
            episode_length += 1
            observation = next_observation

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if verbose:
            iterator.set_postfix(
                {
                    "reward": episode_reward,
                    "avg_reward": np.mean(episode_rewards[-100:]),
                }
            )

        if callback:
            callback(episode, episode_reward, info)

    history = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

    return history


def evaluate_agent(
    env: gym.Env,
    agent: Any,
    num_episodes: int = 10,
    render: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate an agent's performance.

    Args:
        env: Gymnasium environment
        agent: Trained agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        verbose: Whether to print results

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        agent.reset()

        while not done:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if verbose:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }

    if verbose:
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Min/Max Reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
        print(f"  Mean Episode Length: {metrics['mean_length']:.2f}")

    return metrics
