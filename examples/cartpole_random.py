"""
Simple example of using Gymnasium for reinforcement learning.
Gymnasiumを使った簡単な強化学習の例
"""

import gymnasium as gym
import numpy as np


def main():
    """Run a simple random agent in CartPole environment."""
    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Reset environment
    observation, info = env.reset(seed=42)

    episode_rewards = []

    # Run 10 episodes
    for episode in range(10):
        observation, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    print(f"\nAverage reward: {np.mean(episode_rewards):.2f}")
    print(f"Std deviation: {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
