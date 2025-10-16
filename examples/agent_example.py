"""
Example using the RandomAgent class.
RandomAgentクラスを使った例
"""

import gymnasium as gym
import numpy as np

from blackrl import RandomAgent


def main():
    """Run RandomAgent in CartPole environment."""
    # Create environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # Create agent
    agent = RandomAgent(env.observation_space, env.action_space)

    episode_rewards = []

    # Run 10 episodes
    for episode in range(10):
        observation, info = env.reset()
        episode_reward = 0
        done = False

        agent.reset()

        while not done:
            # Agent selects action
            action = agent.select_action(observation)

            # Take action in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update agent (random agent doesn't learn, but good practice)
            agent.update(observation, action, reward, next_observation, done)

            episode_reward += reward
            observation = next_observation

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    print(f"\nAverage reward: {np.mean(episode_rewards):.2f}")
    print(f"Std deviation: {np.std(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
