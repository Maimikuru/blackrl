"""
Example using training utilities.
トレーニングユーティリティを使った例
"""

import gymnasium as gym

from blackrl import RandomAgent, evaluate_agent, train_agent


def main():
    """Train and evaluate a random agent using training utilities."""
    # Create environment
    env = gym.make("CartPole-v1")

    # Create agent
    agent = RandomAgent(env.observation_space, env.action_space)

    print("Training Random Agent on CartPole-v1")
    print("=" * 50)

    # Train agent
    train_agent(env, agent, num_episodes=50, verbose=True)

    print("\n" + "=" * 50)
    print("Evaluation")
    print("=" * 50)

    # Evaluate agent
    evaluate_agent(env, agent, num_episodes=10, verbose=True)

    env.close()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
