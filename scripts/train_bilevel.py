"""Training script for Bi-level RL."""

import pickle
from pathlib import Path

import numpy as np
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnv1_1a
from plot_learning_curves import plot_learning_curves


def create_simple_leader_policy(env_spec):
    """Create a simple leader policy.

    Args:
        env_spec: Environment specification

    Returns:
        Leader policy function

    """

    def leader_policy(observation, deterministic=False):
        """Initial uniform leader policy for exploration.

        Args:
            observation: Current observation (state)
            deterministic: Whether to use deterministic policy

        Returns:
            Leader action (uniformly random)

        """
        # Uniform distribution (50% action 0, 50% action 1)
        # Will be updated to tabular policy during training
        return np.random.randint(0, 2)

    return leader_policy


def main():
    """Main training function."""
    print("Initializing Bi-level RL training...")

    # Create environment
    env = DiscreteToyEnv1_1a()
    print(f"Environment: {env.__class__.__name__}")

    # Create leader policy (initial uniform distribution)
    leader_policy = create_simple_leader_policy(env.spec)
    print("Leader policy created (uniform distribution)")

    # Note: Expert trajectories are generated dynamically during training
    # using the current leader policy at each iteration

    # Define feature function for MDCE IRL
    def feature_fn(state, leader_action, follower_action):
        """Simple feature function: one-hot encoding of (state, leader_action, follower_action).

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Feature vector

        """
        # Simple feature: concatenate one-hot encodings
        state_onehot = np.zeros(3)
        state_onehot[state] = 1.0

        leader_onehot = np.zeros(2)
        leader_onehot[leader_action] = 1.0

        follower_onehot = np.zeros(3)
        follower_onehot[follower_action] = 1.0

        return np.concatenate([state_onehot, leader_onehot, follower_onehot])

    # Initialize Bi-level RL algorithm
    print("Initializing Bi-level RL algorithm...")
    algo = BilevelRL(
        env_spec=env.spec,
        leader_policy=leader_policy,
        reward_fn=feature_fn,  # Use one-hot feature function for MDCE IRL
        discount_leader=0.99,
        discount_follower=0.99,  # RESTORED: 0.8 changes the problem definition
        learning_rate_leader=1e-4,
        learning_rate_follower=0.01,  # REDUCED: 0.2 was too high, caused rapid descent from optimistic init
        mdce_irl_config={
            "max_iterations": 100,
            "tolerance": 0.1,  # REDUCED: 50% → 10% for better FEV matching (more realistic than 5%)
            "n_soft_q_iterations": 100,  # INCREASED: 2x for better convergence (not 10x)
            "n_monte_carlo_samples": 100,  # INCREASED: 1.5x for better FEV estimation (not 2x)
            "n_jobs": -1,
        },
        soft_q_config={
            "learning_rate": 0.01,  # REDUCED: 0.2 caused rapid descent from optimistic init
            "temperature": 1.0,  # Keep original problem definition (but may need to increase for better exploration)
            "optimistic_init": 130.0,  # Optimistic initialization (lowered from 130 for faster convergence)
        },
    )

    # Train
    # Note: Expert trajectories are now generated dynamically each iteration
    # using the current leader policy
    print("Starting training...")
    stats = algo.train(
        env=env,
        n_leader_iterations=100,
        n_follower_iterations=100,  # REALISTIC: 1000 episodes = 100k steps, each of 18 pairs visited ~5,500 times
        verbose=True,
    )

    print("\nTraining completed!")
    print(f"Final leader objective: {stats['leader_objective'][-1]:.4f}" if stats["leader_objective"] else "N/A")

    # Save statistics

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    stats_path = output_dir / "training_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"\nStatistics saved to: {stats_path}")

    # Plot learning curves
    try:
        plot_path = output_dir / "learning_curves.png"
        plot_learning_curves(stats, save_path=plot_path)
        print(f"Learning curves saved to: {plot_path}")

        # Display plot (comment out if running headless)
        # plt.show()
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return algo, stats


if __name__ == "__main__":
    algo, stats = main()
