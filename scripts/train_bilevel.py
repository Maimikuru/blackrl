"""Training script for Bi-level RL."""

import pickle
from pathlib import Path

import numpy as np
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnv1_1a


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
        discount_follower=0.99,
        learning_rate_leader=1e-4,  # REDUCED: Gradient was too large (3M+)
        learning_rate_follower=1e-3,  # REDUCED: 0.1 was too high, Q-values not converging
        mdce_irl_config={
            "max_iterations": 1000,  # INCREASED: Wasn't converging at 500
            "tolerance": 0.5,  # RELAXED: Policy FEV is very different from Expert FEV
            "n_soft_q_iterations": 10000,  # GREATLY INCREASED: Q-values not converging
            "n_monte_carlo_samples": 2000,  # Increased for better FEV estimation
            "n_jobs": -1,  # Use all CPU cores for parallel Monte Carlo sampling
        },
        soft_q_config={
            "learning_rate": 0.01,  # REDUCED: 0.1 was too high, causing Q-value instability
            "temperature": 1.0,
        },
    )

    # Train
    # Note: Expert trajectories are now generated dynamically each iteration
    # using the current leader policy
    print("Starting training...")
    stats = algo.train(
        env=env,
        n_leader_iterations=100,
        n_follower_iterations=20000,  # GREATLY INCREASED: 5000 was insufficient for Q-value convergence (episode length = 100)
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
        from plot_learning_curves import plot_learning_curves

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
