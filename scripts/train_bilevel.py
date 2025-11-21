"""Training script for Bi-level RL."""

import pickle
from pathlib import Path

import numpy as np
from blackrl.algos import BilevelRL
from blackrl.envs import DiscreteToyEnvPaper
from plot_learning_curves import plot_learning_curves


def main():
    """Main training function."""
    print("Initializing Bi-level RL training...")

    # Create environment
    env = DiscreteToyEnvPaper()
    print(f"Environment: {env.__class__.__name__}")

    # Leader policy will be automatically initialized as tabular policy during training
    # No need to define initial policy here

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
    # Store key parameters for filename
    mdce_irl_config = {
        "max_iterations": 500,
        "tolerance": 0.025,  # REDUCED: 50% → 10% for better FEV matching (more realistic than 5%)
        "n_soft_q_iterations": 1000,  # INCREASED: 2x for better convergence (not 10x)
        "n_monte_carlo_samples": 5000,  # INCREASED: 1.5x for better FEV estimation (not 2x)
        "n_jobs": -1,
    }
    algo = BilevelRL(
        env_spec=env.spec,
        feature_fn=feature_fn,  # Use one-hot feature function for MDCE IRL
        discount_leader=0.99,
        discount_follower=0.8,  # RESTORED: 0.8 changes the problem definition
        learning_rate_leader=1e-4,
        learning_rate_follower=0.01,  # REDUCED: 0.2 was too high, caused rapid descent from optimistic init
        mdce_irl_config=mdce_irl_config,
        soft_q_config={
            "learning_rate": 0.1,  # REDUCED: 0.2 caused rapid descent from optimistic init
            "temperature": 1.0,  # Keep original problem definition (but may need to increase for better exploration)
            "optimistic_init": 0,  # Optimistic initialization (lowered from 130 for faster convergence)
        },
    )

    # Train
    # Note: Expert trajectories are now generated dynamically each iteration
    # using the current leader policy
    print("Starting training...")
    stats = algo.train(
        env=env,
        n_leader_iterations=1000,
        n_follower_iterations=500,
        n_episodes_per_iteration=1000,  # REALISTIC: 1000 episodes = 100k steps, each of 18 pairs visited ~5,500 times
        verbose=True,
    )

    print("\nTraining completed!")
    print(f"Final leader objective: {stats['leader_objective'][-1]:.4f}" if stats["leader_objective"] else "N/A")

    # Save statistics

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    stats_path = output_dir / "onlyclip_training_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"\nStatistics saved to: {stats_path}")

    # Plot learning curves
    try:
        # Create filename with key parameters
        lr_irl = mdce_irl_config["learning_rate"]
        lr_irl_after = mdce_irl_config["learning_rate_after_stagnation"]
        threshold = mdce_irl_config["learning_rate_stagnation_threshold"]
        n_soft_q_iterations = mdce_irl_config["n_soft_q_iterations"]
        # Replace dots with underscores for filename compatibility
        filename = f"onlyclip_softq_learning_{n_soft_q_iterations}_learning_curves_lr{lr_irl}_lrAfter{lr_irl_after}_th{threshold}.png".replace(".", "_")
        plot_path = output_dir / filename
        plot_learning_curves(stats, save_path=plot_path)
        print(f"Learning curves saved to: {plot_path}")

        # Display plot (comment out if running headless)
        # plt.show(),
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    return algo, stats


if __name__ == "__main__":
    algo, stats = main()
