"""Training script for Bi-level RL."""

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
        """Simple leader policy that selects action 1 at state 0, otherwise random.

        Args:
            observation: Current observation (state)
            deterministic: Whether to use deterministic policy

        Returns:
            Leader action

        """
        if isinstance(observation, np.ndarray):
            state = int(observation.item() if observation.size == 1 else observation[0])
        else:
            state = int(observation)

        # Simple policy: at state 0 (S), prefer action 1
        if state == 0:
            return 1 if deterministic else np.random.choice([0, 1], p=[0.3, 0.7])
        return np.random.randint(0, 2)

    return leader_policy


def create_demonstration_trajectories(env, n_trajectories=100):
    """Create demonstration trajectories from expert follower.

    Args:
        env: Environment instance
        n_trajectories: Number of trajectories to generate

    Returns:
        List of trajectory dictionaries

    """
    trajectories = []

    for _ in range(n_trajectories):
        obs, _ = env.reset()
        traj = {
            "observations": [],
            "leader_actions": [],
            "follower_actions": [],
            "rewards": [],
        }

        # Simple expert: always choose action that leads to best reward
        # This is a placeholder - in practice, use actual expert policy
        while True:
            leader_act = 1  # Expert leader always chooses action 1
            follower_act = env.get_opt_ag_act_array()[leader_act, obs]

            traj["observations"].append(obs)
            traj["leader_actions"].append(leader_act)
            traj["follower_actions"].append(follower_act)

            env_step = env.step(leader_act, follower_act)
            traj["rewards"].append(env_step.reward)

            obs = env_step.observation

            if env_step.last:
                break

        trajectories.append(traj)

    return trajectories


def main():
    """Main training function."""
    print("Initializing Bi-level RL training...")

    # Create environment
    env = DiscreteToyEnv1_1a()
    print(f"Environment: {env.__class__.__name__}")

    # Create leader policy
    leader_policy = create_simple_leader_policy(env.spec)
    print("Leader policy created")

    # Create demonstration trajectories
    print("Generating demonstration trajectories...")
    trajectories = create_demonstration_trajectories(env, n_trajectories=50)
    print(f"Generated {len(trajectories)} trajectories")

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
        discount_leader=0.99,
        discount_follower=0.99,
        learning_rate_leader=1e-3,
        learning_rate_follower=1e-3,
        mdce_irl_config={
            "learning_rate": 0.01,
            "max_iterations": 100,
            "tolerance": 1e-4,
        },
        soft_q_config={
            "learning_rate": 1e-2,
            "temperature": 1.0,
        },
    )

    # Train
    print("Starting training...")
    stats = algo.train(
        env=env,
        expert_trajectories=trajectories,
        n_leader_iterations=100,
        n_follower_iterations=500,
        verbose=True,
    )

    print("\nTraining completed!")
    print(f"Final leader objective: {stats['leader_objective'][-1]:.4f}" if stats["leader_objective"] else "N/A")

    return algo, stats


if __name__ == "__main__":
    algo, stats = main()
