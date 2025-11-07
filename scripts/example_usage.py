"""Example usage of Bi-level RL components."""

import numpy as np
from blackrl.agents.follower import MDCEIRL, SoftQLearning
from blackrl.envs import DiscreteToyEnv1_1a
from blackrl.policies import JointPolicy


def example_environment():
    """Example: Using the discrete toy environment."""
    print("=== Example: Environment Usage ===")

    env = DiscreteToyEnv1_1a()
    print(f"Environment: {env.__class__.__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Leader action space: {env.leader_action_space}")

    # Reset environment
    obs, episode_info = env.reset()
    print(f"\nInitial observation: {obs}")

    # Step through environment
    leader_action = 1
    follower_action = 0
    env_step = env.step(leader_action, follower_action)

    print(f"Leader action: {leader_action}")
    print(f"Follower action: {follower_action}")
    print(f"Next observation: {env_step.observation}")
    print(f"Reward: {env_step.reward}")
    print(f"Step type: {env_step.step_type}")
    print(f"Last: {env_step.last}")

    env.close()
    print()


def example_mdce_irl():
    """Example: Using MDCE IRL."""
    print("=== Example: MDCE IRL ===")

    # Define feature function
    def feature_fn(state, leader_action, follower_action):
        """Simple feature function."""
        return np.array([state, leader_action, follower_action], dtype=np.float32)

    # Initialize MDCE IRL
    mdce_irl = MDCEIRL(
        feature_fn=feature_fn,
        discount=0.99,
        learning_rate=0.01,
        max_iterations=10,  # Small for example
    )

    print(f"MDCE IRL initialized with discount={mdce_irl.discount}")
    print("Note: To actually fit, call mdce_irl.fit() with trajectories")
    print()


def example_soft_q_learning():
    """Example: Using Soft Q-Learning."""
    print("=== Example: Soft Q-Learning ===")

    env = DiscreteToyEnv1_1a()

    # Simple reward function
    def reward_fn(state, leader_action, follower_action):
        return env.reward_fn(state, leader_action, follower_action)

    # Simple leader policy
    def leader_policy(state):
        return 1

    # Initialize Soft Q-Learning
    soft_q = SoftQLearning(
        env_spec=env.spec,
        reward_fn=reward_fn,
        leader_policy=leader_policy,
        discount=0.99,
        learning_rate=1e-2,
        temperature=1.0,
    )

    print("Soft Q-Learning initialized")
    print(f"Temperature: {soft_q.temperature}")
    print(f"Discount: {soft_q.discount}")

    # Example: Get policy for a state
    state = 0
    leader_act = 1
    policy = soft_q.get_policy(state, leader_act)
    print(f"\nPolicy at state={state}, leader_action={leader_act}: {policy}")

    env.close()
    print()


def example_joint_policy():
    """Example: Using Joint Policy."""
    print("=== Example: Joint Policy ===")

    env = DiscreteToyEnv1_1a()

    # Simple leader policy
    def leader_policy(obs, deterministic=False):
        return 1

    # Simple follower policy
    def follower_policy(obs, leader_act, deterministic=False):
        # Follower observes [observation, leader_action]
        # For simplicity, just return a random action
        return np.random.randint(0, 3)

    # Create joint policy
    joint_policy = JointPolicy(
        env_spec=env.spec,
        leader_policy=leader_policy,
        follower_policy=follower_policy,
    )

    # Get joint action
    obs = 0
    leader_act, follower_act = joint_policy.get_action(obs)

    print(f"Observation: {obs}")
    print(f"Leader action: {leader_act}")
    print(f"Follower action: {follower_act}")

    env.close()
    print()


if __name__ == "__main__":
    example_environment()
    example_mdce_irl()
    example_soft_q_learning()
    example_joint_policy()

    print("All examples completed!")
