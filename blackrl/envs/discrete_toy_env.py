"""Discrete toy environments for bilevel RL."""

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from blackrl.envs.base import (
    Environment,
    EnvStep,
    GlobalEnvSpec,
    StepType,
)


class DiscreteToyEnvBase(Environment):
    """Base class for discrete toy environments.

    This is a simple discrete environment with 3 states (S, A, B),
    2 leader actions (0, 1), and 3 follower actions (s, a, b).
    """

    STATE_TYPE = ["S", "A", "B"]  # 0, 1, 2
    LEADER_ACT_TYPE = ["0", "1"]  # 0, 1
    FOLLOWER_ACT_TYPE = ["s", "a", "b"]  # 0, 1, 2

    def __init__(self):
        """Initialize the discrete toy environment."""
        # Transition function: transition[state][leader_action][follower_action]
        # = 0: Sへ遷移, = 1: Aへ遷移, = 2: Bへ遷移, = -1: 遷移しない
        self.transition = np.ones((3, 2, 3), dtype=np.int32) * -1

        # Reward function: rewards[state][leader_action][follower_action]
        self.rewards = np.zeros((3, 2, 3))
        self.r_range = (-np.inf, np.inf)

        # Target reward function: target_rewards[state][leader_action][follower_action]
        self.target_rewards = np.zeros((3, 2, 3))
        self.target_r_range = (-np.inf, np.inf)

        # Optimal leader policy configuration
        self.key_state = None
        self.optimal_action = 1

        # Leader action costs: costs[state][leader_action]
        self.costs = np.zeros((3, 2))

        # Episode length
        self._max_episode_steps = 100

        # Optimal follower Q-table: opt_ag_qtable[leader_action][state][follower_action]
        self.opt_ag_qtable = np.zeros((2, 3, 3))

        # Define spaces
        self._observation_space = spaces.Discrete(3)
        self._action_space = spaces.Discrete(3)  # Follower action space
        self._leader_action_space = spaces.Discrete(2)

        # Create GlobalEnvSpec
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        # Initialize state
        self.seed()
        self.state = None
        self.steps_n = 0
        self.actions = [None, None]
        self.render_rewards = [0, 0]

    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def action_space(self):
        """Get follower's action space."""
        return self._action_space

    @property
    def leader_action_space(self):
        """Get leader's action space."""
        return self._leader_action_space

    @property
    def observation_space(self):
        """Get observation space."""
        return self._observation_space

    @property
    def spec(self):
        """Get global environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """Get supported render modes."""
        return ["human"]

    def transition_fn(self, state, leader_action, follower_action):
        """Compute next state from transition function.

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Next state

        """
        next_state = self.transition[state, leader_action, follower_action].item()
        if next_state == -1:
            return state
        return next_state

    def reward_fn(self, state, leader_action, follower_action):
        """Compute reward.

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Reward value

        """
        return self.rewards[state, leader_action, follower_action].item()

    def target_reward_fn(self, state, leader_action, follower_action):
        """Compute target reward.

        Args:
            state: Current state
            leader_action: Leader's action
            follower_action: Follower's action

        Returns:
            Target reward value

        """
        return self.target_rewards[state, leader_action, follower_action].item()

    def reset(self, init_state=None):
        """Reset the environment.

        Args:
            init_state: Optional initial state

        Returns:
            tuple: (observation, episode_info)

        """
        if init_state is not None:
            assert self.observation_space.contains(init_state), f"Invalid state {init_state} passed to reset"
            self.state = int(init_state)
        else:
            self.state = 0

        self.actions = [None, None]
        self.render_rewards = [0, 0]
        self.steps_n = 0

        episode_info = {}
        return np.array(self.state, dtype=np.int32), episode_info

    def step(self, leader_action, action):
        """Step the environment.

        Args:
            leader_action: Leader's action
            action: Follower's action

        Returns:
            EnvStep: Environment step result

        """
        follower_action = action

        next_state = self.transition_fn(self.state, leader_action, follower_action)
        reward = self.reward_fn(self.state, leader_action, follower_action)
        target_reward = self.target_reward_fn(self.state, leader_action, follower_action)

        self.steps_n += 1
        self.state = next_state

        # Determine step type
        if self.steps_n >= self._max_episode_steps:
            step_type = StepType.TIMEOUT
        else:
            step_type = StepType.MID

        self.actions = [leader_action, follower_action]
        self.render_rewards = [target_reward, reward]

        env_info = {
            "leader_action": leader_action,
            "target_reward": target_reward,
            "follower_reward": reward,
        }

        return EnvStep(
            env_spec=self.spec,
            action=np.array(follower_action, dtype=np.int32),
            reward=reward,
            observation=np.array(self.state, dtype=np.int32),
            env_info=env_info,
            step_type=step_type,
        )

    def set_state(self, state):
        """Set the current state.

        Args:
            state: State to set

        """
        self.state = state

    def render(self, mode="human"):
        """Render the environment.

        Args:
            mode: Render mode (only 'human' supported)

        Returns:
            str: String representation of current state

        """
        if mode != "human":
            raise ValueError(f"Unsupported render mode: {mode}")

        if self.actions[0] is None or self.actions[1] is None:
            print(f"Step {self.steps_n}: state={self.STATE_TYPE[self.state]}")
        else:
            print(
                f"leader action={self.LEADER_ACT_TYPE[self.actions[0]]}, "
                f"follower action={self.FOLLOWER_ACT_TYPE[self.actions[1]]}, "
                f"reward={self.render_rewards[1]}, "
                f"target_reward={self.render_rewards[0]}",
            )
            print(f"Step {self.steps_n}: state={self.STATE_TYPE[self.state]}")
        return self.STATE_TYPE[self.state]

    def close(self):
        """Close the environment."""

    def get_opt_ag_act_array(self):
        """Get optimal follower action array.

        Returns:
            Array of optimal actions

        """
        return self.opt_ag_qtable.argmax(axis=2)


# Environment variants (same as ptia)
class DiscreteToyEnv1_1a(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(50, -50)
    leader action=1 -> best response return (follower, leader)=(50, 50)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, 0, 1] = 1
        self.transition[0, 1, 1] = 2
        self.transition[0, 0, 2] = 2
        self.transition[0, 1, 2] = 1
        self.transition[1, 0, 0] = 0
        self.transition[1, 1, 0] = 0
        self.transition[2, 0, 0] = 0
        self.transition[2, 1, 0] = 0

        self.rewards[1, :, 0] = 1
        self.rewards[2, :, 0] = -1
        self.r_range = (-1, 1)

        self.target_rewards[0, :, 1] = -1
        self.target_rewards[0, :, 2] = 1
        self.target_r_range = (-1, 1)

        self.key_state = 0
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 0] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 2] = 1
        self.opt_ag_qtable[1, 1, 0] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv1_1b(DiscreteToyEnv1_1a):
    """leader action=0 -> best response return (follower, leader)=(50, 0)
    leader action=1 -> best response return (follower, leader)=(50, 50)
    """

    def __init__(self):
        super().__init__()
        self.target_rewards[0, :, 1] = 0
        self.target_r_range = (0, 1)


class DiscreteToyEnv1_2a(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, 0, 1] = 1
        self.transition[0, 1, 1] = 2
        self.transition[0, 0, 2] = 2
        self.transition[0, 1, 2] = 1
        self.transition[1, 0, 0] = 0
        self.transition[1, 1, 0] = 0
        self.transition[2, 0, 0] = 0
        self.transition[2, 1, 0] = 0
        self.transition[1, 0, 2] = 2
        self.transition[1, 1, 2] = 2

        self.rewards[0, :, 2] = 2
        self.rewards[1, :, 0] = 3
        self.rewards[1, :, 2] = 8
        self.rewards[2, :, 0] = -3
        self.r_range = (-3, 8)

        self.target_rewards[1, :, 0] = 1
        self.target_rewards[1, :, 2] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 0
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 2] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 2] = 1
        self.opt_ag_qtable[1, 1, 0] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv1_2b(DiscreteToyEnv1_2a):
    """leader action=0 -> best response return (follower, leader)=(250, 0)
    leader action=1 -> best response return (follower, leader)=(375, 75)
    """

    def __init__(self):
        super().__init__()
        self.target_rewards[1, :, 2] = 0
        self.target_r_range = (0, 1)


class DiscreteToyEnv1_2c(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, :, 1] = 1
        self.transition[1, :, 0] = 0
        self.transition[1, :, 2] = 2
        self.transition[2, :, 0] = 0

        self.rewards[0, 1, 1] = 2
        self.rewards[1, :, 0] = 3
        self.rewards[2, :, 0] = 5
        self.r_range = (0, 5)

        self.target_rewards[1, :, 0] = 1
        self.target_rewards[1, :, 2] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 0
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 2] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 1] = 1
        self.opt_ag_qtable[1, 1, 0] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv1_2d(DiscreteToyEnv1_2c):
    """leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75)
    """

    def __init__(self):
        super().__init__()
        self.rewards[1, :, 2] = 2
        self.rewards[2, :, 0] = 3
        self.r_range = (0, 3)


class DiscreteToyEnv1_2e(DiscreteToyEnv1_2c):
    """leader action=0 -> best response return (follower, leader)=(250, -50)
    leader action=1 -> best response return (follower, leader)=(375, 75)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, :, 2] = 2

        self.rewards[1, :, 2] = 8
        self.rewards[2, :, 0] = -3
        self.r_range = (-3, 8)


class DiscreteToyEnv1_2f(DiscreteToyEnv1_2e):
    """leader policy f(0|S)=1.0:
        - best response return (follower, leader)=(1451, 195) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(250, -50)
        - S->A->B->S->A->B->...
    leader policy f(1|S)=1.0:
        - best response return (follower, leader)=(1575, 291) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(404, 75)
        - S->A->S->A->...
    """

    def __init__(self):
        super().__init__()
        g = 0.99  # discount rate must be 0.99
        R = (8 * (1 + g**4) - 3 * (g**2 + g**5) - 3 * (g + g**3 + g**5)) / (0.5 * (1 + g**2 + g**4) - 0.5 * (1 + g**3))
        # R = 2.3858910707072356
        self.rewards[0, 1, 1] = R
        self._max_episode_steps = np.inf  # Infinite horizon
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=None,
        )


class DiscreteToyEnv1_2g(DiscreteToyEnv1_2f):
    """leader policy f(0|S)=1.0:
        - best response return (follower, leader)=(1451, 195) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(250, 50)
        - S->A->B->S->A->B->...
    leader policy f(1|S)=1.0:
        - best response return (follower, leader)=(1575, 582) (infinite horizon)
        - horizon=150, discount=1.0: (follower, leader)=(404, 150)
        - S->A->S->A->...
    """

    def __init__(self):
        super().__init__()
        self.target_rewards[1, :, 0] = 2
        self.target_rewards[1, :, 2] = 1
        self.target_r_range = (0, 2)


class DiscreteToyEnv2_1(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(150, -50)
    leader action=1 -> best response return (follower, leader)=(150, 50)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, 0, 1] = 1
        self.transition[0, 1, 1] = 2
        self.transition[0, 0, 2] = 2
        self.transition[0, 1, 2] = 1
        self.transition[1, 0, 0] = 0
        self.transition[1, 1, 0] = 0
        self.transition[2, 0, 0] = 0
        self.transition[2, 1, 0] = 0
        self.transition[1, 0, 2] = 2
        self.transition[1, 1, 2] = 2

        self.rewards[1, :, 0] = 1
        self.rewards[2, :, 0] = -1
        self.rewards[1, :, 2] = 4
        self.r_range = (-1, 4)

        self.target_rewards[0, :, 1] = -1
        self.target_rewards[0, :, 2] = 1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 0
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 2] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 2] = 1
        self.opt_ag_qtable[1, 1, 2] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv2_2(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(150, 0)
    leader action=1 -> best response return (follower, leader)=(150, 50)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, 0, 1] = 1
        self.transition[0, 1, 1] = 1
        self.transition[1, 0, 0] = 0
        self.transition[1, 1, 0] = 2
        self.transition[1, 0, 2] = 2
        self.transition[1, 1, 2] = 0
        self.transition[2, 0, 0] = 0
        self.transition[2, 1, 0] = 0

        self.rewards[0, :, 1] = -1
        self.rewards[2, :, 0] = 4
        self.r_range = (-1, 4)

        self.target_rewards[1, :, 0] = 1
        self.target_r_range = (0, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 1
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 2] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 1] = 1
        self.opt_ag_qtable[1, 1, 0] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv3_1a(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=(75, -75)
    leader action=1 -> best response return (follower, leader)=(-50, 50)
    """

    def __init__(self):
        super().__init__()
        self.transition[0, 0, 1] = 1
        self.transition[0, 1, 1] = 2
        self.transition[0, 0, 2] = 2
        self.transition[0, 1, 2] = 1
        self.transition[1, 0, 0] = 0
        self.transition[1, 1, 0] = 0
        self.transition[2, 0, 0] = 0
        self.transition[2, 1, 0] = 0
        self.transition[1, 0, 2] = 2
        self.transition[1, 1, 2] = 2

        self.rewards = np.ones(self.rewards.shape) * -1
        self.rewards[0, :, 2] = -3
        self.rewards[1, :, 0] = 2
        self.rewards[1, :, 2] = 2
        self.rewards[2, :, 0] = 0
        self.r_range = (-3, 2)

        self.target_rewards[1, :, 2] = 1
        self.target_rewards[1, :, 0] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 0
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 0] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 2] = 1
        self.opt_ag_qtable[1, 1, 2] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv3_1b(DiscreteToyEnv3_1a):
    """leader action=0 -> best response return (follower, leader)=(75, 0)
    leader action=1 -> best response return (follower, leader)=(-50, 50)
    """

    def __init__(self):
        super().__init__()
        self.target_rewards[1, :, 0] = 0
        self.target_r_range = (0, 1)


class DiscreteToyEnv3_2(DiscreteToyEnvBase):
    """leader action=0 -> best response return (follower, leader)=((big_reward+1)*50, 0)
    leader action=1 -> best response return (follower, leader)=(75, 75)
    """

    def __init__(self, big_reward=29):
        super().__init__()
        self.transition[0, 0, 1] = 1
        self.transition[0, 1, 1] = 1
        self.transition[1, 0, 0] = 0
        self.transition[1, 1, 0] = 0
        self.transition[1, 0, 2] = 2
        self.transition[2, 0, 0] = 0
        self.transition[2, 1, 0] = 0

        self.rewards[0, :, 1] = 1
        self.rewards[2, :, 0] = big_reward
        self.r_range = (0, big_reward)

        self.target_rewards[1, :, 0] = 1
        self.target_r_range = (0, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 1
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 2] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 1] = 1
        self.opt_ag_qtable[1, 1, 0] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv4_1a(DiscreteToyEnvBase):
    """optimal action=0
    leader action=0 -> best response return (follower, leader)=(100, 50)
    leader action=1 -> best response return (follower, leader)=(75, -75)
    """

    def __init__(self, R=1.0):
        super().__init__()
        self.transition[0, :, 1] = 1
        self.transition[0, :, 2] = 2
        self.transition[1, 0, 0] = 0
        self.transition[1, 0, 2] = 2
        self.transition[1, 1, :] = 0
        self.transition[2, :, 0] = 0

        self.rewards[0, :, 1] = 1
        self.rewards[0, :, 2] = 1
        self.rewards[1, 1, :] = -1
        self.rewards[1, 0, 2] = R
        self.r_range = (-1, max(1, R))

        self.target_rewards[0, :, 1] = 1
        self.target_rewards[0, :, 2] = -1
        self.target_r_range = (-1, 1)

        self._max_episode_steps = 150
        self._spec = GlobalEnvSpec(
            observation_space=self._observation_space,
            action_space=self._action_space,
            leader_action_space=self._leader_action_space,
            max_episode_length=self._max_episode_steps,
        )

        self.key_state = 1
        self.optimal_action = 1
        self.opt_ag_qtable[0, 0, 1] = 1
        self.opt_ag_qtable[0, 1, 2] = 1
        self.opt_ag_qtable[0, 2, 0] = 1
        self.opt_ag_qtable[1, 0, 2] = 1
        self.opt_ag_qtable[1, 1, 0] = 1
        self.opt_ag_qtable[1, 2, 0] = 1


class DiscreteToyEnv4_1b(DiscreteToyEnv4_1a):
    """optimal action=0
    leader action=0 -> best response return (follower, leader)=(100, 50)
    leader action=1 -> best response return (follower, leader)=(75, 0)
    """

    def __init__(self, R=1.0):
        super().__init__(R)
        self.target_rewards[0, :, 2] = 0
        self.target_r_range = (0, 1)
