from const import *

import gym
import pybullet_envs
import gym_remote.client as grc
import numpy as np


def create_env(env_name, contest=False, human=False):
    if human:
        env = HalfCheetahDiscretizer(gym.make(env_name, scenario="contest", use_restricted_actions=retro.ACTIONS_FILTERED))
    elif not contest:
        env = HalfCheetahDiscretizer(gym.make(env_name))
    else:
        env = HalfCheetahDiscretizer(grc.RemoteEnv('tmp/sock'))
    return env


class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info
    
    def get_act(self, a):
        return self.env.get_act(a)


class HalfCheetahDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(HalfCheetahDiscretizer, self).__init__(env)

        self._actions = []
        self._actions.append(np.zeros((6,), dtype=np.float32))
        self.action_space = gym.spaces.Discrete(6)


    def filter_act(self, a):
        """ Removing weird combos of buttons / useless buttons """
        for i in range(len(a)):
            if abs(a[i]) > 1:
                a[i] = abs(a[i])/a[i]
        return a

    def action(self, a):
        if isinstance(a, np.ndarray):
            a = self.filter_act(a)
            return a.copy()
        return self._actions[a].copy()
    
    
    def get_act(self, a):
        a = self.filter_act(a)
        for i in range(len(self._actions)):
            if np.array_equal(self._actions[i], a):
                return i
