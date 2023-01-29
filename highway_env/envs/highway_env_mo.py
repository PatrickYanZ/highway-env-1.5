import numpy as np
from gym.envs.registration import register
from gym.spaces import Box

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork, BSRoad
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from typing import Dict, Text, Tuple, List  # , Self
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.objects import RF_BS, THz_BS

from ..sinr import *
from ..Shared import *
import pandas as pd

import gym
import gymnasium
import numpy as np
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    Graph,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Text,
    Tuple,
)
from highway_env.envs.highway_env import HighwayEnv, HighwayEnvFast


class MOHighwayEnv(HighwayEnv):
    """A multi-objective version of the HighwayEnv environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = _convert_space(self.observation_space)
        self.action_space = _convert_space(self.action_space)

    def step(self, action):
        # obs, reward, terminated, truncated, info = super().step(action)
        # highway_env 1.5 not implemented truncated so far.
        obs, reward, terminated, info = super().step(action)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"],
                rewards["right_lane_reward"],
                -rewards["collision_reward"],
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        info["original_reward"] = reward
        return obs, vec_reward, terminated, info #, truncated


class MOHighwayEnvFast(HighwayEnvFast):
    """A multi-objective version of the HighwayFastEnv environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = _convert_space(self.observation_space)
        self.action_space = _convert_space(self.action_space)

    def step(self, action):
        # obs, reward, terminated, truncated, info = super().step(action)
        # highway_env 1.5 not implemented truncated so far.
        obs, reward, terminated, info = super().step(action)
        print('info\n',info)
        rewards = info["rewards"]
        vec_reward = np.array(
            [
                rewards["high_speed_reward"],
                rewards["right_lane_reward"],
                -rewards["collision_reward"],
            ],
            dtype=np.float32,
        )
        vec_reward *= rewards["on_road_reward"]
        info["original_reward"] = reward
        return obs, vec_reward, terminated, info #, truncated


def _convert_space(space: gym.Space) -> gymnasium.Space:
    """Converts a gym space to a gymnasium space.
    Args:
        space: the space to convert
    Returns:
        The converted space
    """
    if isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Tuple(spaces=tuple(map(_convert_space, space.spaces)))
    elif isinstance(space, gym.spaces.Dict):
        return Dict(spaces={k: _convert_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, gym.spaces.Sequence):
        return Sequence(space=_convert_space(space.feature_space))
    elif isinstance(space, gym.spaces.Graph):
        return Graph(
            node_space=_convert_space(space.node_space),  # type: ignore
            edge_space=_convert_space(space.edge_space),  # type: ignore
        )
    elif isinstance(space, gym.spaces.Text):
        return Text(
            max_length=space.max_length,
            min_length=space.min_length,
            charset=space._char_str,
        )
    else:
        raise NotImplementedError(f"Cannot convert space of type {space}. Please upgrade your code to gymnasium.")


register(
    id='highway-mofast-v0',
    entry_point='highway_env.envs:MOHighwayEnvFast',
)

register(
    id='highway-mo-v0',
    entry_point='highway_env.envs:MOHighwayEnv',
)
