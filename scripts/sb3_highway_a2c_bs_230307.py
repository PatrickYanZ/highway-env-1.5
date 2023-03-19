import os
from datetime import datetime, timezone, tzinfo

import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np

import highway_env
# from highway_env.envs.highway_obstacle_env import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        # tele_rewards = []
        ho_prob = 1e-9
        # self.tele_rewards = tele_rewards
        self.ho_prob = ho_prob



    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass


    def _on_step(self) -> bool: 

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.logger.record('rollout/ho_prob', self.locals['infos'][0]['agents_ho_prob'][0])
        self.logger.record('rollout/tel_mean', self.locals['infos'][0]['agents_tele_average_rewards'][0])
        self.logger.record('rollout/tran_mean', self.locals['infos'][0]['agents_tran_average_rewards'][0])
        self.logger.record('rollout/tel_total', self.locals['infos'][0]['agents_tele_total_rewards'][0])
        self.logger.record('rollout/tran_total', self.locals['infos'][0]['agents_tran_total_rewards'][0])
        self.logger.record('rollout/steps', self.locals['infos'][0]['agents_self_steps'][0])
        self.logger.record('rollout/distance_travelled', self.locals['infos'][0]['distance_travelled'][0])
        survive = 0
        if not self.locals['infos'][0]['crashed']:
            survive = 1
        else:
            survive = 0
        self.logger.record('rollout/agents_survived', survive)

        return True


TRAIN = True

rpath = "dqn_test/a2c_"+datetime.now().strftime('%Y%m%d_%H%M%S')+"/"
# tmp_path = "/highway_dqn/sb3_log/"+rpath
#tmp_path = r"I:\Research\tcom paper\highway-env-1.5\scripts\dqn_aaa" 
# set up logger
new_logger = configure(rpath, ["stdout", "csv", "tensorboard"])

'''
this will not work since OSError: [Errno 30] Read-only file system: '/highway_dqn'
# tmp_path = "/tmp/sb3_log/"
rpath = "dqn_test/dqn_"+datetime.now().strftime('%Y%m%d_%H%M%S')+"/"
tmp_path = "/highway_dqn/sb3_log/"+rpath
#tmp_path = r"I:\Research\tcom paper\highway-env-1.5\scripts\dqn_aaa" 
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
'''


if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-bs-v0")
    obs = env.reset()

    # # Create the model
    # model = DQN('MlpPolicy', env,
    #             policy_kwargs=dict(net_arch=[64,64,64]),#32,
    #             learning_rate=5e-2,
    #             buffer_size=15000,
    #             learning_starts=500,
    #             batch_size=512,#512
    #             gamma=0.8,
    #             train_freq=1,
    #             gradient_steps=1,
    #             target_update_interval=50,
    #             exploration_fraction = 0.5,
    #             verbose=1,
    #             tensorboard_log="highway_dqn/")

    if __name__ == "__main__":
        train = True
        if train:
            n_cpu = 8
            batch_size = 256
            env = make_vec_env("highway-bs-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
            model = A2C("MlpPolicy",
                        env,
                        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                        n_steps=256 * 12 // n_cpu,
                        # batch_size=batch_size,
                        # n_epochs=10,
                        learning_rate=5e-4,
                        gamma=0.8,
                        verbose=2,
                        tensorboard_log="highway_a2c/")
            # model = A2C("MlpPolicy", env, device="cpu")
            # Train the agent
            model.set_logger(new_logger)
            model.learn(total_timesteps=int(1e5), callback=TensorboardCallback())
            # Save the agent
            model.save("highway_ppo/model/bs230315a2c-20-30")

    # # Train the model
    # if TRAIN:
    #     # model.learn(total_timesteps=int(3e2), callback=TensorboardCallback())#2e4 1e5
    #     model.set_logger(new_logger)
    #     model.learn(int(1e3), callback=TensorboardCallback())#2e4 1e5
    #     model.save("highway_dqn/model/bs230307")
    #     # del model
