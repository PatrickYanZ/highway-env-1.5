import gym
import torch as th
from stable_baselines3 import PPO, A2C
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env

import sys
sys.path.append(r'G:\00temp\code\highway-env-1.5')

# import gym
# from gym.wrappers import RecordVideo
# from stable_baselines3 import DQN

# import highway_env
# from highway_env.envs.highway_obstacle_env import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

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

        tele_total_rewards = []
        tran_total_rewards = []

        self.tele_total_rewards = tele_total_rewards
        self.tran_total_rewards = tran_total_rewards


    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass


    def _on_step(self) -> bool: 

        # print(self.locals['infos'],type(self.locals['infos']))
        # idx = self.locals['infos'].index('agents_te_rewards')

        # tel_reward = self.locals['infos'][0]['agents_te_rewards']
        # self.tele_rewards.append(tel_reward)

        # print(self.locals['infos'])
        # print(tel_reward)

        tel_reward_all = self.locals['infos'][0]['agents_tele_all_rewards']
        self.tele_total_rewards.append(tel_reward_all)

        tran_reward_all = self.locals['infos'][0]['agents_tran_all_rewards']
        self.tran_total_rewards.append(tran_reward_all)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # tele_reward = np.mean(self.tele_rewards)
        # self.logger.record('rollout/tele_reward', tele_reward)
        # tele_reward = 0
        # self.tele_reward = 0
        # self.tele_rewards = []

        self.ho_prob = self.locals['infos'][0]['agents_ho_prob']
        self.logger.record('rollout/ho_prob', self.ho_prob[0])
        # self.ho_prob = 1e-9

        tel_reward_all_mean = np.mean(self.tele_total_rewards)
        tran_reward_all_mean = np.mean(self.tran_total_rewards)

        self.logger.record('rollout/tel_mean', tel_reward_all_mean)
        self.logger.record('rollout/tran_mean', tran_reward_all_mean)

        self.tele_total_rewards = []
        self.tran_total_rewards = []
        

        return True


# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("highway-bs-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = A2C("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    # batch_size=batch_size,
                    # n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="highway_a2c/")
        # Train the agent
        model.learn(total_timesteps=int(5e4), callback=TensorboardCallback())
        # Save the agent
        model.save("highway_ppo/model")

    # model = PPO.load("highway_ppo/model")
    # env = gym.make("highway-fast-v0")
    # for _ in range(5):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         action, _ = model.predict(obs)
    #         obs, reward, done, info = env.step(action)
    #         env.render()
