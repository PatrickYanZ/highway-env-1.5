import gym
import torch as th
from stable_baselines3 import DQN,PPO, A2C
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv,VecVideoRecorder, DummyVecEnv
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

def train_env():
    env = gym.make('highway-bs-v0')
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
    })
    env.reset()
    return env

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        model = DQN('CnnPolicy', DummyVecEnv([train_env]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="highway_cnn/")
        model.learn(total_timesteps=int(1e5), callback=TensorboardCallback())
        model.save("highway_cnn/model")

    # # Record video
    # model = DQN.load("highway_cnn/model")

    # env = DummyVecEnv([test_env])
    # video_length = 2 * env.envs[0].config["duration"]
    # env = VecVideoRecorder(env, "highway_cnn/videos/",
    #                        record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                        name_prefix="dqn-agent")
    # obs, info = env.reset()
    # for _ in range(video_length + 1):
    #     action, _ = model.predict(obs)
    #     obs, _, _, _, _ = env.step(action)
    # env.close()

    # model = PPO.load("highway_ppo/model")
    # env = gym.make("highway-fast-v0")
    # for _ in range(5):
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         action, _ = model.predict(obs)
    #         obs, reward, done, info = env.step(action)
    #         env.render()
