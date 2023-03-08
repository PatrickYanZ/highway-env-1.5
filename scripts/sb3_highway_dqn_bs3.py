import sys
sys.path.append(r'G:\00temp\code\highway-env-1.5')

import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env
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


TRAIN = True

if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-bs-v0")
    obs = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[64,64]),#32,
                learning_rate=8e-2,
                buffer_size=15000,
                learning_starts=500,
                batch_size=512,#512
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction = 0.7,
                verbose=1,
                tensorboard_log="highway_dqn/")

    # Train the model
    if TRAIN:
        # model.learn(total_timesteps=int(3e2), callback=TensorboardCallback())#2e4 1e5
        model.learn(int(4e4), callback=TensorboardCallback())#2e4 1e5
        model.save("highway_dqn/model/bs230108-normalize-model-v1020")
        del model

    # # Run the trained model and record video
    # model = DQN.load("highway_dqn/model/bs", env=env)
    # env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    # for videos in range(10):
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         # Predict
    #         action, _states = model.predict(obs, deterministic=True)
    #         # Get reward
    #         obs, reward, done, info = env.step(action)
    #         # Render
    #         env.render()
    # env.close()
