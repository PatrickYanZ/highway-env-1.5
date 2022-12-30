import sys
sys.path.append(r'G:\00temp\code\highway-env-1.5')

import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env
# from highway_env.envs.highway_obstacle_env import *

TRAIN = True

if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-bs-v0")
    obs = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="highway_dqn/")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e2))#2e4
        model.save("highway_dqn/model/bs")
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
