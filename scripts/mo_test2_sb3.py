import stable_baselines3 as sb3
import numpy as np
import gymnasium as gym
import mo_gymnasium as mo_gym
import gym
import highway_env

# Linear scalarizes the environment
env = mo_gym.LinearReward(mo_gym.make("resource-gathering-v0"), weight=np.array([0.8, 0.2, 0.2]))

# Run DQN agent!
agent = sb3.DQN("MlpPolicy", env)
agent.learn(1000)