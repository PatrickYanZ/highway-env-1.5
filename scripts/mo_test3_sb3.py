import stable_baselines3 as sb3
import numpy as np
# import mo_gym
import mo_gymnasium as mo_gym


# Linear scalarizes the environment
# env = mo_gym.LinearReward(mo_gym.make("mo-mountaincar-v0"), weight=np.array([0.9, 0.1, 0.0]))
# Create the environment
# env = mo_gym.make("minecart-v0")
env = mo_gym.make("mo-mountaincar-v0")
obs = env.reset()

# Create the model
env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))
model = sb3.DQN('MlpPolicy', env)
# Run DQN agent!
model.learn(1000)