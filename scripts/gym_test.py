import stable_baselines3 as sb3
import numpy as np
import gymnasium as gym
# import mo_gymnasium as mo_gym
# import mo_gymnasium as mo_gym
# import highway_env

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

model = sb3.DQN('MlpPolicy', env)
# Run DQN agent!
model.learn(1000)