import stable_baselines3 as sb3
import numpy as np
import gymnasium as gym
# import mo_gymnasium as mo_gym
import mo_gymnasium as mo_gym
import highway_env
# import mo_gym

env = mo_gym.make("mo-mountaincar-v0", render_mode="human")
print(env.observation_space)
print(env.observation_space)
print(env.action_space)
print(env.reward_space)
# with a random agent
env.reset()
done = False

while not done:
    obs, vec_reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    
env.close()
