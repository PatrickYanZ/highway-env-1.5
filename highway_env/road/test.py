# Import the necessary libraries
import gym
from stable_baselines3 import A2C, PPO, DDPG

# Create the highway environment
env = gym.make('highway-v0')

# Train the agent using A2C
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Train the agent using PPO
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Train the agent using DDPG
model = DDPG('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)