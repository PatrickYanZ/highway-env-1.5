import gymnasium
import mo_gymnasium as mo_gym

env = mo_gym.make('minecart-v0') # It follows the original Gymnasium API ...

obs = env.reset()
next_obs, vector_reward, terminated, truncated, info = env.step(your_agent.act(obs))  # but vector_reward is a numpy array!

# Optionally, you can scalarize the reward function with the LinearReward wrapper
env = mo_gym.LinearReward(env, weight=np.array([0.8, 0.2, 0.2]))