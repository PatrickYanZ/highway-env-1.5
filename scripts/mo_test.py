import gym
import mo_gym

env = mo_gym.make("mo-mountaincar-v0")#, render_move="human"
env.observation_space
env.action_space
env.reward_space

env.reset()
done = False

while not done:
    obs, vec_reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print(obs, vec_reward, terminated, truncated, info)
env.close()