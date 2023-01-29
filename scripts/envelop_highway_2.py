import mo_gymnasium as mo_gym
from mo_gymnasium.utils import MORecordEpisodeStatistics
from gym import spaces
# from mo_gym import spaces
from morl_baselines.multi_policy.envelope.envelope import Envelope

import highway_env

def main():
    def make_env():
        # env = mo_gym.make("minecart-v0")
        # env = mo_gym.make("highway-mofast-v0")
        env = mo_gym.make("mo-highway-fast-v0")
        env = MORecordEpisodeStatistics(env, gamma=0.98)
        env = mo_gym.LinearReward(env)
        return env

    env = make_env()
    print('obs space',env.observation_space)
    print('act space',env.action_space)
    print('is instance ? ',isinstance(env.action_space, (spaces.Discrete, spaces.MultiBinary)))
    print('act space shape',env.action_space.shape)
    # print('act space shape',env.action_space.shape[0])
    print('reward space',env.reward_space)
    eval_env = make_env()
    # RecordVideo(make_env(), "videos/minecart/", episode_trigger=lambda e: e % 1000 == 0)

    agent = Envelope(
        env,
        max_grad_norm=0.1,
        learning_rate=3e-4,
        gamma=0.98,
        batch_size=64,
        net_arch=[256, 256],#, 256, 256
        buffer_size=int(2e6),
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=50000,
        initial_homotopy_lambda=0.0,
        final_homotopy_lambda=1.0,
        homotopy_decay_steps=10000,
        learning_starts=100,
        envelope=True,
        gradient_updates=1,
        target_net_update_freq=1000,  # 1000,  # 500 reduce by gradient updates
        tau=1,
        log=True,
        project_name="MORL-Baselines",
        experiment_name="Envelope - minecart-v0",
    )

    agent.train(
        total_timesteps=100000,
        total_episodes=None,
        weight=None,
        eval_env=eval_env,
        eval_freq=1000,
        reset_num_timesteps=False,
        reset_learning_starts=False,
    )


if __name__ == "__main__":
    main()
