import sys
sys.path.append(r'G:\00temp\code\highway-env-1.5')

import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN

import highway_env

TRAIN = True

env = gym.make("highway-bs-v0")
obs = env.reset()

import cProfile
import pstats
import os
os.environ['PROFILING'] = 'y'
# 性能分析装饰器定义
def do_cprofile(filename):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = os.getenv("PROFILING")
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper


# @do_cprofile("./mkm_run.prof")
def test():
    for i in range(100):
        print(f'\n{i}\n')
        _, _, over, _ = env.step(env.action_space.sample())
        if over:
            env.reset()
    


'''
状态:
1. 给定AV和当前车道中的前AV在x方向上的相对距离 dfc，
2. 给定AV与x方向相邻车道上的前AV之间的相对距离 dft
3. 给定AV与x方向相邻车道上的后AV之间的相对距离 drt
4. x方向上当前车道上给定AV和前AV之间的相对速度 vfc
5. x方向上相邻车道上给定AV和前AV之间的相对速度 vrc
6. 给定的AV和在x方向上的相邻车道中的后AV之间的相对速度 vrt，
7. 在当前AV周围提供最佳数据速率的三个BS中，满足给定AV的期望数据速率的BS的数量 c (0, 1, 2, 3)
8. 给定AV停留的车道 lid

距离离散为3种状态(<=>): 最小安全距离dc, 最大安全距离df
速度也是3种离散状态. v <0 =0 >0 对应接近 同速 分开




'''

if __name__ == '__main__':
    test()
    