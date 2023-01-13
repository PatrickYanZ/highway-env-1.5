from highway_env.envs.highway_env import *
from highway_env.envs.merge_env import *
from highway_env.envs.parking_env import *
from highway_env.envs.summon_env import *
from highway_env.envs.roundabout_env import *
from highway_env.envs.two_way_env import *
from highway_env.envs.intersection_env import *
from highway_env.envs.lane_keeping_env import *
from highway_env.envs.u_turn_env import *
from highway_env.envs.exit_env import *
from highway_env.envs.racetrack_env import *

register(
    id='highway-bs-v0',
    entry_point='highway_env.envs:HighwayEnvBS',
)

# register(
#     id='highway-mo-v0',
#     entry_point='highway_env.envs:MOHighwayEnv',
# )

# register(
#     id='highwayfast-mo-v0',
#     entry_point='highway_env.envs:MOHighwayEnvFast',
# )