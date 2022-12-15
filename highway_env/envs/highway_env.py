import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

from typing import Dict, Text
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.objects import RF_BS, THz_BS

from ..sinr import *
import pandas as pd

Observation = np.ndarray

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                # "type": "DiscreteMetaAction",
                "type": "DiscreteDualObjectMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "rf_reward":1,
            "dr_reward": 0.2,
            "ho_reward":-0.2,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class HighwayEnvObstacle(HighwayEnvFast):

    @classmethod
    def default_config(cls) -> dict:
        conf = super().default_config()
        conf.update({
            "obstacle_count": 20,
            # https://github.com/eleurent/highway-env/issues/35#issuecomment-1206427869
            "termination_agg_fn": 'any'
        })
        return conf

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        # vehicle_dist = 0.0
        # lanes = [4 * lane for lane in range(self.config["lanes_count"])]
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            if self.config['controlled_vehicles']:
                # vehicle_lane = np.random.choice(lanes)
                # To make sure the agents doesn't collide on the start itself because of the random obstacles.
                # vehicle.position = np.array([vehicle_dist, vehicle_lane])
                # vehicle_dist += 25
                lanes = [4 * lane for lane in range(self.config["lanes_count"])]
                vehicle_lane = np.random.choice(lanes)
                vehicle.position = np.array([vehicle_dist, vehicle_lane])
                vehicle_dist += 25
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                self.controlled_vehicles.append(vehicle)
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        # Adding obstacles at random places on the lanes
        for i in range(1, self.config['obstacle_count']):
            lanes = [4 * lane for lane in range(self.config["lanes_count"])]
            obstacle_lane = np.random.choice(lanes)
            obstacle_dist = np.random.randint(300, 10000)
            self.road.objects.append(Obstacle(self.road, [obstacle_dist, obstacle_lane]))

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info['other_vehicle_collision'] = \
            sum(vehicle.crashed for vehicle in self.road.vehicles if vehicle not in self.controlled_vehicles)
        # changed
        info['agents_rewards'] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info['agents_collided'] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info['distance_travelled'] = tuple(vehicle.position[0] for vehicle in self.controlled_vehicles)
        info['agents_survived'] = self._is_truncated()
        return info

    # To check if a single agent has collided
    def _agent_is_terminal(self, vehicle) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return vehicle.crashed or \
            (self.config["offroad_terminal"] and not vehicle.on_road)

    # To terminate when the duration limit has reached.
    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

    # To terminate training based on any or all agent has collided.
    def _is_terminated(self) -> bool:
        # https://github.com/eleurent/highway-env/issues/35#issuecomment-1206427869
        agent_terminal = [self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles]
        agg_fn = {'any': any, 'all': all}[self.config['termination_agg_fn']]
        return agg_fn(agent_terminal)

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents"""
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }


    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road)
        }

class HighwayEnvBS(HighwayEnvFast):

    @classmethod
    def default_config(cls) -> dict:
        conf = super().default_config()
        conf.update({
            "obstacle_count": 20,
            # https://github.com/eleurent/highway-env/issues/35#issuecomment-1206427869
            # https://github.com/eleurent/highway-env/pull/352/files
            "termination_agg_fn": 'any',
            'rf_bs_count':20,
            'thz_bs_count':100
        })
        return conf

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        vehicle_dist = 0.0
        # lanes = [4 * lane for lane in range(self.config["lanes_count"])]
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            if self.config['controlled_vehicles']:
                # vehicle_lane = np.random.choice(lanes)
                # To make sure the agents doesn't collide on the start itself because of the random obstacles.
                # vehicle.position = np.array([vehicle_dist, vehicle_lane])
                # vehicle_dist += 25
                lanes = [4 * lane for lane in range(self.config["lanes_count"])]
                vehicle_lane = np.random.choice(lanes)
                vehicle_dist += 25
                vehicle.position = np.array([vehicle_dist, vehicle_lane])
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                self.controlled_vehicles.append(vehicle)
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        print('lanes count', self.config["lanes_count"])#debug
        # Adding obstacles at random places on the lanes
        for i in range(1, self.config['obstacle_count']):
            lanes = [4 * lane for lane in range(self.config["lanes_count"])]
            # print('lanes count', self.config["lanes_count"])#debug
            # print('lanes are ',lanes)#debug
            obstacle_lane = np.random.choice([0,8]) #random generate lane number (integer) obstacle_lane = np.random.choice(lanes)
            # obstacle_lane = 0
            # print('obstacle_lane is ', obstacle_lane) #debug
            obstacle_dist = np.random.randint(300, 10000)
            # print('obstacle_dist is ', obstacle_dist) #debug
            self.road.objects.append(Obstacle(self.road, [obstacle_dist, obstacle_lane]))
        
            '''creating RF bss'''
        for i in range(1, self.config['rf_bs_count']):
            rf_bs_lane = np.random.choice([0,8]) #random generate lane number (integer) obstacle_lane = np.random.choice(lanes)
            rf_bs_dist = np.random.randint(300, 10000)
            # self.road.objects.append(RF_BS(self.road, [rf_bs_dist, rf_bs_lane]))
            self.road.rf_bss.append(RF_BS(self.road, [rf_bs_dist, rf_bs_lane]))

            '''creating thz bss'''
        for i in range(1, self.config['thz_bs_count']):
            thz_bs_lane = np.random.choice([0,8]) #random generate lane number (integer) obstacle_lane = np.random.choice(lanes)
            thz_bs_dist = np.random.randint(300, 10000)
            # self.road.objects.append(RF_BS(self.road, [rf_bs_dist, rf_bs_lane]))
            self.road.thz_bss.append(THz_BS(self.road, [thz_bs_dist, thz_bs_lane]))

    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info['other_vehicle_collision'] = \
            sum(vehicle.crashed for vehicle in self.road.vehicles if vehicle not in self.controlled_vehicles)
        # changed
        # info['agents_rewards'] = tuple(self._agent_reward(action, vehicle)[0] for vehicle in self.controlled_vehicles)

        # tran changed
        
        # tele changed
        info['agents_te_rewards'] = tuple(self._agent_rewards(action, vehicle)['rf_reward'] for vehicle in self.controlled_vehicles)
        info['agents_rewards'] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        # info['agents_tr_rewards'] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) to be implemented
        info['agents_collided'] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info['distance_travelled'] = tuple(vehicle.position[0] for vehicle in self.controlled_vehicles)
        info['agents_survived'] = self._is_truncated()
        return info

    # To check if a single agent has collided
    def _agent_is_terminal(self, vehicle) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return vehicle.crashed or \
            (self.config["offroad_terminal"] and not vehicle.on_road)

    # To terminate when the duration limit has reached.
    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

    # To terminate training based on any or all agent has collided.
    def _is_terminated(self) -> bool:
        # https://github.com/eleurent/highway-env/issues/35#issuecomment-1206427869
        agent_terminal = [self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles]
        agg_fn = {'any': any, 'all': all}[self.config['termination_agg_fn']]
        return agg_fn(agent_terminal)

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents"""
        # reward,reward_tr,reward_te = self._agent_reward(action, vehicle)
        # sum_total_reward = sum(self._agent_reward(action, vehicle)[0] for vehicle in self.controlled_vehicles) \
        #        / len(self.controlled_vehicles)
        # sum_tr_reward = sum(self._agent_reward(action, vehicle)[1] for vehicle in self.controlled_vehicles) \
        #        / len(self.controlled_vehicles)
        # sum_te_reward = sum(self._agent_reward(action, vehicle)[2] for vehicle in self.controlled_vehicles) \
        #        / len(self.controlled_vehicles)
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)
        # return sum_total_reward #,sum_tr_reward,sum_te_reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        agents_rewards = [self._agent_rewards(action, vehicle) for vehicle in self.controlled_vehicles]
        return {
            name: sum(agent_rewards[name] for agent_rewards in agents_rewards) / len(agents_rewards)
            for name in agents_rewards[0].keys()
        }


    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """Per-agent reward signal."""
        rewards = self._agent_rewards(action, vehicle)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward += rewards['rf_reward']
        reward *= rewards['on_road_reward']
        reward_tr = reward - rewards['rf_reward']
        reward_te = rewards['rf_reward']
        return reward #,reward_tr,reward_te



    def _agent_rewards(self, action: int, vehicle: Vehicle) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
        lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
            else vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = vehicle.speed * np.cos(vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        distance_matrix_rf, vehicles,bss_rf = HighwayEnvBS._get_distance_rf_matrix(self)
        # UNIMPLEMENTED
        distance_matrix_thz, vehicles,bss_thz = HighwayEnvBS._get_distance_thz_matrix(self)

        vid = vehicle._get_vehicle_id()
        # rf_sinr = rf_sinr_matrix(distance_matrix,vehicles,bss)
        rf_dr = get_rf_dr(distance_matrix_rf,vehicles,bss_rf)
        result_rf = rf_dr.loc[vid] # current vehicle dr

        thz_dr = get_thz_dr(distance_matrix_thz,vehicles,bss_thz)
        result_thz = thz_dr.loc[vid]
        # print("result is ",type(result),result)
        # nearst_bs_id,nearst_rf_bs_distance = HighwayEnvBS._get_min_bs(result) # min(distance_matrix["v"+str(vid)])
        bs_min_name,min_rate,bs_max_name,max_rate = HighwayEnvBS._get__max_min_dr_bs(result_rf)
        print('shape before and after ',distance_matrix_rf.shape,rf_dr.shape)
        return {
            "collision_reward": float(vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(vehicle.on_road),
            "rf_reward": float(max_rate)
            # "rf_reward": float(1/rf_sinr_specific_vehicle)
        }

    def _get__max_min_dr_bs(ser):
        '''
        Input
        rb448    2486.719716
        rb496    4441.200057
        rb544    1544.227621
        rb592    6483.191725
        rb640    9397.511902
        rb504    4013.541322
        rb552    6649.137008
        rb184    7769.828695
        rb136    2415.887817
        rb88     8791.616117
        rb40     7742.835072
        rb992    8767.620541
        rb896    3340.255970
        rb352    4663.047710
        rb304    2038.979633
        rb256    4400.229874
        rb208    2968.789306
        rb160    6204.290285
        rb112    2258.299791

        return rb544, 1544.227621

        '''
        bs_min_name = ser.idxmin()
        min_rate = np.min(ser)
        bs_max_name = ser.idxmax()
        max_rate = np.max(ser)
        # print("bs name and min rate",bs_name,min_rate,max_rate)
        return bs_min_name,min_rate,bs_max_name,max_rate
    
    def _get_n_max_bs(ser,n):
        '''
        return n maximum data rate with corresponding bs id
        Also useful for distance 
        '''
        return ser.nlargest(n)

    def _get_n_min_bs(ser,n):
        '''
        return n minimum data rate with corresponding bs id
        '''
        return ser.nsmallest(n)


    def list_drop_duplicate(x):
        return list(dict.fromkeys(x))

    def _get_distance_rf_matrix(self):
        '''
        distance matrice between AVs and RF BSs.
        '''
        bss = self.road.rf_bss
        vehicles = self.road.vehicles
        # vehicles = self.controlled_vehicles
        distance_matrix = pd.DataFrame() 

        vehicle_list = []
        bs_list = []

        for v in vehicles:
            x2,y2 = v.position
            vid = v._get_vehicle_id()
            # vid_str = "v" + str(vid)
            vehicle_list.append(vid)
            # distance_matrix[vid_str] = []
            bs_list = []
            for bs in bss:
                x1,y1 = bs.position
                distance = utils.relative_distance(x1,x2,y1,y2)
                bid = bs._get_rf_bs_id()
                bs_list.append(bid)
                distance_matrix.at[vid,bid]=distance

            
        # print(distance_matrix)
        # print('before vehicle len',len(vehicle_list),vehicle_list)
        # print('before bss len',len(bs_list),bs_list)
        vehicle_list = list( dict.fromkeys(vehicle_list) )
        bs_list = list( dict.fromkeys(bs_list) )
        # print('after vehicle len',len(vehicle_list),vehicle_list)
        # print('after bss len',len(bs_list),bs_list)
        return distance_matrix,vehicle_list,bs_list


    def _get_distance_thz_matrix(self):
        '''
        distance matrice between AVs and Thz BSs.
        '''
        bss = self.road.thz_bss
        vehicles = self.road.vehicles
        # vehicles = self.controlled_vehicles
        distance_matrix = pd.DataFrame() 

        vehicle_list = []
        bs_list = []

        for v in vehicles:
            x2,y2 = v.position
            vid = v._get_vehicle_id()
            vehicle_list.append(vid)
            bs_list = []
            for bs in bss:
                x1,y1 = bs.position
                distance = utils.relative_distance(x1,x2,y1,y2)
                bid = bs._get_thz_bs_id()
                bs_list.append(bid)
                distance_matrix.at[vid,bid]=distance

        vehicle_list = list( dict.fromkeys(vehicle_list) )
        bs_list = list( dict.fromkeys(bs_list) )
        return distance_matrix,vehicle_list,bs_list

    # def _get_sinr_rf_matrix(self):
    #     '''
    #     distance matrice between AVs and RF BSs.
    #     '''
    #     bss = self.road.rf_bss
    #     vehicles = self.road.vehicles
    #     distance_matrix = {} #dict()


    #     for v in vehicles:
    #         x2,y2 = v.position
    #         vid = int(v._get_vehicle_id())
    #         distance_matrix[vid] = {}
    #         for bs in bss:
    #             x1,y1 = bs.position
    #             distance = utils.relative_distance(x1,x2,y1,y2)
    #             bid = int(bs._get_rf_bs_id())
    #             distance_matrix[vid][bid] = distance
    #             # print("vehicle is is ",vid,"rf bs id is",bid,'relative distance is ',distance)
    #     return distance_matrix


    # def _get_distance_thz_matrix():
    #     '''
    #     distance matrice between AVs and THz BSs.
    #     '''
    #     return 0
    
    def _get_3_nearst_bss():
        '''
        we iterate the list of all base stations and current vehicle
        keep the set of base stations set which capacity is 3.
        Iterate the function to get the distance between base station and insert corresponding id to the the current set.
        return the 3 nearst bs id
        '''
        return 0
    
    def _get_3_maximum_sinr_bss():
        '''
        we iterate the list of all base stations and current vehicle
        keep the set of base stations set which capacity is 3.
        Iterate the function to get the sinr between base station and insert corresponding id to the the current set.
        return the 3 nearst bs id
        '''
        return 0
    
    def _relative_distance(x1,x2,y1,y2):
        return utils.relative_distance(x1,x2,y1,y2)



register(
    id='highway-bs-v0',
    entry_point='highway_env.envs:HighwayEnvBS',
)

register(
    id='highway-obstacle-v0',
    entry_point='highway_env.envs:HighwayEnvObstacle',
)

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)
