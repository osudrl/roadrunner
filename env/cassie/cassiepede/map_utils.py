import logging
import re

import math
# import cv2
import numpy as np
from bs4 import BeautifulSoup

from util.quaternion import euler2quat

logging.basicConfig(level=logging.INFO)


class MapUtils:
    def __init__(self):
        self.goal_name = self._get_geom_name('goal')
        # assert len(self.goal_name) == 1
        # self.goal_name = self.goal_name[0]

        self.obstacles_name = self._get_geom_name('obs_*')
        # logging.info(f'There are {len(self.obstacles_name)} obstacles:{self.obstacles_name}')

    def _get_geom_name(self, query):
        with open('sim/cassie_sim/cassiemujoco/cassie_obstacle.xml') as r:
            bs = BeautifulSoup(r, features="html.parser")
            geoms_name = [geom.attrs['name'] for geom in bs.find_all('geom', {'name': re.compile(query)})]
            return geoms_name

    def get_goal_properties(self, sim):
        goal_name = self._get_geom_name('goal')

        assert len(goal_name) == 1, "There must be one geom named goal in cassie.xml"

        goal_name = goal_name[0]

        pos = sim.get_geom_pos(goal_name)
        size = sim.get_geom_size(goal_name)

        return pos, size

    def get_obstacle_bbox(self, sim):
        positions = []
        for obstacle_name in self.obstacles_name:
            positions.append(
                np.concatenate((sim.get_geom_pos(obstacle_name)[:2], sim.get_geom_size(obstacle_name)[:1])))

        return np.array(positions)

    def set_random_goal(self, sim):
        x, y = np.random.randint(low=-4, high=4, size=2)
        # x, y = np.random.randint(low=[-4, -4], high=[5, 5], size=2)
        # x, y = 3, 0

        # x, y = 5, 0
        # z = 0.25
        z = 0.001

        pos = [x, y, z]
        self.set_goal_position(sim, pos)

        return np.array(pos)

    @staticmethod
    def get_non_colliding_points(min_dist, low, high, n, default_points):
        def valid(p):
            for _p in default_points:
                if np.linalg.norm(p - _p) < min_dist:
                    return False
            return True

        for i in range(n):
            while True:
                p = np.random.uniform(low=low + min_dist / 2, high=high - min_dist / 2, size=2)

                if valid(p):
                    default_points.append(p)
                    break
        return np.array(default_points)

    def set_obstacles_position(self, sim, positions):
        for i in range(len(self.obstacles_name)):
            obstacle_name = self.obstacles_name[i]
            sim.model.geom(obstacle_name).pos = np.array([*positions[i], 0.5])

    def check_collision(self, self_pose, obstacle_positions):
        for obstacle_position in obstacle_positions:
            if np.linalg.norm(self_pose[:2] - obstacle_position[:2]) < 0.5:
                return True
        return False

    def set_goal_position(self, sim, position):
        sim.set_geom_pose(self.goal_name, np.array([*position, 0.001]))
