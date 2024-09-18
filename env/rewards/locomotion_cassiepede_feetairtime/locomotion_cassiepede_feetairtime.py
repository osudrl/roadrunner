from collections import defaultdict

import numpy as np
import mujoco as mj

from env.util.quaternion import *
from scipy.spatial.transform import Rotation as R
from util.check_number import is_variable_valid
from util.colors import FAIL, ENDC

def kernel(x):
  return np.exp(-x)

def compute_rewards(self, action):
    rewards = []
    for i in range(self.num_cassie):
        q = {}

        # Retrieve states
        l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][0]])
        r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][1]])
        l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][0])
        r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][1])
        base_pose = self.sim.get_body_pose(self.sim.base_body_name[i])

        ### Feet air time rewards ###
        # Define constants:
        min_air_time = 0.5 # seconds

        contact = np.array([l_foot_force, r_foot_force]) > 0.1 # e.g. [0, 1]
        first_contact = np.logical_and(contact, self.feet_air_time[i] > 0) # e.g. [0, 1]
        self.feet_air_time[i] += 1 # increment airtime by one timestep

        feet_air_time_rew = first_contact.dot(self.feet_air_time[i]/self.policy_rate - min_air_time) # e.g. [0, 1] * [0, 9] = 9
        zero_cmd = (self.x_velocity_poi, self.y_velocity_poi, self.turn_rate_poi) == (0, 0, 0)
        feet_air_time_rew *= (1 - zero_cmd) # only give reward when there is a nonzero velocity command
        q["feet_air_time"] = feet_air_time_rew
        self.feet_air_time[i] *= ~contact # reset airtime to 0 if contact is 1

        # feet single contact reward. This is to avoid hops. if zero command reward double contact
        n_feet_in_contact = contact.sum()
        if zero_cmd:
            # Adding 0.5 for single contact seems to work pretty good
            # Makes stepping to stabilize standing stance less costly
            q["feet_contact"] = (n_feet_in_contact == 2) + 0.5*(n_feet_in_contact == 1)
        else:
            q["feet_contact"] = n_feet_in_contact == 1

        if zero_cmd:
            l_foot_in_base = self.sim.get_relative_pose(base_pose, l_foot_pose)
            r_foot_in_base = self.sim.get_relative_pose(base_pose, r_foot_pose)
            q["stance_x"] = np.abs(l_foot_pose[0] - r_foot_pose[0])
            q["stance_y"] = np.abs((l_foot_in_base[1] - r_foot_in_base[1]) - 0.385)

        ### Speed rewards ###
        poi_vel = self.get_poi_linear_velocity(local_frame=False)
        # This is in global frame

        target_vel_in_local = np.array([self.x_velocity_poi, self.y_velocity_poi, 0])

        quat = R.from_euler('xyz', [0, 0, self.orient_add[0]])
        target_vel = quat.apply(target_vel_in_local)

        x_vel = np.abs(poi_vel[0] - target_vel[0])
        y_vel = np.abs(poi_vel[1] - target_vel[1])

        if x_vel < 0.05:
            x_vel = 0
        if y_vel < 0.05:
            y_vel = 0
        q["x_vel"] = x_vel
        q["y_vel"] = y_vel

        ### Orientation rewards (base and feet) ###
        poi_orient = self.get_poi_orientation()
        target_quat = np.array([1, 0, 0, 0])
        if self.orient_add != 0:
            command_quat = R.from_euler('xyz', [0, 0, self.orient_add[0]])
            target_quat = R.from_quat(us2scipy(target_quat)) * command_quat
            target_quat = scipy2us(target_quat.as_quat())
        orientation_error = quaternion_distance(poi_orient, target_quat)

        if orientation_error < 5e-3 and not zero_cmd:
            orientation_error = 0

        # Foot has to face the same direction as base (pelvis)
        foot_orientation_error = quaternion_distance(base_pose[3:], l_foot_pose[3:]) + \
                                 quaternion_distance(base_pose[3:], r_foot_pose[3:])
        q["orientation"] = orientation_error + foot_orientation_error

        ### Sim2real stability rewards ###
        if self.simulator_type == "libcassie" and self.state_est:
            base_acc = self.sim.robot_estimator_state.pelvis.translationalAcceleration[:]
        else:
            base_acc = self.sim.get_body_acceleration(self.sim.base_body_name[i])
        q["stable_base"] = np.abs(base_acc).sum()
        if self.last_action is not None:
            q["ctrl_penalty"] = sum(np.abs(self.last_action[i] - action[i])) / len(action[i])
        else:
            q["ctrl_penalty"] = 0
        if self.simulator_type == "libcassie" and self.state_est:
            torque = self.sim.get_torque(state_est = self.state_est)
        else:
            torque = self.sim.get_torque()[i*10:(i+1)*10]

        # Normalized by torque limit, sum worst case is 10, usually around 1 to 2
        q["trq_penalty"] = sum(np.abs(torque) / self.sim.output_torque_limit[i * 10:(i + 1) * 10])

        rewards.append(q)

    return rewards

# Termination condition: If orientation too far off terminate
def compute_done(self):
    dones = []
    base_positions = self.get_base_position()
    base_orientations = self.get_base_orientation()
    poi_orientation = R.from_quat(us2scipy(self.get_poi_orientation())).as_euler('xyz')

    # Check if the orientation of deck is too far off
    if np.abs(poi_orientation[1]) > 20 / 180 * np.pi or np.abs(poi_orientation[0]) > 20 / 180 * np.pi:
        return np.ones(self.num_cassie, dtype=bool)

    for i in range(self.num_cassie):
        base_position = base_positions[i]
        base_orientation = base_orientations[i]
        base_height = base_position[2]
        base_euler = R.from_quat(us2scipy(base_orientation)).as_euler('xyz')
        height_limit = 0.65
        if np.abs(base_euler[1]) > 20/180*np.pi or np.abs(base_euler[0]) > 20/180*np.pi or base_height < height_limit:
            dones.append(True)
        else:
            dones.append(False)
    return np.array(dones, dtype=bool)
