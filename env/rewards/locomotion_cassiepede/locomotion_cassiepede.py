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
        assert hasattr(self, "clock"), \
            f"{FAIL}Environment {self.__class__.__name__} does not have a clock object.{ENDC}"
        assert self.clock is not None, \
            f"{FAIL}Clock has not been initialized, is still None.{ENDC}"
        assert self.clock_type == "von_mises", \
            f"{FAIL}locomotion_vonmises_clock_reward should be used with von mises clock type, but " \
            f"clock type is {self.clock_type}.{ENDC}"

        q = {}

        ### Cyclic foot force/velocity reward ###
        l_force, r_force = self.clock[i].get_von_mises_values()
        l_stance = 1 - l_force
        r_stance = 1 - r_force

        # Retrieve states
        l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][0]])
        r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][1]])
        l_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[i][0]])
        r_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[i][1]])
        l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][0])
        r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][1])

        # print('l_foot_force:', l_foot_force, 'r_foot_force:', r_foot_force)

        q["left_force"] = l_force * l_foot_force
        q["right_force"] = r_force * r_foot_force
        q["left_speed"] = l_stance * l_foot_vel
        q["right_speed"] = r_stance * r_foot_vel

        # print('l_force', l_force, 'r_force', r_force, 'l_stance',l_stance,'r_stance',r_stance)


        ### Speed rewards ###
        poi_vel = self.get_poi_linear_velocity(local_frame=False)
        # This is in global frame

        target_vel_in_local = np.array([self.x_velocity_poi, self.y_velocity_poi, 0])

        quat = R.from_euler('xyz', [0, 0, self.orient_add[0]])
        target_vel = quat.apply(target_vel_in_local)

        x_vel = np.abs(poi_vel[0] - target_vel[0])
        y_vel = np.abs(poi_vel[1] - target_vel[1])

        z_vel = np.abs(self.get_poi_angular_velocity()[2] - self.turn_rate_poi)

        if x_vel < 0.05:
            x_vel = 0
        if y_vel < 0.05:
            y_vel = 0

        q["x_vel"] = x_vel
        q["y_vel"] = y_vel
        q['z_vel'] = z_vel

        ### Orientation rewards (base and feet) ###
        poi_orient = self.get_poi_orientation()
        base_pose = self.sim.get_body_pose(self.sim.base_body_name[i])
        target_quat = np.array([1, 0, 0, 0])
        if self.orient_add != 0:
            command_quat = R.from_euler('xyz', [0, 0, self.orient_add[0]])
            target_quat = R.from_quat(us2scipy(target_quat)) * command_quat
            target_quat = scipy2us(target_quat.as_quat())

        # poi_yaw = np.degrees(quaternion2euler(np.array(self.get_poi_orientation()).reshape(1, -1))[0, -1])
        # target_yaw = np.degrees(quaternion2euler(np.array(target_quat).reshape(1, -1))[0, -1])

        orientation_error = quaternion_distance(poi_orient, target_quat)

        if orientation_error < 5e-3:
            orientation_error = 0

        # Foot has to face the same direction as base (pelvis)
        foot_orientation_error = quaternion_distance(base_pose[3:], l_foot_pose[3:]) + \
                                 quaternion_distance(base_pose[3:], r_foot_pose[3:])
        q["orientation"] = orientation_error + foot_orientation_error

        ### Hop symmetry reward (keep feet equidistant) ###
        period_shifts = self.clock[i].get_period_shifts()
        l_foot_pose_in_base = self.sim.get_relative_pose(base_pose, l_foot_pose)
        r_foot_pose_in_base = self.sim.get_relative_pose(base_pose, r_foot_pose)
        xdif = np.sqrt(np.power(l_foot_pose_in_base[[0, 2]] - r_foot_pose_in_base[[0, 2]], 2).sum())
        pdif = np.exp(-5 * np.abs(np.sin(np.pi * (period_shifts[0] - period_shifts[1]))))
        q['hop_symmetry'] = pdif * xdif

        ### Sim2real stability rewards ###
        if self.simulator_type == "libcassie" and self.state_est:
            base_acc = self.sim.robot_estimator_state.pelvis.translationalAcceleration[:]
        else:
            base_acc = self.sim.get_body_acceleration(self.sim.base_body_name[i])
        q["stable_base"] = np.abs(base_acc).sum()
        q["action_penalty"] = np.abs(action[i]).sum()  # Only penalize hip roll/yaw
        if self.last_action is not None:
            q["ctrl_penalty"] = sum(np.abs(self.last_action[i] - action[i])) / len(action[i])
        else:
            q["ctrl_penalty"] = 0
        if self.simulator_type == "libcassie" and self.state_est:
            torque = self.sim.get_torque(state_est=self.state_est)
        else:
            torque = self.sim.get_torque()[i * 10:(i + 1) * 10]

        # print('torque:', torque.sum())

        # Normalized by torque limit, sum worst case is 10, usually around 1 to 2
        q["trq_penalty"] = sum(np.abs(torque) / self.sim.output_torque_limit[i * 10:(i + 1) * 10])
        #
        # ### Add up all reward components ###
        # reward = defaultdict(lambda: 0.)
        # for name in q:
        #     if not is_variable_valid(q[name]):
        #         raise RuntimeError(f"Reward {name} has Nan or Inf values as {q[name]}.\n"
        #                            f"Training stopped.")
        #     reward[name] += self.reward_weight[name]["weighting"] * \
        #                     kernel(self.reward_weight[name]["scaling"] * q[name])
        #     # print out reward name and values in a block format
        #     # print(f"{name:15s} : {q[name]:.3f} | {self.reward_weight[name]['scaling'] * q[name]:.3f} | {kernel(self.reward_weight[name]['scaling'] * q[name]):.3f}", end="\n")
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

        height_bound = (0.70, 1.0 if self.curr_step > 30 else 1.5)
        if np.abs(base_euler[1]) > 20 / 180 * np.pi or np.abs(
                base_euler[0]) > 20 / 180 * np.pi or base_height < height_bound[0] or base_height > height_bound[1]:
            dones.append(True)
        else:
            dones.append(False)
    return np.array(dones, dtype=bool)
