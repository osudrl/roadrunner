import numpy as np
from scipy.spatial.transform import Rotation as R

from env.util.quaternion import *


def kernel(x):
    return np.exp(-x)


def compute_rewards(self, action):
    rewards = []

    height_base = self.get_base_position()[:, -1]

    # slides = self.sim.data.qpos[7:].reshape(self.num_cassie, -1)[:, :2]

    joint_forces = np.linalg.norm(self.get_base_force()[..., :2], axis=-1)

    for i in range(self.num_cassie):
        q = {}

        # Retrieve states
        l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][0]])
        r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][1]])
        l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][0])
        r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][1])
        base_pose = self.sim.get_body_pose(self.sim.base_body_name[i])

        # l_foot_acc = self.sim.get_body_acceleration(self.sim.feet_body_name[i][0])
        # r_foot_acc = self.sim.get_body_acceleration(self.sim.feet_body_name[i][1])

        ### Feet air time rewards ###

        # When foot contact happens, contact is 1, else 0 for each foot. E.g. [0, 1]
        contact = np.array([l_foot_force, r_foot_force]) > 0.1

        # If foot was swinging and now in contact, then contact_after_swing is 1, else 0. E.g. [0, 1]
        contact_after_swing = contact & (self.feet_air_time[i] > 0)

        # Once there is contact, compute for how long the foot was swinging or in the air
        feet_air_time = self.feet_air_time[i] / self.policy_rate

        # Reward for maximizing air time but stay in bounds
        feet_air_time_rew = min(
            contact_after_swing.dot(feet_air_time - self.air_time_bounds[0]),
            contact_after_swing.dot(self.air_time_bounds[1] - feet_air_time)
        )

        # if contact_after_swing.dot(self.feet_air_time[i] / self.policy_rate) > 0:
        #     print('feet_air_time:', contact_after_swing.dot(self.feet_air_time[i] / self.policy_rate),
        #             'feet_air_time_rew:', feet_air_time_rew)

        zero_cmd = (self.x_velocity_poi[i], self.y_velocity_poi[i], self.turn_rate_poi[i]) == (0, 0, 0)

        # if contact_after_swing.dot(self.feet_air_time[i] / self.policy_rate) > 0:
        #     print('feet_air_time:', contact_after_swing.dot(self.feet_air_time[i] / self.policy_rate))

        # print('zero_cmd:', zero_cmd)
        n_feet_in_contact = contact.sum()
        if zero_cmd:

            # If commanded is zero, give reward for double contact, and partially for single contact for recovery
            q["feet_contact"] = (n_feet_in_contact == 2) + 0.5 * (n_feet_in_contact == 1)

            # If commanded is zero, give no reward for air time
            q['feet_air_time'] = 0

            l_foot_in_base = self.sim.get_relative_pose(base_pose, l_foot_pose)
            r_foot_in_base = self.sim.get_relative_pose(base_pose, r_foot_pose)

            # If commanded is zero, penalize for misalignment of feet
            q["stance_x"] = np.abs(l_foot_pose[0] - r_foot_pose[0])
            q["stance_y"] = np.abs((l_foot_in_base[1] - r_foot_in_base[1]) - 0.385)

            # Prevent leaning while standing
            # avg_foot_pos = (l_foot_pose[:2] + r_foot_pose[:2]) / 2
            # q['com'] = -np.linalg.norm(base_pose[:2] - avg_foot_pos)
        else:
            # If commanded is nonzero, give reward for single contact
            q["feet_contact"] = n_feet_in_contact == 1

            # If the foot was swinging for less than min_air_time, give negative reward, else reward for maximizing
            # air time
            q['feet_air_time'] = feet_air_time_rew

            # Give no penalty for stance if commanded is nonzero
            q['stance_x'] = 0
            q['stance_y'] = 0

        # q['foot_acc'] = np.linalg.norm(l_foot_acc) + np.linalg.norm(r_foot_acc)

        # Increment feet air time if foot is swinging.
        self.feet_air_time[i] += 1

        # Reset feet air time if foot is in contact
        self.feet_air_time[i] *= ~contact

        ### Speed rewards ###
        poi_vel = self.get_poi_linear_velocity(local_frame=False)
        # This is in global frame

        target_vel_in_local = np.array([self.x_velocity_poi[i], self.y_velocity_poi[i], 0])

        quat = R.from_euler('xyz', [0, 0, self.orient_add[i]])
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
        if self.orient_add[i] != 0:
            command_quat = R.from_euler('xyz', [0, 0, self.orient_add[i]])
            target_quat = R.from_quat(us2scipy(target_quat)) * command_quat
            target_quat = scipy2us(target_quat.as_quat())
        orientation_error = quaternion_distance(poi_orient, target_quat)

        if orientation_error < 5e-3 and not zero_cmd:
            orientation_error = 0

        # Foot has to face the same direction as base (pelvis)
        foot_orientation_error = quaternion_distance(base_pose[3:], l_foot_pose[3:]) + \
                                 quaternion_distance(base_pose[3:], r_foot_pose[3:])
        q["orientation"] = orientation_error + foot_orientation_error

        # print(poi_height, self.height_poi[i], abs(poi_height - self.height_poi[i]))

        q['height'] = -abs(height_base[i] - self.height_base[i])

        if self.num_cassie > 1:
            # Supplement reward for minimizing orientation error
            q['poi_orientation'] = -orientation_error
            q['base_yaw_poi'] = -np.abs(self.state_dict['base_yaw_poi'][i].item()) / np.pi

        # q['slides'] = -np.abs(slides[i]).sum()

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
            torque = self.sim.get_torque(state_est=self.state_est)
        else:
            torque = self.sim.get_torque()[i * 10:(i + 1) * 10]

        # Normalized by torque limit, sum worst case is 10, usually around 1 to 2
        q["trq_penalty"] = sum(np.abs(torque) / self.sim.output_torque_limit[i * 10:(i + 1) * 10])

        # Penalize for exerting forces on joint if perturbation force is zero while standing for multiple cassies env
        # print( zero_cmd , self.num_cassie > 1,  (not self.perturbation_force or (np.linalg.norm(
        #         self.force_vector) == 0 and np.linalg.norm(self.torque_vector) == 0)))
        if zero_cmd and self.num_cassie > 1 and (not self.perturbation_force or (np.linalg.norm(
                self.force_vector) == 0 and np.linalg.norm(self.torque_vector) == 0)):
            q['joint_forces'] = joint_forces[i]

        rewards.append(q)

    return rewards


# Termination condition: If orientation too far off terminate
def compute_done(self):
    dones = []
    base_positions = self.get_base_position()
    base_orientations = self.get_base_orientation()
    poi_orientation = R.from_quat(us2scipy(self.get_poi_orientation())).as_euler('xyz')

    # Check if the orientation of deck is too far off
    if np.abs(poi_orientation[0]) > 30 / 180 * np.pi:
        return np.ones(self.num_cassie, dtype=bool), ['Deck pitch too far off']
    if np.abs(poi_orientation[1]) > 30 / 180 * np.pi:
        return np.ones(self.num_cassie, dtype=bool), ['Deck roll too far off']

    done_reasons = []
    for i in range(self.num_cassie):

        # Check for knee collision
        knee_collision = False
        for b in self.sim.knee_walking_list[i]:
            if self.sim.is_body_collision(b):
                knee_collision = True
                break

        if knee_collision:
            done_reasons.append(f'Cassie {i} knee collision')
            dones.append(True)
            continue

        base_position = base_positions[i]
        base_orientation = base_orientations[i]
        base_height = base_position[2]
        base_euler = R.from_quat(us2scipy(base_orientation)).as_euler('xyz')

        # If height control is on, then the lower bound is 0.1 less lower bound of height command, 0.7 otherwise
        # The upper bound is 1.0 if the current step is greater than 30, 1.5 otherwise regardless of height command
        height_bounds = (
            self._height_bounds[0] - 0.1 if self._height_control else 0.7,  # Lower bound
            1.0 if self.curr_step > 30 else 1.5  # Upper bound
        )

        if np.abs(base_euler[0]) > 30 / 180 * np.pi:
            done_reasons.append(f'Cassie {i} pitch too far off')
            dones.append(True)
        elif np.abs(base_euler[1]) > 30 / 180 * np.pi:
            done_reasons.append(f'Cassie {i} roll too far off')
            dones.append(True)
        elif np.abs(self.state_dict['base_yaw_poi'][i]) > 30 / 180 * np.pi:
            done_reasons.append(f'Cassie {i} yaw too far off')
            dones.append(True)
        elif not (height_bounds[0] < base_height < height_bounds[1]):
            done_reasons.append(f'Cassie {i} height out of bounds')
            dones.append(True)
        else:
            dones.append(False)
    return np.array(dones, dtype=bool), done_reasons
