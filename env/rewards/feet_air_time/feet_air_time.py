import numpy as np

from env.util.quaternion import *
from scipy.spatial.transform import Rotation as R
from util.check_number import is_variable_valid
from util.colors import FAIL, ENDC

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    rewards = []
    for i in range(self.num_cassie):
        q = {}

        l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][0]])
        r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[i][1]])
        l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][0])
        r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[i][1])

        ### Feet air time rewards ###
        # TODO maybe only check Fz instead of norm
        contact = (np.array([l_foot_force, r_foot_force]) > 0.1) # e.g. [0, 1]
        first_contact = contact * (self.feet_air_time[i] > 0) # e.g. [0, 1]
        self.feet_air_time[i] += 1 # increment airtime by one timestep

        feet_air_time_rew = first_contact.dot(self.feet_air_time[i]/self.policy_rate - 0.4) # e.g. [0, 1] * [0, 9] = 9 # min step is 0.4 seconds
        zero_cmd = np.linalg.norm(np.array([self.x_velocity[i], self.y_velocity[i], self.turn_rate[i]])) < 0.01
        feet_air_time_rew *= (1-zero_cmd) # only give reward when there is a nonzero velocity command
        q["feet_air_time"] = feet_air_time_rew
        self.feet_air_time[i] *= (1-contact) # reset airtime to 0 if contact is 1

        # feet single contact reward. This is to avoid hops. if zero command reward double contact
        if zero_cmd:
            q["feet_single_contact"] = (np.array([l_foot_force, r_foot_force]) > 0.1).sum() == 2
            # TODO maybe check for >=1 instead of ==2 to ease stance fixation
            # maybe == 1 just gets half of == 2 idk
        else:
            q["feet_single_contact"] = (np.array([l_foot_force, r_foot_force]) > 0.1).sum() == 1

        ### Speed rewards ###
        base_vel = self.sim.get_base_linear_velocity()[i]
        # Offset velocity in local frame by target orient_add to get target velocity in world frame
        target_vel_in_local = np.array([self.x_velocity[i], self.y_velocity[i], 0])
        euler = R.from_quat(mj2scipy(self.sim.get_base_orientation()[i])).as_euler('xyz')
        quat = R.from_euler('xyz', [0,0,euler[2]])
        target_vel = quat.apply(target_vel_in_local)
        # Compare velocity in the same frame
        x_vel = np.abs(base_vel[0] - target_vel[0])
        y_vel = np.abs(base_vel[1] - target_vel[1])
        # We have deadzones around the speed reward since it is impossible (and we actually don't want)
        # for base velocity to be constant the whole time.
        if x_vel < 0.05:
            x_vel = 0
        if y_vel < 0.05:
            y_vel = 0
        q["x_vel"] = x_vel
        q["y_vel"] = y_vel

        ### Orientation rewards (base and feet) ###
        base_pose = self.sim.get_body_pose(self.sim.base_body_name[i])
        target_quat = np.array([1, 0, 0, 0])
        if self.orient_add[i] != 0:
            command_quat = R.from_euler('xyz', [0,0,self.orient_add[i]])
            target_quat = R.from_quat(scipy2us(target_quat)) * command_quat
            target_quat = scipy2us(target_quat.as_quat())
        orientation_error = quaternion_distance(base_pose[3:], target_quat)
        # Deadzone around quaternion as well
        if orientation_error < 5e-3:
            orientation_error = 0

        # Foor orientation target in global frame. Want to be flat and face same direction as base all
        # the time. So compare to the same orientation target as the base.
        foot_orientation_error = quaternion_distance(target_quat, l_foot_pose[3:]) + \
                                 quaternion_distance(target_quat, r_foot_pose[3:])
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
        q["trq_penalty"] = sum(np.abs(torque)/self.sim.output_torque_limit[i*10:(i+1)*10])

        if "digit" in self.__class__.__name__.lower():
            l_hand_pose = self.sim.get_site_pose(self.sim.hand_site_name[0])
            r_hand_pose = self.sim.get_site_pose(self.sim.hand_site_name[1])
            l_hand_in_base = self.sim.get_relative_pose(base_pose, l_hand_pose)
            r_hand_in_base = self.sim.get_relative_pose(base_pose, r_hand_pose)
            l_hand_target = np.array([[0.15, 0.3, -0.1]])
            r_hand_target = np.array([[0.15, -0.3, -0.1]])
            l_hand_distance = np.linalg.norm(l_hand_in_base[:3] - l_hand_target)
            r_hand_distance = np.linalg.norm(r_hand_in_base[:3] - r_hand_target)
            q['arm'] = l_hand_distance + r_hand_distance

        ### Add up all reward components ###
        reward = 0
        for name in q:
            if not is_variable_valid(q[name]):
                raise RuntimeError(f"Reward {name} has Nan or Inf values as {q[name]}.\n"
                                   f"Training stopped.")
            if name in ["feet_air_time", "feet_single_contact"]:
                reward += self.reward_weight[name]["weighting"] * q[name]
            else:
                reward += self.reward_weight[name]["weighting"] * kernel(self.reward_weight[name]["scaling"] * q[name])
            # print out reward name and values in a block format
            # print(f"{name:15s} : {q[name]:.3f} | {self.reward_weight[name]['scaling'] * q[name]:.3f} | {kernel(self.reward_weight[name]['scaling'] * q[name]):.3f}", end="\n")
        rewards.append(reward)
    return rewards

# Termination condition: If orientation too far off terminate
def compute_done(self):
    dones = []
    for i in range(self.num_cassie):
        base_pose = self.sim.get_body_pose(self.sim.base_body_name[i])
        base_height = base_pose[2]
        base_euler = R.from_quat(us2scipy(base_pose[3:])).as_euler('xyz')
        for b in self.sim.knee_walking_list[i]:
            collide = self.sim.is_body_collision(b)
            if collide:
                break
        if np.abs(base_euler[1]) > 20 / 180 * np.pi or np.abs(
                base_euler[0]) > 20 / 180 * np.pi or collide or base_height < 0.65:
            dones.append(True)
        else:
            dones.append(False)
    return dones
