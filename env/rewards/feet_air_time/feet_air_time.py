import numpy as np
from scipy.spatial.transform import Rotation as R

from env.tasks.locomotionenv.locomotionenv import LocomotionEnv
from util.quaternion import quaternion_distance, mj2scipy, scipy2mj


def compute_rewards(self: LocomotionEnv, action):
    q = {}

    l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[0])
    r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[1])
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)

    ### Feet air time rewards ###
    # Define constants:
    min_air_time = 0.5 # seconds

    contact = np.array([l_foot_force, r_foot_force]) > 0.1 # e.g. [0, 1]
    first_contact = np.logical_and(contact, self.feet_air_time > 0) # e.g. [0, 1]
    self.feet_air_time += 1 # increment airtime by one timestep

    feet_air_time_rew = first_contact.dot(self.feet_air_time/self.policy_rate - min_air_time) # e.g. [0, 1] * [0, 9] = 9
    zero_cmd = (self.x_velocity, self.y_velocity, self.turn_rate) == (0, 0, 0)
    feet_air_time_rew *= (1 - zero_cmd) # only give reward when there is a nonzero velocity command
    q["feet_air_time"] = feet_air_time_rew
    self.feet_air_time *= ~contact # reset airtime to 0 if contact is 1

    # feet single contact reward. This is to avoid hops. if zero command reward double contact
    n_feet_in_contact = contact.sum()
    if zero_cmd:
        # Adding 0.5 for single contact seems to work pretty good
        # Makes stepping to stabilize standing stance less costly
        q["feet_contact"] = (n_feet_in_contact == 2) + 0.5*(n_feet_in_contact == 1)
    else:
        q["feet_contact"] = n_feet_in_contact == 1

    # Foot stance location reward
    if zero_cmd:
        l_foot_in_base = self.sim.get_relative_pose(base_pose, l_foot_pose)
        r_foot_in_base = self.sim.get_relative_pose(base_pose, r_foot_pose)
        q["stance_x"] = np.abs(l_foot_pose[0] - r_foot_pose[0])
        q["stance_y"] = np.abs((l_foot_in_base[1] - r_foot_in_base[1]) - 0.385)

    ### Speed rewards ###
    base_vel = self.sim.get_base_linear_velocity()
    # Offset velocity in local frame by target orient_add to get target velocity in world frame
    target_vel_in_local = np.array([self.x_velocity, self.y_velocity, 0])
    euler = R.from_quat(mj2scipy(self.sim.get_base_orientation())).as_euler('xyz')
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
    target_quat = np.array([1, 0, 0, 0])
    if self.orient_add != 0:
        command_quat = R.from_euler('xyz', [0,0,self.orient_add])
        target_quat = R.from_quat(mj2scipy(target_quat)) * command_quat
        target_quat = scipy2mj(target_quat.as_quat())
    orientation_error = quaternion_distance(base_pose[3:], target_quat)
    # Deadzone around quaternion, but not for standing
    if orientation_error < 5e-3 and not zero_cmd:
        orientation_error = 0

    # Foot orientation target in global frame. Want to be flat and face same direction as base all
    # the time. So compare to the same orientation target as the base.
    foot_orientation_error = quaternion_distance(target_quat, l_foot_pose[3:]) + \
                             quaternion_distance(target_quat, r_foot_pose[3:])

    q["orientation"] = orientation_error + foot_orientation_error

    ### Sim2real stability rewards ###
    if self.simulator_type == "libcassie" and self.state_est:
        base_acc = self.sim.robot_estimator_state.pelvis.translationalAcceleration[:]
    else:
        base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["stable_base"] = np.abs(base_acc).sum()
    if self.last_action is not None:
        q["ctrl_penalty"] = sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    if self.simulator_type == "libcassie" and self.state_est:
        torque = self.sim.get_torque(state_est = self.state_est)
    else:
        torque = self.sim.get_torque()
    # Normalized by torque limit, sum worst case is 10, usually around 1 to 2
    q["trq_penalty"] = sum(np.abs(torque)/self.sim.output_torque_limit)

    if self.robot.robot_name == "digit":
        l_hand_pose = self.sim.get_site_pose(self.sim.hand_site_name[0])
        r_hand_pose = self.sim.get_site_pose(self.sim.hand_site_name[1])
        l_hand_in_base = self.sim.get_relative_pose(base_pose, l_hand_pose)
        r_hand_in_base = self.sim.get_relative_pose(base_pose, r_hand_pose)
        l_hand_target = np.array([[0.15, 0.3, -0.1]])
        r_hand_target = np.array([[0.15, -0.3, -0.1]])
        l_hand_distance = np.linalg.norm(l_hand_in_base[:3] - l_hand_target)
        r_hand_distance = np.linalg.norm(r_hand_in_base[:3] - r_hand_target)
        q['arm'] = l_hand_distance + r_hand_distance

    return q

def compute_done(self: LocomotionEnv):
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    base_height = base_pose[2]
    base_euler = R.from_quat(mj2scipy(base_pose[3:])).as_euler('xyz')
    for b in self.sim.knee_walking_list:
        if self.sim.is_body_collision(b):
            return True
    if np.abs(base_euler[1]) > 20/180*np.pi or np.abs(base_euler[0]) > 20/180*np.pi or base_height < self.robot.min_base_height:
        return True
    return False
