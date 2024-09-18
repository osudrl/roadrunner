import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from env.tasks.locomotionclockenv.locomotionclockenv import LocomotionClockEnv
from util.colors import FAIL, ENDC
from util.quaternion import quaternion_distance, mj2scipy, scipy2mj
from scipy.spatial.transform import Rotation as R


def compute_rewards(self: LocomotionClockEnv, action):
    assert hasattr(self, "clock"), \
        f"{FAIL}Environment {self.__class__.__name__} does not have a clock object.{ENDC}"
    assert self.clock is not None, \
        f"{FAIL}Clock has not been initialized, is still None.{ENDC}"
    assert self.clock_type == "linear", \
        f"{FAIL}locomotion_linear_clock_reward should be used with linear clock type, but clock " \
        f"type is {self.clock_type}.{ENDC}"

    q = {}

    ### Cyclic foot force/velocity reward ###
    # Get foot cost clock weightings, linear replacement for the von mises function. There are two
    # separate clocks with different transition timings for the force/velocity cost and the position
    # (foot height) cost. Force/velocity cost transitions between stance and swing quicker to try
    # and make sure get full swing phase in there, especially important for higher speeds where we
    # try to enforce longer swing time for aerial phase. However, don't need foot height to reach
    # clearance level as quickly, so can use smoother transition (higher percent_trans)
    l_force, r_force = self.clock.linear_clock(percent_transition = 0.2)
    l_swing, r_swing = self.clock.linear_clock(percent_transition = 0.7)
    l_stance = 1 - l_force
    r_stance = 1 - r_force

    # Retrieve states
    l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[0])
    r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[1])

    if self.x_velocity <= 1:
        des_foot_height = 0.1
    elif 1 < self.x_velocity <= 3:
        des_foot_height = 0.1 + 0.2 * (self.x_velocity - 1) / 2
    else:
        des_foot_height = 0.3

    l_force_cost = l_foot_force / 75
    r_force_cost = r_foot_force / 75
    l_height_cost = (des_foot_height - l_foot_pose[2])**2
    r_height_cost = (des_foot_height - r_foot_pose[2])**2

    q["l_foot_cost_forcevel"] = l_force * l_force_cost + l_stance * l_foot_vel
    q["r_foot_cost_forcevel"] = r_force * r_force_cost + r_stance * r_foot_vel
    q["l_foot_cost_pos"] = l_swing * l_height_cost
    q["r_foot_cost_pos"] = r_swing * r_height_cost

    ### Speed rewards ###
    base_vel = self.sim.get_base_linear_velocity()
    # Offset velocity in local frame by target orient_add to get target velocity in world frame
    target_vel_in_local = np.array([self.x_velocity, self.y_velocity, 0])
    quat = scipy2mj(R.from_euler('xyz', [0,0,self.orient_add]).as_quat())
    target_vel = np.zeros(3)
    mj.mju_rotVecQuat(target_vel, target_vel_in_local, quat)
    # Compare velocity in the same frame
    x_vel = np.abs(base_vel[0] - target_vel[0])
    y_vel = np.abs(base_vel[1] - target_vel[1])
    # We have deadzones around the speed reward since it is impossible (and we actually don't want)
    # for base velocity to be constant the whole time.
    if x_vel < 0.05:
        x_vel = 0
    if y_vel < 0.05:
        y_vel = 0

    base_quat = self.sim.get_body_pose(self.sim.base_body_name)[3:]
    target_quat = np.array([1, 0, 0, 0])
    if self.orient_add != 0:
        command_quat = R.from_euler('xyz', [0,0,self.orient_add])
        target_quat = R.from_quat(mj2scipy(target_quat)) * command_quat
        target_quat = scipy2mj(target_quat.as_quat())
    orientation_error = quaternion_distance(base_quat, target_quat)
    # Deadzone around quaternion as well
    if orientation_error < 5e-3:
        orientation_error = 0
    q["x_vel"] = x_vel
    q["y_vel"] = y_vel
    q["base_orientation"] = orientation_error

    ### Foot orientation rewards ###
    # Foor orientation target in global frame. Want to be flat and face same direction as base all
    # the time. So compare to the same orientation target as the base.
    q["l_foot_orientation"] = quaternion_distance(target_quat, l_foot_pose[3:])
    q["r_foot_orientation"] = quaternion_distance(target_quat, r_foot_pose[3:])

    ### Stable base reward terms.  Don't want base to rotate or accelerate too much ###
    if self.simulator_type == "libcassie" and self.state_est:
        base_acc = self.sim.robot_estimator_state.pelvis.translationalAcceleration[:]
    else:
        base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)[0:3]
    q["base_rotvel"] = np.linalg.norm(base_vel[3:])
    q["base_transacc"] = np.linalg.norm(base_acc[0:2]) # Don't care about z acceleration

    ### Sim2real stability rewards ###
    q["action_penalty"] = np.abs(action[[0, 1, 5, 6]]).sum() # Only penalize hip roll/yaw
    if self.last_action is not None:
        q["ctrl_penalty"] = sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    if self.simulator_type == "libcassie" and self.state_est:
        torque = self.sim.get_torque(state_est = self.state_est)
    else:
        torque = self.sim.get_torque()
    q["trq_penalty"] = sum(np.abs(torque[[0, 1, 5, 6]])) / 4 # Only penalize hip roll/yaw

    return q

# Termination condition: If reward is too low or height is too low (cassie fell down) terminate
def compute_done(self: LocomotionClockEnv):
    base_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if base_height < 0.6 or self.reward < 0.4:
        return True
    else:
        return False
