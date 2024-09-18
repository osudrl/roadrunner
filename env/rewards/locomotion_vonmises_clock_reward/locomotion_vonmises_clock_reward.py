import numpy as np
import mujoco as mj

from env.util.quaternion import *
from scipy.spatial.transform import Rotation as R
from util.check_number import is_variable_valid
from util.colors import FAIL, ENDC

def kernel(x):
  return np.exp(-x)

def compute_reward(self, action):
    assert hasattr(self, "clock"), \
        f"{FAIL}Environment {self.__class__.__name__} does not have a clock object.{ENDC}"
    assert self.clock is not None, \
        f"{FAIL}Clock has not been initialized, is still None.{ENDC}"
    assert self.clock_type == "von_mises", \
        f"{FAIL}locomotion_vonmises_clock_reward should be used with von mises clock type, but " \
        f"clock type is {self.clock_type}.{ENDC}"

    q = {}

    ### Cyclic foot force/velocity reward ###
    l_force, r_force = self.clock.get_von_mises_values()
    l_stance = 1 - l_force
    r_stance = 1 - r_force

    # Retrieve states
    l_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_force = np.linalg.norm(self.feet_grf_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[0])
    r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[1])

    q["left_force"] = l_force * l_foot_force
    q["right_force"] = r_force * r_foot_force
    q["left_speed"] = l_stance * l_foot_vel
    q["right_speed"] = r_stance * r_foot_vel

    ### Speed rewards ###
    base_vel = self.sim.get_base_linear_velocity()
    # Offset velocity in local frame by target orient_add to get target velocity in world frame
    target_vel_in_local = np.array([self.x_velocity, self.y_velocity, 0])
    # quat = euler2quat(z = self.orient_add, y = 0, x = 0)
    # target_vel = np.zeros(3)
    # mj.mju_rotVecQuat(target_vel, target_vel_in_local, quat)
    # new way
    quat = R.from_euler('xyz', [0,0,self.orient_add])
    target_vel = quat.apply(target_vel_in_local)
    # no orient
    # euler = R.from_quat(us2scipy(self.sim.get_base_orientation())).as_euler('xyz')
    # quat = R.from_euler('xyz', [0,0,euler[2]])
    # target_vel = quat.apply(target_vel_in_local)
    # Compare velocity in the same frame
    x_vel = np.abs(base_vel[0] - target_vel[0])
    y_vel = np.abs(base_vel[1] - target_vel[1])
    # print("actual x vel: ", base_vel[0], "actual y vel: ", base_vel[1])
    # print("target x vel: ", target_vel[0], "target y vel: ", target_vel[1])
    # print()
    z_vel = np.abs(self.sim.get_base_angular_velocity()[2] - self.turn_rate)
    # We have deadzones around the speed reward since it is impossible (and we actually don't want)
    # for base velocity to be constant the whole time.
    if x_vel < 0.05:
        x_vel = 0
    if y_vel < 0.05:
        y_vel = 0
    q["x_vel"] = x_vel
    q["y_vel"] = y_vel
    q['z_vel'] = z_vel

    ### Orientation rewards (base and feet) ###
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    target_quat = np.array([1, 0, 0, 0])
    if self.orient_add != 0:
        # old way
        # command_quat = euler2quat(z = self.orient_add, y = 0, x = 0)
        # target_quat = quaternion_product(target_quat, command_quat)
        # new way
        command_quat = R.from_euler('xyz', [0,0,self.orient_add])
        target_quat = R.from_quat(us2scipy(target_quat)) * command_quat
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

    ### Hop symmetry reward (keep feet equidistant) ###
    period_shifts = self.clock.get_period_shifts()
    l_foot_pose_in_base = self.sim.get_relative_pose(base_pose, l_foot_pose)
    r_foot_pose_in_base = self.sim.get_relative_pose(base_pose, r_foot_pose)
    xdif = np.sqrt(np.power(l_foot_pose_in_base[[0, 2]] - r_foot_pose_in_base[[0, 2]], 2).sum())
    pdif = np.exp(-5 * np.abs(np.sin(np.pi * (period_shifts[0] - period_shifts[1]))))
    q['hop_symmetry'] = pdif * xdif

    ### Sim2real stability rewards ###
    if self.simulator_type == "libcassie" and self.state_est:
        base_acc = self.sim.robot_estimator_state.pelvis.translationalAcceleration[:]
    else:
        base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["stable_base"] = np.abs(base_acc).sum()
    q["action_penalty"] = np.abs(action).sum() # Only penalize hip roll/yaw
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
    self.reward = 0
    for name in q:
        if not is_variable_valid(q[name]):
            raise RuntimeError(f"Reward {name} has Nan or Inf values as {q[name]}.\n"
                               f"Training stopped.")
        self.reward += self.reward_weight[name]["weighting"] * \
                       kernel(self.reward_weight[name]["scaling"] * q[name])
        # print out reward name and values in a block format
        # print(f"{name:15s} : {q[name]:.3f} | {self.reward_weight[name]['scaling'] * q[name]:.3f} | {kernel(self.reward_weight[name]['scaling'] * q[name]):.3f}", end="\n")

    return self.reward

# Termination condition: If orientation too far off terminate
def compute_done(self):
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    base_height = base_pose[2]
    base_euler = R.from_quat(us2scipy(base_pose[3:])).as_euler('xyz')
    height_limit = 0.5 if "cassie" in self.__class__.__name__.lower() else 0.7
    if np.abs(base_euler[1]) > 20/180*np.pi or np.abs(base_euler[0]) > 20/180*np.pi or base_height < height_limit:
        return True
    else:
        return False
