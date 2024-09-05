import numpy as np

from env.tasks.stoneenv.stoneenv import StoneEnv
from util.colors import FAIL, ENDC
from util.quaternion import quaternion_distance


def compute_rewards(self: StoneEnv, action):
    assert hasattr(self, "clock"), \
        f"{FAIL}Environment {self.__class__.__name__} does not have a clock object.{ENDC}"
    assert self.clock is not None, \
        f"{FAIL}Clock has not been initialized, is still None.{ENDC}"

    vars_to_check = ['steps_target_global', 'steps_active_idx', 'last_base_position',
                     'touchdown_by_clock_flag', 'steps_order']
    for var in vars_to_check:
        assert hasattr(self, var), \
            f"{FAIL}Environment {self.__class__.__name__} does not have a {var} object.{ENDC}"

    q = {}

    ### Cyclic foot force/velocity reward ###
    l_force, r_force = self.clock.linear_clock()
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

    ### Floating-base rewards ###
    base_vel = self.sim.get_body_velocity(self.sim.base_body_name)
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    base_disance2target_prev = \
        np.linalg.norm(self.steps_target_global[self.steps_active_idx][0:2] -\
                       self.last_base_position[0:2])
    base_disance2target_curr = \
        np.linalg.norm(self.steps_target_global[self.steps_active_idx][0:2] - base_pose[0:2])
    base_change = base_disance2target_prev - base_disance2target_curr
    footstep_distance = np.linalg.norm(self.steps_commands_pelvis[0:2])
    if footstep_distance < 0.1:
        q['base_progress'] = np.abs(base_vel[1]) + np.abs(base_vel[0])
    else:
        q['base_progress'] = np.abs(base_disance2target_curr)

    ### Orientation rewards (base and feet) ###
    target_quat = np.array([1, 0, 0, 0])
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
    base_acc = self.sim.get_body_acceleration(self.sim.base_body_name)
    q["stable_base"] = np.abs(base_vel[3:]).sum() + np.abs(base_acc[0:3]).sum()
    if self.last_action is not None:
        q["ctrl_penalty"] = sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    torque = self.sim.get_torque()
    q["trq_penalty"] = sum(np.abs(torque)) / len(torque)

    ### Stepping stone sparse reward ###
    if any(self.touchdown_by_clock_flag):
        side = self.steps_order[self.steps_active_idx]
        footstep_error = np.linalg.norm(\
            self.sim.get_site_pose(self.sim.feet_site_name[side])[0:2] - \
            self.steps_target_global[self.steps_active_idx][0:2])
        q['footstep'] = footstep_error
        footstep_reward = self.reward_weight['footstep']["weighting"] * \
                       self.kernel(self.reward_weight['footstep']["scaling"] * q['footstep'])
        print(f"episode index = {self.traj_idx}. TD clock {self.touchdown_by_clock_flag}\n"
              f"target footstep {self.steps_target_global[self.steps_active_idx][0:2]} "
              f"actual footstep {self.sim.get_site_pose(self.sim.feet_site_name[side])[0:2]}\n"
              f"check side={side}, distance error {footstep_error}, reward is {footstep_reward}.\n")
        print()

    return q

# Termination condition: If orientation too far off terminate
def compute_done(self: StoneEnv):
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    target_quat = np.array([1, 0, 0, 0])
    orientation_error = 3 * quaternion_distance(base_pose[3:], target_quat)
    base_height = base_pose[2]
    if np.exp(-orientation_error) < 0.8 or base_height < 0.5:
        return True
    else:
        return False
