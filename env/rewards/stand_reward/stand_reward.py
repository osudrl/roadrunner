import numpy as np

from env.genericenv import GenericEnv
from util.quaternion import quaternion_distance


def compute_rewards(self: GenericEnv, action):
    q = {}

    ### Height penalty, match the desired standing height ###
    base_pose = self.sim.get_body_pose(self.sim.base_body_name)
    if hasattr(self, "cmd_height"):
        q['height_penalty'] = np.abs(base_pose[2] - self.cmd_height)
    else:
        q['height_penalty'] = np.abs(base_pose[2] - 0.9)

    ### Arm/Hand position reward.
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

    ### Orientation rewards, base and feet ###
    # Retrieve states
    l_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[0]])
    r_foot_vel = np.linalg.norm(self.feet_velocity_tracker_avg[self.sim.feet_body_name[1]])
    l_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[0])
    r_foot_pose = self.sim.get_site_pose(self.sim.feet_site_name[1])
    avg_foot_pos = (l_foot_pose[0:2] + r_foot_pose[0:2]) / 2
    q["com"] = np.linalg.norm(self.sim.data.subtree_com[0, 0:2] - avg_foot_pos)

    q["stance_width"] = np.abs((l_foot_pose[1] - r_foot_pose[1]) - 0.385)
    q["stance_x"] = np.abs(l_foot_pose[0] - r_foot_pose[0])

    orient_target = np.array([1, 0, 0, 0])
    q["base_orientation"] = quaternion_distance(base_pose[3:], orient_target)
    q["l_foot_orientation"] = quaternion_distance(orient_target, l_foot_pose[3:])
    q["r_foot_orientation"] = quaternion_distance(orient_target, r_foot_pose[3:])

    ### Static rewards. Want feet and motor velocities to be zero ###
    motor_vel = self.sim.get_motor_velocity()
    q['motor_vel_penalty'] = np.linalg.norm(motor_vel) / len(motor_vel)
    if self.max_foot_vel < l_foot_vel + r_foot_vel:
        self.max_foot_vel = l_foot_vel + r_foot_vel
    q['foot_vel_penalty'] = self.max_foot_vel

    ### Control rewards ###
    if self.last_action is not None:
        q["ctrl_penalty"] = sum(np.abs(self.last_action - action)) / len(action)
    else:
        q["ctrl_penalty"] = 0
    torque = self.sim.get_torque()
    q["trq_penalty"] = sum(np.abs(torque)) / len(torque)

    if self.sim.is_self_collision():
        q["self_collision"] = -0.1
    else:
        q["self_collision"] = 0

    return q

# Termination condition: If height is too low (cassie fell down) terminate
def compute_done(self: GenericEnv):
    base_height = self.sim.get_body_pose(self.sim.base_body_name)[2]
    if base_height < 0.4:
        return True

    for b in self.sim.knee_walking_list:
        collide = self.sim.is_body_collision(b)
        if collide:
            return True

    return False
