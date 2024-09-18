import mujoco as mj
import numpy as np
import pathlib

from sim import MujocoSim

WARNING = '\033[93m'
ENDC = '\033[0m'


class MjCassieSim(MujocoSim):
    """
    Wrapper for Cassie Mujoco. This class only defines several specifics for Cassie.
    """

    def __init__(self, model_name: str = "cassiemujoco/cassie.xml", terrain=None, num_cassie=1):
        if terrain == 'hfield':
            model_name = "cassiemujoco/cassie_hfield.xml"
        elif terrain == 'obstacle':
            model_name = "cassiemujoco/cassie_obstacle.xml"
        elif terrain.startswith('cassiepede'):
            model_name = f"cassiemujoco/{terrain}{num_cassie}.xml"

        model_path = pathlib.Path(__file__).parent.resolve() / model_name
        # Torque delay, i.e. size of the torque buffer. Note that "delay" of 1 corresponds to no
        # delay. So torque_delay_cycles should be the number of sim steps before commanded torque is
        # actually applied + 1
        self.torque_delay_cycles = 6 + 1
        self.torque_efficiency = 1.0

        self.motor_position_inds = np.array([[7, 8, 9, 14, 20, 21, 22, 23, 28, 34]])

        # Number of DOF added due to new joint. For example (for each hinge, +1 pos and +1 vel DOF and for each
        # ball joint, +4 pos and +3 vel)
        match num_cassie:
            case 1:
                # Zero dof for the deck (because one cassie can move with whole deck that has 6dof already)
                dof_pos_offset = 0
                dof_vel_offset = 0
            case 2:
                # 4 dof for the deck (pitch and yaw)
                dof_pos_offset = 2
                dof_vel_offset = 2
            case _:
                # 5 dof for the deck (pitch, yaw and roll) for 3+ cassies
                dof_pos_offset = 3
                dof_vel_offset = 3

        # self.motor_position_inds += dof_pos_offset + 7
        self.motor_position_inds += dof_pos_offset

        # 35 and 32 is the number of qpos and qvel in the original cassie. We have removed 3 slide joint and one ball
        # joint so we need to subtract (3 + 4 = 7) from the original qpos and (3 + 3 = 6) from the original qvel
        # Additionally, we need to add the offset for the new joints that is added inside each cassie
        position_offset = 35 - 7 + dof_pos_offset
        velocity_offset = 32 - 6 + dof_vel_offset

        for _ in range(1, num_cassie):
            self.motor_position_inds = np.vstack(
                [self.motor_position_inds, self.motor_position_inds[-1] + position_offset])

        self.joint_position_inds = np.array([[15, 16, 29, 30]])
        # self.joint_position_inds += dof_pos_offset + 7
        self.joint_position_inds += dof_pos_offset
        for _ in range(1, num_cassie):
            self.joint_position_inds = np.vstack(
                [self.joint_position_inds, self.joint_position_inds[-1] + position_offset])

        self.motor_velocity_inds = np.array([[6, 7, 8, 12, 18, 19, 20, 21, 25, 31]])
        # self.motor_velocity_inds += dof_vel_offset + 6
        self.motor_velocity_inds += dof_vel_offset
        for _ in range(1, num_cassie):
            self.motor_velocity_inds = np.vstack(
                [self.motor_velocity_inds, self.motor_velocity_inds[-1] + velocity_offset])

        self.joint_velocity_inds = np.array([[13, 14, 26, 27]])
        # self.joint_velocity_inds += dof_vel_offset + 6
        self.joint_velocity_inds += dof_vel_offset
        for _ in range(1, num_cassie):
            self.joint_velocity_inds = np.vstack(
                [self.joint_velocity_inds, self.joint_velocity_inds[-1] + velocity_offset])

        self.base_body_name = ["cassie-pelvis"]
        self.feet_body_name = [["left-foot", "right-foot"]]  # force purpose
        self.feet_site_name = [["left-foot-mid", "right-foot-mid"]]  # pose purpose

        for i in range(1, num_cassie):
            self.base_body_name.append(f'c{i + 1}_{self.base_body_name[0]}')
            self.feet_body_name.append([f'c{i + 1}_{feet_body_name}' for feet_body_name in self.feet_body_name[0]])
            self.feet_site_name.append([f'c{i + 1}_{feet_site_name}' for feet_site_name in self.feet_site_name[0]])

        self.num_actuators = self.motor_position_inds.shape
        self.num_joints = self.joint_position_inds.shape

        # self.reset_qpos = dict(deck=np.array([0,0,0,0,0,0,0,0, 0, 1.01, 1, 0, 0, 0]),
        self.reset_qpos = dict(deck=np.array([0, 0, 1.01, 1, 0, 0, 0]),
                               base=np.array([0] * dof_pos_offset +
                                             [0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                                              -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                                              -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                                              -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968]))

        self.reset_qpos['base'] = np.repeat(self.reset_qpos['base'].reshape(1, -1), repeats=num_cassie, axis=0)

        # List of bodies that cannot (prefer not) collide with environment
        # Complete list
        # self.body_collision_list = ['left-tarsus', 'left-achilles-rod', 'left-heel-spring', 'left-foot-crank',\
        #     'left-plantar-rod',\
        #     'right-tarsus', 'right-achilles-rod', 'right-heel-spring', 'right-foot-crank',\
        #     'right-plantar-rod',
        #     'cassie-pelvis', 'left-hip-pitch', 'right-hip-pitch']

        self.body_collision_list = [['cassie-pelvis', 'left-hip-pitch', 'right-hip-pitch',
                                     'left-heel-spring', 'left-tarsus', 'right-heel-spring', 'right-tarsus']]

        # minimal list of unwanted collisions to avoid knee walking
        self.knee_walking_list = [['left-heel-spring', 'right-heel-spring', 'left-foot-crank', 'right-foot-crank']]

        for i in range(1, num_cassie):
            self.body_collision_list.append([f'c{i + 1}_{body_name}' for body_name in self.body_collision_list[0]])
            self.knee_walking_list.append([f'c{i + 1}_{body_name}' for body_name in self.knee_walking_list[0]])

        # Input motor velocity limit is in RPM, ordered in Mujoco motor
        # XML already includes this attribute as 'user' under <actuator>, can be queried as
        # self.model.actuator_user[:, 0]
        self.input_motor_velocity_max = np.array([2900.0, 2900.0, 1300.0, 1300.0, 5500.0, \
                                                  2900.0, 2900.0, 1300.0, 1300.0, 5500.0])
        self.input_motor_velocity_max = np.repeat(self.input_motor_velocity_max, repeats=num_cassie, axis=0)

        self.output_torque_limit = np.array([112.5, 112.5, 195.2, 195.2, 45.0, \
                                             112.5, 112.5, 195.2, 195.2, 45.0])

        self.output_torque_limit = np.repeat(self.output_torque_limit, repeats=num_cassie, axis=0)

        # NOTE: Have to call super init AFTER index arrays and constants are defined
        super().__init__(model_path=model_path, terrain=terrain)

        self.simulator_rate = int(1 / self.model.opt.timestep)
