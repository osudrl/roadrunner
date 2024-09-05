import mujoco as mj
import numpy as np
import pathlib

from sim.mujoco_sim import MujocoSim

WARNING = '\033[93m'
ENDC = '\033[0m'

class MjCassieSim(MujocoSim):

    """
    Wrapper for Cassie Mujoco. This class only defines several specifics for Cassie.
    """
    def __init__(self, model_name: str = "cassiemujoco/cassie.xml", terrain=None, fast=True):
        if terrain == 'hfield':
            model_name = "cassiemujoco/cassie_hfield.xml"
        if fast:
            model_name = model_name[:-4] + "_fast.xml"
        model_path = pathlib.Path(__file__).parent.resolve() / model_name
        # Torque delay, i.e. size of the torque buffer. Note that "delay" of 1 corresponds to no
        # delay. So torque_delay_cycles should be the number of sim steps before commanded torque is
        # actually applied + 1
        self.torque_delay_cycles = 6 + 1
        self.torque_efficiency = 1.0

        self.motor_position_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_position_inds = [15, 16, 29, 30]
        self.motor_velocity_inds = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_velocity_inds = [13, 14, 26, 27]

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        self.base_body_name = "cassie-pelvis"
        self.feet_body_name = ["left-foot", "right-foot"] # force purpose
        self.feet_site_name = ["left-foot-mid", "right-foot-mid"] # pose purpose

        self.num_actuators = len(self.motor_position_inds)
        self.num_joints = len(self.joint_position_inds)
        self.reset_qpos = np.array([0, 0, 1.01, 1, 0, 0, 0,
                    0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                    -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])

        self.offset = self.reset_qpos[self.motor_position_inds]

        self.kp = np.array([80, 80, 110, 110, 50,
                            80, 80, 110, 110, 50])
        self.kd = np.array([8, 8, 10, 10, 5,
                            8, 8, 10, 10, 5])

        # NOTE: Have to call super init AFTER index arrays and constants are defined
        super().__init__(model_path=model_path, terrain=terrain)

        self.simulator_rate = int(1 / self.model.opt.timestep)

        # List of bodies that cannot (prefer not) collide with environment
        self.body_collision_list = ['left-tarsus', 'left-heel-spring', 'left-plantar-rod',\
            'right-tarsus', 'right-heel-spring', 'right-plantar-rod']

        # minimal list of unwanted collisions to avoid knee walking
        self.knee_walking_list = ['left-heel-spring', 'right-heel-spring',\
                                  'left-foot-crank', 'right-foot-crank']

        # Input motor velocity limit is in RPM, ordered in Mujoco motor
        # XML already includes this attribute as 'user' under <actuator>, can be queried as
        # self.model.actuator_user[:, 0]
        self.input_motor_velocity_max = [2900, 2900, 1300, 1300, 5500,\
                                         2900, 2900, 1300, 1300, 5500]

        self.output_torque_limit = [112.5, 112.5, 195.2, 195.2, 45.0,\
                                    112.5, 112.5, 195.2, 195.2, 45.0]
