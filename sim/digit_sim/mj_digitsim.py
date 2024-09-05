import mujoco as mj
import numpy as np
import pathlib

from sim.mujoco_sim import MujocoSim


class MjDigitSim(MujocoSim):

    """
    Wrapper for Digit Mujoco. This class only defines several specifics for Digit.
    """
    def __init__(self, model_name: str = "digit-v3.xml", terrain=None, fast=True):
        if terrain == 'hfield':
            model_name = "digit-v3-hfield.xml"
        if fast:
            model_name = model_name[:-4] + "-fast.xml"
        model_path = pathlib.Path(__file__).parent.resolve() / model_name
        # Number of sim steps before commanded torque is actually applied
        self.torque_delay_cycles = 6
        self.torque_efficiency = 1.0

        self.motor_position_inds = [7, 8, 9, 14, 18, 23, 30, 31, 32, 33, 34, 35, 36, 41, 45, 50, 57, 58, 59, 60]
        self.motor_velocity_inds = [6, 7, 8, 12, 16, 20, 26, 27, 28, 29, 30, 31, 32, 36, 40, 44, 50, 51, 52, 53]
        self.joint_position_inds = [15, 16, 17, 28, 29, 42, 43, 44, 55, 56]
        self.joint_velocity_inds = [13, 14, 15, 24, 25, 37, 38, 39, 48, 49]

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        self.arm_position_inds = [30, 31, 32, 33, 57, 58, 59, 60]
        self.arm_velocity_inds = [26, 27, 28, 29, 50, 51, 52, 53]
        self.base_body_name = "torso/base"
        self.feet_site_name = ["left-foot-mid", "right-foot-mid"] # pose purpose
        self.feet_body_name = ["left-leg/toe-roll", "right-leg/toe-roll"] # force purpose
        self.hand_body_name = ["left-arm/elbow", "right-arm/elbow"] # force purpose
        self.hand_site_name = ["left-hand", "right-hand"] # pose purpose

        self.num_actuators = len(self.motor_position_inds)
        self.num_joints = len(self.joint_position_inds)
        self.reset_qpos = np.array([0, 0, 1,  1, 0 ,0 ,0,
                        3.33020155e-01, -2.66178730e-02, 1.92369587e-01,
                        9.93409734e-01, -1.04126145e-03, 1.82534311e-03, 1.14597921e-01,
                        2.28971047e-01, 1.48527831e-03, -2.31455693e-01, 4.55857916e-04,
                        -1.29734322e-02,
                        9.89327705e-01, 1.45524756e-01, 1.73630859e-03, 7.08678995e-03,
                        -2.03852305e-02,
                        9.88035432e-01, 1.53876629e-01, 6.59769560e-05, 1.03905844e-02,
                        -3.52778547e-03, -5.54992074e-02,
                        -1.05542715e-01, 8.94852532e-01, -8.63756398e-03, 3.44780280e-01,
                        -3.33020070e-01, 2.66195360e-02, -1.92382190e-01,
                        9.93409659e-01, 1.04481446e-03, 1.82489637e-03, -1.14598546e-01,
                        -2.28971188e-01, -1.48636971e-03, 2.31454977e-01, -4.53425792e-04,
                        1.41299940e-02,
                        9.89323654e-01, -1.45521550e-01, 1.80177609e-03, -7.67726135e-03,
                        1.93767895e-02,
                        9.88041418e-01, -1.53872399e-01, 1.26073837e-05, -9.87117790e-03,
                        2.36874210e-03, 5.55559678e-02,
                        1.05444698e-01, -8.94890429e-01, 8.85979401e-03, -3.44723293e-01
                        ])

        # NOTE: Have to call super init AFTER index arrays are defined
        super().__init__(model_path=model_path, terrain=terrain)

        self.simulator_rate = int(1 / self.model.opt.timestep)

        self.offset = self.reset_qpos[self.motor_position_inds]
        self.kp = np.array([80, 80, 110, 140, 40, 40, 80, 80, 50, 80,
                            80, 80, 110, 140, 40, 40, 80, 80, 50, 80])
        self.kd = np.array([8, 8, 10, 12, 6, 6, 9, 9, 7, 9,
                            8, 8, 10, 12, 6, 6, 9, 9, 7, 9])

        # List of bodies that cannot (prefer not) collide with environment
        self.body_collision_list = \
            ['left-leg/shin', 'left-leg/tarsus', 'left-leg/heel-spring', 'left-leg/toe-a', \
             'left-leg/toe-a-rod', 'left-leg/toe-b', 'left-leg/toe-b-rod',\
             'right-leg/shin', 'right-leg/tarsus', 'right-leg/heel-spring', 'right-leg/toe-a', \
             'right-leg/toe-a-rod', 'right-leg/toe-b', 'right-leg/toe-b-rod']

        # minimal list of unwanted collisions to avoid knee walking
        self.knee_walking_list = \
            ['left-leg/toe-a-rod', 'left-leg/toe-b-rod', \
             'right-leg/toe-a-rod', 'right-leg/toe-b-rod']

        # Map from mj motor indices to llapi motor indices
        self.digit_motor_llapi2mj_index = [0, 1, 2, 3, 4, 5,\
                                           12, 13, 14, 15,\
                                           6, 7, 8, 9, 10, 11,\
                                           16, 17, 18, 19]

        # Followings are ordered in LLAPI motor indices
        # Output torque limit is in Nm
        self.output_torque_limit_llapi = np.array([126.682458, 79.176536, 216.927898, 231.31695, 41.975942, 41.975942,\
                                            126.682458, 79.176536, 216.927898, 231.31695, 41.975942, 41.975942,\
                                            126.682458, 126.682458, 79.176536, 126.682458,\
                                            126.682458, 126.682458, 79.176536, 126.682458])
        self.output_torque_limit = self.output_torque_limit_llapi[self.digit_motor_llapi2mj_index]

        # Output damping limit is in Nm/(rad/s)
        self.output_damping_limit = np.array([66.849046, 26.112909, 38.05002, 38.05002, 28.553161, 28.553161,\
                                            66.849046, 26.112909, 38.05002, 38.05002, 28.553161, 28.553161,\
                                            66.849046, 66.849046, 26.112909, 66.849046,\
                                            66.849046, 66.849046, 26.112909, 66.849046])
        # Output velocity limit is in rad/s
        self.output_motor_velocity_limit = np.array([4.5814, 7.3303, 8.5084, 8.5084, 11.5191, 11.5191,\
                                                    4.5814, 7.3303, 8.5084, 8.5084, 11.5191, 11.5191,\
                                                    4.5814, 4.5814, 7.3303, 4.5814,\
                                                    4.5814, 4.5814, 7.3303, 4.5814])
        # Input motor velocity limit is in RPM, ordered in Mujoco motor
        # XML already includes this attribute as 'user' under <actuator>, can be queried as
        # self.model.actuator_user[:, 0]
        self.input_motor_velocity_max = \
            self.output_motor_velocity_limit[self.digit_motor_llapi2mj_index] * \
            self.model.actuator_gear[:, 0] * 60 / (2 * np.pi)
