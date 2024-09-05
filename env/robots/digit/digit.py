import numpy as np

from env.robots.base_robot import BaseRobot
from util.colors import FAIL, WARNING, ENDC
from sim import MjDigitSim
from testing.common import DIGIT_JOINT_LLAPI2MJ_INDEX, DIGIT_MOTOR_LLAPI2MJ_INDEX


class Digit(BaseRobot):
    def __init__(
        self,
        simulator_type: str,
        terrain: str,
        state_est: bool,
    ):
        """Robot class for Digit defining robot and sim.
        This class houses all bot specific stuff for Digit.

        Args:
            simulator_type (str): "mujoco" or "ar_async"
            dynamics_randomization (bool): True, enable dynamics randomization.
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                          each subenv class to support individual use case.
        """
        super().__init__(robot_name="digit", simulator_type=simulator_type)

        self.kp = np.array([80, 80, 110, 140, 40, 40, 80, 80, 50, 80,
                            80, 80, 110, 140, 40, 40, 80, 80, 50, 80])
        self.kd = np.array([8, 8, 10, 12, 6, 6, 9, 9, 7, 9,
                            8, 8, 10, 12, 6, 6, 9, 9, 7, 9])
        self._min_base_height = 0.85
        self._offset = np.array([0.33302015, -0.02661787, 0.19236959, 0.22897105, -0.01297343, -0.02038523,
                -0.10554271, 0.89485253, -0.00863756, 0.34478028,
                -0.33302007, 0.02661954, -0.19238219, -0.22897119, 0.01412999, 0.01937679,
                0.1054447, -0.89489043, 0.00885979, -0.34472329])


        # Mirror indices and make sure complete test_mirror when changes made below
        # Readable string format listed in /testing/commmon.py
        # Digit's motor order is different between XML and Agility's header, here uses XML
        self._motor_mirror_indices = [
            -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, # right leg/arm
            -0.1, -1, -2, -3, -4, -5, -6, -7, -8, -9          # left leg/arm
        ]

        # Proprioceptive state mirror inds should be synced up with get_robot_state()
        self._robot_state_mirror_indices = [
            0.01, -1, 2, -3,              # base orientation
            -4, 5, -6,                    # base rotational vel
            -17, -18, -19, -20, -21, -22, # right leg motor pos
            -23, -24, -25, -26,           # right arm motor pos
            -7,  -8,  -9,  -10, -11, -12, # left leg motor pos
            -13, -14, -15, -16,           # left arm motor pos
            -37, -38, -39, -40, -41, -42, # right leg motor vel
            -43, -44, -45, -46,           # right arm motor vel
            -27, -28, -29, -30, -31, -32, # left leg motor vel
            -33, -34, -35, -36,           # left arm motor vel
            -52, -53, -54, -55, -56,      # right joint pos
            -47, -48, -49, -50, -51,      # left joint pos
            -62, -63, -64, -65, -66,      # right joint vel
            -57, -58, -59, -60, -61,      # left joint vel
        ]

        # Define names for robot state input and action output
        self.output_names = [
            "left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-toe-a", "left-toe-b",
            "left-shoulder-roll", "left-shoulder-pitch", "left-shoulder-yaw", "left-elbow",
            "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-toe-a", "right-toe-b",
            "right-shoulder-roll", "right-shoulder-pitch", "right-shoulder-yaw", "right-elbow"
        ]
        self.robot_state_names = [
            "base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
            "left-hip-roll-pos", "left-hip-yaw-pos", "left-hip-pitch-pos", "left-knee-pos", "left-foot-a-pos", "left-foot-b-pos",
            "left-shoulder-roll-pos", "left-shoulder-pitch-pos", "left-shoulder-yaw-pos", "left-elbow-pos",
            "right-hip-roll-pos", "right-hip-yaw-pos", "right-hip-pitch-pos", "right-knee-pos", "right-foot-a-pos", "right-foot-b-pos",
            "right-shoulder-roll-pos", "right-shoulder-pitch-pos", "right-shoulder-yaw-pos", "right-elbow-pos",
            "left-hip-roll-vel", "left-hip-yaw-vel", "left-hip-pitch-vel", "left-knee-vel", "left-foot-a-vel", "left-foot-b-vel",
            "left-shoulder-roll-vel", "left-shoulder-pitch-vel", "left-shoulder-yaw-vel", "left-elbow-vel",
            "right-hip-roll-vel", "right-hip-yaw-vel", "right-hip-pitch-vel", "right-knee-vel", "right-foot-a-vel", "right-foot-b-vel",
            "right-shoulder-roll-vel", "right-shoulder-pitch-vel", "right-shoulder-yaw-vel", "right-elbow-vel",
            "left-shin-pos", "left-tarsus-pos", "left-heel-spring-pos", "left-toe-pitch-pos", "left-toe-roll-pos",
            "right-shin-pos", "right-tarsus-pos", "right-heel-spring-pos", "right-toe-pitch-pos", "right-toe-roll-pos",
            "left-shin-vel", "left-tarsus-vel", "left-heel-spring-vel", "left-toe-pitch-vel", "left-toe-roll-vel",
            "right-shin-vel", "right-tarsus-vel", "right-heel-spring-vel", "right-toe-pitch-vel", "right-toe-roll-vel"
        ]

        # Select simulator
        self.state_est = state_est
        if "mesh" in simulator_type:
            fast = False
            simulator_type = simulator_type.replace("_mesh", "")
        else:
            fast = True
        if simulator_type == "mujoco":
            self._sim = MjDigitSim(terrain=terrain, fast=fast)
        else:
            raise RuntimeError(f"{FAIL}Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco'.{ENDC}")

    def get_raw_robot_state(self):
        states = {}
        # NOTE: do not use floating base angular velocity and it's bad on hardware
        states['base_orient'] = self.sim.get_base_orientation()
        states['base_ang_vel'] = self.sim.data.sensor('torso/base/imu-gyro').data
        states['motor_pos'] = self.sim.get_motor_position()
        states['motor_vel'] = self.sim.get_motor_velocity()
        states['joint_pos'] = self.sim.get_joint_position()
        states['joint_vel'] = self.sim.get_joint_velocity()
        return states
