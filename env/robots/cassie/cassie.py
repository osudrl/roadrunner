import numpy as np

from types import SimpleNamespace
from env.robots.base_robot import BaseRobot
from sim.cassie_sim.mj_cassiesim import MjCassieSim
from sim.cassie_sim.lib_cassiesim import LibCassieSim
from sim.cassie_sim.cassiemujoco import state_out_t


class Cassie(BaseRobot):
    def __init__(
        self,
        simulator_type: str,
        terrain: str,
        state_est: bool,
    ):
        """Robot class for Cassie defining robot and sim.
        This class houses all bot specific stuff for Cassie.

        Args:
            simulator_type (str): "mujoco" or "libcassie"
            dynamics_randomization (bool): True, enable dynamics randomization.
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                           each subenv class to support individual use case.
        """
        super().__init__(robot_name="cassie", simulator_type = simulator_type)

        self._offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp = np.array([80, 80, 110, 110, 50,
                            80, 80, 110, 110, 50])
        self.kd = np.array([8, 8, 10, 10, 5,
                            8, 8, 10, 10, 5])
        self._min_base_height = 0.65

        # Mirror indices and make sure complete test_mirror when changes made below
        # Readable string format listed in /testing/commmon.py
        self._motor_mirror_indices = [
            -5, -6, 7, 8, 9,
            -0.1, -1, 2, 3, 4
        ]

        # Proprioceptive state mirror inds should be synced up with get_robot_state()
        self._robot_state_mirror_indices = [
            0.01, -1, 2, -3,      # base orientation
            -4, 5, -6,             # base rotational vel
            -12, -13, 14, 15, 16,  # right motor pos
            -7,  -8,  9,  10,  11, # left motor pos
            -22, -23, 24, 25, 26,  # right motor vel
            -17, -18, 19, 20, 21,  # left motor vel
            29, 30, 27, 28,        # joint pos
            33, 34, 31, 32,        # joint vel
        ]

        # Define names for robot state input and action output
        self.output_names = [
            "left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-foot",
            "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-foot",
        ]
        self.robot_state_names = [
            "base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
            "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
            "left-hip-roll-pos", "left-hip-yaw-pos", "left-hip-pitch-pos", "left-knee-pos", "left-foot-pos",
            "right-hip-roll-pos", "right-hip-yaw-pos", "right-hip-pitch-pos", "right-knee-pos", "right-foot-pos",
            "left-hip-roll-vel", "left-hip-yaw-vel", "left-hip-pitch-vel", "left-knee-vel", "left-foot-vel",
            "right-hip-roll-vel", "right-hip-yaw-vel", "right-hip-pitch-vel", "right-knee-vel", "right-foot-vel",
            "left-shin-pos", "left-tarsus-pos",
            "right-shin-pos", "right-tarsus-pos",
            "left-shin-vel", "left-tarsus-vel",
            "right-shin-vel", "right-tarsus-vel"
        ]

        # Select simulator
        if "mesh" in simulator_type:
            fast = False
            simulator_type = simulator_type.replace("_mesh", "")
        else:
            fast = True
        if state_est and not simulator_type == 'libcassie':
            raise RuntimeError(f"State estimator input can only be used with libcassie sim.")
        if simulator_type == "mujoco":
            self._sim = MjCassieSim(terrain=terrain, fast=fast)
            self.state_est = False
        elif simulator_type == 'libcassie':
            self._sim = LibCassieSim()
            self.state_est = state_est
        elif simulator_type == 'real':
            self.robot_estimator_state = state_out_t()
            self.robot_estimator_state.pelvis.orientation[:] = [1, 0, 0, 0]
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'libcassie'.")

    def get_raw_robot_state(self):
        states = {}
        if self.simulator_type == "libcassie" and self.state_est:
            states['base_orient'] = self.sim.get_base_orientation(state_est = self.state_est)
            states['base_ang_vel'] = self.sim.get_base_angular_velocity(state_est = self.state_est)
            states['motor_pos'] = self.sim.get_motor_position(state_est = self.state_est)
            states['motor_vel'] = np.array(self.sim.get_motor_velocity(state_est = self.state_est))
            states['joint_pos'] = self.sim.get_joint_position(state_est = self.state_est)
            states['joint_vel'] = np.array(self.sim.get_joint_velocity(state_est = self.state_est))
        elif self.simulator_type == "real":
            states['base_orient'] = self.robot_estimator_state.pelvis.orientation[:]
            states['base_ang_vel'] = self.robot_estimator_state.pelvis.rotationalVelocity[:]
            states['motor_pos'] = self.robot_estimator_state.motor.position[:]
            states['motor_vel'] = self.robot_estimator_state.motor.velocity[:]
            joint_pos = self.robot_estimator_state.joint.position[:]
            joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
            states['joint_pos'] = joint_pos
            joint_vel = self.robot_estimator_state.joint.velocity[:]
            joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])
            states['joint_vel'] = joint_vel
        else:
            states['base_orient'] = self.sim.get_base_orientation()
            states['base_ang_vel'] = self.sim.get_base_angular_velocity()
            states['motor_pos'] = self.sim.get_motor_position()
            states['motor_vel'] = self.sim.get_motor_velocity()
            states['joint_pos'] = self.sim.get_joint_position()
            states['joint_vel'] = self.sim.get_joint_velocity()

        return states
