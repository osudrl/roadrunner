import copy
import json
import numpy as np
import os
import time
import mujoco as mj

from env.genericenv import GenericEnv
from sim import MjCassieSim
from env.util.quaternion import *
from scipy.spatial.transform import Rotation as R
from util.colors import FAIL, WARNING, ENDC
from pathlib import Path

class CassieEnv(GenericEnv):
    def __init__(self,
                 simulator_type: str,
                 num_cassie: int,
                 terrain: str,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 state_noise: float,
                 velocity_noise: float,
                 state_est: bool):
        """Template class for Cassie with common functions.
        This class intends to capture all signals under simulator rate (2kHz).

        Args:
            simulator_type (str): "mujoco" or "libcassie"
            clock (bool): "linear" or "von-Mises" or None
            policy_rate (int): Control frequency of the policy in Hertz
            dynamics_randomization (bool): True, enable dynamics randomization.
            terrain (str): Type of terrain generation [stone, stair, obstacle...]. Initialize inside
                           each subenv class to support individual use case.
        """
        super().__init__()
        self.num_cassie = num_cassie
        self.dynamics_randomization = dynamics_randomization
        self.default_policy_rate = policy_rate
        self.terrain = terrain
        # Select simulator
        if state_est and not simulator_type == 'libcassie':
            raise RuntimeError(f"State estimator input can only be used with libcassie sim.")
        self.simulator_type = simulator_type
        if simulator_type == "mujoco":
            self.sim = MjCassieSim(terrain=terrain, num_cassie=self.num_cassie)
            self.state_est = False
        elif simulator_type == 'libcassie':
            from sim import LibCassieSim
            self.sim = LibCassieSim()
            self.state_est = state_est
        else:
            raise RuntimeError(f"Simulator type {simulator_type} not correct!"
                               "Select from 'mujoco' or 'libcassie'.")

        # Low-level control specifics
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.kp = np.array([[80.0, 80.0, 110.0, 110.0, 50.0,
                            80.0, 80.0, 110.0, 110.0, 50.0]])

        self.kp = np.repeat(self.kp, repeats=num_cassie, axis=0)

        self.kd = np.array([[8.0, 8.0, 10.0, 10.0, 5.0,
                            8.0, 8.0, 10.0, 10.0, 5.0]])

        self.kd = np.repeat(self.kd, repeats=num_cassie, axis=0)

        # Init trackers to weigh/avg 2kHz signals and containers for each signal
        self.orient_add = np.zeros(num_cassie)
        self.trackers = {self.update_tracker_grf: {"frequency": 50},
                         self.update_tracker_velocity: {"frequency": 50},
                         self.update_tracker_torque: {"frequency": 50},
                        }
        if terrain == 'hfield':
            self.trackers[self.update_tracker_touch_sensor] = {"frequency": 2000}
        # if self.simulator_type == "mujoco":
        #     self.trackers[self.update_tracker_cop] = {"frequency": 50}
        # Double check tracker frequencies and convert to number of sim steps
        for tracker, tracker_dict in self.trackers.items():
            freq = tracker_dict["frequency"]
            steps = int(self.sim.simulator_rate // freq)
            if steps != self.sim.simulator_rate / freq:
                print(f"{WARNING}WARNING: tracker frequency for {tracker.__name__} of {freq}Hz " \
                      f"does not fit evenly into simulator rate of {self.sim.simulator_rate}. " \
                      f"Rounding to {self.sim.simulator_rate / steps:.2f}Hz instead.{ENDC}")
            tracker_dict["num_step"] = steps

        self.torque_tracker_avg = np.zeros(self.sim.num_actuators) # log torque in 2kHz
        self.feet_grf_tracker_avg = {} # log GRFs in 2kHz
        self.feet_velocity_tracker_avg = {} # log feet velocity in 2kHz
        self.feet_touch_sensor_tracker_avg = {'left-toe':0, 'right-toe':0} # log feet touch sensor in 2kHz
        for i in range(len(self.sim.feet_body_name)):
            for foot in self.sim.feet_body_name[i]:
                self.feet_grf_tracker_avg[foot] = self.sim.get_body_contact_force(name=foot)
                self.feet_velocity_tracker_avg[foot] = self.sim.get_body_velocity(name=foot)
        self.cop = None
        self.cop_marker_id = None

        # Dynamics randomization ranges
        # If any joints/bodies are missing from the json file they just won't be randomized,
        # DR will still run. Get default ranges for each param too. We grab the indicies of the
        # relevant joints/bodies to avoid using named access later (vectorized access is faster)
        if self.__class__.__name__.lower() != "cassieenv":
            dyn_rand_file = open(Path(__file__).parent /
                                 f"{self.__class__.__name__.lower()}/dynamics_randomization.json")
            dyn_rand_data = json.load(dyn_rand_file)
            self.dr_ranges = {}
            # Damping
            damp_inds = []
            damp_ranges = []
            for joint_name, rand_range in dyn_rand_data["damping"].items():
                num_dof = len(self.sim.get_dof_damping(joint_name))
                for i in range(num_dof):
                    damp_inds.append(self.sim.get_joint_dof_adr(joint_name) + i)
                    damp_ranges.append(rand_range)
            damp_ranges = np.array(damp_ranges)
            self.dr_ranges["damping"] = {"inds":damp_inds,
                                        "ranges":damp_ranges}
            # Mass
            mass_inds = []
            mass_ranges = []
            for body_name, rand_range in dyn_rand_data["mass"].items():
                mass_inds.append(self.sim.get_body_adr(body_name))
                mass_ranges.append(rand_range)
            mass_ranges = np.array(mass_ranges)
            self.dr_ranges["mass"] = {"inds":mass_inds,
                                    "ranges":mass_ranges}
            # CoM location
            ipos_inds = []
            ipos_ranges = []
            for body_name, rand_range in dyn_rand_data["ipos"].items():
                ipos_inds.append(self.sim.get_body_adr(body_name))
                ipos_ranges.append(np.repeat(np.array(rand_range)[:, np.newaxis], 3, axis=1))
            ipos_ranges = np.array(ipos_ranges)
            self.dr_ranges["ipos"] = {"inds":ipos_inds,
                                    "ranges":ipos_ranges}
            # Friction
            self.dr_ranges["friction"] = {"ranges": dyn_rand_data["friction"]}
            self.dr_ranges["encoder-noise"] = {"ranges": dyn_rand_data["encoder-noise"]}
            dyn_rand_file.close()
        self.state_noise = state_noise
        self.velocity_noise = velocity_noise
        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(4)

        # Mirror indices and make sure complete test_mirror when changes made below
        # Readable string format listed in /testing/commmon.py
        self.motor_mirror_indices = [-5, -6, 7, 8, 9,
                                     -0.1, -1, 2, 3, 4]
        # Proprioceptive state mirror inds should be synced up with get_robot_state()
        self.robot_state_mirror_indices = [0.01, -1, 2, -3,      # base orientation
                                          -4, 5, -6,             # base rotational vel
                                          -12, -13, 14, 15, 16,  # right motor pos
                                          -7,  -8,  9,  10,  11, # left motor pos
                                          -22, -23, 24, 25, 26,  # right motor vel
                                          -17, -18, 19, 20, 21,  # left motor vel
                                          29, 30, 27, 28,        # joint pos
                                          33, 34, 31, 32, ]      # joint vel
        self.robot_state_feet_mirror_indices = [3, -4, 5,                # right foot position
                                                0.1, -1, 2,              # left foot position
                                                6, -7, 8, -9,            # base orientation
                                                -15, -16, 17, 18, 19,    # right motor pos
                                                -10, -11, 12, 13, 14,    # left motor pos
                                                20, -21, 22,             # base translational velocity
                                                -23, 24, -25,            # rotational velocity
                                                -31, -32, 33, 34, 35,    # right motor vel
                                                -26, -27, 28, 29, 30,    # left motor vel
                                                38, 39, 36, 37,          # joint pos
                                                42, 43, 40, 41]          # joint vel
        # Display menu of available commands for interactive control
        self._init_interactive_key_bindings()

        if self.simulator_type == 'mujoco':
            self.sensor_id = []
            for i, sensor_id in enumerate(self.sim.model.sensor_type):
                if sensor_id == mj.mjtSensor.mjSENS_RANGEFINDER:
                    self.sensor_id.append(i)
                    self.sim.model.sensor_type[i] = mj.mjtSensor.mjSENS_USER

    def reset_simulation(self, qpos=None):
        """Reset simulator.
        Depending on use cases, child class can override this as well.
        """
        if self.dynamics_randomization:
            self.sim.randomize_dynamics(self.dr_ranges)
            self.motor_encoder_noise = np.random.uniform(*self.dr_ranges["encoder-noise"]["ranges"], size=10)
            self.joint_encoder_noise = np.random.uniform(*self.dr_ranges["encoder-noise"]["ranges"], size=4)
            # NOTE: this creates very wrong floor slipperiness
            # if self.terrain != "hfield":
            #     rand_euler = np.random.uniform(-.05, .05, size=2)
            #     rand_quat = euler2quat(z=0, y=rand_euler[0], x=rand_euler[1])
            #     self.sim.set_geom_quat("floor", rand_quat)
        else:
            self.sim.default_dynamics()
            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(4)
        self.sim.reset(qpos=qpos)

    def step_simulation(self, action: np.ndarray, simulator_repeat_steps: int, integral_action: bool = False):
        """This loop sends actions into control interfaces, update torques, simulate step,
        and update 2kHz simulation states.
        User should add any 2kHz signals inside this function as member variables and
        fetch them inside each specific env.

        Args:
            action (np.ndarray): Actions from policy inference.
        """
        # Reset trackers
        for tracker_fn, tracker_dict in self.trackers.items():
            tracker_fn(weighting = 0, sim_step = 0)
        if integral_action:
            setpoint = action + self.sim.get_motor_position()
        else:
            # Explore around neutral offset
            setpoint = action + self.offset
        # If using DR, need to subtract the motor encoder noise that we added in the robot_state
        if self.dynamics_randomization:
            setpoint -= self.motor_encoder_noise
        for sim_step in range(simulator_repeat_steps):
            # Send control setpoints and update torques
            self.sim.set_PD(setpoint=setpoint, velocity=np.zeros(action.shape), \
                            kp=self.kp, kd=self.kd)
            if sim_step + 1 == simulator_repeat_steps:
                for i in self.sensor_id:
                    self.sim.model.sensor_type[i] = mj.mjtSensor.mjSENS_RANGEFINDER
            else:
                for i in self.sensor_id:
                    self.sim.model.sensor_type[i] = mj.mjtSensor.mjSENS_USER
            # step simulation
            self.sim.sim_forward()
            # Update simulation trackers (signals higher than policy rate, like GRF, etc)
            if sim_step > 0:
                for tracker_fn, tracker_dict in self.trackers.items():
                    # Save time by stop specific trackers if certain conditions are met
                    if self.feet_touch_sensor_tracker_avg['left-toe'] > 0.0 and self.feet_touch_sensor_tracker_avg['right-toe'] > 0.0:
                        continue
                    if (sim_step + 1) % tracker_dict["num_step"] == 0 or sim_step + 1 == simulator_repeat_steps:
                        tracker_fn(weighting = 1 / np.ceil(simulator_repeat_steps / tracker_dict["num_step"]),
                                sim_step = sim_step)

    def get_robot_state(self):
        """Get standard robot prorioceptive states. Sub-env can override this function to define its
        own get_robot_state().

        Returns:
            robot_state (np.ndarray): robot state
        """
        if self.simulator_type == "libcassie" and self.state_est:
            base_orient = self.rotate_to_heading(self.sim.get_base_orientation(state_est = self.state_est))
            base_ang_vel = self.sim.get_base_angular_velocity(state_est = self.state_est)
            motor_pos = self.sim.get_motor_position(state_est = self.state_est)
            motor_vel = np.array(self.sim.get_motor_velocity(state_est = self.state_est))
            joint_pos = self.sim.get_joint_position(state_est = self.state_est)
            joint_vel = np.array(self.sim.get_joint_velocity(state_est = self.state_est))
        else:
            base_orient = self.rotate_to_heading(self.sim.get_base_orientation())
            base_ang_vel = self.sim.get_base_angular_velocity()
            motor_pos = self.sim.get_motor_position()
            motor_vel = self.sim.get_motor_velocity()
            joint_pos = self.sim.get_joint_position()
            joint_vel = self.sim.get_joint_velocity()

        # Add noise to motor and joint encoders per episode
        if self.dynamics_randomization:
            motor_pos += self.motor_encoder_noise
            joint_pos += self.joint_encoder_noise

        # Apply noise to proprioceptive states per step
        motor_vel += np.random.normal(0, self.velocity_noise, size = self.sim.num_actuators)
        joint_vel += np.random.normal(0, self.velocity_noise, size = self.sim.num_joints)
        if isinstance(self.state_noise, list):
            orig_euler = quaternion2euler(base_orient)
            noise_euler = orig_euler + np.random.normal(0, self.state_noise[0], size=(len(self.sim.num_actuators), 3))
            noise_quat = euler2quat(x=noise_euler[:, 0], y=noise_euler[:, 1], z=noise_euler[:, 2])
            base_orient = noise_quat
            base_ang_vel = base_ang_vel + np.random.normal(0, self.state_noise[1], size = 3)
            motor_pos = motor_pos + np.random.normal(0, self.state_noise[2], size = self.sim.num_actuators)
            motor_vel = motor_vel + np.random.normal(0, self.state_noise[3], size = self.sim.num_actuators)
            joint_pos = joint_pos + np.random.normal(0, self.state_noise[4], size = self.sim.num_joints)
            joint_vel = joint_vel + np.random.normal(0, self.state_noise[5], size = self.sim.num_joints)
        else:
            pass
            # raise NotImplementedError("state_noise must be a list of 6 elements")

        # joint_pos[1] = 0.0
        # joint_vel[1] = 0.0
        robot_state = np.concatenate([
            base_orient,
            base_ang_vel,
            motor_pos,
            motor_vel,
            joint_pos,
            joint_vel
        ])
        return robot_state

    def update_tracker_grf(self, weighting: float, sim_step: int):
        """Keep track of 2khz signals, aggragate, and average uniformly.

        Args:
            weighting (float): weightings of each signal at simulation step to aggregate total
            sim_step (int): indicate which simulation step
        """
        for foot in self.feet_grf_tracker_avg.keys():
            if sim_step == 0: # reset at first sim step
                self.feet_grf_tracker_avg[foot] = np.zeros(3)
            else:
                self.feet_grf_tracker_avg[foot] += \
                    weighting * self.sim.get_body_contact_force(name=foot)

    def update_tracker_velocity(self, weighting: float, sim_step: int):
        for foot in self.feet_velocity_tracker_avg.keys():
            if sim_step == 0: # reset at first sim step
                self.feet_velocity_tracker_avg[foot] = np.zeros(6)
            else:
                self.feet_velocity_tracker_avg[foot] += \
                    weighting * self.sim.get_body_velocity(name=foot)

    def update_tracker_torque(self, weighting: float, sim_step: int):
        if sim_step == 0:   # reset at first sim step
            self.torque_tracker_avg = np.zeros(10 * self.num_cassie)
        else:
            if self.simulator_type == "libcassie" and self.state_est:
                curr_torque = self.sim.get_torque(state_est = self.state_est)
            else:
                curr_torque = self.sim.get_torque()
            self.torque_tracker_avg += weighting * curr_torque

    def update_tracker_touch_sensor(self, weighting: float, sim_step: int):
        if sim_step == 0:
            self.feet_touch_sensor_tracker_avg['left-toe'] = 0
            self.feet_touch_sensor_tracker_avg['right-toe'] = 0
        else:
            self.feet_touch_sensor_tracker_avg['left-toe'] += weighting * self.sim.data.sensor('left-toe').data
            self.feet_touch_sensor_tracker_avg['right-toe'] += weighting * self.sim.data.sensor('right-toe').data

    def update_tracker_cop(self, weighting: float, sim_step: int):
        if sim_step == 0:   # reset at first sim step
            self.cop = None
        else:
            self.cop = self.sim.compute_cop()

    def update_tracker_data_base_linear_accel(self, weighting: float, sim_step: int):
        if sim_step == 0:   # reset at first sim step
            self.tracker_base_linear_accel = self.sim.data.qacc[:3]
        else:
            self.tracker_base_linear_accel = np.vstack((self.tracker_base_linear_accel,
                                                        self.sim.data.qacc[:3]))

    def update_tracker_data_base_angular_vel(self, weighting: float, sim_step: int):
        if sim_step == 0:   # reset at first sim step
            self.tracker_base_angular_vel = self.sim.get_base_angular_velocity()
        else:
            self.tracker_base_angular_vel = np.vstack((self.tracker_base_angular_vel,
                                                       self.sim.get_base_angular_velocity()))

    def update_tracker_data_feet_position(self, weighting: float, sim_step: int):
        if sim_step == 0:   # reset at first sim step
            self.tracker_feet_position = self.sim.get_feet_position_in_base()
        else:
            self.tracker_feet_position = np.vstack((self.tracker_feet_position,
                                                    self.sim.get_feet_position_in_base()))

    def add_high_rate_trackers(self):
        self.trackers[self.update_tracker_data_base_linear_accel] = {"frequency": 1000}
        self.trackers[self.update_tracker_data_base_angular_vel] = {"frequency": 1000}
        self.trackers[self.update_tracker_data_feet_position] = {"frequency": 1000}
        self.trackers[self.update_tracker_touch_sensor] = {"frequency": 50} # speedup sim
        for tracker, tracker_dict in self.trackers.items():
            freq = tracker_dict["frequency"]
            steps = int(self.sim.simulator_rate // freq)
            if steps != self.sim.simulator_rate / freq:
                print(f"{WARNING}WARNING: tracker frequency for {tracker.__name__} of {freq}Hz " \
                      f"does not fit evenly into simulator rate of {self.sim.simulator_rate}. " \
                      f"Rounding to {self.sim.simulator_rate / steps:.2f}Hz instead.{ENDC}")
            tracker_dict["num_step"] = steps

    def rotate_to_heading(self, orientation: np.ndarray, yaw_offset=None):
        """Offset robot heading in world frame by self.orient_add amount

        Args:
            orientation (list): current robot heading in world frame

        Returns:
            new_orient (list): Offset orientation
        """
        if yaw_offset is None:
            quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        else:
            quaternion = euler2quat(z=yaw_offset, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if orientation.shape[1] == 3:
            return rotate_by_quaternion(orientation, iquaternion)

        elif orientation.shape[1] == 4:
            new_orient = quaternion_product(iquaternion, orientation)
            return new_orient

    def check_observation_action_size(self):
        """Check the size of observation/action/mirror. Subenv needs to define
        self.observation_size, self.action_size, self.get_state(),
        self.get_observation_mirror_indices(), self.get_action_mirror_indices().
        """
        _, indices = np.unique(self.get_observation_mirror_indices(), return_index=True)
        assert len(indices) == len(self.get_observation_mirror_indices()), \
            f"Observation mirror indices {self.get_observation_mirror_indices()} contains duplicates."
        _, indices = np.unique(self.get_action_mirror_indices(), return_index=True)
        assert len(indices) == len(self.get_action_mirror_indices()), \
            f"Action mirror indices {self.get_action_mirror_indices()} contains duplicates."
        assert self.observation_size == len(self.get_state()), \
            f"Check observation size = {self.observation_size}," \
            f"but get_state() returns with size {len(self.get_state())}"
        assert len(self.get_observation_mirror_indices()) == self.observation_size, \
            f"State mirror inds size {len(self.get_observation_mirror_indices())} mismatch " \
            f"with observation size {self.observation_size}."
        assert len(self.get_action_mirror_indices()) == self.action_size, \
            "Action mirror inds size mismatch with action size."

    def _init_interactive_key_bindings(self):
        pass