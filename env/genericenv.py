from abc import ABC, abstractmethod
from decimal import Decimal
from importlib import import_module
import json
import os
from pathlib import Path
import traceback
from types import MethodType

import numpy as np
from scipy.spatial.transform import Rotation as R

from env.robots.base_robot import BaseRobot
from util.colors import BLUE, WHITE, ORANGE, FAIL, ENDC, WARNING
from util.check_number import is_variable_valid
from util.quaternion import scipy2mj, mj2scipy


class GenericEnv(ABC):
    """
    Define generic environment functions that are needed for RL. Should define (not implement) all
    of the functions that sampling uses.

    This env is intended to unify all common env code that was in envs like CassieEnvClock, DigitEnvClock, etc.
    Its also intended to be agnostic of any locomotion specific code, these should be implemented in
    the child classes, so that we can use this class as a super for non-locomotion envs too.
    """

    def __init__(
        self,
        robot_name: str,
        reward_name: str,
        simulator_type: str,
        terrain: str,
        policy_rate: int,
        dynamics_randomization: bool,
        state_noise: float,
        state_est: bool,
        integral_action: bool = False,
        **kwargs,
    ):
        self._robot: BaseRobot = self.load_robot(robot_name)(
            simulator_type=simulator_type,
            terrain=terrain,
            state_est=state_est,
        )
        self.reward_name = reward_name
        self.simulator_type = simulator_type
        self.terrain = terrain
        self.dynamics_randomization = dynamics_randomization
        self.default_policy_rate = policy_rate
        self.policy_rate = policy_rate
        self.state_est = state_est,
        self.integral_action = integral_action

        self.dynamics_randomization = dynamics_randomization
        if 'dynamics_randomization_file_path' in kwargs:
            self.dynamic_randomization_file_path = kwargs['dynamics_randomization_file_path']
        else:
            self.dynamic_randomization_file_path = \
                os.path.join(Path(__file__).parent, 'robots', self.robot.robot_name.lower(), "dynamics_randomization.json")

        self.input_keys_dict = {}
        self.input_xbox_dict = {}
        self.xbox_scale_factor = 1.0
        self.control_commands_dict = {}
        self.num_menu_backspace_lines = None
        self.reward_dict = {}
        self.reward = 0
        self.orient_add = 0

        # Configure menu of available commands for interactive control
        self._init_interactive_key_bindings()
        self._init_interactive_xbox_bindings()

        if self.simulator_type in ["ar_async", "real"]:
            return # Can't do anything else if using AR

        # Init trackers to weigh/avg 2kHz signals and containers for each signal
        self.trackers = {
            self.update_tracker_grf: {"frequency": 50},
            self.update_tracker_velocity: {"frequency": 50},
            self.update_tracker_torque: {"frequency": 50},
        }
        if "mujoco" in self.simulator_type:
            self.trackers[self.update_tracker_cop] = {"frequency": 50}

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
        if simulator_type != "real":
            for foot in self.sim.feet_body_name:
                self.feet_grf_tracker_avg[foot] = self.sim.get_body_contact_force(name=foot)
                self.feet_velocity_tracker_avg[foot] = self.sim.get_body_velocity(name=foot)
        self.cop = None
        self.cop_marker_id = None

        if simulator_type != "real":
            self.dr_ranges = self.get_dr_ranges()

        assert isinstance(state_noise, list), \
                f"{FAIL}Env {self.__class__.__name__} received 'state_noise' arg that was not a " \
                f"list. State noise must be given as a 6 long list.{ENDC}"
        assert len(state_noise) == 6, \
            f"{FAIL}Env {self.__class__.__name__} received 'state_noise' arg that was not 6 long. " \
            f"State noise must be given as a 6 long list.{ENDC}"
        if all(noise == 0 for noise in state_noise):
            self.state_noise = None
        else:
            self.state_noise = state_noise
        self.motor_encoder_noise = np.zeros(self.robot.n_actuators)
        self.joint_encoder_noise = np.zeros(self.robot.n_unactuated_joints)

        # Load reward module
        self.load_reward_module()

    @property
    def robot(self):
        return self._robot

    @property
    def sim(self):
        return self._robot.sim

    @property
    def action_size(self):
        return self.sim.num_actuators

    @property
    def observation_size(self):
        return len(self.get_robot_state())

    @property
    def extra_input_names(self):
        return []

    @staticmethod
    def load_robot(robot_name):
        """Load robot class from robot folder.

        Args:
            robot_name (str): robot name, e.g. "cassie", "digit", etc.

        Returns:
            robot_class (class): robot class
        """
        try:
            module = import_module(f"env.robots.{robot_name}.{robot_name}")
            robot_class = getattr(module, robot_name.capitalize())
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Robot {robot_name} not found in env.robot.{robot_name}.")
        except:
            print(traceback.format_exc())
            exit(1)
        return robot_class

    @abstractmethod
    def reset(self, interactive_evaluation=False):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        pass

    @abstractmethod
    def _get_state(self):
        """Get state from robot and simulation.

        Returns:
            state (np.ndarray): state
        """
        pass

    def get_state(self):
        """Get state from robot and simulation.
        Wrapper for _get_state() to catch errors and print stack trace.
        """
        out = self._get_state()
        if not is_variable_valid(out):
            raise RuntimeError(f"States has Nan or Inf values. Training stopped.\n get_state returns: {out}")
        return out

    def reset_simulation(self):
        """Reset simulator.
        Depending on use cases, child class can override this as well.
        """
        if self.simulator_type == "ar_async":
            self.robot.ar_sim.reset()
            return
        elif self.simulator_type == "real":
            return
        if self.dynamics_randomization:
            self.sim.randomize_dynamics(self.dr_ranges)
            self.motor_encoder_noise = np.random.uniform(*self.dr_ranges["encoder-noise"]["ranges"], size=self.robot.n_actuators)
            self.joint_encoder_noise = np.random.uniform(*self.dr_ranges["encoder-noise"]["ranges"], size=self.robot.n_unactuated_joints)
            self.sim.torque_delay_cycles = np.random.randint(*self.dr_ranges["torque-delay"]["ranges"])
            self.sim.torque_efficiency = np.random.uniform(*self.dr_ranges["torque-efficiency"]["ranges"])
            if self.terrain != "hfield" and "mujoco" in self.simulator_type:
                rand_euler = np.random.uniform(*self.dr_ranges["slope"]["ranges"], size=2)
                rand_quat = scipy2mj(R.from_euler("xyz", [rand_euler[0], rand_euler[1], 0]).as_quat())
                if self.robot.robot_name == "digit":
                    self.sim.model.geom("floor").sameframe = 0
                self.sim.set_geom_quat("floor", rand_quat)
        else:
            self.sim.default_dynamics()
            self.motor_encoder_noise = np.zeros(self.robot.n_actuators)
            self.joint_encoder_noise = np.zeros(self.robot.n_unactuated_joints)
            if self.terrain != "hfield" and "mujoco" in self.simulator_type:
                self.sim.model.opt.disableflags = 0
                self.sim.model.geom("floor").sameframe = 1
                self.sim.set_geom_quat("floor", np.array([1, 0, 0, 0]))
        self.sim.reset()

    def randomize_base_orientation(self):
        self.orient_add = np.random.uniform(-np.pi, np.pi)
        q = R.from_euler(seq='xyz', angles=[0,0,self.orient_add], degrees=False)
        quaternion = scipy2mj(q.as_quat())
        self.sim.set_base_orientation(quaternion)

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
            setpoint = action + self.robot.offset
        # If using DR, need to subtract the motor encoder noise that we added in the robot_state
        if self.dynamics_randomization:
            setpoint -= self.motor_encoder_noise
        for sim_step in range(simulator_repeat_steps):
            # Send control setpoints and update torques
            self.sim.set_PD(setpoint=setpoint, velocity=np.zeros(action.shape), \
                            kp=self.robot.kp, kd=self.robot.kd)
            # step simulation
            self.sim.sim_forward()
            # Update simulation trackers (signals higher than policy rate, like GRF, etc)
            if sim_step > 0:
                for tracker_fn, tracker_dict in self.trackers.items():
                    if (sim_step + 1) % tracker_dict["num_step"] == 0 or sim_step + 1 == simulator_repeat_steps:
                        tracker_fn(weighting = 1 / np.ceil(simulator_repeat_steps / tracker_dict["num_step"]), sim_step = sim_step)

    def get_robot_state(self):
        states = self.robot.get_raw_robot_state()

        # orient to heading
        states['base_orient'] = self.robot.rotate_to_heading(states['base_orient'], orient_add=self.orient_add)

        # Add noise to motor and joint encoders per episode
        if self.dynamics_randomization:
            states['motor_pos'] -= self.motor_encoder_noise
            states['joint_pos'] -= self.joint_encoder_noise

        # Apply noise to proprioceptive states per step
        if self.simulator_type not in ["ar_async", "real"] and isinstance(self.state_noise, list):
            noise_euler = np.random.normal(0, self.state_noise[0], size = 3)
            noise_quat_add = R.from_euler('xyz', noise_euler)
            noise_quat = noise_quat_add * R.from_quat(mj2scipy(states['base_orient']))
            states['base_orient'] = scipy2mj(noise_quat.as_quat())
            states['base_ang_vel'] += np.random.normal(0, self.state_noise[1], size = 3)
            states['motor_pos'] += np.random.normal(0, self.state_noise[2], size = self.sim.num_actuators)
            states['motor_vel'] += np.random.normal(0, self.state_noise[3], size = self.sim.num_actuators)
            states['joint_pos'] += np.random.normal(0, self.state_noise[4], size = self.sim.num_joints)
            states['joint_vel'] += np.random.normal(0, self.state_noise[5], size = self.sim.num_joints)
        return np.concatenate(list(states.values()))

    @staticmethod
    def kernel(x):
        return np.exp(-x)

    def compute_reward(self, action: np.ndarray):
        # Get raw rewards from reward module
        q = self._compute_reward_components(action)

        # Add up all reward components
        self.reward_dict = {}
        for name in q:
            if not is_variable_valid(q[name]):
                raise RuntimeError(f"Reward {name} has Nan or Inf values as {q[name]}.\nTraining stopped.")
            if "kernel" in self.reward_weight[name] and self.reward_weight[name]['kernel'] == False:
                self.reward_dict[name] = self.reward_weight[name]["weighting"] * q[name]
            else:
                self.reward_dict[name] = self.reward_weight[name]["weighting"] * self.kernel(self.reward_weight[name]["scaling"] * q[name])
        self.reward = sum(self.reward_dict.values())

    def compute_done(self):
        return self._compute_done()

    @abstractmethod
    def get_action_mirror_indices(self):
        """Get a list of indices that mirror the action space.

        Returns:
            action_mirror_indices (list): action mirror indices
        """
        pass

    @abstractmethod
    def get_observation_mirror_indices(self):
        """Get a list of indices that mirror the observation space.

        Returns:
            observation_mirror_indices (list): observation mirror indices
        """
        pass

    def check_observation_action_size(self):
        """Check the size of observation/action/mirror. Subenv needs to define
        self.observation_size, self.action_size, self.get_state(),
        self.get_observation_mirror_indices(), self.get_action_mirror_indices().
        """
        assert self.observation_size == len(self.get_state()), \
            f"Check observation size = {self.observation_size}, " \
            f"but get_state() returns with size {len(self.get_state())}"
        assert len(self.get_observation_mirror_indices()) == self.observation_size, \
            f"State mirror inds size {len(self.get_observation_mirror_indices())} mismatch " \
            f"with observation size {self.observation_size}."
        assert len(self.get_action_mirror_indices()) == self.action_size, \
            "Action mirror inds size mismatch with action size."

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
            self.torque_tracker_avg = np.zeros(self.robot.n_actuators)
        else:
            if self.simulator_type == "libcassie" and self.state_est:
                self.torque_tracker_avg += weighting * self.sim.get_torque(state_est = self.state_est)
            else:
                self.torque_tracker_avg += weighting * self.sim.get_torque()

    def update_tracker_cop(self, weighting: float, sim_step: int):
        """Keep track of 2khz signals, aggragate, and average uniformly.

        Args:
            weighting (float): weightings of each signal at simulation step to aggregate total
            sim_step (int): indicate which simulation step
        """
        if sim_step == 0:
            self.cop = None
        else:
            self.cop = self.sim.compute_cop()

    def display_controls_menu(self,):
        """
        Method to pretty print menu of available controls.
        """
        if ((type(self.input_keys_dict) is dict) and (len(self.input_keys_dict) > 0)):
            print("")
            self.print_command("Key", "Function", color=BLUE)
            for key, value in self.input_keys_dict.items():
                assert isinstance(key, str) and len(key) == 1, (
                    f"{FAIL}input_keys_dict key must be a length-1 string corresponding \
                    to the desired keybind{ENDC}")
                assert isinstance(value["description"], str), (
                    f"{FAIL}control command description must be of type string{ENDC}")
                self.print_command(key, value["description"], color=WHITE)
            print("")

    def display_xbox_controls_menu(self,):
        """
        Method to pretty print menu of available controls.
        """
        if ((type(self.input_xbox_dict) is dict) and (len(self.input_xbox_dict) > 0)):
            print("")
            self.print_xbox_command("Key", "Function", color=BLUE)
            for key, value in self.input_xbox_dict.items():
                assert isinstance(key, str) or isinstance(key, tuple), (
                    f"{FAIL}input_xbox_dict key must be either a single string corresponding \
                    to the desired xbox input or a tuple of such strings.{ENDC}")
                assert isinstance(value["description"], str), (
                    f"{FAIL}control command description must be of type string{ENDC}")
                self.print_xbox_command(key, value["description"], color=WHITE)
            print("")

    def display_control_commands(self, erase : bool = False):
        """
        Method to pretty print menu of current commands.
        """
        if erase:
            print(f"\033[J", end='\r')
        elif ((type(self.input_keys_dict) is dict) and (len(self.input_keys_dict)>0)):
            print("")
            self.print_command("Control Input", "Commanded value",color=BLUE)
            for key, value in self.control_commands_dict.items():
                assert type(key) is str, (f"{FAIL}ctrl_dict key must be of type string{ENDC}")
                self.print_command(key, value, color=WHITE)
            print("")
            num_backspace_lines = len(self.control_commands_dict) + 3
            print(f"\033[{num_backspace_lines}A\033[K", end='\r')

    def print_command(self, char, info, color=ENDC):
        char += " " * (10 - len(char))
        print(f"\033[K", end='')
        if isinstance(info, float):
            print(f"{color}{char}\t{info:.3f}{ENDC}")
        else:
            print(f"{color}{char}\t{info}{ENDC}")

    def print_xbox_command(self, xbox_input, info, color=ENDC):
        if isinstance(xbox_input, tuple):
            char = xbox_input[0]
            for i in range(1, len(xbox_input)):
                char += " & " + xbox_input[i]
        else:
            char = xbox_input

        char += " " * (20 - len(char))
        print(f"\033[K", end='')
        if isinstance(info, float):
            print(f"{color}{char}\t{info:.3f}{ENDC}")
        else:
            print(f"{color}{char}\t{info}{ENDC}")

    @abstractmethod
    def _init_interactive_key_bindings(self):
        """
        Updates data used by the interactive control menu print functions to display the menu of available commands
        as well as the table of command inputs sent to the policy.
        """
        pass

    @abstractmethod
    def _update_control_commands_dict(self):
        pass

    def interactive_control(self, c):
        if c in self.input_keys_dict:
            self.input_keys_dict[c]["func"](self)
            self._update_control_commands_dict()
            self.display_control_commands()

    def interactive_xbox_control(self, xbox):
        for xbox_input, func_dict in self.input_xbox_dict.items():
            # Handle multiple inputs
            if isinstance(xbox_input, tuple):
                if len(xbox_input) == 2:    # Layer 2 and 3
                    if getattr(xbox, xbox_input[0]) != 0:
                        self._xbox_control_helper(xbox, xbox_input[1], func_dict["func"])
                elif len(xbox_input) == 3:  # Layer 4
                    if getattr(xbox, xbox_input[0]) != 0 and getattr(xbox, xbox_input[1]) != 0:
                        self._xbox_control_helper(xbox, xbox_input[2], func_dict["func"])
                else:
                    raise RuntimeError(f"{FAIL}Invalid xbox control input tuple length "
                        f"{len(xbox_input)} for xbox control. Should either be single string or 2 or "
                        f" 3 element tuple.{ENDC}")
            # Handle single input
            else:
                if xbox.RightBumper == 0 and xbox.LeftBumper == 0:
                    self._xbox_control_helper(xbox, xbox_input, func_dict["func"])

    def _xbox_control_helper(self, xbox, xbox_input, func):
        print_update = False
        # Handle continuous inputs (joysticks and triggers)
        if "Joystick" in xbox_input or "Trigger" in xbox_input:
            if getattr(xbox, xbox_input) != 0:
                # print("nonzero input!")
                func(self, getattr(xbox, xbox_input) * self.xbox_scale_factor)
                print_update = True
        # Handle button inputs
        else:
            if getattr(xbox, xbox_input) != 0 and not getattr(xbox, f"{xbox_input}_pressed"):
                setattr(xbox, f"{xbox_input}_pressed", True)
                func(self, getattr(xbox, xbox_input))
                print_update = True
            elif getattr(xbox, f"{xbox_input}_pressed") and getattr(xbox, xbox_input) == 0:
                setattr(xbox, f"{xbox_input}_pressed", False)
        if print_update:
            self._update_control_commands_dict()
            self.display_control_commands()

    def viewer_update_cop_marker(self):
        # Update CoP marker
        if self.sim.viewer is not None and self.sim.__class__.__name__ != "LibCassieSim":
            if self.cop_marker_id is None:
                so3 = R.from_euler(seq='xyz', angles=[0,0,0]).as_matrix()
                self.cop_marker_id = self.sim.viewer.add_marker("sphere", "", [0, 0, 0], [0.03, 0.03, 0.03], [0.99, 0.1, 0.1, 1.0], so3)
            if self.cop is not None:
                self.sim.viewer.update_marker_position(self.cop_marker_id, self.cop)

    def load_reward_module(self):
        try:
            reward_module  = import_module(f"env.rewards.{self.reward_name}.{self.reward_name}")
            self._compute_reward_components = MethodType(reward_module.compute_rewards, self)
            self._compute_done = MethodType(reward_module.compute_done, self)
            reward_path = Path(__file__).parent / "rewards" / self.reward_name / "reward_weight.json"
            reward_weight = json.load(open(reward_path))
            self.reward_weight = reward_weight['weights']
            if reward_weight['normalize']:
                self.normalize_reward_weightings()
        except ModuleNotFoundError:
            print(f"{FAIL}ERROR: No such reward '{self.reward_name}'.{ENDC}")
            exit(1)
        except:
            print(traceback.format_exc())
            exit(1)

    def normalize_reward_weightings(self):
        # Double check that reward weights add up to 1
        weight_sum = Decimal('0')
        for name, weight_dict in self.reward_weight.items():
            weighting = weight_dict["weighting"]
            weight_sum += Decimal(f"{weighting}")
        if weight_sum != 1:
            print(f"{WARNING}WARNING: Reward weightings do not sum up to 1, renormalizing.{ENDC}")
            for name, weight_dict in self.reward_weight.items():
                weight_dict["weighting"] /= float(weight_sum)

    def get_dr_ranges(self):
        # Dynamics randomization ranges
        # If any joints/bodies are missing from the json file they just won't be randomized,
        # DR will still run. Get default ranges for each param too. We grab the indicies of the
        # relevant joints/bodies to avoid using named access later (vectorized access is faster)
        with open(self.dynamic_randomization_file_path) as dyn_rand_file:
            dyn_rand_data = json.load(dyn_rand_file)

        dr_ranges = {}
        # Damping
        damp_inds = []
        damp_ranges = []
        for joint_name, rand_range in dyn_rand_data["damping"].items():
            num_dof = len(self.sim.get_dof_damping(joint_name))
            for i in range(num_dof):
                damp_inds.append(self.sim.get_joint_dof_adr(joint_name) + i)
                damp_ranges.append(rand_range)
        damp_ranges = np.array(damp_ranges)
        dr_ranges["damping"] = {"inds":damp_inds,
                                "ranges":damp_ranges}
        # Mass
        mass_inds = []
        mass_ranges = []
        for body_name, rand_range in dyn_rand_data["mass"].items():
            mass_inds.append(self.sim.get_body_adr(body_name))
            mass_ranges.append(rand_range)
        mass_ranges = np.array(mass_ranges)
        dr_ranges["mass"] = {"inds":mass_inds,
                            "ranges":mass_ranges}
        # CoM location
        ipos_inds = []
        ipos_ranges = []
        for body_name, rand_range in dyn_rand_data["ipos"].items():
            ipos_inds.append(self.sim.get_body_adr(body_name))
            ipos_ranges.append(np.repeat(np.array(rand_range)[:, np.newaxis], 3, axis=1))
        ipos_ranges = np.array(ipos_ranges)
        dr_ranges["ipos"] = {"inds":ipos_inds,
                            "ranges":ipos_ranges}
        # Spring stiffness
        spring_inds = []
        spring_ranges = []
        for joint_name, rand_range in dyn_rand_data["spring"].items():
            spring_inds.append(self.sim.get_joint_adr(joint_name))
            spring_ranges.append(rand_range)
        spring_ranges = np.array(spring_ranges)
        dr_ranges["spring"] = {"inds":spring_inds,
                                "ranges":spring_ranges}

        # Solref
        if not self.simulator_type == 'libcassie':
            solref_inds = []
            solref_ranges = []
            for geom_name, rand_range in dyn_rand_data["solref"].items():
                solref_inds.append(self.sim.get_geom_adr(geom_name))
                solref_ranges.append(rand_range)
            solref_ranges = np.array(solref_ranges)
            dr_ranges["solref"] = {"inds":solref_inds,
                                    "ranges":solref_ranges}

        # Friction
        dr_ranges["friction"] = {"ranges": dyn_rand_data["friction"]}
        dr_ranges["encoder-noise"] = {"ranges": dyn_rand_data["encoder-noise"]}

        # slope
        dr_ranges["slope"] = {"ranges": dyn_rand_data["friction"]}

        # torque
        dr_ranges["torque-delay"] = {"ranges": dyn_rand_data["torque-delay"]}
        dr_ranges["torque-efficiency"] = {"ranges": dyn_rand_data["torque-efficiency"]}

        return dr_ranges

    @abstractmethod
    def get_env_args():
        pass
