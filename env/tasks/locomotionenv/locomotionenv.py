import numpy as np

from env.genericenv import GenericEnv
from util.colors import FAIL, WARNING, ENDC


class LocomotionEnv(GenericEnv):
    """This is the no-clock locomotion env. It implements the bare minimum for locomotion, such as
    velocity commands. More complex no-clock locomotion envs can inherit from this class
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
        integral_action: bool = False
    ):
        super().__init__(
            robot_name=robot_name,
            reward_name=reward_name,
            simulator_type=simulator_type,
            terrain=terrain,
            policy_rate=policy_rate,
            dynamics_randomization=dynamics_randomization,
            state_noise=state_noise,
            state_est=state_est,
            integral_action=integral_action
        )

        # Command randomization ranges
        self._x_velocity_bounds = [-0.5, 2.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._turn_rate_bounds = [-0.4, 0.4] # rad/s
        self._randomize_commands_bounds = [50, 250] # in episode length

        self.x_velocity = 0
        self.y_velocity = 0
        self.turn_rate = 0

        # feet air time tracking
        self.feet_air_time = np.array([0, 0]) # 2 feet

        # Only check obs if this envs is inited, not when it is parent:
        if self.__class__.__name__ == "LocomotionEnv" and self.simulator_type not in ["ar_async", "real"]:
            self.check_observation_action_size()

    @property
    def observation_size(self):
        return super().observation_size + 3 # XY velocity and turn command

    @property
    def extra_input_names(self):
        return ['x-velocity', 'y-velocity', 'turn-rate']

    def reset(self, interactive_evaluation=False):
        self.randomize_commands_at = np.random.randint(*self._randomize_commands_bounds)
        self.randomize_commands()

        self.reset_simulation()
        self.randomize_base_orientation()

        self.interactive_evaluation = interactive_evaluation
        if interactive_evaluation:
            self._update_control_commands_dict()

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        self.feet_air_time = np.array([0, 0])
        self.max_foot_vel = 0

        return self.get_state()

    def step(self, action: np.ndarray):
        self.policy_rate = self.default_policy_rate
        if self.dynamics_randomization:
            self.policy_rate += np.random.randint(0, 6)
            # Apply random forces when standing
            self.apply_random_forces()

        # Offset global zero heading by turn rate per policy step
        self.orient_add += self.turn_rate / self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps, integral_action=self.integral_action)

        # Reward for taking current action before changing quantities for new state
        self.compute_reward(action)

        self.traj_idx += 1
        self.last_action = action

        if self.traj_idx % self.randomize_commands_at == 0 and not self.interactive_evaluation:
            self.randomize_commands()

        return self.get_state(), self.reward, self.compute_done(), {'rewards': self.reward_dict}

    def hw_step(self):
        self.orient_add += self.turn_rate / self.default_policy_rate

    def _get_state(self):
        return np.concatenate((
            self.get_robot_state(),
            [self.x_velocity, self.y_velocity, self.turn_rate],
        ))

    def randomize_commands(self):
        self.x_velocity = np.random.uniform(*self._x_velocity_bounds)
        self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
        choices = ['in-place-stand', 'in-place-turn', 'walk', 'walk-sideways', 'walk-turn']
        mode = np.random.choice(choices, p=[0.2, 0.2, 0.3, 0.1, 0.2])
        match mode:
            case 'in-place-stand':
                self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
            case 'in-place-turn':
                self.x_velocity, self.y_velocity = 0, 0
            case 'walk':
                self.turn_rate = 0
            case 'walk-sideways':
                self.x_velocity, self.turn_rate = 0, 0
        # Clip to avoid useless commands
        if np.abs(self.x_velocity) <= 0.1:
            self.x_velocity = 0
        if np.abs(self.y_velocity) <= 0.1:
            self.y_velocity = 0
        if np.abs(self.turn_rate) <= 0.1:
            self.turn_rate = 0

    def apply_random_forces(self):
        base = self.sim.get_body_adr(self.sim.base_body_name)
        if (self.x_velocity, self.y_velocity, self.turn_rate) == (0, 0, 0) and np.random.random() < 0.05:
            self.sim.data.xfrc_applied[base, :3] = np.random.randint(-1, 1, 3) * 250
        else:
            self.sim.data.xfrc_applied[base, :3] = np.zeros(3)

    def get_action_mirror_indices(self):
        return self.robot.motor_mirror_indices

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot.robot_state_mirror_indices
        mirror_inds += [len(mirror_inds), -(len(mirror_inds) + 1), -(len(mirror_inds) + 2)] # XY velocity commands
        return mirror_inds

    def _init_interactive_key_bindings(self):
        self.input_keys_dict["w"] = {
            "description": "increment x velocity",
            "func": lambda self: setattr(self, "x_velocity", self.x_velocity + 0.1)
        }
        self.input_keys_dict["s"] = {
            "description": "decrement x velocity",
            "func": lambda self: setattr(self, "x_velocity", self.x_velocity - 0.1)
        }
        self.input_keys_dict["d"] = {
            "description": "increment y velocity",
            "func": lambda self: setattr(self, "y_velocity", self.y_velocity + 0.1)
        }
        self.input_keys_dict["a"] = {
            "description": "decrement y velocity",
            "func": lambda self: setattr(self, "y_velocity", self.y_velocity - 0.1)
        }
        self.input_keys_dict["e"] = {
            "description": "decrease turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate - 0.1)
        }
        self.input_keys_dict["q"] = {
            "description": "increase turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate + 0.1)
        }
        def zero_command(self):
            self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
        self.input_keys_dict["0"] = {
            "description": "reset all commands to zero",
            "func": zero_command,
        }

    def _init_interactive_xbox_bindings(self):
        self.input_xbox_dict["LeftJoystickY"] = {
            "description": "in/decrement x velocity",
            "func": lambda self, joystick: setattr(self, "x_velocity", self.x_velocity + joystick / self.default_policy_rate)
        }
        self.input_xbox_dict["LeftJoystickX"] = {
            "description": "in/decrement y velocity",
            "func": lambda self, joystick: setattr(self, "y_velocity", self.y_velocity - joystick / self.default_policy_rate)
        }
        self.input_xbox_dict["RightJoystickX"] = {
            "description": "in/decrement turn rate",
            "func": lambda self, joystick: setattr(self, "turn_rate", self.turn_rate - joystick / self.default_policy_rate)
        }
        def zero_command(self, back):
            self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
        self.input_xbox_dict["Back"] = {
            "description": "reset all commands to zero",
            "func": zero_command
        }

    def _update_control_commands_dict(self):
        self.control_commands_dict["x velocity"] = self.x_velocity
        self.control_commands_dict["y velocity"] = self.y_velocity
        self.control_commands_dict["turn rate"] = self.turn_rate

    @staticmethod
    def get_env_args():
        return {
            "robot-name"         : ("cassie", "Which robot to use (\"cassie\" or \"digit\")"),
            "simulator-type"     : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\" or \"ar\")"),
            "terrain"            : ("", "What terrain to train with (default is flat terrain)"),
            "policy-rate"        : (50, "Rate at which policy runs in Hz"),
            "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
            "state-noise"        : ([0,0,0,0,0,0], "Amount of noise to add to proprioceptive state."),
            "state-est"          : (False, "Whether to use true sim state or state estimate. Only used for libcassie sim."),
            "reward-name"        : ("feet_air_time", "Which reward to use"),
            "integral-action"    : (False, "Whether to use integral action in the clock (default is False)"),
        }
