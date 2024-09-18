import argparse
import copy
import json
import numpy as np
import os
import traceback

from decimal import Decimal
from env.util.periodicclock import PeriodicClock
from env.util.quaternion import scipy2us
from scipy.spatial.transform import Rotation as R
from env.digit.digitenv import DigitEnv
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from util.colors import FAIL, WARNING, ENDC
from util.check_number import is_variable_valid

class DigitEnvClock(DigitEnv):

    def __init__(self,
                 clock_type: str,
                 reward_name: str,
                 simulator_type: str,
                 terrain: str,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 state_noise: float,
                 velocity_noise: float,
                 state_est: bool,
                 full_clock: bool = False,
                 full_gait: bool = False,
                 integral_action: bool = False,
                 actor_feet: bool = False,):
        assert clock_type == "linear" or clock_type == "von_mises", \
            f"{FAIL}CassieEnvClock received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"

        super().__init__(simulator_type=simulator_type,
                         terrain=terrain,
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization,
                         state_noise=state_noise,
                         velocity_noise=velocity_noise,
                         state_est=state_est)

        self.integral_action = integral_action
        self.actor_feet = actor_feet

        # Clock variables
        self.clock_type = clock_type
        self.full_clock = full_clock

        # Command randomization ranges
        self._x_velocity_bounds = [-0.5, 2.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._turn_rate_bounds = [-np.pi / 8, np.pi / 8] # rad/s
        self.full_gait = full_gait
        if self.full_gait:
            self._swing_ratio_bounds = [0.35, 0.8]
            self._period_shift_bounds = [0.0, 0.5]
            self._cycle_time_bounds = [0.8, 0.8]
        else:
            self._swing_ratio_bounds = [0.4, 0.7]
            self._period_shift_bounds = [-0.5, 0.5]
            self._cycle_time_bounds = [0.7, 1.1]
        self._randomize_commands_bounds = [150, 200] # in episode length

        # Load reward module
        self.reward_name = reward_name
        try:
            reward_module = import_module(f"env.rewards.{self.reward_name}.{self.reward_name}")
            reward_path = Path(__file__).parents[2] / "rewards" / self.reward_name / "reward_weight.json"
            self.reward_weight = json.load(open(reward_path))
            # Double check that reward weights add up to 1
            weight_sum = Decimal('0')
            for name, weight_dict in self.reward_weight.items():
                weighting = weight_dict["weighting"]
                weight_sum += Decimal(f"{weighting}")
            if weight_sum != 1:
                print(f"{WARNING}WARNING: Reward weightings do not sum up to 1, renormalizing.{ENDC}")
                for name, weight_dict in self.reward_weight.items():
                    weight_dict["weighting"] /= float(weight_sum)
            self._compute_reward = reward_module.compute_reward
            self._compute_done = reward_module.compute_done
        except ModuleNotFoundError:
            print(f"{FAIL}ERROR: No such reward '{self.reward_name}'.{ENDC}")
            exit(1)
        except:
            print(traceback.format_exc())
            exit(1)

        self.reset()

        # Define env specifics after reset
        self.observation_size = len(self.get_robot_state())
        self.observation_size += 3 # XY velocity and turn command
        self.observation_size += 2 # swing ratio
        self.observation_size += 2 # period shift
        # input clock
        if self.full_clock:
            self.observation_size += 4
        else:
            self.observation_size += 2
        if self.actor_feet:
            self.observation_size += 6
        self.action_size = self.sim.num_actuators
        # Only check sizes if calling current class. If is child class, don't need to check
        if os.path.basename(__file__).split(".")[0] == self.__class__.__name__.lower():
            self.check_observation_action_size()

    def reset(self, interactive_evaluation: bool = False):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        self.randomize_commands(init=True)
        self.randomize_commands_at = np.random.randint(*self._randomize_commands_bounds)
        self.orient_add = np.random.uniform(-np.pi, np.pi)
        q = R.from_euler(seq='xyz', angles=[0,0,self.orient_add], degrees=False)
        quaternion = scipy2us(q.as_quat())
        self.sim.set_base_orientation(quaternion)

        # Update clock
        self.randomize_clock(init=True)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Interactive control/evaluation
        self._update_control_commands_dict()
        self.interactive_evaluation = interactive_evaluation

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        self.cop = None
        return self.get_state()

    def randomize_clock(self, init=False):
        phase_add = 1 / self.default_policy_rate
        if init:
            swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
            swing_ratios = [swing_ratio, swing_ratio]
            if np.random.random() < 0.3: # 50% chance of rand shifts
                period_shifts = [0   + np.random.uniform(*self._period_shift_bounds),
                                 0.5 + np.random.uniform(*self._period_shift_bounds)]
            else:
                period_shifts = [0, 0.5]
            self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
            self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        else:
            swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
            self.clock.set_swing_ratios([swing_ratio, swing_ratio])
            if np.random.random() < 0.3: # 50% chance of rand shifts
                period_shifts = [0   + np.random.uniform(*self._period_shift_bounds),
                                 0.5 + np.random.uniform(*self._period_shift_bounds)]
            else:
                period_shifts = [0, 0.5]
            self.clock.set_period_shifts(period_shifts)
            self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
            self.clock.set_cycle_time(self.cycle_time)

    def step(self, action: np.ndarray):
        if self.dynamics_randomization:
            self.policy_rate = self.default_policy_rate + np.random.randint(0, 6)
        else:
            self.policy_rate = self.default_policy_rate

        # Offset global zero heading by turn rate per policy step
        self.orient_add += self.turn_rate / self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps, integral_action=self.integral_action)

        # Update CoP marker
        if self.sim.viewer is not None:
            if self.cop_marker_id is None:
                so3 = R.from_euler(seq='xyz', angles=[0,0,0]).as_matrix()
                self.cop_marker_id = self.sim.viewer.add_marker("sphere", "", [0, 0, 0], [0.03, 0.03, 0.03], [0.99, 0.1, 0.1, 1.0], so3)
            if self.cop is not None:
                cop_pos = np.concatenate([self.cop, [0]])
                self.sim.viewer.update_marker_position(self.cop_marker_id, cop_pos)

        # Reward for taking current action before changing quantities for new state
        r = self.compute_reward(action)

        self.traj_idx += 1
        self.last_action = action

        # Increment clock at last for updating s'
        self.clock.increment()

        # Randomize commands
        if self.traj_idx % self.randomize_commands_at == 0 and not self.interactive_evaluation:
            self.randomize_commands()
            if self.full_gait:
                self.randomize_clock()

        return self.get_state(), r, self.compute_done(), {}

    def randomize_commands(self, init=False):
        # Randomize commands
        self.x_velocity = np.random.uniform(*self._x_velocity_bounds)
        self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
        choices = ['in-place', 'in-place-turn', 'walk', 'walk-turn']
        mode = np.random.choice(choices, p=[0.1, 0.1, 0.3, 0.5])
        match mode:
            case 'in-place':
                self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
            case 'in-place-turn':
                self.x_velocity, self.y_velocity = 0, 0
                self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
            case 'walk':
                self.turn_rate = 0
            case 'walk-turn':
                self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
        # Clip to avoid useless commands
        if self.x_velocity <= 0.1:
            self.x_velocity = 0
        if self.y_velocity <= 0.1:
            self.y_velocity = 0

    def compute_reward(self, action: np.ndarray):
        return self._compute_reward(self, action)

    def compute_done(self):
        return self._compute_done(self)

    def get_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()
        robot_state = self.get_robot_state()
        if self.actor_feet:
            feetinbase = self.sim.get_feet_position_in_base() + np.random.uniform(-0.02, 0.02, size=6)
            robot_state = np.concatenate((robot_state, feetinbase))
        out = np.concatenate((robot_state,
                              [self.x_velocity, self.y_velocity, self.turn_rate],
                              [self.clock.get_swing_ratios()[0], 1 - self.clock.get_swing_ratios()[0]],
                              self.clock.get_period_shifts(),
                              input_clock))
        if not is_variable_valid(out):
            raise RuntimeError(f"States has Nan or Inf values. Training stopped.\n"
                               f"get_state returns {out}")
        return out

    def get_action_mirror_indices(self):
        return self.motor_mirror_indices

    def get_observation_mirror_indices(self):
        mirror_inds = [x for x in self.robot_state_mirror_indices]
        if self.actor_feet:
            mirror_inds += [len(mirror_inds) + 3, -(len(mirror_inds) + 4), len(mirror_inds) + 5,
                            len(mirror_inds),     -(len(mirror_inds) + 1), len(mirror_inds) + 2]
        # XY velocity command
        mirror_inds += [len(mirror_inds), -(len(mirror_inds) + 1), -(len(mirror_inds) + 2)]
        # swing ratio
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)]
        # period shift
        mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)]
        # input clock sin/cos
        if self.full_clock:
            # mirror_inds += [-len(mirror_inds), -(len(mirror_inds) + 1),
            #                 -(len(mirror_inds) + 2), -(len(mirror_inds) + 3)]
            mirror_inds += [len(mirror_inds) + 2, len(mirror_inds) + 3,
                            len(mirror_inds), len(mirror_inds) + 1]
        else:
            mirror_inds += [-len(mirror_inds), -(len(mirror_inds) + 1)]
        return mirror_inds

    def _init_interactive_key_bindings(self,):
        """
        Updates data used by the interactive control menu print functions to display the menu of available commands
        as well as the table of command inputs sent to the policy.
        """

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
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate - 0.01 * np.pi/4)
        }
        self.input_keys_dict["q"] = {
            "description": "increase turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate + 0.01 * np.pi/4)}
        self.input_keys_dict["o"] = {
            "description": "increase clock cycle time",
            "func": lambda self: setattr(self.clock, "_cycle_time", np.clip(
                self.clock._cycle_time + 0.01,
                self._cycle_time_bounds[0],
                self._cycle_time_bounds[1]
            ))
        }
        self.input_keys_dict["u"] = {
            "description": "decrease clock cycle time",
            "func": lambda self: setattr(self.clock, "_cycle_time", np.clip(
                self.clock._cycle_time - 0.01,
                self._cycle_time_bounds[0],
                self._cycle_time_bounds[1]
            ))
        }
        self.input_keys_dict["-"] = {
            "description": "increase swing ratio",
            "func": lambda self: setattr(self.clock, "_swing_ratios",
                np.full((2,), np.clip(self.clock._swing_ratios[0] + 0.1,
                    self._swing_ratio_bounds[0],
                    self._swing_ratio_bounds[1])))
        }
        self.input_keys_dict["="] = {
            "description": "decrease swing ratio",
            "func": lambda self: setattr(self.clock, "_swing_ratios",
                np.full((2,), np.clip(self.clock._swing_ratios[0] - 0.1,
                    self._swing_ratio_bounds[0],
                    self._swing_ratio_bounds[1])))
        }
        self.input_keys_dict["k"] = {
            "description": "increase period shift",
            "func": lambda self: setattr(self.clock, "_period_shifts",
                np.array([0, np.clip(self.clock._period_shifts[1] + 0.05,
                    self._period_shift_bounds[0],
                    self._period_shift_bounds[1])]
                    ))
        }
        self.input_keys_dict["l"] = {
            "description": "decrease period shift",
            "func": lambda self: setattr(self.clock, "_period_shifts",
                np.array([0, np.clip(self.clock._period_shifts[1] - 0.05,
                    self._period_shift_bounds[0],
                    self._period_shift_bounds[1])
                    ]))
        }

        self.control_commands_dict["x velocity"] = None
        self.control_commands_dict["y velocity"] = None
        self.control_commands_dict["turn rate"] = None
        self.control_commands_dict["clock cycle time"] = None
        self.control_commands_dict["swing ratios"] = None
        self.control_commands_dict["period shifts"] = None
        # # in order to update values without printing a new table to terminal at every step
        # # equal to the length of control_commands_dict plus all other prints for the table, i.e table header
        self.num_menu_backspace_lines = len(self.control_commands_dict) + 3

    def _update_control_commands_dict(self,):
        self.control_commands_dict["x velocity"] = self.x_velocity
        self.control_commands_dict["y velocity"] = self.y_velocity
        self.control_commands_dict["turn rate"] = self.turn_rate
        self.control_commands_dict["clock cycle time"] = self.clock._cycle_time
        self.control_commands_dict["swing ratios"] = tuple(round(x, 2) for x in (
            self.clock._swing_ratios[0], self.clock._swing_ratios[1]))
        self.control_commands_dict["period shifts"] = tuple(round(x, 2) for x in (
            self.clock._period_shifts[0], self.clock._period_shifts[1]))

    def interactive_control(self, c):
        if c in self.input_keys_dict:
            self.input_keys_dict[c]["func"](self)
            self._update_control_commands_dict()
            self.display_control_commands()
        if c == '0':
            self.x_velocity = 0
            self.y_velocity = 0
            self.turn_rate = 0
            self.clock.set_cycle_time(0.7)
            self.clock.set_swing_ratios([0.5, 0.5])
            self.clock.set_period_shifts([0, 0.5])
            self._update_control_commands_dict()
            self.display_control_commands()

def add_env_args(parser: argparse.ArgumentParser | SimpleNamespace | argparse.Namespace):
    """
    Function to add handling of arguments relevant to this environment construction. Handles both
    the case where the input is an argument parser (in which case it will use `add_argument`) and
    the case where the input is just a Namespace (in which it will just add to the namespace with
    the default values) Note that arguments that already exist in the namespace will not be
    overwritten. To add new arguments if needed, they can just be added to the `args` dictionary
    which should map arguments to the tuple pair (default value, help string).

    Args:
        parser (argparse.ArgumentParser or SimpleNamespace, or argparse.Namespace): The argument
            parser or Namespace object to add arguments to

    Returns:
        argparse.ArgumentParser or SimpleNamespace, or argparse.Namespace: Returns the same object
            as the input but with added arguments.
    """
    args = {
        "simulator-type" : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\")"),
        "terrain" : ("", "What terrain to train with (default is flat terrain)"),
        "policy-rate" : (50, "Rate at which policy runs in Hz"),
        "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
        "state-noise" : (0.0, "Amount of noise to add to proprioceptive state."),
        "velocity-noise" : (0.0, "Amount of noise to add to motor and joint state."),
        "state-est" : (False, "Whether to use true sim state or state estimate. Only used for \
                       libcassie sim."),
        "reward-name" : ("locomotion_linear_clock_reward", "Which reward to use"),
        "clock-type" : ("linear", "Which clock to use (\"linear\" or \"von_mises\")"),
        "full-clock" : (False, "Whether to input the full clock (sine/cosine for each leg) or just \
                        single sine/cosine pair (default is False)"),
        "full-gait" : (False, "Whether to train on all gait parameters or just train walking \
                       (default is False)"),
        "integral-action" : (False, "Whether to use integral action in the clock (default is False)"),
        "actor-feet" : (False, "Whether to use actor feet or not (default is False)"),
    }
    if isinstance(parser, argparse.ArgumentParser):
        env_group = parser.add_argument_group("Env arguments")
        for arg, (default, help_str) in args.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                env_group.add_argument("--" + arg, action=argparse.BooleanOptionalAction)
            else:
                env_group.add_argument("--" + arg, default = default, type = type(default), help = help_str)
        env_group.set_defaults(dynamics_randomization=True)
        env_group.set_defaults(state_est=False)
        env_group.set_defaults(full_clock=False)
        env_group.set_defaults(full_gait=False)
        env_group.set_defaults(integral_action=False)
        env_group.set_defaults(actor_feet=False)
    elif isinstance(parser, (SimpleNamespace, argparse.Namespace)):
        for arg, (default, help_str) in args.items():
            arg = arg.replace("-", "_")
            if not hasattr(parser, arg):
                setattr(parser, arg, default)
    else:
        raise RuntimeError(f"{FAIL}Environment add_env_args got invalid object type when trying " \
                           f"to add environment arguments. Input object should be either an " \
                           f"ArgumentParser or a SimpleNamespace.{ENDC}")

    return parser