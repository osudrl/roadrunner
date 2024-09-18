import copy
import numpy as np
import os

from env.genericenv import GenericEnv
from util.colors import FAIL, WARNING, ENDC


class StandEnv(GenericEnv):
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
        if robot_name == "digit":
            self._height_bounds = [0.6, 1.3]
            self.reset_states = np.load(os.path.dirname(os.path.realpath(__file__)) + "/digit_init_data.npz")
        elif robot_name == "cassie":
            self._height_bounds = [0.6, 1.1]
            self.reset_states = np.load(os.path.dirname(os.path.realpath(__file__)) + "/cassie_init_data.npz")
        else:
            raise ValueError(f"{FAIL}Unknown robot name: {robot_name}{ENDC}")
        self.num_reset = self.reset_states["pos"].shape[0]

        self._randomize_commands_bounds = [100, 200] # in episode length

        self.cmd_height = 0.9
        self.base_adr = self.sim.get_body_adr(self.sim.base_body_name)

        # Only check obs if this envs is inited, not when it is parent:
        if self.__class__.__name__ == "LocomotionEnv" and self.simulator_type not in ["ar_async", "real"]:
            self.check_observation_action_size()

    @property
    def observation_size(self):
        return super().observation_size + 1 # height command

    @property
    def extra_input_names(self):
        return ['cmd-height']

    def reset(self, interactive_evaluation=False):
        self.randomize_commands_at = np.random.randint(*self._randomize_commands_bounds)
        self.randomize_commands()

        self.push_force = np.random.uniform(0, 30, size = 2)
        self.push_duration = np.random.randint(5, 10)
        self.push_start_time = np.random.uniform(100, 200)

        self.reset_simulation()
        rand_ind = np.random.randint(self.num_reset)
        reset_qpos = copy.deepcopy(self.reset_states["pos"][rand_ind, :])
        reset_qpos[0:2] = np.zeros(2)
        self.sim.reset(qpos = reset_qpos, qvel = self.reset_states["vel"][rand_ind, :])

        self.interactive_evaluation = interactive_evaluation
        if interactive_evaluation:
            self._update_control_commands_dict()

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        self.max_foot_vel = 0

        return self.get_state()

    def step(self, action: np.ndarray):
        self.policy_rate = self.default_policy_rate
        if self.dynamics_randomization:
            self.policy_rate += np.random.randint(0, 6)

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps, integral_action=self.integral_action)

        # Reward for taking current action before changing quantities for new state
        self.compute_reward(action)

        self.traj_idx += 1
        self.last_action = action

        if self.traj_idx % self.randomize_commands_at == 0 and not self.interactive_evaluation:
            self.randomize_commands()

        if not self.interactive_evaluation:
            if self.push_start_time <= self.traj_idx < self.push_start_time + self.push_duration:
                self.sim.data.xfrc_applied[self.base_adr, 0:2] = self.push_force
            elif self.traj_idx == self.push_start_time + self.push_duration:
                self.sim.data.xfrc_applied[self.base_adr, 0:2].xfrc_applied[0:2] = np.zeros(2)

        return self.get_state(), self.reward, self.compute_done(), {'rewards': self.reward_dict}

    def hw_step(self):
        pass

    def _get_state(self):
        return np.concatenate((
            self.get_robot_state(),
            [self.cmd_height],
        ))

    def randomize_commands(self):
        self.cmd_height = np.random.uniform(*self._height_bounds)

    def get_action_mirror_indices(self):
        return self.robot.motor_mirror_indices

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot.robot_state_mirror_indices
        mirror_inds += [len(mirror_inds)] # height commands
        return mirror_inds

    def _init_interactive_key_bindings(self):
        self.input_keys_dict["w"] = {
            "description": "increment cmd height",
            "func": lambda self: setattr(self, "cmd_height", self.cmd_height + 0.1)
        }
        self.input_keys_dict["s"] = {
            "description": "decrement cmd height",
            "func": lambda self: setattr(self, "cmd_height", self.cmd_height - 0.1)
        }
        def zero_command(self):
            self.cmd_height = 0.9
        self.input_keys_dict["0"] = {
            "description": "reset all height command to nominal",
            "func": zero_command,
        }

    def _init_interactive_xbox_bindings(self):
        self.input_xbox_dict["LeftJoystickY"] = {
            "description": "in/decrement cmd height",
            "func": lambda self, joystick: setattr(self, "cmd_height", self.cmd_height + joystick / self.default_policy_rate)
        }
        def zero_command(self, back):
            self.cmd_height
        self.input_xbox_dict["Back"] = {
            "description": "reset all commands to zero",
            "func": zero_command
        }

    def _update_control_commands_dict(self):
        self.control_commands_dict["cmd height"] = self.cmd_height

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
            "reward-name"        : ("stand_reward", "Which reward to use"),
            "integral-action"    : (False, "Whether to use integral action in the clock (default is False)"),
        }
