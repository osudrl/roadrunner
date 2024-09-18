import os

import numpy as np

from env.util.periodicclock import PeriodicClock
from env.tasks.locomotionclockenv.locomotionclockenv import LocomotionClockEnv
from util.colors import FAIL, WARNING, ENDC


class StoneEnv(LocomotionClockEnv):
    def __init__(
        self,
        robot_name: str,
        clock_type: str,
        reward_name: str,
        simulator_type: str,
        terrain: str,
        policy_rate: int,
        dynamics_randomization: bool,
        state_noise: float,
        state_est: bool,
        z_step: bool
    ):
        # Override default Cassie dynamics randomization ranges with custom version:
        if robot_name == "cassie":
            kwargs = {'dynamics_randomization_file_path': \
                os.path.join(os.path.dirname(__file__), "cassie_dynamics_randomization.json")}
        super().__init__(
            robot_name=robot_name,
            reward_name=reward_name,
            simulator_type=simulator_type,
            terrain=terrain,
            policy_rate=policy_rate,
            dynamics_randomization=dynamics_randomization,
            state_noise=state_noise,
            state_est=state_est,
            clock_type=clock_type,
            **kwargs,
        )

        # Env args
        self.z_step = z_step

        # Command randomization ranges
        self._x_velocity_bounds = [0.0, 3.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._swing_ratio_bounds = [0.4, 0.8]
        self._period_shift_bounds = [0.0, 0.5]
        self._cycle_time_bounds = [0.75, 1.5]

        stones = self.generate_fixed_stones(z_step=self.z_step)
        for s in range(len(self.sim.geom_generator.geoms)):
            stone_size = 0.15 if s < 2 else 0.1
            self.sim.geom_generator._create_geom(f'box{s}', *stones[s], rise=.01,
                                                  length=stone_size, width=stone_size)
        self.sim.adjust_robot_pose()
        self.touchdown_by_clock_flag = [False, False]
        self.is_stance_previous = [False, False]
        self.steps_active_idx = 0
        self.steps_commands_pelvis = self.steps_target_global[self.steps_active_idx] - \
                                     self.sim.get_body_pose(self.sim.base_body_name)[0:3]

        # self.state_dim = len(self.get_robot_state())
        # self.nonstate_dim = 7

        # Only check obs if this envs is inited, not when it is parent:
        if self.__class__.__name__ == "StoneEnv" and self.simulator_type not in ["ar_async", "real"]:
            self.check_observation_action_size()

    @property
    def action_size(self):
        return super().action_size + 1 # phase_add

    @property
    def observation_size(self):
        return super().observation_size - 2 # TODO @helei

    def reset(self):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        self.reset_simulation()
        self.sim.set_geom_quat("floor", np.array([1,0,0,0]))
        self.orient_add = 0

        # Update clock
        # NOTE: Both cycle_time and phase_add are in terms in raw time in seconds
        swing_ratios = [0.5, 0.5]#np.random.uniform(*self._swing_ratio_bounds, 2)
        period_shifts = [0, 0.5]#np.random.uniform(*self._period_shift_bounds, 2)
        self.cycle_time = 0.8#np.random.uniform(*self._cycle_time_bounds)
        phase_add = 1 / self.default_policy_rate
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Stone reset
        stones = self.generate_fixed_stones(z_step=self.z_step)
        for s in range(len(self.sim.geom_generator.geoms)):
            stone_size = 0.15 if s < 2 else 0.1
            self.sim.geom_generator._create_geom(f'box{s}', *stones[s], rise=.01,
                                                  length=stone_size, width=stone_size)
        # self.sim.set_geom_color('box0',np.array([255, 0, 0, 1]))
        self.sim.adjust_robot_pose()

        self.touchdown_by_clock_flag = [False, False]
        self.is_stance_previous = [False, False]
        self.steps_active_idx = 0
        self.steps_commands_pelvis = self.steps_target_global[self.steps_active_idx] - \
                                     self.sim.get_body_pose(self.sim.base_body_name)[0:3]

        # Reset env counter variables
        self.traj_idx = 0
        self.last_action = None
        self.last_base_position = self.sim.get_base_position()

        return self.get_state()

    def step(self, action: np.ndarray):
        # Unpack actions besides motor actions
        phase_add_residual = action[-1]
        action = action[:-1]

        # Step simulation by n steps. This call will update self.tracker_fn.
        if self.dynamics_randomization:
            self.policy_rate = self.default_policy_rate + np.random.randint(-5, 6)
        else:
            self.policy_rate = self.default_policy_rate
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps)

        # Check time-based clock for gait event
        is_stance = self.clock.is_stance()
        # print(f"clock value, {self.clock.linear_clock()}, stance {is_stance} "
        #       f"previous stance {self.is_stance_previous} ")
        for i, stance in enumerate(is_stance):
            if stance:
                if not self.touchdown_by_clock_flag[i] and not self.is_stance_previous[i]:
                    self.touchdown_by_clock_flag[i] = True
                    # print(self.traj_idx, "at steps_active_idx #", self.steps_active_idx, \
                    # f"side of TD {self.touchdown_by_clock_flag}\n"
                    # "This TD loc ", self.steps_target_global[self.steps_active_idx][0:2],\
                    # "Next TD side ", self.steps_order[self.steps_active_idx+1])
                else:
                    self.touchdown_by_clock_flag[i] = False

        # Reward for taking current action before changing quantities for new state
        self.compute_reward(action)

        # Update footstep targets at TD event
        if any(self.touchdown_by_clock_flag):
            self.update_footstep_target()

        # Update counter variable after reward
        self.is_stance_previous = self.clock.is_stance()
        self.traj_idx += 1
        self.last_action = action
        self.last_base_position = self.sim.get_base_position()

        # Increment clock at the last place and update s' with get_state()
        new_phase_add = self.clock.get_phase_add() + \
                        self.scale_number(phase_add_residual, min=-15/2000, max=15/2000, smoothness=3)
        self.clock.increment()

        return self.get_state(), self.reward, self.compute_done(), {}

    def update_footstep_target(self):
        """Update inputs for s', assuming touchdown event on any side is triggered
        """
        self.steps_active_idx += 1 # increment for next step info
        # update the target, global is based on the curr TD pos + the relative commands
        self.steps_com_target[self.steps_active_idx] = self.steps_target_global[self.steps_active_idx]
        self.steps_commands_pelvis = self.steps_target_global[self.steps_active_idx] -\
                                     self.sim.get_body_pose(self.sim.base_body_name)[0:3]
        # print(f"Update input as {self.steps_target_global[self.steps_active_idx][0:2]}")
        # input()

    def _get_state(self):
        # self.steps_commands_pelvis = [0, 0.135 * -np.power(-1, self.steps_order[self.steps_active_idx]), -1]
        # print(self.steps_commands_pelvis)
        # input()
        command_target2pelvis = self.steps_commands_pelvis / np.linalg.norm(self.steps_commands_pelvis)
        command_target2pelvis = np.append(command_target2pelvis, np.linalg.norm(self.steps_commands_pelvis))
        return np.concatenate((
            self.get_robot_state(),
            command_target2pelvis,
            [0],
            self.clock.input_sine_only_clock()
        ))

    def get_action_mirror_indices(self):
        mirror_inds = super().get_action_mirror_indices()
        mirror_inds += [len(mirror_inds)] # phase_add
        return mirror_inds

    def get_observation_mirror_indices(self):
        mirror_inds = self.robot.robot_state_mirror_indices
        # XYZ footstep + distance
        mirror_inds += [len(mirror_inds), -(len(mirror_inds)+1), len(mirror_inds) + 2, len(mirror_inds) + 3]
        # orientation
        mirror_inds += [len(mirror_inds)]
        # input clock sin/cos
        mirror_inds += [- len(mirror_inds), - (len(mirror_inds) + 1)]
        return mirror_inds

    def generate_fixed_stones(self, z_step=False):
        """
        Generate 20 consecutive footsteps in 3D global coordinates
        Return:
            A dict of step commands representing from cmd_t to cmd_{t+1}
            A dict of contact sequence
            A dict of the step target in global coordinates
        Mode:
            saggital: next target has to be larger in abs(x) values
            frontal-left: next target has to be y larger for the next left side
            frontal-right: next target has to be y smaller for the next right side
        Method:
            Lay (x, y) in global coordinates according to the modes and their rules.
            The final sequence has to choose 2 modes by first and second half.
            Each half will follow the rule of the mode on generating 10 footsteps.
            Then stiching the two halfs together to return the final footstep sequence.
        """
        # initialize the first step
        fpos = [self.sim.get_site_pose(self.sim.feet_site_name[0])[0:3],
                self.sim.get_site_pose(self.sim.feet_site_name[1])[0:3]]
        self.steps_order           = {}
        self.steps_commands        = {}
        self.steps_target_relative = {}
        self.steps_target_global   = {}
        self.steps_com_target      = {}

        self.steps_order.clear()
        self.steps_commands.clear()
        self.steps_target_relative.clear()
        self.steps_target_global.clear()
        self.steps_com_target.clear()

        initial_phase = self.clock.get_phase()
        self.steps_order[0] = self.get_init_contact_order(initial_phase) # first step target side
        # print("first contact side ", self.steps_order[0])
        self.steps_commands[0] = {"sl": np.random.uniform(0.2, 0.35),
                                  "sd": np.pi/2*np.power(-1, self.steps_order[0])}
        # used for rewards
        self.steps_target_relative[0] = \
            np.array([self.steps_commands[0]['sl']*np.cos(self.steps_commands[0]['sd']),
            self.steps_commands[0]['sl']*np.sin(self.steps_commands[0]['sd']),
            0])
        self.steps_target_global[0] = fpos[self.steps_order[0]]
        self.steps_com_target[0] = self.steps_target_global[0]

        # pre-compute the rest steps
        step_n1 = np.random.randint(2, 5)
        step_n2 = 12 - step_n1 # avoid key error
        step_n3 = 30 - (step_n1+step_n2)
        modes = ["inplace"]*step_n1 + ["front"]*step_n2 + ["inplace"] * step_n3
        for i in range(step_n1+step_n2+step_n3): # precompute contact orders
            self.steps_order[i+1] = 1-self.steps_order[i]
        # print(self.steps_order)
        for i, mode in enumerate(modes):
            i=i+1
            side = self.steps_order[i] # side is the next on index
            if mode == 'front':
                a = np.random.uniform(20/180*np.pi,60/180*np.pi) * np.power(-1, side) # forward 20-60deg
                l = np.random.uniform(0.2, 0.6)
                h = np.random.uniform(-0.05, 0.05) if i < 5 else np.random.uniform(-0.15, 0.15) # avoid initial geom overlap
            elif mode == 'inplace':
                a = np.pi/2 * np.power(-1, side) # backward
                l = np.random.uniform(0.2, 0.35)
                h = 0
            # append the signals
            self.steps_target_global[i]   = self.steps_target_global[i-1] + np.array([l*np.cos(a), l*np.sin(a), 0])
            if z_step:
                self.steps_target_global[i][2] = h + self.steps_target_global[0][2] # offset by initial robot height
            self.steps_target_relative[i] = np.array([l*np.cos(a), l*np.sin(a), 0])
            self.steps_commands[i]        = {"sl": l, "sd": a}
            self.steps_com_target[i]      = self.steps_target_global[i]
            # if i < 5:
            #     print(mode, side, l, np.array([l*np.cos(a), l*np.sin(a), 0]), self.steps_target_global[i])
            # print()
        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # x=[]
        # y=[]
        # colors = cm.Blues(np.linspace(0, 1, len(self.steps_order)))
        # for key, value in self.steps_target_global.items():
        #     x.append(value[0])
        #     y.append(value[1])
        # print("first TD is ", self.steps_order[0], "step count", len(self.steps_order), "turn angle", turn_angle/np.pi*180)
        # plt.scatter(x, y, color=colors)
        # plt.scatter(x[turn_step], y[turn_step],color='red')
        # plt.axis('equal')
        # plt.show()
        # exit()
        return self.steps_target_global

    def scale_number(self, nn, min, max, smoothness):
        return (max - min) / (1 + np.exp(- smoothness * nn)) + min

    def get_init_contact_order(self, init_phase):
        # print("initial phase ", self.clock._phase)
        lf=[]
        rf=[]
        cnt=0
        while True:
            lswing, rswing = self.clock.linear_clock(percent_transition=0.2)
            lf.append(lswing)
            rf.append(rswing)
            # print(lswing, rswing)
            if lswing < 0.05: # current left TD
                side = 0
                cnt+=1
            elif rswing < 0.05: # right TD
                side = 1
                cnt+=1
            if cnt ==1:
                break
            self.clock.increment()
        # print(side)
        # import matplotlib.pyplot as plt
        # plt.plot(lf, color='red')
        # plt.plot(rf, color='blue')
        # plt.show()
        # exit()
        self.clock.set_phase(init_phase)
        return side

    def get_env_args():
        return {
            "robot-name" : ("cassie", "Which robot to use (\"cassie\" or \"digit\")"),
            "simulator-type" : ("mujoco", "Which simulator to use (\"mujoco\" or \"libcassie\""),
            "terrain" : ("", "What terrain to train with (default is flat terrain)"),
            "policy-rate" : (50, "Rate at which policy runs in Hz"),
            "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
            "state-noise" : ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Amount of noise to add to proprioceptive state."),
            "state-est" : (False, "Whether to use true sim state or state estimate. Only used for \
                        libcassie sim."),
            "reward-name" : ("stepping_stone", "Which reward to use"),
            "clock-type" : ("linear", "Which clock to use (\"linear\" or \"von_mises\")"),
            "z-step" : (False, ""),
        }