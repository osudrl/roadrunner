import argparse
import json
import numpy as np
import os
import traceback
import time

from decimal import Decimal
from env.util.periodicclock import PeriodicClock
from env.util.quaternion import euler2so3, euler2quat, quaternion2euler
from env.digit.digitenv import DigitEnv
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from util.colors import FAIL, WARNING, ENDC
from util.check_number import is_variable_valid
from scipy.spatial.transform import Rotation as R


class DigitHfield(DigitEnv):

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
                 full_clock: bool,
                 offscreen: bool,
                 depth_vis: bool,
                 depth_input: bool,
                 autoclock: bool,
                 hfield_name: str = 'stair',
                 reverse_heightmap: bool = False,
                 collision_negative_reward: bool = False,
                 contact_patch: bool = False,
                 mix_terrain: bool = False,
                 feetmap: bool = False,
                 rangesensor: bool = False,
                 autoclock_simple: bool = False,
                 hfield_noise: bool = False,
                 integral_action: bool = False,
                 actor_feet: bool = False,
                 debug: bool = False):
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

        self.debug = False
        self.reverse_heightmap = reverse_heightmap
        self.feetmap = feetmap
        self.hfield_noise = hfield_noise
        self.rangesensor = rangesensor
        self.autoclock = autoclock
        self.autoclock_simple = autoclock_simple
        self.actor_feet = actor_feet
        self.offscreen = offscreen
        self.depth_vis = depth_vis
        self.depth_input = depth_input
        self.hfield_name = hfield_name
        self.collision_negative_reward = collision_negative_reward
        self.contact_patch = contact_patch
        self.mix_terrain = mix_terrain
        self.integral_action = integral_action
        self.use_basevel = False
        # Constants for sim2real experiments
        self.perception_policy = True
        self.hardware_mode = False
        self.hardware_perception_state = None
        self.offline_collection = False

        # Clock variables
        self.clock_type = clock_type
        self.full_clock = full_clock

        # Command randomization ranges
        self._x_velocity_bounds = [0.0, 0.6]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._turn_rate_bounds = [-np.pi/8, np.pi/8] # rad/s
        self._init_orient_add = [-np.pi, np.pi] # rad
        self._period_shift_bounds = [0.5, 0.5]
        if self.autoclock or self.autoclock_simple:
            self._swing_ratio_bounds = [0.5, 0.5]
            self._cycle_time_bounds = [0.8, 0.8]
        else:
            self._swing_ratio_bounds = [0.4, 0.6]
            self._cycle_time_bounds = [0.7, 1]
        self.command_randomize_freq_bounds = [500, 550] # in terms of trajectory steps

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

        # Height field constants
        # Define spatial heightmap points relative to robot base without rotation
        self.map_x, self.map_y = 1.5, 1 # m
        self.hfield_offset = 0.0 # offset heightmap by x distance
        # self.map_x, self.map_y = 2, 1 # m
        # self.hfield_offset = -0.5 # offset heightmap by x distance
        self.map_points_x, self.map_points_y = int(self.map_x/0.05), int(self.map_y/0.05) # pixels
        self.map_dim = int(self.map_points_x*self.map_points_y)
        x = np.linspace(0 + self.hfield_offset, self.map_x + self.hfield_offset, num=self.map_points_x, dtype=float)
        y = np.linspace(-self.map_y/2, self.map_y/2, num=self.map_points_y, dtype=float)
        grid_x, grid_y = np.meshgrid(x, y)
        self.heightmap_num_points = grid_x.size
        self.local_grid_unrotated = np.zeros((self.heightmap_num_points, 3))
        self.local_grid_unrotated[:, 0] = grid_x.flatten()
        self.local_grid_unrotated[:, 1] = grid_y.flatten()
        # Mirror indices for heightmap to account for saggital symmetry
        ind = np.arange(self.heightmap_num_points).reshape(self.map_points_y, self.map_points_x)
        self.mirror_map_inds = np.fliplr(ind.T).T.flatten().tolist()
        if self.feetmap:
            feet_grid_x, feet_grid_y = np.meshgrid(np.linspace(-0.5, 0.5, num=20, dtype=float),
                                                   np.linspace(-0.3, 0.3, num=10, dtype=float))
            self.feet_grid_unrotated = np.zeros((200, 3))
            self.feet_grid_unrotated[:, 0] = feet_grid_x.flatten()
            self.feet_grid_unrotated[:, 1] = feet_grid_y.flatten()

        # Define spatial heightmap points relative to robot base without rotation
        x_2 = np.linspace(-0.5, 1.5, num=40, dtype=float)
        y_2 = np.linspace(-0.5, 0.5, num=20, dtype=float)
        grid_x_2, grid_y_2 = np.meshgrid(x_2, y_2)
        self.big_grid_unrotated = np.zeros((grid_x_2.size, 3))
        self.big_grid_unrotated[:, 0] = grid_x_2.flatten()
        self.big_grid_unrotated[:, 1] = grid_y_2.flatten()

        # Depth constants
        # A list of depth cameras to avoid re-compile simulator
        # Sim does not allow to change camera intrinsics after init
        self.camera_ranges = ["depth-0", "depth-1", "depth-2", "depth-3", "depth-4"]
        self.camera_name = np.random.choice(self.camera_ranges)
        # Depth constants
        # self.camera_name = "forward-chest-realsense-d435/depth/image-rect"
        # self.camera_name = "forward-pelvis-realsense-d430/depth/image-rect"
        self.camera_name = "xshift-0-yshift-0-zshift-0-fovy-0-tilt-0"
        self.depth_image_dim = (848, 480)
        self.depth_image_size = int(self.depth_image_dim[0]*self.depth_image_dim[1])
        self.depth_clip_range = [0.0, 3.0]

        # Define env specifics
        # NOTE: Hardccode some input values to be able to initialize NN modules without using any simulator
        # Unfortuantely a result of using env stuff for UDP sim2real
        self.observation_size = 67#len(self.get_robot_state())
        self.observation_size += 3 # XY velocity command + turn rate
        self.observation_size += 2 # swing ratio
        self.observation_size += 2 # period shift
        # input clock
        if self.full_clock:
            self.observation_size += 4
        else:
            self.observation_size += 2
        if self.depth_input:
            self.map_dim = self.depth_image_size # depth
        else:
            self.map_dim = self.map_dim # hfield
        self.observation_size += self.map_dim
        self.privilege_obs_size = self.observation_size + 6 + 1 + 4 # footpos + height + foot force + touch sensor
        if self.feetmap:
            self.privilege_obs_size += 400
        self.privilege_obs_size += 7 # NOTE: length of collision list varies
        if self.rangesensor:
            self.privilege_obs_size += 4 # range sensor
        if self.actor_feet:
            self.observation_size += 6 # foot positions
        if self.use_basevel:
            self.privilege_obs_size += 3 # base velocity
        self.state_dim = self.observation_size - self.map_dim
        self.action_size = self.sim.num_actuators
        if self.autoclock:
            self.action_size += 3 # phase_add and shift_add, left, right
        if self.autoclock_simple:
            self.action_size += 2

        # Clock is cheap to initialize
        # Update clock
        # NOTE: Both cycle_time and phase_add are in terms in raw time in seconds
        swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
        swing_ratios = [swing_ratio, swing_ratio]
        period_shifts = [0, np.random.uniform(*self._period_shift_bounds)]
        self.cycle_time = np.random.uniform(*self._cycle_time_bounds)
        phase_add = 1 / self.default_policy_rate
        self.clock = PeriodicClock(self.cycle_time, phase_add, swing_ratios, period_shifts)
        if self.clock_type == "von_mises":
            self.clock.precompute_von_mises()

        # Only want to init renderer once so it checks for the existence of the renderer
        if (self.depth_vis or self.depth_input) and not self.hardware_mode:
            if self.sim.renderer is not None:
                self.sim.renderer.close()
            self.sim.init_renderer(offscreen=self.offscreen,
                                   width=self.depth_image_dim[0], height=self.depth_image_dim[1])

        # Visualization stuff
        self.vis_marker_keys = {}

    def reset_info(self, difficulty=None):
        base_position = self.sim.get_base_position()
        # Randomize terrain, robot position, and update height map/depth image
        if self.mix_terrain:
            hfield_name = np.random.choice(['flat', 'noisy', 'stair-small', 'bump', 'platform'],
                                           p=[0.03, 0.07, 0.35, 0.2, 0.35])
        else:
            hfield_name = self.hfield_name
        # hfield_name = 'bump'
        match hfield_name:
            case 'stair':
                base_position[0] = -5
                self.sim.set_base_position(position=base_position)
                mode = np.random.choice(['up', 'down'], p=[0.5, 0.5])
                height_map = self.sim.hfield_generator.create_stair(difficulty=1, mode=mode)
                self.x_velocity = np.random.uniform(0.1, 1)
                self.turn_rate = 0
            case 'stair-small':
                base_position[0] = np.random.uniform(-5, -1)
                self.sim.set_base_position(position=base_position)
                height_map = self.sim.hfield_generator.create_stair_even(difficulty=0.5)
                self._x_velocity_bounds = [0.3, 1.0]
                self._y_velocity_bounds = [0.0, 0.0]
            case 'bump':
                base_position[0] = -5
                self.sim.set_base_position(position=base_position)
                height_map = self.sim.hfield_generator.create_bump(difficulty=0)
                self._x_velocity_bounds = [0.3, 1.0]
                self._y_velocity_bounds = [0.0, 0.0]
            case 'stone':
                base_position[0] = -6
                self.sim.set_base_position(position=base_position)
                height_map = self.sim.hfield_generator.create_stone(difficulty=0)
                self.x_velocity = np.random.uniform(0.4, 1)
                self._turn_rate_bounds = [0, 0]
            case 'noisy':
                base_position[0] = -3
                self.sim.set_base_position(position=base_position)
                height_map = self.sim.hfield_generator.create_noisy()
                self.x_velocity = np.random.uniform(*[-0.5, 1.0])
                self.y_velocity = np.random.uniform(*[-0.3, 0.3])
            case 'block':
                base_position[0] = 0
                self.sim.set_base_position(position=base_position)
                self.orient_add = np.random.uniform(*self._init_orient_add)
                quaternion = euler2quat(z=self.orient_add, y=0, x=0)
                self.sim.set_base_orientation(quaternion)
                height_map = self.sim.hfield_generator.create_block(difficulty=1,
                                                                    base_pos=base_position)
            case 'platform':
                base_position[0] = 0
                self.sim.set_base_position(position=base_position)
                self.orient_add = np.random.uniform(*self._init_orient_add)
                quaternion = euler2quat(z=self.orient_add, y=0, x=0)
                self.sim.set_base_orientation(quaternion)
                height_map = self.sim.hfield_generator.create_platform(difficulty=0.5,
                                                                       base_pos=base_position)
                self._x_velocity_bounds = [0.3, 1.0]
                self._y_velocity_bounds = [0.0, 0.0]
            case 'single-block':
                self.turn_rate = 0
                self.x_velocity = np.random.uniform(0.1, 0.5)
                self.y_velocity = 0
                radius = np.random.uniform(0.5, 1)
                angle = np.random.uniform(0, 2 * np.pi)
                base_position[0] = radius * np.cos(angle)
                base_position[1] = radius * np.sin(angle)
                self.sim.set_base_position(position=base_position)
                vec1 = np.array([1, 0])
                vec2 = np.array([base_position[0], base_position[1]])
                heading_yaw = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                self.orient_add = np.pi - heading_yaw % np.pi if base_position[1] < 0 else np.pi + heading_yaw % np.pi
                quaternion = euler2quat(z=self.orient_add, y=0, x=0)
                self.sim.set_base_orientation(quaternion)
                height_map = self.sim.hfield_generator.create_single_block(difficulty=1,
                                                                           base_pos=base_position)
            case 'flat':
                height_map = self.sim.hfield_generator.create_flat()
                self.x_velocity = np.random.uniform(*[-0.5, 1.0])
                self.y_velocity = np.random.uniform(*[-0.3, 0.3])
        self.sim.randomize_hfield(data=height_map)
        # Lift the robot up based on how high are feet and flatten the initial spawn hfield
        z_deltas = []
        for (x,y,z) in [self.sim.get_site_pose(self.sim.feet_site_name[0])[:3],
                        self.sim.get_site_pose(self.sim.feet_site_name[1])[:3]]:
            tl_hgt = self.sim.get_hfield_height(x + 0.10, y + 0.05)
            tr_hgt = self.sim.get_hfield_height(x + 0.10, y - 0.05)
            bl_hgt = self.sim.get_hfield_height(x - 0.10, y + 0.05)
            br_hgt = self.sim.get_hfield_height(x - 0.10, y - 0.05)
            z_hgt = max(tl_hgt, tr_hgt, bl_hgt, br_hgt)
            z_deltas.append((z_hgt))
        z_offset = max(z_deltas)
        height_map = self.sim.hfield_generator.flat_spawn_hfield(base_position, height_map, z_offset)
        self.sim.randomize_hfield(data=height_map)
        self.update_hfield_map()
        if self.depth_vis or self.depth_input:
            self.update_depth_image()

    def reset(self, difficulty=None, interactive_evaluation=False):
        """Reset simulator and env variables.

        Returns:
            state (np.ndarray): the s in (s, a, s')
        """
        # self.sim.torque_efficiency = np.random.uniform(0.90, 1.0)
        self.reset_simulation()

        # Reset env counter variables
        self.orient_add = 0
        self.traj_idx = 0
        self.last_action = None
        self.body_collision_with_env = [False] * len(self.sim.body_collision_list)
        self.foot_scuffed = False
        # Per episode noise model for perception
        self.perception_update_rate = 1 # in terms of policy steps
        self.perception_buffer = []
        self.perception_buffer_size = np.random.randint(2, 6) # 1 means no buffer, +1 means 1/50=20ms delay
        x_shift = np.random.randint(-1, 2) # each pixel = 5cm
        y_shift = np.random.randint(-1, 2)
        z_shift = np.random.uniform(-0.1, 0.1) # meters
        self.hfield_shift = {'x': x_shift, 'y': y_shift, 'z': z_shift}

        # Reset heightmap, alter commands and update the perception state from simulation
        self.reset_info(difficulty=difficulty)

        # Reset control commands
        self.randomize_commands(init=True)
        self.command_randomize_freq = np.random.randint(*self.command_randomize_freq_bounds)

        # Since we are hard coding these values at init(), catch here if wrong
        assert self.privilege_obs_size == len(self.get_privilege_state()),\
            f"Privilege observation size mismatch. Expected: {self.privilege_obs_size}, " \
            f"Got: {len(self.get_privilege_state())}"
        assert self.observation_size == len(self.get_state()),\
            f"Observation size mismatch. Expected: {self.observation_size}, " \
            f"Got: {len(self.get_state())}"
        # Only check sizes if calling current class. If is child class, don't need to check
        if os.path.basename(__file__).split(".")[0] == self.__class__.__name__.lower():
            self.check_observation_action_size()

        # Visualization init
        self.update_markers()
        self.interactive_evaluation = interactive_evaluation
        if self.interactive_evaluation:
            self._update_control_commands_dict()

        return self.get_state()

    def step(self, action: np.ndarray):
        assert action.shape == (self.action_size,),\
            f"Mismatch action shape. Expected: {(self.action_size,)}, Got: {action.shape}"

        if self.dynamics_randomization and not self.offline_collection:
            self.policy_rate = self.default_policy_rate + np.random.randint(0, 6)
        else:
            self.policy_rate = self.default_policy_rate

        # Offset global zero heading by turn rate per policy step
        self.orient_add += self.turn_rate / self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action[:self.sim.num_actuators], simulator_repeat_steps, integral_action=self.integral_action)

        # Reward for taking current action before changing quantities for new state
        r = self.compute_reward(action)

        # Update perception information
        if self.traj_idx % self.perception_update_rate == 0:
            if self.depth_vis or self.depth_input:
                self.update_depth_image()
            self.update_hfield_map()

        # Increment episode counter and update previous attributes
        self.traj_idx += 1
        self.last_action = action

        # Increment clock at last for updating s'
        if self.autoclock or self.autoclock_simple:
            self.update_clock(clock_action=action[10:self.action_size])
        self.clock.increment()
        # print("clock", self.clock.is_stance(), "clock value ", self.clock.input_sine_only_clock())

        # Check foot height at touchdown to encourage full contact
        # Only have to check toe and heel height relative to the map
        contact_patch_mismatch = False
        if self.contact_patch:
            for i, foot in enumerate(['left-foot', 'right-foot']):
                if self.feet_grf_tracker_avg[self.sim.feet_body_name[i]][2] > 0:
                    toe = self.sim.get_site_pose(foot+'-toe')
                    mid = self.sim.get_site_pose(foot+'-mid')
                    heel = self.sim.get_site_pose(foot+'-heel')
                    # check height diff of toe and heel
                    if np.abs(self.sim.get_hfield_height(*toe[:2]) - self.sim.get_hfield_height(*mid[:2])) > 0.05 or \
                       np.abs(self.sim.get_hfield_height(*heel[:2]) - self.sim.get_hfield_height(*mid[:2])) > 0.05:
                        contact_patch_mismatch = True
                        break
                    # print("foot", foot, "toe hfield height ", self.sim.get_hfield_height(*toe[:2]), "heel hfield height ", self.sim.get_hfield_height(*heel[:2]))
            # Dense reward for matching contact patch
            # free to give 0.1 during swing on top of walking reward
            if contact_patch_mismatch:
                contact_patch_reward = -1
            else:
                contact_patch_reward = 0.0
            r += contact_patch_reward
        # Check any other physics, if any foot is scuffed on ground or edge of terrain
        foot_scuffed = False
        if self.feet_touch_sensor_tracker_avg['left-toe'] > 0 or \
           self.feet_touch_sensor_tracker_avg['right-toe'] > 0:
            foot_scuffed = True
            self.foot_scuffed = True
        for i, b in enumerate(self.sim.body_collision_list):
            self.body_collision_with_env[i] = self.sim.is_body_collision(b)
        if foot_scuffed:
            r -= 5
        done = self.compute_done() or any(self.body_collision_with_env)
        # Only terminate episode after 30 steps
        # done = done and self.traj_idx > 50
        # Terminate when robot is too far away from origin
        if np.linalg.norm(self.sim.get_base_position()[:2]) > 8:
            done = True

        if self.debug:
            if foot_scuffed:
                print(self.get_privilege_state()[-11:])
                print(f"{self.traj_idx}, Foot scuffed wtih toe touch sensor. Left toe: {self.feet_touch_sensor_tracker_avg['left-toe']}, "
                      f"Right toe: {self.feet_touch_sensor_tracker_avg['right-toe']}")
                print(f"{self.traj_idx}, Foot scuffed wtih toe touch sensor. Left toe: {self.sim.data.sensor('left-range-0').data, self.sim.data.sensor('left-range-1').data}, ",
                      f"Right toe: {self.sim.data.sensor('right-range-0').data, self.sim.data.sensor('right-range-1').data}")
                input("Press Enter to continue...")
            if any(self.body_collision_with_env):
                print(self.get_privilege_state()[-11:])
                print(f"body_collision_with_env: {np.array(self.sim.body_collision_list)[np.where(self.body_collision_with_env)]}")
                input("Press Enter to continue...")

        # Randomize commands
        if self.traj_idx % self.command_randomize_freq == 0 and not self.interactive_evaluation:
            self.randomize_commands()

        # Visualization
        self.update_markers()

        return self.get_state(), r, done, {}

    def update_clock(self, clock_action):
        if self.autoclock_simple:
            shift_add_left, shift_add_right = clock_action[0], clock_action[1]
            shift_left  = 0.0 + self.scale_nn(shift_add_left, -0.5, 0.5, smoothness=3)
            shift_right = 0.5 + self.scale_nn(shift_add_right, -0.5, 0.5, smoothness=3)
        else:
            phase_add, shift_add_left, shift_add_right = clock_action[0], clock_action[1], clock_action[2]
            # Scale phase add by multiplier
            phase_add = 1 / self.default_policy_rate * self.scale_nn(phase_add, 0.75, 1.0, smoothness=3)
            shift_left  = 0.0 + self.scale_nn(shift_add_left, -0.5, 0.5, smoothness=3)
            shift_right = 0.5 + self.scale_nn(shift_add_right, -0.5, 0.5, smoothness=3)
            self.clock.set_phase_add(phase_add=phase_add)
        self.clock.set_period_shifts(period_shifts=[shift_left, shift_right])

    def randomize_commands(self, init=False):
        self.x_velocity = np.random.uniform(*self._x_velocity_bounds)
        self.y_velocity = np.random.uniform(*self._y_velocity_bounds)
        self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
        choices = ['in-place', 'in-place-turn', 'side-walk', 'walk', 'walk-turn']
        mode = np.random.choice(choices, p=[0.05, 0.05, 0.0, 0.6, 0.3])
        match mode:
            case 'in-place':
                self.x_velocity, self.y_velocity, self.turn_rate = 0, 0, 0
            case 'in-place-turn':
                self.x_velocity, self.y_velocity = 0, 0
                self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
            case 'side-walk':
                self.turn_rate, self.x_velocity = 0, 0
            case 'walk':
                self.turn_rate, self.y_velocity = 0, 0
            case 'walk-turn':
                self.turn_rate = np.random.uniform(*self._turn_rate_bounds)
                self.y_velocity = 0
        # Clip to avoid useless commands
        if self.x_velocity <= 0.1:
            self.x_velocity = 0
        if self.y_velocity <= 0.1:
            self.y_velocity = 0

    def compute_reward(self, action: np.ndarray):
        return self._compute_reward(self, action)

    def compute_done(self):
        return self._compute_done(self)

    def update_depth_image(self):
        # Keep self.depth_image as a 2D array and same reference inside env
        self.depth_image = self.sim.get_depth_image(self.camera_name)
        # Clip depth image in case of long range depth
        self.depth_image = np.clip(self.depth_image, *self.depth_clip_range)

    def update_hfield_map(self):
        # Keep self.hfield_map as a 2D array and same reference inside env
        # Random shift per step [-1, 0, 1] for XY
        shift = [np.random.randint(-1, 2), np.random.randint(-1, 2)]
        total_shift = [shift[0] + self.hfield_shift['y'], shift[1] + self.hfield_shift['x']]
        self.hfield_map_realtime, self.hfield_ground_truth, self.local_grid_rotated = \
            self.sim.get_hfield_map(self.local_grid_unrotated, shift=total_shift)
        self.perception_buffer.append(self.hfield_map_realtime)
        if len(self.perception_buffer) >= self.perception_buffer_size:
            self.hfield_map = self.perception_buffer[0]
            self.perception_buffer.pop(0)
        else:
            self.hfield_map = self.perception_buffer[0]
        if self.feetmap:
            self.feet_hfield_map_l, _, self.feet_grid_rotated_l = self.sim.get_hfield_map(self.feet_grid_unrotated, frame='left')
            self.feet_hfield_map_r, _, self.feet_grid_rotated_r = self.sim.get_hfield_map(self.feet_grid_unrotated, frame='right')
        if self.offline_collection:
            _, self.hfield_map_big, self.big_grid_rotated = self.sim.get_hfield_map(self.big_grid_unrotated)
        self.noisify_hfield_map()

    def noisify_hfield_map(self):
        # Z direction noise, adding per step and per episode
        self.hfield_map_noise = self.hfield_map + \
            np.random.uniform(-0.02, 0.02, self.hfield_map.shape) + self.hfield_shift['z']

    def get_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()

        if self.hardware_mode:
            assert self.hardware_perception_state is not None, "Hardware perception input is None!"
            perception_inputs = self.hardware_perception_state.flatten()
        else:
            if self.depth_input:
                # Normalize depth image to [0, 1]
                perception_inputs = self.depth_image.flatten() / self.depth_clip_range[1]
            else:
                perception_inputs = self.hfield_map.flatten()
                if self.hfield_noise:
                    perception_inputs = self.hfield_map_noise.flatten()
                if self.reverse_heightmap:
                    perception_inputs = - (self.sim.get_base_position()[2] - perception_inputs)
        robot_state = self.get_robot_state()
        if self.actor_feet:
            feetinbase = self.sim.get_feet_position_in_base() + np.random.uniform(-0.02, 0.02, size=6)
            robot_state = np.concatenate((robot_state, feetinbase))
        out = np.concatenate((robot_state,
                            [self.x_velocity, self.y_velocity, self.turn_rate],
                            self.clock.get_swing_ratios(),
                            self.clock.get_period_shifts(),
                            input_clock,
                            perception_inputs))
        # if not is_variable_valid(out):
        #     raise RuntimeError(f"States has Nan or Inf values. Training stopped.\n"
        #                        f"get_state returns {out}")
        return out

    def get_privilege_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()
        hfield = self.hfield_ground_truth.flatten()
        if self.feetmap:
            hfield = np.concatenate((hfield, self.feet_hfield_map_l.flatten(), self.feet_hfield_map_r.flatten()))
        out = np.concatenate((self.get_robot_state(),
                            [self.x_velocity, self.y_velocity, self.turn_rate],
                            self.clock.get_swing_ratios(),
                            self.clock.get_period_shifts(),
                            input_clock,
                            hfield,
                            self.sim.get_feet_position_in_base(),
                            [self.sim.get_base_position()[2] - self.sim.get_hfield_height(*self.sim.get_base_position()[:2])],
                            [1] if self.feet_touch_sensor_tracker_avg['left-toe'] > 0 else [0],
                            [1] if self.feet_touch_sensor_tracker_avg['right-toe'] > 0 else [0],
                            [1] if any(self.feet_grf_tracker_avg[self.sim.feet_body_name[0]]) > 0 else [0],
                            [1] if any(self.feet_grf_tracker_avg[self.sim.feet_body_name[1]]) > 0 else [0],
                            self.body_collision_with_env,
                            ))
        if self.rangesensor:
            out = np.concatenate((out, self.sim.data.sensor('left-range-0').data,
                                       self.sim.data.sensor('left-range-1').data,
                                       self.sim.data.sensor('right-range-0').data,
                                       self.sim.data.sensor('right-range-1').data))
        if self.use_basevel:
            out = np.concatenate((out, self.sim.get_base_linear_velocity()))
        # if not is_variable_valid(out):
        #     raise RuntimeError(f"States has Nan or Inf values. Training stopped.\n"
        #                        f"get_state returns {out}")
        return out

    def get_teacher_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()
        perception_inputs = self.hfield_map.flatten()
        if self.reverse_heightmap:
            perception_inputs = - (self.sim.get_base_position()[2] - perception_inputs)
        out = np.concatenate((self.get_robot_state(),
                            [self.x_velocity, self.y_velocity, self.turn_rate],
                            self.clock.get_swing_ratios(),
                            self.clock.get_period_shifts(),
                            input_clock,
                            perception_inputs))
        return out

    def get_nonperception_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()
        out = np.concatenate((self.get_robot_state(),
                            [self.x_velocity, self.y_velocity, self.turn_rate],
                            self.clock.get_swing_ratios(),
                            self.clock.get_period_shifts(),
                            input_clock))
        return out

    def get_imu_state(self, high_rate_data: bool = False):
        if high_rate_data:
            return np.concatenate((self.tracker_base_linear_accel, self.tracker_base_angular_vel), axis=1)
        else:
            return np.concatenate((self.sim.data.qacc[:3], self.sim.get_base_angular_velocity()))

    def get_feet_state(self, high_rate_data: bool = False):
        if high_rate_data:
            return self.tracker_feet_position
        else:
            return self.sim.get_feet_position_in_base()

    def get_motor_state(self):
        return np.concatenate((self.sim.get_motor_position(),
                               self.sim.get_motor_velocity(),
                               self.sim.get_joint_position(),
                               self.sim.get_joint_velocity()))

    def get_proprioceptive_state(self, include_joint: bool = False, include_feet: bool = False):
        """Used for hardware inference
        """
        assert self.hardware_mode == True, "Proprioceptive state is only available in hardware mode."
        assert self.simulator_type == "libcassie", "Proprioceptive state is only available in libcassie simulator."
        assert self.state_est == True, "Proprioceptive state is only available when state estimation is enabled."

        out = np.concatenate((self.sim.robot_estimator_state.pelvis.translationalAcceleration[:],
                              self.sim.robot_estimator_state.pelvis.rotationalVelocity[:]))
        if include_joint:
            out = np.concatenate((out,
                                  self.sim.get_motor_position(state_est=True),
                                  self.sim.get_motor_velocity(state_est=True),
                                  self.sim.get_joint_position(state_est=True),
                                  self.sim.get_joint_velocity(state_est=True)))
        if include_feet:
            out = np.concatenate((out, self.sim.get_feet_position_in_base(state_est=True)))
        return out

    def get_robot_height(self):
        base_pos = self.sim.get_base_position()
        pelvis_global_height = base_pos[2]
        pelvis_on_terrain = base_pos[2] - self.sim.get_hfield_height(*base_pos[:2])
        out = np.concatenate(([pelvis_global_height], [pelvis_on_terrain]))
        return out

    def get_blind_state(self):
        if self.full_clock:
            input_clock = self.clock.input_full_clock()
        else:
            input_clock = self.clock.input_clock()
        out = np.concatenate((self.get_robot_state(),
                              [self.x_velocity, self.y_velocity, self.turn_rate],
                              [self.clock.get_swing_ratios()[0], 1 - self.clock.get_swing_ratios()[0]],
                              self.clock.get_period_shifts(),
                              input_clock))
        return out

    def get_info_dict(self):
        out = {}
        out['teacher_states'] = self.get_teacher_state()
        out['privilege_states'] = self.get_privilege_state()
        return out

    def get_action_mirror_indices(self):
        mirror_inds = [x for x in self.motor_mirror_indices]
        if self.autoclock:
            # phase add and shift add
            mirror_inds += [len(mirror_inds), len(mirror_inds) + 2, len(mirror_inds) + 1]
        if self.autoclock_simple:
            mirror_inds += [len(mirror_inds) + 1, len(mirror_inds)]
        return mirror_inds

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
        if self.depth_input:
            for _ in range(self.depth_image_size):
                mirror_inds += [len(mirror_inds)]
        else:
            non_heightmap_length = len(mirror_inds)
            for i in range(self.map_dim):
                mirror_inds += [self.mirror_map_inds[i] + non_heightmap_length]
        return mirror_inds

    def scale_nn(self, nn, min, max, smoothness):
        return (max - min) / (1 + np.exp(- smoothness * nn)) + min

    def update_markers(self):
        velocity_marker_position = self.sim.get_base_position()
        velocity_marker_position[2] += 0.3
        if self.sim.viewer is not None:
            if not self.vis_marker_keys:
                self.vis_marker_keys['velocity'] = self.sim.viewer.add_marker("arrow", "",
                    velocity_marker_position, [0.01, 0.01, 0.5], [0.0, 0.0, 1.0, 1.0],
                    euler2so3(z=self.orient_add, x=0, y=np.pi/2))
            for k,v in self.vis_marker_keys.items():
                if k == 'velocity':
                    self.sim.viewer.update_marker_position(v, velocity_marker_position)
                    self.sim.viewer.update_marker_so3(v, euler2so3(z=self.orient_add, x=0, y=np.pi/2))
                    q = self.sim.get_base_orientation()
                    quat = R.from_quat([*q[1:4], q[0]]) # quat definiton convention is xyzw
                    vel = quat.apply(self.sim.get_base_linear_velocity(), inverse=True)
                    label = f"x: {self.x_velocity:.2f}vs{vel[0]:.2f} | "
                    label += f"y: {self.y_velocity:.2f}vs{vel[1]:.2f} | "
                    label += f"turn: {self.turn_rate:.2f} | "
                    cycle_time = self.cycle_time if not self.autoclock else 1/(self.default_policy_rate*self.clock._phase_add)
                    phaseadd = self.clock.get_phase_add()
                    label += f"phase add: {phaseadd:.3f} | "
                    label += f"shift: {self.clock._period_shifts[0]:.2f},{self.clock._period_shifts[1]:.2f}"
                    # label += f"swing: {self.clock._swing_ratios[0]:.2f}"
                    self.sim.viewer_update_marker_name(v, label)
            # self.sim.viewer.render_hfield(self.hfield_map_big, self.big_grid_rotated,rgba = np.array([0.8, 0, 0.8, 1.0]), name='big')
            # self.sim.viewer.render_hfield(self.hfield_map, self.local_grid_rotated)
            if hasattr(self, 'hfield_map_pred'):
                self.sim.viewer.render_hfield(self.hfield_map_pred, self.local_grid_rotated)
            # else:
            #     self.sim.viewer.render_hfield(self.hfield_ground_truth, self.local_grid_rotated, rgba=np.array([0, 0, 0.5, 1.0]), name='realtime')
            # self.sim.viewer.render_hfield(self.hfield_map_noise, self.local_grid_rotated, name='noise')
            # self.sim.viewer.render_hfield(self.hfield_ground_truth, self.local_grid_rotated, rgba=np.array([0, 0, 0.5, 1.0]), name='ground-truth')
            # if self.feetmap:
            #     self.sim.viewer.render_hfield(self.feet_hfield_map_l, self.feet_grid_rotated_l,
            #                                   rgba=np.array([1, 0, 0, 1.0]), name='left')
            #     self.sim.viewer.render_hfield(self.feet_hfield_map_r, self.feet_grid_rotated_r,
            #                                   rgba=np.array([0, 0, 1, 1.0]), name='right')

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
        self.input_keys_dict["q"] = {
            "description": "decrease turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate + 0.1 * np.pi/4)
        }
        self.input_keys_dict["e"] = {
            "description": "decrease turn rate",
            "func": lambda self: setattr(self, "turn_rate", self.turn_rate - 0.1 * np.pi/4)
        }

        self.control_commands_dict["x velocity"] = None
        self.control_commands_dict["turn rate"] = None
        # # in order to update values without printing a new table to terminal at every step
        # # equal to the length of control_commands_dict plus all other prints for the table, i.e table header
        self.num_menu_backspace_lines = len(self.control_commands_dict) + 3

    def _update_control_commands_dict(self,):
        self.control_commands_dict["x velocity"] = self.x_velocity
        self.control_commands_dict["turn rate"] = self.turn_rate

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
        "terrain" : ("hfield", "What terrain to train with (default is flat terrain)"),
        "policy-rate" : (50, "Rate at which policy runs in Hz"),
        "dynamics-randomization" : (True, "Whether to use dynamics randomization or not (default is True)"),
        "state-noise" : (0.0, "Amount of noise to add to proprioceptive state."),
        "velocity-noise" : (0.0, "Amount of noise to add to motor and joint state."),
        "state-est" : (False, "Whether to use true sim state or state estimate. Only used for \
                       libcassie sim."),
        "reward-name" : ("depth", "Which reward to use"),
        "clock-type" : ("linear", "Which clock to use (\"linear\" or \"von_mises\")"),
        "full-clock" : (False, "Whether to input the full clock (sine/cosine for each leg) or just \
                        single sine/cosine pair (default is False)"),
        "offscreen" : (False, "Onscreen rendering of depth image."),
        "depth-vis" : (False, "Use depth for vis."),
        "depth-input" : (False, "Use depth as perception input."),
        "autoclock" : (False, "Use autoclock to self-change clock"),
        "hfield-name" : ("flat", "Name of hfield file to load"),
        "reverse-heightmap" : (False, "Whether to reverse the heightmap or not"),
        "collision-negative-reward" : (False, "Whether to use negative reward for collisions"),
        "debug" : (False, "Whether to print debug statements"),
        "contact-patch" : (False, "Whether to use contact patch termination"),
        "mix-terrain" : (False, "Whether to use mixed terrain"),
        "feetmap" : (False, "Whether to use feetmap"),
        "autoclock-simple" : (False, "Whether to use autoclock simple"),
        "hfield-noise" : (False, "Whether to use noise input"),
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
        env_group.set_defaults(depth_vis=False)
        env_group.set_defaults(depth_input=False)
        env_group.set_defaults(offscreen=False)
        env_group.set_defaults(autoclock=False)
        env_group.set_defaults(reverse_heightmap=False)
        env_group.set_defaults(collision_negative_reward=False)
        env_group.set_defaults(debug=False)
        env_group.set_defaults(contact_patch=False)
        env_group.set_defaults(mix_terrain=False)
        env_group.set_defaults(feetmap=False)
        env_group.set_defaults(autoclock_simple=False)
        env_group.set_defaults(hfield_noise=False)
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