import copy
import json
import traceback
from collections import OrderedDict
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from types import MethodType
from typing import Optional

import mujoco as mj
import numpy as np
import torch
from shapely.geometry import Polygon, Point

from env.cassie.cassieenv import CassieEnv
from env.util.periodicclock import PeriodicClock
from env.util.quaternion import *
from util.check_number import is_variable_valid
from util.colors import FAIL, ENDC


class Cassiepede(CassieEnv):

    def __init__(self,
                 clock_type: Optional[str],
                 reward_name: str,
                 simulator_type: str,
                 policy_rate: int,
                 dynamics_randomization: bool,
                 state_noise: list | float,
                 velocity_noise: float,
                 state_est: bool,
                 full_clock: bool = False,
                 full_gait: bool = False,
                 integral_action: bool = False,
                 num_cassie: int = 1,
                 position_offset: float = 0.0,
                 poi_heading_range: float = 0.0,
                 poi_position_offset: float = 0.0,
                 perturbation_force: float = 0.0,
                 force_prob=0.0,
                 cmd_noise=(0.0, 0.0, 0.0),
                 cmd_noise_prob=0.0,
                 mask_tarsus_input=False,
                 com_vis=False,
                 custom_terrain=None,
                 only_deck_force=False,
                 height_control=False,
                 merge_states=False,
                 depth_input=False,
                 offscreen=False):

        assert clock_type == "linear" or clock_type == "von_mises" or clock_type == None, \
            f"{FAIL}CassieEnvClock received invalid clock type {clock_type}. Only \"linear\" or " \
            f"\"von_mises\" are valid clock types.{ENDC}"

        self.num_cassie = num_cassie
        self.position_offset = position_offset
        self.poi_position_offset = poi_position_offset
        self.poi_heading_range = poi_heading_range
        self.custom_terrain = custom_terrain
        self.perturbation_force = perturbation_force
        self.force_prob = force_prob
        self.clock_in_state = True
        assert len(cmd_noise) == 3, f"Command noise should be a tuple of 3 values. Got {cmd_noise}."
        self.cmd_noise = cmd_noise
        self.cmd_noise_prob = cmd_noise_prob
        self.mask_tarsus_input = mask_tarsus_input

        super().__init__(simulator_type=simulator_type,
                         num_cassie=self.num_cassie,
                         terrain=custom_terrain or 'cassiepede',
                         policy_rate=policy_rate,
                         dynamics_randomization=dynamics_randomization,
                         state_noise=state_noise,
                         velocity_noise=velocity_noise,
                         state_est=state_est)

        self.integral_action = integral_action
        self.hardware_mode = False
        self.com_vis = com_vis
        self.depth_input = depth_input
        self.depth_input = depth_input
        self.offscreen = offscreen
        self.eval_ = False

        self._merge_states = merge_states

        self._height_control = height_control

        self._only_deck_force = only_deck_force

        # Clock variables
        self.clock_type = clock_type
        self.full_clock = full_clock

        # Command randomization ranges
        self._x_velocity_bounds = [-0.5, 2.0]
        self._y_velocity_bounds = [-0.3, 0.3]
        self._turn_rate_bounds = [-np.pi / 8, np.pi / 8]  # rad/s
        self._height_bounds = [0.5, 0.8]
        self.full_gait = full_gait

        if self.clock_type:
            # if self.full_gait:
            #     self._swing_ratio_bounds = [0.35, 0.8]
            #     self._period_shift_bounds = [0.0, 0.5]
            #     self._cycle_time_bounds = [0.8, 0.8]
            # else:
            #     self._swing_ratio_bounds = [0.4, 0.7]
            #     self._period_shift_bounds = [-0.5, 0.5]
            #     self._cycle_time_bounds = [0.6, 1.0]
            #
            # Fix clock for simplicity
            self._swing_ratio_bounds = [0.5, 0.5]
            self._period_shift_bounds = [0.0, 0.0]
            self._cycle_time_bounds = [0.6, 0.6]
        else:
            self.air_time_bounds = np.array([0.35, 0.35])

        self._randomize_commands_bounds = [100, 450]  # in episode length
        self._randomize_force_bounds = [50, 200]  # in episode length
        self._randomize_height_bounds = [100, 450]  # in episode length

        # Payload mass ranges for 1, 2 and n-cassie
        match self.num_cassie:
            case 1:
                self._randomize_payload_mass_bounds = [0, 10]
            case 2:
                self._randomize_payload_mass_bounds = [0, 20]
            case _:
                self._randomize_payload_mass_bounds = [0, 15]

        # Load reward module
        self.reward_name = reward_name
        self.load_reward_module()

        # Define env specifics after reset
        if self.reward_name == "locomotion_cassiepede_clock_stand":
            if not self.clock_in_state:
                self.observation_size = 42
                self.privilege_observation_size = 42
            else:
                self.observation_size = 50
                self.privilege_observation_size = 50
        elif self.reward_name == "locomotion_cassiepede":
            self.observation_size = 49
            self.privilege_observation_size = 49
        elif self.reward_name in ["locomotion_cassiepede_feetairtime",
                                  "locomotion_cassiepede_feetairtime_modified",
                                  # "locomotion_cassiemultirobot",
                                  ]:
            self.observation_size = 41 + self._height_control
            self.privilege_observation_size = 41 + self._height_control

            self.observation_size = OrderedDict(
                base_orient_cmd=(4,),
                base_yaw_poi=(1,),
                base_ang_vel=(3,),
                motor_pos=(10,),
                motor_vel=(10,),
                joint_pos=(4,),
                joint_vel=(4,),
                cmd=(3 + self._height_control,),
                encoding=(2,)
            )

            if self._merge_states:
                for k in self.observation_size:
                    self.observation_size[k] = (self.observation_size[k][0] * self.num_cassie,)
        else:
            raise ValueError(f"Invalid reward name {self.reward_name}")

        if type(self.sim.num_actuators) == int:
            self.action_size = self.sim.num_actuators
        else:
            self.action_size = self.sim.num_actuators[1]

        if self._merge_states:
            self.action_size *= self.num_cassie

        self.vis_marker_keys = {}

        self.encoding = None

        # self.running_pelvis_position = ExponentialMovingStd(shape=(3,), alpha=0.9)

        self.target_pertub_bodies = [self.sim.get_body_adr(f'deck')]
        if not self._only_deck_force:
            for i in range(self.num_cassie):
                i = '' if i == 0 else f'c{i + 1}_'
                self.target_pertub_bodies.append(self.sim.get_body_adr(f'{i}cassie-pelvis'))
        self.force_vector = np.zeros((len(self.target_pertub_bodies), 3), dtype=float)
        self.torque_vector = np.zeros((len(self.target_pertub_bodies), 3), dtype=float)

        self.robot_state_mirror_indices = dict(
            base_orient_cmd=[0.01, -1, 2, -3],
            base_yaw_poi=[-0.01],
            base_ang_vel=[-0.01, 1, -2],
            motor_pos=[-5, -6, 7, 8, 9, -0.01, -1, 2, 3, 4],
            motor_vel=[-5, -6, 7, 8, 9, -0.01, -1, 2, 3, 4],
            joint_pos=[2, 3, 0.01, 1],
            joint_vel=[2, 3, 0.01, 1],
            cmd=[0.01, -1, -2] + [3] if self._height_control else [],
            encoding=[0.01, -1]
        )

    def get_action_mirror_indices(self):
        return self.motor_mirror_indices

    def get_mirror_dict(self):
        # # State mirror indices
        # state_mirror_indices = torch.tensor(self.robot_state_mirror_indices, dtype=torch.float32, device=device)
        # privilege_state_mirror_indices = state_mirror_indices.clone()
        #
        # # Action mirror indices
        # action_mirror_indices = torch.tensor(self.motor_mirror_indices, dtype=torch.float32, device=device)

        mirror_dict = {
            'state_mirror_indices': self.robot_state_mirror_indices,
            'action_mirror_indices': self.motor_mirror_indices,
        }
        return mirror_dict

    @staticmethod
    def kernel(x):
        return np.exp(-x)

    def compute_reward(self, action: np.ndarray):
        rewards = []

        # Get raw rewards from reward module
        rewards_dict_raw = self._compute_reward_components(action)
        for q in rewards_dict_raw:

            # Add up all reward components
            reward_dict = {}
            for name in q:
                if not is_variable_valid(q[name]):
                    raise RuntimeError(f"Reward {name} has Nan or Inf values as {q[name]}.\nTraining stopped.")
                if "kernel" in self.reward_weight[name] and self.reward_weight[name]['kernel'] == False:
                    reward_dict[name] = self.reward_weight[name]["weighting"] * q[name]
                else:
                    reward_dict[name] = self.reward_weight[name]["weighting"] * self.kernel(
                        self.reward_weight[name]["scaling"] * q[name])

            rewards.append(reward_dict)

        return rewards, rewards_dict_raw

    def compute_done(self, *args, **kwargs):
        return self._compute_done(*args, **kwargs)

    def load_reward_module(self):
        try:
            reward_module = import_module(f"env.rewards.{self.reward_name}.{self.reward_name}")
            self._compute_reward_components = MethodType(reward_module.compute_rewards, self)
            self._compute_done = MethodType(reward_module.compute_done, self)
            reward_path = Path(__file__).parents[2] / "rewards" / self.reward_name / "reward_weight.json"
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
            # logging.warning(f"{WARNING}WARNING: Reward weightings do not sum up to 1, renormalizing.{ENDC}")
            for name, weight_dict in self.reward_weight.items():
                weight_dict["weighting"] /= float(weight_sum)

    def eval(self, value):
        self.eval_ = value

    def _get_vertices(self, centroid, n=3, d=1.):
        if n == 1:
            return np.array([centroid])

        centroid = np.array(centroid)
        vertices = []

        central_angle = 2 * np.pi / n

        radius = (d / 2) / np.sin(central_angle / 2)

        for i in range(n):
            theta = i * central_angle

            vertex = centroid + radius * np.array([np.cos(theta), np.sin(theta)])

            vertices.append(vertex)

        return np.array(vertices)

    def _sample_poi(self, points, poi_position_offset, n=1):
        match len(points):
            case 1:
                # If there are only one point, sample a point away from the point at distance noise
                poi_position = points[0] + np.random.uniform(-poi_position_offset, poi_position_offset, (2,))
                return poi_position
            case 2:
                # If there are only two points, sample a point from their mid-point with radius of their half distance
                # mid = (points[0] + points[1]) / 2.0

                radius = np.random.uniform(0, poi_position_offset, size=n)

                angle = np.random.uniform(-np.pi, np.pi, size=n)

                # result = mid[:, None] + radius * np.stack((np.cos(angle), np.sin(angle)), axis=0)

                # Sample a poi around the cassie
                result = points[np.random.randint(0, len(points))][:, None] + radius * np.stack(
                    (np.cos(angle), np.sin(angle)), axis=0)

                if n == 1:
                    return result[:, 0]
            # case 2:
            #     # If there are only two points, sample a point on the line between them
            #     x_rand = np.random.uniform(points[0][0], points[1][0], size=n)
            #
            #     m = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0] + 1e-10)
            #
            #     y_rand = (x_rand - points[0][0]) * m + points[0][1]
            #
            #     result = np.stack((x_rand, y_rand), axis=-1)
            #
            #     if n == 1:
            #         return result[0]
            case _:
                # If there are more than two points, sample a point inside the polygon
                polgyon = Polygon(points)

                result = []

                min_bound = np.min(points, axis=0)
                max_bound = np.max(points, axis=0)

                while len(result) < n:
                    point = np.random.uniform(min_bound, max_bound)

                    if polgyon.contains(Point(point)):
                        result.append(point)

                if n == 1:
                    return result[0]

                return np.array(result)

    # All position are computed on local frame of frame
    def _randomize_cassie_position(self, noise):
        if self.custom_terrain:
            # If custom terrain, load the position from the terrain. No need to compute position in polygon
            mj.mj_forward(self.sim.model, self.sim.data)

            # Save original position for restoring at reset
            if not hasattr(self, 'local_base_position'):
                self.local_base_position = self.get_base_position()[:, :2]
                local_base_position = self.local_base_position.copy()
            else:
                local_base_position = self.local_base_position.copy()
        else:
            # If non-custom terrain, compute position as vertices of a polygon
            local_base_position = self._get_vertices([0, 0], n=self.num_cassie, d=1.5)

        # Initialize orientation of poi same as randomized commanded heading
        assert not np.any(self.orient_add != self.orient_add[0])
        poi_orientation = self.orient_add[0]

        if self.num_cassie > 1:
            # Randomize local position of cassie with noise for non-custom terrain and more than 1 cassie env
            local_base_position += np.random.uniform(-noise / 2, noise / 2, (self.num_cassie, 2))

        # Update position of cassie based on recompute position
        for i in range(self.num_cassie):
            name = '' if i == 0 else f'c{i + 1}_'

            self.sim.model.body(f'{name}cassie-pelvis').pos[:2] = local_base_position[i]

            self.sim.model.body(f'{name}cassie-pelvis').quat = euler2quat(z=self.orient_add[[i]], y=0, x=0)[0]

        if self.poi_position_offset:
            poi_position = self._sample_poi(local_base_position, self.poi_position_offset)
        elif self.custom_terrain:
            # Get point of interest position directly from terrain
            if not hasattr(self, 'poi_position'):
                self.poi_position = self.get_poi_position()[:2].copy()
            poi_position = self.poi_position
        else:
            poi_position = local_base_position[0]

        # Update the poi position
        self._update_poi_position(poi_position=poi_position, poi_orientation=poi_orientation)

        # Compute encoding
        self._compute_encoding(poi_position=poi_position, poi_orientation=poi_orientation,
                               base_positions=local_base_position)

        if self.custom_terrain or self.num_cassie == 1:
            # If custom terrain, no need to update the geom because terrain is fixed.
            # Also, if only one cassie, no need to update connector position as it is always fixed in its frame
            return

        # Update connector position
        for i in range(self.num_cassie):
            if self.num_cassie == 2 and i > 0:
                break

            j = (i + 1) % self.num_cassie

            connector_name = f'connector_{i + 1}_{j + 1}'

            mid = (local_base_position[i] + local_base_position[j]) / 2.
            rotation = np.arctan2(local_base_position[j][1] - local_base_position[i][1],
                                  local_base_position[j][0] - local_base_position[i][0])

            # Change size of connector
            self.sim.model.geom(connector_name).size[0] = np.linalg.norm(
                local_base_position[j] - local_base_position[i]) / 2.

            # Change rotation of connector
            self.sim.model.body(connector_name).quat = euler2quat(z=rotation.reshape(-1), y=0, x=0)[0]

            # Change position of connector
            self.sim.model.body(connector_name).pos[:-1] = mid

            # Recompute inertia of connector
            self.recompute_inertia(connector_name)

    def recompute_inertia(self, connector_name):
        """Recompute inertia of the connector. Might need to update other properties as well."""

        # Get size of connector
        size = self.sim.model.geom(connector_name).size

        # Get mass of connector
        mass = self.sim.get_body_mass(connector_name)[0]

        # Compute inertia - only for box shape
        self.sim.model.body(connector_name).inertia[0] = mass * (size[1] ** 2 + size[2] ** 2) / 3
        self.sim.model.body(connector_name).inertia[1] = mass * (size[0] ** 2 + size[2] ** 2) / 3
        self.sim.model.body(connector_name).inertia[2] = mass * (size[0] ** 2 + size[1] ** 2) / 3

    def reset(self, interactive_evaluation=False):
        self.orient_add = np.random.uniform(-self.poi_heading_range, self.poi_heading_range)
        self.orient_add = np.repeat(self.orient_add, self.num_cassie, axis=0)

        self._randomize_cassie_position(noise=self.position_offset)

        self.reset_simulation()
        self.randomize_commands()
        self.randomize_height()
        if self.custom_terrain is None:
            self.update_payload_mass()
            self.sim.reset()
        self.randomize_commands_at = np.random.randint(*self._randomize_commands_bounds)
        self.randomize_force_at = np.random.randint(*self._randomize_force_bounds,
                                                    size=len(self.target_pertub_bodies))
        self.randomize_torque_at = np.random.randint(*self._randomize_force_bounds,
                                                     size=len(self.target_pertub_bodies))
        self.randomize_height_at = np.random.randint(*self._randomize_height_bounds)

        # Update clock
        if self.clock_type:
            self.randomize_clock(init=True)
            if self.clock_type == "von_mises":
                for i in range(self.num_cassie):
                    self.clock[i].precompute_von_mises()
                # self.clock.precompute_von_mises()
        else:
            self.feet_air_time = np.zeros((self.num_cassie, 2))

        # Interactive control/evaluation
        self.interactive_evaluation = interactive_evaluation

        # Reset env counter variables
        self.curr_step = 0
        self.last_action = None
        self.cop = None

        return self.get_state()

    def randomize_clock(self, init=False):
        phase_add = 1 / self.default_policy_rate
        if init:
            self.clock = []
            for i in range(self.num_cassie):
                swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
                swing_ratios = [swing_ratio, swing_ratio]
                if np.random.random() < 0.3:  # 50% chance of rand shifts
                    period_shifts = [0 + np.random.uniform(*self._period_shift_bounds),
                                     0.5 + np.random.uniform(*self._period_shift_bounds)]
                else:
                    period_shifts = [0, 0.5]
                cycle_time = np.random.uniform(*self._cycle_time_bounds)
                self.clock.append(PeriodicClock(cycle_time, phase_add, swing_ratios, period_shifts))
        else:
            for i in range(self.num_cassie):
                swing_ratio = np.random.uniform(*self._swing_ratio_bounds)
                self.clock[i].set_swing_ratios([swing_ratio, swing_ratio])
                if np.random.randint(3) == 0:  # 1/3 chance of hopping
                    period_shifts = [0, 0]
                else:
                    period_shifts = [0, np.random.uniform(*self._period_shift_bounds)]
                self.clock[i].set_period_shifts(period_shifts)
                cycle_time = np.random.uniform(*self._cycle_time_bounds)
                self.clock[i].set_cycle_time(cycle_time)

    def update_payload_mass(self):
        for i in range(self.num_cassie):
            if self.num_cassie == 2 and i > 0:
                break

            j = (i + 1) % self.num_cassie
            connector_name = f'connector_{i + 1}_{j + 1}'

            self._connector_mass = np.random.uniform(*self._randomize_payload_mass_bounds)

            self.sim.set_body_mass(self._connector_mass, connector_name)

            # Recompute inertia manually
            self.recompute_inertia(connector_name)

    def step(self, action: np.ndarray):

        if self._merge_states:
            action = action.reshape(self.num_cassie, -1)

        if self.dynamics_randomization:
            self.policy_rate = self.default_policy_rate + np.random.randint(0, 6)
        else:
            self.policy_rate = self.default_policy_rate

        if self.perturbation_force and self.force_prob:
            for i in range(len(self.target_pertub_bodies)):
                if self.curr_step % self.randomize_force_at[i] == 0:

                    if np.random.random() < self.force_prob:
                        magnitude = np.random.uniform(0.0, self.perturbation_force)

                        theta = np.random.uniform(-np.pi, np.pi)  # polar angle

                        z_force_limit = 5.0

                        # Compute azimuthal delta for limited force in upward direction
                        azimuth_delta = np.arccos(np.minimum(1, z_force_limit / magnitude))

                        phi = np.random.uniform(azimuth_delta, np.pi)  # azimuthal angle

                        x = magnitude * np.sin(phi) * np.cos(theta)
                        y = magnitude * np.sin(phi) * np.sin(theta)
                        z = magnitude * np.cos(phi)

                        self.force_vector[i] = x, y, z
                    else:
                        self.force_vector[i] = 0.0, 0.0, 0.0

                    self.sim.data.xfrc_applied[self.target_pertub_bodies[i], :3] = self.force_vector[i]

                    if np.random.random() < self.force_prob:
                        # Apply half the linear force as torque
                        magnitude = np.random.uniform(0.0, self.perturbation_force / 2.0)

                        theta = np.random.uniform(-np.pi, np.pi)  # polar angle

                        phi = np.random.uniform(0, np.pi)  # azimuthal angle

                        x = magnitude * np.sin(phi) * np.cos(theta)
                        y = magnitude * np.sin(phi) * np.sin(theta)
                        z = magnitude * np.cos(phi)

                        self.torque_vector[i] = x, y, z
                    else:
                        self.torque_vector[i] = 0.0, 0.0, 0.0

                    self.sim.data.xfrc_applied[self.target_pertub_bodies[i], 3:] = self.torque_vector[i]
        #
        # self.sim.data.xfrc_applied[self.sim.get_body_adr(f'cassie-pelvis'),:3] = -self.sim.data.cfrc_int[self.sim.get_body_adr(f'cassie-pelvis'),:3]
        #
        # print("self.sim.data.qfrc_smooth",self.sim.data.qfrc_smooth[self.sim.get_body_adr(f'cassie-pelvis')])
        # print("self.sim.data.qfrc_constraint",self.sim.data.qfrc_constraint[self.sim.get_body_adr(f'cassie-pelvis')])
        # print("cfrc_int",self.sim.data.cfrc_int[self.sim.get_body_adr(f'cassie-pelvis')])

        # Offset global zero heading by turn rate per policy step
        self.orient_add += self.turn_rate_poi / self.default_policy_rate

        # Step simulation by n steps. This call will update self.tracker_fn.
        simulator_repeat_steps = int(self.sim.simulator_rate / self.policy_rate)
        self.step_simulation(action, simulator_repeat_steps, integral_action=self.integral_action)

        # Update CoP marker
        if self.sim.viewer is not None:
            if self.cop_marker_id is None:
                so3 = euler2so3(z=0, x=0, y=0)
                self.cop_marker_id = self.sim.viewer.add_marker("sphere", "", [0, 0, 0], [0.03, 0.03, 0.03],
                                                                [0.99, 0.1, 0.1, 1.0], so3)
            if self.cop is not None:
                cop_pos = np.concatenate([self.cop, [0]])
                self.sim.viewer.update_marker_position(self.cop_marker_id, cop_pos)

        # Reward for taking current action before changing quantities for new state
        r, r_raw = self.compute_reward(action)

        rewards = np.array([sum(kv.values()) for kv in r])

        if self._merge_states:
            rewards = np.mean(rewards, keepdims=True)

        # Increment episode counter and update previous attributes
        self.curr_step += 1
        self.last_action = action

        # Increment clock at last for updating s'
        if self.clock_type:
            for i in range(self.num_cassie):
                self.clock[i].increment()
            # self.clock.increment()

        # Randomize commands
        if self.curr_step % self.randomize_commands_at == 0 and not self.interactive_evaluation:
            self.randomize_commands()
            # self.randomize_commands_poi()
            if self.clock_type:
                if self.full_gait:
                    self.randomize_clock()

        if self.curr_step % self.randomize_height_at == 0 and not self.interactive_evaluation:
            self.randomize_height()

        if self.eval_:
            self._update_markers()

        state = self.get_state()

        done, done_info = self.compute_done()

        if self._merge_states:
            done = np.any(done, keepdims=True)

        if self.eval_:
            info = {'reward': r, 'reward_raw': r_raw, 'done_info': done_info}
        else:
            # Save memory during training
            info = {}

        return state, rewards, done, info

    # def _get_force_euler(self, force_vector):
    #     Fx, Fy, Fz = force_vector
    #
    #     # Yaw (rotation about the vertical axis)
    #     yaw = math.atan2(Fy, Fx)
    #
    #     # Pitch (rotation about the lateral axis)
    #     pitch = math.atan2(-Fz, math.sqrt(Fx ** 2 + Fy ** 2))
    #
    #     # Roll (rotation about the longitudinal axis)
    #     Fxy = math.sqrt(Fx ** 2 + Fy ** 2)
    #     roll = math.atan2(Fxy, Fz)
    #
    #     print('roll:', np.degrees(roll), 'yaw:',np.degrees(yaw), 'pitch:', np.degrees(pitch))
    #
    #
    #
    #     return roll, yaw, pitch
    def _calculate_rotation_matrix(self, v):
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm

        # Normalize the direction vector
        v = normalize(v)

        # Create an orthogonal basis
        # Assume the object's original forward direction is along the z-axis (0, 0, 1)
        forward = np.array([0, 0, 1])

        if np.allclose(v, forward):
            # If the vector is already aligned with the forward direction, use identity matrix
            return np.eye(3)

        # Find a vector orthogonal to v for constructing the rotation matrix
        right = np.cross(forward, v)
        right = normalize(right)

        # The third vector in the basis (orthogonal to both v and right)
        up = np.cross(v, right)
        up = normalize(up)

        # Construct the rotation matrix
        rotation_matrix = np.column_stack((right, up, v))

        return rotation_matrix

    def _update_markers(self, show_text=False):
        if not hasattr(self.sim, 'viewer') or self.sim.viewer is None:
            return

        poi_position = self.get_poi_position()
        poi_orientation = self.get_poi_orientation()
        poi_orientation_euler = quaternion2euler(np.array([poi_orientation]))[0]
        poi_linear_velocity = self.get_poi_linear_velocity(local_frame=True)
        base_positions = self.get_base_position()
        base_orientations = self.get_base_orientation()
        # base_orientations = self.rotate_to_heading(base_orientations)

        # poi_yaw = quaternion2euler(np.array(self.get_poi_orientation()).reshape(1, -1))[0, -1]
        # base_orientations = self.rotate_to_heading(base_orientations, yaw_offset=np.array([-poi_yaw] * self.num_cassie))

        base_orientations_euler = quaternion2euler(base_orientations)
        base_linear_velocity = self.get_base_linear_velocity()

        for i, body_name in enumerate(self.target_pertub_bodies):
            position = self.sim.get_body_pose(self.target_pertub_bodies[i])[:3]

            magnitude = np.linalg.norm(self.force_vector[i])
            magnitude_norm = magnitude / 50.0
            so3 = self._calculate_rotation_matrix(self.force_vector[i])

            position = [position[0], position[1], position[2] + 0.4]

            if show_text:
                label = f"Perturbation force (linear): {magnitude:.2f}"
            else:
                label = ''

            if f'force_vectors_{i}' not in self.vis_marker_keys:
                self.vis_marker_keys[f'force_vectors_{i}'] = self.sim.viewer.add_marker(
                    geom_type="arrow",
                    name=label,
                    position=position,
                    size=[0.01, 0.01, magnitude_norm],
                    rgba=[1.0, 1.0, 1.0, 0.8],
                    so3=so3)

            else:
                self.sim.viewer.update_marker_position(self.vis_marker_keys[f'force_vectors_{i}'], position)
                self.sim.viewer.update_marker_so3(self.vis_marker_keys[f'force_vectors_{i}'], so3)
                self.sim.viewer.update_marker_size(self.vis_marker_keys[f'force_vectors_{i}'],
                                                   [0.01, 0.01, magnitude_norm])
                self.sim.viewer_update_marker_name(self.vis_marker_keys[f'force_vectors_{i}'], label)

        marker_position = [poi_position[0], poi_position[1], poi_position[2]]

        if f'poi_velocity_actual' not in self.vis_marker_keys:
            self.vis_marker_keys['poi_velocity_actual'] = self.sim.viewer.add_marker(
                geom_type="arrow",
                name="",
                position=marker_position,
                size=[0.01, 0.01, 0.5],
                rgba=[0.93333333, 0.29411765, 0.16862745, 1.0],
                so3=euler2so3(z=poi_orientation_euler[-1], x=0, y=np.pi / 2))

        for k, v in self.vis_marker_keys.items():
            if k == 'poi_velocity_actual':
                self.sim.viewer.update_marker_position(v, marker_position)
                self.sim.viewer.update_marker_so3(v, euler2so3(z=poi_orientation_euler[-1], x=0, y=np.pi / 2))
                # q = poi_orientation
                # quat = R.from_quat([*q[1:4], q[0]])
                # poi_linear_velocity = quat.apply(poi_linear_velocity, inverse=True)
                label = ''
                self.sim.viewer_update_marker_name(v, label)

        for i in range(self.num_cassie):
            if f'poi_velocity_{i}' not in self.vis_marker_keys:
                self.vis_marker_keys[f'poi_velocity_{i}'] = self.sim.viewer.add_marker(
                    geom_type="arrow",
                    name="",
                    position=marker_position,
                    size=[0.01, 0.01, 0.8],
                    # rgba=[0.93333333, 0.29411765, 0.16862745, 0.2],
                    rgba=[0.0, 1.0, 0.0, 0.6],
                    so3=euler2so3(z=self.orient_add[i], x=0, y=np.pi / 2))

            for k, v in self.vis_marker_keys.items():
                if k == f'poi_velocity_{i}':
                    self.sim.viewer.update_marker_position(v, marker_position)
                    self.sim.viewer.update_marker_so3(v, euler2so3(z=self.orient_add[i], x=0, y=np.pi / 2))

                    if show_text:
                        label = "POI | "
                        label += f"x: {self.x_velocity_poi[i]:.2f}vs{poi_linear_velocity[0]:.2f} | "
                        label += f"y: {self.y_velocity_poi[i]:.2f}vs{poi_linear_velocity[1]:.2f} | "
                        label += f"turn: {self.turn_rate_poi[i]:.2f}"
                        if hasattr(self, '_connector_mass'):
                            label += f" | connector-mass: {self._connector_mass:.2f}"
                    else:
                        label = ''

                    self.sim.viewer_update_marker_name(v, label)

        for i in range(self.num_cassie):
            if self.sim.viewer is not None:
                marker_position = [base_positions[i, 0], base_positions[i, 1], base_positions[i, 2] + 0.3]
                if f'velocity_{i}' not in self.vis_marker_keys:
                    self.vis_marker_keys[f'velocity_{i}'] = self.sim.viewer.add_marker(
                        geom_type="arrow",
                        name="",
                        position=marker_position,
                        size=[0.01, 0.01, 0.3],
                        rgba=[0.9, 0.5, 0.1, 1.0],
                        so3=euler2so3(z=base_orientations_euler[i, -1], x=0, y=np.pi / 2))

                for k, v in self.vis_marker_keys.items():
                    if k == f'velocity_{i}':
                        self.sim.viewer.update_marker_position(v, marker_position)
                        self.sim.viewer.update_marker_so3(v,
                                                          euler2so3(z=base_orientations_euler[i, -1], x=0, y=np.pi / 2))
                        # q = base_orientations[i]
                        # quat = R.from_quat([*q[1:4], q[0]])
                        # vel = quat.apply(self.get_base_linear_velocity()[i], inverse=True)
                        # label = f"({base_positions[i, 0]:.2f}, {base_positions[i, 1]:.2f})"
                        # label += f"dx: {vel[0]:.2f} | "
                        # label += f"dy: {vel[1]:.2f}"
                        if show_text:
                            label = f'Cassie {i} | x: {base_linear_velocity[i, 0]:.2f} | y: {base_linear_velocity[i, 1]:.2f}'
                            if hasattr(self, 'height_base'):
                                label += f" | height: {self.height_base[i]:.2f}vs{base_positions[i, -1]:.2f}"
                        else:
                            label = ''
                        self.sim.viewer_update_marker_name(v, label)

        # Visualize Center of Mass
        if self.com_vis and 'com' not in self.vis_marker_keys:
            for i, com in enumerate(self.sim.data.subtree_com):
                self.vis_marker_keys['com'] = self.sim.viewer.add_marker(
                    geom_type="sphere",
                    name=f'{i}:{mj.mj_id2name(self.sim.model, mj.mjtObj.mjOBJ_BODY, i)}',
                    position=com,
                    size=[0.05, 0.05, 0.05],
                    rgba=[1.0, 1.0, 1.0, 1.0],
                    so3=euler2so3(z=0, x=0, y=0))

        if self.com_vis and 'com' in self.vis_marker_keys:
            for i, com in enumerate(self.sim.data.subtree_com):
                self.sim.viewer.update_marker_position(self.vis_marker_keys['com'], com)

            # self.sim.viewer.update_marker_position(v, marker_position)
            # self.sim.viewer.update_marker_so3(v, euler2so3(z=poi_orientation_euler[-1], x=0, y=np.pi / 2))
            # # q = poi_orientation
            # # quat = R.from_quat([*q[1:4], q[0]])
            # # poi_linear_velocity = quat.apply(poi_linear_velocity, inverse=True)
            # label = f"x: {self.x_velocity_poi:.2f}vs{poi_linear_velocity[0]:.2f} | "
            # label += f"y: {self.y_velocity_poi:.2f}vs{poi_linear_velocity[1]:.2f} | "
            # label += f"turn: {self.turn_rate_poi:.2f} | "
            # self.sim.viewer_update_marker_name(v, label)

    def _compute_encoding(self, poi_position, poi_orientation, base_positions):
        # each cassie is represented by tuple of length 2.
        # (<distance_from_poi>, <angle_with_poi>)

        r = np.linalg.norm(poi_position - base_positions, axis=1)
        theta = np.arctan2(poi_position[1] - base_positions[:, 1],
                           poi_position[0] - base_positions[:, 0]) - poi_orientation

        # Normalize theta to be between -pi and pi
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self.encoding = np.stack((r, theta), axis=1)

        # self.encoding.fill(0)

    def _update_poi_position(self, poi_position, poi_orientation):
        # This line flags to change poi position dynamically
        self.sim.model.site_sameframe[0] = 0

        self.sim.model.site('poi_site').pos[:2] = poi_position

        poi_euler = quaternion2euler(np.array(self.get_poi_orientation()).reshape(1, -1))

        poi_euler[:, -1] = poi_orientation

        self.sim.model.site('poi_site').quat = euler2quat(x=0, y=0, z=poi_euler[:, 2])[0]

    def randomize_commands(self):
        # # temporary override [REMOVE THIS LATER]
        # self.x_velocity_poi = 0
        # self.y_velocity_poi = 0
        # self.turn_rate_poi = 0
        # return

        choices = OrderedDict(
            in_place=0.16,
            in_place_turn=0.16,
            walk=0.2,
            walk_sideways=0.16,
            stand=0.16,
            walk_turn=0.16
        )

        if self.reward_name == "locomotion_cassiepede_clock_stand":
            self.stand = False
        else:
            choices['stand'] = 0.0

        # Normalize
        total = sum(choices.values())
        for k in choices:
            choices[k] /= total

        mode = np.random.choice(list(choices.keys()), p=list(choices.values()))

        match mode:
            case 'in_place':
                self.x_velocity_poi = 0
                self.y_velocity_poi = 0
                self.turn_rate_poi = 0
            case 'in_place_turn':
                self.x_velocity_poi = 0
                self.y_velocity_poi = 0
                self.turn_rate_poi = np.random.uniform(*self._turn_rate_bounds)
            case 'walk':
                self.x_velocity_poi = np.random.uniform(*self._x_velocity_bounds)
                self.y_velocity_poi = np.random.uniform(*self._y_velocity_bounds)
                self.turn_rate_poi = 0
            case 'walk_sideways':
                self.x_velocity_poi = 0
                self.y_velocity_poi = np.random.uniform(*self._y_velocity_bounds)
                self.turn_rate_poi = 0
            case 'stand':
                self.stand = True
                self.x_velocity_poi = 0
                self.y_velocity_poi = 0
                self.turn_rate_poi = 0
            case 'walk_turn':
                self.x_velocity_poi = np.random.uniform(*self._x_velocity_bounds)
                self.y_velocity_poi = np.random.uniform(*self._y_velocity_bounds)
                self.turn_rate_poi = np.random.uniform(*self._turn_rate_bounds)
            case _:
                raise ValueError(f"Invalid mode {mode}")

        # Add offset to cmd for each cassie, so they depend less on each other
        if np.random.random() < self.cmd_noise_prob:
            self.x_velocity_poi += np.random.uniform(-self.cmd_noise[0], self.cmd_noise[0], self.num_cassie)
            self.y_velocity_poi += np.random.uniform(-self.cmd_noise[1], self.cmd_noise[1], self.num_cassie)
            self.turn_rate_poi += np.random.uniform(-np.radians(self.cmd_noise[2]), np.radians(self.cmd_noise[2]),
                                                    self.num_cassie)
        else:
            self.x_velocity_poi = np.array([self.x_velocity_poi]).repeat(self.num_cassie)
            self.y_velocity_poi = np.array([self.y_velocity_poi]).repeat(self.num_cassie)
            self.turn_rate_poi = np.array([self.turn_rate_poi]).repeat(self.num_cassie)

        # Clip to avoid small commands
        self.x_velocity_poi[np.abs(self.x_velocity_poi) <= 0.1] = 0
        self.y_velocity_poi[np.abs(self.y_velocity_poi) <= 0.1] = 0
        self.turn_rate_poi[np.abs(self.turn_rate_poi) <= 0.1] = 0

    def randomize_height(self):
        self.height_base = np.random.uniform(*self._height_bounds)
        self.height_base = np.array([self.height_base]).repeat(self.num_cassie)

    def get_base_position(self):
        data = []
        for i in range(self.num_cassie):
            i = '' if i == 0 else f'c{i + 1}_'
            data.append(self.sim.data.sensor(f'{i}pelvis-position').data)
            # data.append(self.sim.data.body(f'{i}cassie-pelvis').xpos)
        return np.stack(data)

    def get_base_orientation(self):
        data = []
        for i in range(self.num_cassie):
            i = '' if i == 0 else f'c{i + 1}_'
            data.append(self.sim.data.sensor(f'{i}pelvis-orientation').data)
        return np.stack(data)

    def get_base_angular_velocity(self):
        data = []
        for i in range(self.num_cassie):
            i = '' if i == 0 else f'c{i + 1}_'
            data.append(self.sim.data.sensor(f'{i}pelvis-angular-velocity').data)

        return np.stack(data)

    def get_base_linear_velocity(self):
        data = []
        for i in range(self.num_cassie):
            i = '' if i == 0 else f'c{i + 1}_'
            data.append(self.sim.data.sensor(f'{i}pelvis-linear-velocity').data)

        return np.stack(data)

    def get_base_acceleration(self):
        data = []
        for i in range(self.num_cassie):
            i = '' if i == 0 else f'c{i + 1}_'
            data.append(self.sim.data.sensor(f'{i}pelvis-linear-acceleration').data)

        return np.stack(data)

    def get_base_force(self):
        data = []
        for i in range(self.num_cassie):
            i = '' if i == 0 else f'c{i + 1}_'
            data.append(self.sim.data.sensor(f'{i}pelvis-force').data)

        return np.stack(data)

    def get_poi_position(self):
        return self.sim.data.sensor('poi_position').data

    def get_poi_orientation(self):
        return self.sim.data.sensor('poi_orientation').data

    def get_poi_linear_velocity(self, local_frame=False):
        # This gives linear velocity of poi in local frame
        # return self.sim.data.sensor('poi_linear_velocity').data

        # This gives linear velocity of poi in global frame
        return self.sim.get_body_velocity('poi_site', type=mj.mjtObj.mjOBJ_SITE, local_frame=local_frame)

    def get_poi_angular_velocity(self):
        return self.sim.data.sensor('poi_angular_velocity').data

    def get_poi_linear_acceleration(self):
        return self.sim.data.sensor('poi_linear_acceleration').data

    def get_encoder_yaw(self, poi_yaw=None):

        if self.simulator_type == "mujoco":

            # Global orientation of poi
            poi_yaw = quaternion2euler(np.array(self.get_poi_orientation()).reshape(1, -1))[0, [-1]]

            base_orient = self.get_base_orientation()
        else:
            if poi_yaw is None:
                return np.zeros((self.num_cassie, 1))

            poi_yaw = np.array([poi_yaw])

            base_orient = np.array(self.sim.get_base_orientation(state_est=self.state_est)).reshape(self.num_cassie, -1)

        # Relative yaw of cassie w.r.t to yaw given by encoder attached at yaw hinge
        base_yaw_poi = quaternion2euler(self.rotate_to_heading(base_orient, yaw_offset=poi_yaw))[:, [-1]]

        # base_yaw_poi.fill(0)
        return base_yaw_poi

    def get_base_pose(self):
        """Get standard robot prorioceptive states. Sub-env can override this function to define its
        own get_robot_state().

        Returns:
            robot_state (np.ndarray): robot state
        """
        if self.simulator_type == "libcassie" and self.state_est:
            base_orient = np.array(self.sim.get_base_orientation(state_est=self.state_est)).reshape(self.num_cassie, -1)
            base_ang_vel = np.array(self.sim.get_base_angular_velocity(state_est=self.state_est)).reshape(
                self.num_cassie, -1)
        else:
            base_orient = self.get_base_orientation()
            base_ang_vel = self.get_base_angular_velocity()

        # Base orientation in command frame
        base_orient_cmd = self.rotate_to_heading(base_orient)

        # Relative yaw of cassie w.r.t to yaw given by encoder attached at yaw hinge
        base_yaw_poi = self.get_encoder_yaw()

        # Apply noise to proprioceptive states per step
        if isinstance(self.state_noise, list):
            orig_euler_cmd = quaternion2euler(base_orient_cmd)
            noise_euler_cmd = orig_euler_cmd + np.random.normal(0, self.state_noise[0], size=(self.num_cassie, 3))
            noise_quat_cmd = euler2quat(x=noise_euler_cmd[:, 0], y=noise_euler_cmd[:, 1], z=noise_euler_cmd[:, 2])
            base_orient_cmd = noise_quat_cmd

            base_ang_vel = base_ang_vel + np.random.normal(0, self.state_noise[1], size=3)

            base_yaw_poi += np.random.normal(0, self.state_noise[6], size=(self.num_cassie, 1))

        # print('base orient cmd ',np.degrees( quaternion2euler(base_orient_cmd)))
        return OrderedDict(base_orient_cmd=base_orient_cmd,
                           base_yaw_poi=base_yaw_poi,
                           base_ang_vel=base_ang_vel)

    def get_motor_state(self):
        """Get standard robot prorioceptive states. Sub-env can override this function to define its
            own get_robot_state().

            Returns:
                robot_state (np.ndarray): robot state
            """
        if self.simulator_type == "libcassie" and self.state_est:
            motor_pos = np.array(self.sim.get_motor_position(state_est=self.state_est)).reshape(self.num_cassie, -1)
            motor_vel = np.array(self.sim.get_motor_velocity(state_est=self.state_est)).reshape(self.num_cassie, -1)
            joint_pos = np.array(self.sim.get_joint_position(state_est=self.state_est)).reshape(self.num_cassie, -1)
            joint_vel = np.array(self.sim.get_joint_velocity(state_est=self.state_est)).reshape(self.num_cassie, -1)
        else:
            motor_pos = self.sim.get_motor_position()
            motor_vel = self.sim.get_motor_velocity()
            joint_pos = self.sim.get_joint_position()
            joint_vel = self.sim.get_joint_velocity()

        # Add noise to motor and joint encoders per episode
        if self.dynamics_randomization:
            motor_pos += self.motor_encoder_noise
            joint_pos += self.joint_encoder_noise

        # Apply noise to proprioceptive states per step
        motor_vel += np.random.normal(0, self.velocity_noise, size=self.sim.num_actuators)
        joint_vel += np.random.normal(0, self.velocity_noise, size=self.sim.num_joints)
        if isinstance(self.state_noise, list):
            motor_pos = motor_pos + np.random.normal(0, self.state_noise[2], size=self.sim.num_actuators)
            motor_vel = motor_vel + np.random.normal(0, self.state_noise[3], size=self.sim.num_actuators)
            joint_pos = joint_pos + np.random.normal(0, self.state_noise[4], size=self.sim.num_joints)
            joint_vel = joint_vel + np.random.normal(0, self.state_noise[5], size=self.sim.num_joints)
        else:
            pass
            # raise NotImplementedError("state_noise must be a list of 6 elements")

        # Since tarsus encoder is broken in cassie, we mask and training it
        if self.mask_tarsus_input:
            left_tarsus_pos_idx = 1
            left_tarsus_vel_idx = 1

            joint_pos[:, left_tarsus_pos_idx] = 0.0
            joint_vel[:, left_tarsus_vel_idx] = 0.0

        # print('joint_pos:', joint_pos, 'motor_pos:', motor_pos)

        return OrderedDict(motor_pos=motor_pos, motor_vel=motor_vel, joint_pos=joint_pos, joint_vel=joint_vel)

    def _get_command_poi(self):
        if self._height_control:
            # stand = (self.x_velocity_poi == 0) & (self.y_velocity_poi == 0) & (self.turn_rate_poi == 0)
            cmd = np.stack((self.x_velocity_poi, self.y_velocity_poi, self.turn_rate_poi, self.height_base), axis=1)
        else:
            cmd = np.stack((self.x_velocity_poi, self.y_velocity_poi, self.turn_rate_poi), axis=1)

        return cmd

    def compute_state_dict(self):
        if self.clock_type:
            input_clock = []
            for i in range(self.num_cassie):
                if self.full_clock:
                    input_clock.append(self.clock[i].input_full_clock())
                else:
                    input_clock.append(self.clock[i].input_clock())

        base_pose = self.get_base_pose()

        motor_state = self.get_motor_state()

        cmd = self._get_command_poi()

        if isinstance(self.state_noise, list):
            encoding_noise = np.concatenate([np.random.normal(0, self.state_noise[7], size=(self.num_cassie, 1)),
                                             np.random.normal(0, self.state_noise[8], size=(self.num_cassie, 1))],
                                            axis=-1)
        else:
            encoding_noise = 0

        if self.clock_type:
            clocks = []
            for i in range(self.num_cassie):
                clocks.append([self.clock[i].get_swing_ratios()[0], 1 - self.clock[i].get_swing_ratios()[0],
                               *self.clock[i].get_period_shifts(), *input_clock[i]])

            clocks = np.stack(clocks, axis=0)

            if self.reward_name == "locomotion_cassiepede_clock_stand":

                if self.stand:
                    # Mask the clock values if stand is True
                    clocks.fill(0.0)

                stand = np.array([self.stand]).reshape(1, -1).repeat(self.num_cassie, axis=0)

                self.state_dict = OrderedDict(**base_pose,
                                              **motor_state,
                                              stand=stand,
                                              cmd=cmd,
                                              encoding=self.encoding + encoding_noise,
                                              clock=clocks)
            else:
                self.state_dict = OrderedDict(
                    **base_pose,
                    **motor_state,
                    cmd=cmd,
                    encoding=self.encoding + encoding_noise,
                    clock=clocks
                )
            if not self.clock_in_state:
                self.state_dict.pop('clock')
        else:
            self.state_dict = OrderedDict(
                **base_pose,
                **motor_state,
                # stand=np.all(cmd == 0).astype(int).reshape(self.num_cassie, -1),
                cmd=cmd,
                encoding=self.encoding + encoding_noise,
            )

    def get_state(self):
        self.compute_state_dict()
        # print('state_dict:', dict([(k,v.shape[1]) for k,v in self.state_dict.items()]))
        # state = np.concatenate(list(self.state_dict.values()), axis=-1)
        # return dict(base=state, privilege=state)

        if self._merge_states:
            state_dict = {}
            for k in self.state_dict.keys():
                state_dict[k] = self.state_dict[k].reshape(1, -1)
            return state_dict

        return self.state_dict
