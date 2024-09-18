import argparse
from collections import OrderedDict
from typing import Optional
import mujoco as mj
import numpy as np
import torch

from algo.common.network import Actor_LSTM_v2
from env.cassie.cassiepede.cassiepede import Cassiepede
from env.util.quaternion import *
import copy

class CassiepedeHL:

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
                 depth_input=False,
                 offscreen=False):

        self.base_env = Cassiepede(clock_type=clock_type,
                                   reward_name=reward_name,
                                   simulator_type=simulator_type,
                                   policy_rate=policy_rate,
                                   dynamics_randomization=dynamics_randomization,
                                   state_noise=state_noise,
                                   velocity_noise=velocity_noise,
                                   state_est=state_est,
                                   full_clock=full_clock,
                                   full_gait=full_gait,
                                   integral_action=integral_action,
                                   num_cassie=num_cassie,
                                   position_offset=position_offset,
                                   poi_heading_range=poi_heading_range,
                                   poi_position_offset=poi_position_offset,
                                   perturbation_force=perturbation_force,
                                   force_prob=force_prob,
                                   cmd_noise=cmd_noise,
                                   cmd_noise_prob=cmd_noise_prob,
                                   mask_tarsus_input=mask_tarsus_input,
                                   com_vis=com_vis,
                                   custom_terrain=custom_terrain,
                                   depth_input=depth_input,
                                   offscreen=offscreen)

        self.base_actor = self._load_base_env()

        self.base_env._update_markers = self._update_markers

        # self.observation_size = 42
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
        self.action_size = 3

    def _load_base_env(self):
        args = argparse.Namespace(
            state_dim=OrderedDict(
                base_orient_cmd=(4,),
                base_yaw_poi=(1,),
                base_ang_vel=(3,),
                motor_pos=(10,),
                motor_vel=(10,),
                joint_pos=(4,),
                joint_vel=(4,),
                cmd=(3 + self._height_control,),
                encoding=(2,)
            ),
            lstm_num_layers=2,
            lstm_hidden_dim=64,
            action_dim=10,
            std=0.13,
            use_orthogonal_init=True,
        )

        model = Actor_LSTM_v2(args)

        checkpoint = 'checkpoints/checkpoint-2024-04-13 21:56:36.012260.pt'

        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))['actor_state_dict']

        # # Backward compatibility
        # checkpoint['mean_layer.weight'] = checkpoint['means.weight']
        # checkpoint['mean_layer.bias'] = checkpoint['means.bias']
        # checkpoint.pop('means.weight')
        # checkpoint.pop('means.bias')

        model.load_state_dict(checkpoint, strict=True)

        return model

    # Access base_env methods if not overridden
    def __getattr__(self, item):
        return getattr(self.base_env, item)

    # Write base_env methods if not overridden
    def __setattr__(self, key, value):
        if key == 'base_env':
            super().__setattr__(key, value)
        else:
            setattr(self.base_env, key, value)

    def _get_base_orientation(self, offset):
        base_orient_cmd = self.rotate_to_heading(self.get_base_orientation(), offset)

        if isinstance(self.state_noise, list):
            orig_euler_cmd = quaternion2euler(base_orient_cmd)
            noise_euler_cmd = orig_euler_cmd + np.random.normal(0, self.state_noise[0], size=(self.num_cassie, 3))
            noise_quat_cmd = euler2quat(x=noise_euler_cmd[:, 0], y=noise_euler_cmd[:, 1], z=noise_euler_cmd[:, 2])
            base_orient_cmd = noise_quat_cmd

        return base_orient_cmd

    def step(self, action):

        # action.fill(0.0)
        # command: It is the command produced by high level policy for each cassie, [num_cassie, 3]

        self._base_cmd = action[:, :3]
        # residual_action = action[:, 3:]

        # CREATE STATE FOR LOW-LEVEL POLICY

        # # Override cmd for each cassie with command produced by high level policy
        # self.state_dict['cmd'][:, :3] = self._base_cmd

        # Add residual cmd to each cassie with command produced by high level policy
        self.state_dict['cmd'][:, :3] += self._base_cmd

        # Update orient offset based on turn rate (produced by high-level policy) for each cassie
        self.orient_add_bases += self.state_dict['cmd'][:, :3][:, -1] / self.default_policy_rate

        # Compute base orientation in each cassie's commanded frame
        self.state_dict['base_orient_cmd'] = self._get_base_orientation(offset=self.orient_add_bases)

        # This base policy doesn't take clock as input
        clock = self.state_dict.pop('clock', None)

        # Combine state for low level policy to evaluate
        # base_state = torch.tensor(np.concatenate(list(self.state_dict.values()), axis=-1),
        #                           dtype=torch.float32).unsqueeze(1)

        base_state = OrderedDict()
        for k, v in self.state_dict.items():
            base_state[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(1)

        with torch.inference_mode():
            # Get action from low level policy
            base_action, _ = self.base_actor(base_state)
            base_action = base_action.squeeze(1).numpy()

        # Step low level policy. This update the state_dict as well
        _, reward, done, info = self.base_env.step(base_action)  # + residual_action)

        # CREATE STATE FOR HIGH-LEVEL POLICY

        # Option 1
        OPTION = 2

        match OPTION:
            case 1:
                # Option 1
                state = OrderedDict(
                    cmd=self._get_command_poi(),
                    encoding=self.state_dict['encoding'].copy(),
                    base_yaw_poi=self.state_dict['base_yaw_poi'].copy()
                )
            case 2:
                # Option 2

                # High-level state has command in POI frame, not in each cassie's frame
                self.state_dict['cmd'] = self._get_command_poi()

                # Put back clock for high-level policy in input state if it exists
                if clock is not None:
                    self.state_dict['clock'] = clock

                # Base orientation has to be computed in POI frame
                self.state_dict['base_orient_cmd'] = self._get_base_orientation(offset=self.orient_add)

                # # Combine state for high level policy
                # state = np.concatenate(list(self.state_dict.values()), axis=-1)
                state = OrderedDict({k: v.copy() for k, v in self.state_dict.items()})
            case _:
                raise ValueError("Invalid option")

        # Remove state that is not needed for low-level policy. This is used in next call to step
        self.state_dict['encoding'].fill(0.0)
        self.state_dict['base_yaw_poi'].fill(0.0)

        # print(f"State: {state['encoding']}")

        return state, reward, done, info

    def _update_markers(self):
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

        # Offset all cassie orientation by poi orientation.

        marker_position = [poi_position[0], poi_position[1], poi_position[2] + 0.2]
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

                    label = "POI | "
                    label += f"x: {self.x_velocity_poi[i]:.2f}vs{poi_linear_velocity[0]:.2f} | "
                    label += f"y: {self.y_velocity_poi[i]:.2f}vs{poi_linear_velocity[1]:.2f} | "
                    label += f"turn: {self.turn_rate_poi[i]:.2f}"

                    self.sim.viewer_update_marker_name(v, label)

        for i in range(self.num_cassie):
            if self.sim.viewer is not None:
                marker_position = [base_positions[i, 0], base_positions[i, 1], base_positions[i, 2] + 0.3]
                if f'velocity_actual_{i}' not in self.vis_marker_keys:
                    self.vis_marker_keys[f'velocity_actual_{i}'] = self.sim.viewer.add_marker(
                        geom_type="arrow",
                        name="",
                        position=marker_position,
                        size=[0.01, 0.01, 0.3],
                        rgba=[0.9, 0.5, 0.1, 1.0],
                        so3=euler2so3(z=base_orientations_euler[i, -1], x=0, y=np.pi / 2))

                if f'velocity_{i}' not in self.vis_marker_keys:
                    self.vis_marker_keys[f'velocity_{i}'] = self.sim.viewer.add_marker(
                        geom_type="arrow",
                        name="",
                        position=marker_position,
                        size=[0.01, 0.01, 0.3],
                        rgba=[0.0, 1.0, 0.0, 0.5],
                        so3=euler2so3(z=self.orient_add_bases[i], x=0, y=np.pi / 2))

                for k, v in self.vis_marker_keys.items():
                    if k == f'velocity_actual_{i}':
                        self.sim.viewer.update_marker_position(v, marker_position)
                        self.sim.viewer.update_marker_so3(v,
                                                          euler2so3(z=base_orientations_euler[i, -1], x=0, y=np.pi / 2))
                        label = ''
                        self.sim.viewer_update_marker_name(v, label)

                    elif k == f'velocity_{i}':
                        self.sim.viewer.update_marker_position(v, marker_position)
                        self.sim.viewer.update_marker_so3(v,
                                                          euler2so3(z=self.orient_add_bases[i], x=0, y=np.pi / 2))

                        label = f'Cassie {i} | '
                        label += f'x: {self._base_cmd[i, 0]:.2f}vs{base_linear_velocity[i, 0]:.2f} | '
                        label += f'y: {self._base_cmd[i, 1]:.2f}vs{base_linear_velocity[i, 1]:.2f}'
                        label += f"turn: {self._base_cmd[i, 2]:.2f}"
                        if hasattr(self, 'height_base'):
                            label += f" | height: {self.height_base[i]:.2f}vs{base_positions[i, -1]:.2f}"
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

    def reset(self, interactive_evaluation=False):
        self.base_actor.init_hidden_state(device='cpu', batch_size=self.num_cassie)

        # Reset the base env
        self.base_env.reset(interactive_evaluation)

        # Each cassie offset is originally to POI's offset
        self.orient_add_bases = self.orient_add.copy()

        # CREATE STATE FOR HIGH-LEVEL POLICY
        OPTION = 2

        match OPTION:
            case 1:
                # Option 1
                state = OrderedDict(
                    cmd=self._get_command_poi(),
                    encoding=self.state_dict['encoding'].copy(),
                    base_yaw_poi=self.state_dict['base_yaw_poi'].copy()
                )
            case 2:
                # Option 2
                # High-level state has command in POI frame, not in each cassie's frame
                self.state_dict['cmd'] = self._get_command_poi()

                # Base orientation has to be computed in POI frame
                self.state_dict['base_orient_cmd'] = self._get_base_orientation(offset=self.orient_add)

                # # Combine state for high level policy
                # state = np.concatenate(list(self.state_dict.values()), axis=-1)
                state = self.state_dict.copy()
            case _:
                raise ValueError("Invalid option")

        # Remove state that is not needed for low-level policy. This is used in next call to step
        self.state_dict['encoding'].fill(0.0)
        self.state_dict['base_yaw_poi'].fill(0.0)

        return state
