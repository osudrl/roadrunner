import argparse
import copy
import time
from collections import OrderedDict
from multiprocessing import Manager, Process

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from algo.common.tarsus_patch_wrapper import TarsusPatchWrapper
from env.cassie.cassiepedeHL.cassiepedeHL import CassiepedeHL
from env.util.quaternion import quaternion2euler

from itertools import cycle

plt.ion()

from env.cassie.cassiepede.cassiepede import Cassiepede
from algo.common.network import *

import sys
import tty
import select
from algo.common.utils import *
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)


def plotter(data):
    min_reward, max_reward = np.inf, -np.inf
    while True:
        if 'clear' in data:
            for fig_num in plt.get_fignums():
                plt.figure(fig_num)
                plt.clf()
            data.clear()
            continue

        if 'vis_position_plot' in data:
            plt.figure(0)
            plt.clf()

            poi_x, poi_y = data['vis_position_plot']['poi_position'][:2]
            base_positions = data['vis_position_plot']['base_positions'][:, :2]

            plt.scatter(0, 0, c='g', s=500)
            plt.scatter(base_positions[:, 0] - poi_x, base_positions[:, 1] - poi_y, c='orange', s=500)
            plt.title('Global positions heatmap')
            plt.axis('equal')

        if 'vis_encoding_plot' in data:
            plt.figure(1)
            plt.clf()

            # The structure of deck and connection can be composed using following three components. We will use
            # these to try to decompose the actual structure of the deck and connection
            r, theta = data['vis_encoding_plot']['encoding'].T
            base_yaws_poi = data['vis_encoding_plot']['base_yaws_poi']
            base_orient_cmd = data['vis_encoding_plot']['base_orient_cmd']

            x, y = r * np.cos(theta), r * np.sin(theta)

            for i in range(data['vis_encoding_plot']['num_cassie']):
                # Plot a line connecting poi and each cassie
                plt.plot((0, y[i]), (0, -x[i]), '--')

                # Plot poi axis (always fixed)
                plt.quiver(0, 0, 0, 0.4,
                           color='red',
                           angles="xy",
                           scale_units='xy',
                           scale=1.)

                # Plot relative orientation with poi axis
                plt.quiver(y[i],
                           -x[i], 0.4 * np.cos(base_yaws_poi[i] + np.pi / 2),
                           0.4 * np.sin(base_yaws_poi[i] + np.pi / 2),
                           color='orange',
                           angles="xy",
                           scale_units='xy',
                           scale=1.)

                # Plot relative orientation with poi axis
                plt.quiver(0, 0, 0.4 * np.cos(base_yaws_poi[i] - base_orient_cmd[i] + np.pi / 2),
                           0.4 * np.sin(base_yaws_poi[i] - base_orient_cmd[i] + np.pi / 2),
                           color='green',
                           angles="xy",
                           scale_units='xy',
                           scale=1.)

                plt.annotate(f'Cassie {i}', (y[i], -x[i]))

            plt.scatter(0, 0, c='g')
            plt.scatter(y, -x, c='r')
            plt.title('Encoding')
            plt.axis('equal')

            # Now plot actual poi and cassie positions to verify if above decomposition is correct.

            plt.figure(2)
            plt.clf()
            poi_x, poi_y, _ = data['vis_encoding_plot']['poi_position']
            poi_yaw = quaternion2euler(np.array(data['vis_encoding_plot']['poi_orientation']).reshape(1, -1))[0, [-1]]

            base_positions = data['vis_encoding_plot']['base_positions']
            base_yaws = quaternion2euler(data['vis_encoding_plot']['base_orientations'])[:, -1]

            plt.quiver(-poi_y,
                       poi_x,
                       0.4 * np.cos(poi_yaw + np.pi / 2),
                       0.4 * np.sin(poi_yaw + np.pi / 2),
                       color='red',
                       angles="xy",
                       scale_units='xy',
                       scale=1.)

            plt.quiver(-poi_y,
                       poi_x,
                       0.4 * np.cos(data['vis_encoding_plot']['orient_add'][0] + np.pi / 2),
                       0.4 * np.sin(data['vis_encoding_plot']['orient_add'][0] + np.pi / 2),
                       color='green',
                       angles="xy",
                       scale_units='xy',
                       scale=1.)

            for i in range(data['vis_encoding_plot']['num_cassie']):
                base_y = base_positions[i, 1]
                base_x = base_positions[i, 0]

                plt.plot((-poi_y, -base_y), (poi_x, base_x), '--')
                plt.quiver(-base_y,
                           base_x,
                           0.4 * np.cos(base_yaws[i] + np.pi / 2),
                           0.4 * np.sin(base_yaws[i] + np.pi / 2),
                           color='orange',
                           angles="xy",
                           scale_units='xy',
                           scale=1.)

                plt.annotate(f'Cassie {i}', (-base_positions[i, 1], base_positions[i, 0]))
            plt.scatter(-poi_y, poi_x, c='g')
            plt.scatter(-base_positions[:, 1], base_positions[:, 0], c='r')
            plt.title('Global positions')
            plt.axis('equal')

        if 'vis_reward_dict_plot' in data:
            num_cassie = data['vis_reward_dict_plot']['num_cassie']
            reward = data['vis_reward_dict_plot']['reward']
            plt.figure(3)
            plt.clf()

            for i in range(num_cassie):
                plt.subplot(num_cassie, 1, i + 1)

                y_pos = np.arange(len(reward[i]))

                plt.barh(y_pos, reward[i].values())

                plt.yticks(y_pos, labels=reward[i].keys())

                plt.title('Cassie ' + str(i) + ' reward')

                # print('feet_air_time', reward[i]['feet_air_time'])

                min_reward = min(min_reward, min(reward[i].values()))
                max_reward = max(max_reward, max(reward[i].values()))

                plt.xlim(min_reward, max_reward)
                # plt.xlim(0, 1 / len(reward[i]))

        if 'vis_individual_reward' in data:
            reward = data['vis_individual_reward']

            plt.figure(4)
            plt.clf()

            plt.subplot(1, 1, 1)

            plt.plot(reward)

            plt.legend(['Cassie ' + str(i) for i in range(len(reward[0]))])
            # print('feet air time:', np.mean(reward), 'feet air time std:', np.std(reward))

            plt.title('Cassie reward')

        if 'vis_power_plot' in data:
            power = np.stack(data['vis_power_plot'])

            plt.figure(5)
            plt.clf()

            plt.plot(power)

            labels = [f'Cassie {i}' for i in range(power.shape[1])]

            plt.legend(labels)

            plt.title('Power')

            plt.figure(6)
            plt.clf()
            plt.barh(np.arange(power.shape[1]), np.mean(power, axis=0))
            plt.yticks(np.arange(power.shape[1]), labels=labels)
            plt.title('Total Power')

        if 'base_acc' in data:
            plt.figure(6)
            # plt.clf()
            plt.plot(data['base_acc'])
            plt.title('Base acceleration (z)')

            # print('base acc mean:', np.mean(data['base_acc']), 'base acc std:', np.std(data['base_acc']))
            # plt.pause(0.0001)

        if 'vis_motion_plot' in data:
            stable_base = np.array(data['vis_motion_plot']['base_positions'])

            # base_velocities = np.diff(base_positions, axis=0)
            #
            # base_accelerations = np.diff(base_velocities, axis=0)

            # base_jerks = np.diff(base_positions, axis=0, n=1)

            # plt.figure(5)
            # plt.clf()
            # plt.plot(base_positions[:, 0], label='x')
            # plt.title('Base positions')
            #
            # plt.figure(6)
            # plt.clf()
            # plt.plot(base_velocities[:, 0], label='x')
            # plt.title('Base velocities')
            #
            # plt.figure(7)
            # plt.clf()
            # plt.plot(base_accelerations[:, 0], label='x')
            # plt.title('Base accelerations')

            plt.figure(8)
            plt.clf()
            plt.plot(stable_base[:, -1])
            plt.grid()
            plt.title('motion')

        if 'vis_force_plot' in data:
            force = np.stack(data['vis_force_plot'])

            plt.figure(9)
            plt.clf()

            # print('force:', force[...,0].shape)

            plt.plot(force)

            labels = [f'Cassie {i}' for i in range(force.shape[1])]

            plt.legend(labels)

            plt.title('Force')

        if plt.gca().has_data():
            plt.pause(0.0001)


def apply_linear_perturbation(env, magnitude):
    if magnitude == 0:
        # Save computation
        env.force_vector[:] = 0.0, 0.0, 0.0
    else:
        theta = np.random.uniform(-np.pi, np.pi)  # polar angle

        z_force_limit = 5.0

        # Compute azimuthal delta for limited force in upward direction
        azimuth_delta = np.arccos(np.minimum(1, z_force_limit / magnitude))

        phi = np.random.uniform(azimuth_delta, np.pi)  # azimuthal angle

        x = magnitude * np.sin(phi) * np.cos(theta)
        y = magnitude * np.sin(phi) * np.sin(theta)
        z = magnitude * np.cos(phi)

        # Override env's force vector just for visualization purpose
        env.force_vector[:] = x, y, z

    for i in range(len(env.target_pertub_bodies)):
        env.sim.data.xfrc_applied[env.target_pertub_bodies[i], :3] = env.force_vector[i]


def main():
    if torch.backends.mps.is_available():
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    keyboard = True
    offscreen = False
    legacy_actor = False
    dummy_actor = False
    vis_reward_plot = False
    vis_clock_plot = False
    vis_encoding_plot = False
    vis_center_of_mass = False
    vis_reward_dict_plot = False
    vis_individual_reward = None #'joint_forces'
    vis_power_plot = False
    vis_position_plot = False
    vis_force_plot = False
    done_enable = False
    actor_model = Actor_LSTM_v2
    compute_mirror_loss = False
    single_windowed_force = False
    centralized_policy = False
    record_video = False

    custom_terrain = None
    # custom_terrain = 'cassiepede_sim2real_'
    # custom_terrain = 'cassiepede_t_shape_b'

    # custom_terrain = 'cassiepede_rectangle'
    # custom_terrain = 'cassiepede_rectangle_dynamic_'
    # custom_terrain = 'cassiepede_t_shape_b'
    # custom_terrain = 'cassiepede_config_b'
    # custom_terrain = 'cassiepede_rectangle_30kg_'
    # custom_terrain = 'cassiepede_real_exp'
    # custom_terrain = 'cassiepede_centralized'
    # custom_terrain = 'cassiepede_weight_c'
    # custom_terrain = 'cassiepede_couch_flat_weight'
    # custom_terrain = 'cassiepede_couch'
    # custom_terrain = 'cassiepede_pivot'
    # custom_terrain = 'cassiepede_fixed_poi'

    plotter_data = Manager().dict()
    remote_func = Process(target=plotter, args=(plotter_data,))
    remote_func.start()

    sampling_freq = np.array(args.num_cassie_prob, dtype=float) / np.sum(args.num_cassie_prob)

    if single_windowed_force:
        perturbation_force_start = np.random.randint(50, args.time_horizon)
        perturbation_force_end = np.random.randint(perturbation_force_start, args.time_horizon) + 1

    env = Cassiepede(
        clock_type=args.clock_type,
        reward_name=args.reward_name,
        simulator_type='mujoco',
        policy_rate=50,
        dynamics_randomization=True,
        # state_noise=[0.05, 0.1, 0.01, 0.2, 0.01, 0.2, 0.05, 0.05, 0.05],
        state_noise=0,
        velocity_noise=0.0,
        state_est=False,
        full_clock=True,
        full_gait=False,
        integral_action=False,
        com_vis=vis_center_of_mass,
        depth_input=False,
        num_cassie=np.random.choice(np.arange(1, len(sampling_freq) + 1), p=sampling_freq),
        custom_terrain=custom_terrain,
        poi_position_offset=args.poi_position_offset,
        perturbation_force=(not single_windowed_force) * args.perturbation_force,
        force_prob=(not single_windowed_force) * args.force_prob,
        position_offset=args.position_offset,
        only_deck_force=False,
        height_control=True,
        poi_heading_range=args.poi_heading_range,
        cmd_noise=args.cmd_noise,
        merge_states=centralized_policy,
        cmd_noise_prob=args.cmd_noise_prob,
        mask_tarsus_input=args.mask_tarsus_input,
        offscreen=offscreen)

    env.eval(True)
    # env._randomize_commands_bounds = [args.time_horizon, args.time_horizon + 1]

    args.state_dim = env.observation_size
    args.action_dim = env.action_size

    if keyboard:
        tty.setcbreak(sys.stdin.fileno())

    actors = []
    for run_name in args.runs_name:
        args_ = copy.deepcopy(args)
        args_.run_name = run_name

        if dummy_actor:
            actor = actor_model(args_)
        elif legacy_actor:
            actor = load_legacy_actor(args_, device, actor_model)
        else:
            actor = load_actor(args_, device, actor_model)

            # actor = TarsusPatchWrapper(actor)

            # actor = actor.to(device)

        print('actor', sum(p.numel() for p in actor.parameters()))

        actors.append(actor)

    batch_size = (1 if len(actors) > 1 else (1 if env._merge_states else env.num_cassie))

    done = np.zeros(env.num_cassie, dtype=bool)
    episode_length = 0
    episode_reward = defaultdict(lambda: 0)

    # This is penalty (before kernel applied to reward)
    episode_reward_raw = defaultdict(lambda: 0)

    for actor in actors:
        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state(device=device, batch_size=batch_size * (1 + compute_mirror_loss))

    state = env.reset(interactive_evaluation=args.evaluation_mode == 'interactive')

    mirror_dict = env.get_mirror_dict()

    for k in mirror_dict['state_mirror_indices'].keys():
        mirror_dict['state_mirror_indices'][k] = torch.tensor(mirror_dict['state_mirror_indices'][k],
                                                              dtype=torch.float32,
                                                              device=device)

    mirror_dict['action_mirror_indices'] = torch.tensor(mirror_dict['action_mirror_indices'],
                                                        dtype=torch.float32,
                                                        device=device)

    match args.evaluation_mode:
        case 'interactive':
            env.x_velocity_poi = np.zeros(env.num_cassie, dtype=float)
            env.y_velocity_poi = np.zeros(env.num_cassie, dtype=float)
            env.turn_rate_poi = np.zeros(env.num_cassie, dtype=float)
            env.stand = np.zeros(env.num_cassie, dtype=bool)
            env.height_base = np.full(env.num_cassie, 0.7, dtype=float)

            if env.clock_type:
                swing_ratio = 0.5
                period_shifts = [0, 0.5]
                cycle_time = 0.6

                for i in range(env.num_cassie):
                    env.clock[i].set_swing_ratios([swing_ratio, swing_ratio])
                    env.clock[i].set_period_shifts(period_shifts)
                    env.clock[i].set_cycle_time(cycle_time)
        case 'navigation':

            # turn = lambda x: (
            #     (abs(x) / 10.0) * env.default_policy_rate, 0.0, 0.0, np.pi * 10.0 * np.sign(x) / 180.0, 0.75)
            # forward = lambda x: (abs(x) * env.default_policy_rate, np.sign(x) * 1.0, 0.0, 0.0, 0.75)
            #
            # command_schedule = [
            #     forward(10),
            #     turn(90),
            #     forward(6),
            #     turn(90),
            #     forward(3),
            #     turn(90),
            #     forward(3),
            #     turn(-90),
            #     forward(7),
            #     turn(90),
            #     forward(3)
            # ]
            #
            # command_schedule_lookup = np.repeat(np.arange(len(command_schedule)),
            #                                     [command[0] for command in command_schedule], axis=0)
            #
            # args.time_horizon = len(command_schedule_lookup)

            assert args.navigation is not None, ""

            with open(args.navigation, 'r') as file:
                navigations = yaml.safe_load(file)['benchmarks']

            navigations = cycle(navigations)

            navigation = next(navigations)

            command_schedule = navigation['command_schedule']

            command_schedule_lookup = np.repeat(np.arange(len(command_schedule)),
                                                [command['duration'] * env.default_policy_rate for command in
                                                 command_schedule],
                                                axis=0)

            args.time_horizon = len(command_schedule_lookup)

            print(
                f"Running navigation {navigation['idx']}:{navigation['name']} with {len(command_schedule)} for total of {args.time_horizon} steps / {args.time_horizon / env.default_policy_rate} seconds.")

        case 'random':
            pass
        case _:
            raise NotImplementedError

            # env.min_air_time, env.max_air_time = np.full((env.num_cassie, 1), 0.3), np.full((env.num_cassie, 1), 0.5)

    render_state = offscreen

    if not offscreen:
        env.sim.viewer_init(record_video=record_video)
        render_state = env.sim.viewer_render()

    rewards = np.zeros((100, env.num_cassie))
    clocks = np.zeros((100, 8, env.num_cassie))

    done_sum = np.zeros(env.num_cassie)

    poi_idx = 0

    reset = False

    total_reward = 0

    total_power = np.zeros(env.num_cassie, dtype=float)

    initial_poi_position = env.get_poi_position().copy()

    while render_state:

        # env._compute_encoding(poi_position=np.array([0, 0]),
        #                       poi_orientation=0,
        #                       base_positions=np.array([[1.0, 0]]))

        # env.encoding.fill(0)

        start_time = time.time()
        if offscreen or not env.sim.viewer_paused():

            # Numpy to tensor
            for k in state.keys():
                state[k] = torch.tensor(state[k], dtype=torch.float32, device=device)

            # Add seq dimension
            for k in state.keys():
                state[k] = state[k].unsqueeze(1)

            if compute_mirror_loss:
                for k in state.keys():
                    s_mirrored = mirror_tensor(state[k], mirror_dict['state_mirror_indices'][k])

                    state[k] = torch.cat([state[k], s_mirrored], dim=0)

            actions = []

            # print(state.keys(), [v.shape for v in state.values()])

            for i, actor in enumerate(actors):
                if actor is None:
                    action = np.random.uniform(-0.2, 0.2, args.action_dim)
                else:
                    with torch.inference_mode():
                        if not compute_mirror_loss:
                            if len(actors) > 1:
                                action, _ = actor.forward(OrderedDict((k, v[[i]]) for k, v in state.items()))
                            else:
                                action, _ = actor.forward(state)
                            action = action.cpu().numpy().squeeze(1)
                        else:
                            if len(actors) > 1:
                                action, _ = actor.forward(OrderedDict((k, v[[i, i + env.num_cassie]])
                                                                      for k, v in state.items()))

                                mirrored_action = mirror_tensor(action[-1:], mirror_dict['action_mirror_indices'])
                                action = action[:1]
                            else:
                                action, _ = actor.forward(state)

                                mirrored_action = mirror_tensor(action[-env.num_cassie:],
                                                                mirror_dict['action_mirror_indices'])

                                action = action[:env.num_cassie]

                            mirror_loss = (0.5 * F.mse_loss(action, mirrored_action)).item()

                            logging.info(f'Mirror loss: {mirror_loss}')

                            action = mirrored_action.cpu().numpy().squeeze(1)

                actions.append(action)

            action = np.concatenate(actions, axis=0)
            #
            # print('Action before:', action)
            #
            # action[:] = np.array([[env.x_velocity_poi, env.y_velocity_poi, env.turn_rate_poi]])
            #

            if args.evaluation_mode == 'navigation':
                args.time_horizon = len(command_schedule_lookup)

                env.x_velocity_poi = np.full(env.num_cassie,
                                             command_schedule[command_schedule_lookup[episode_length]]['x'],
                                             dtype=float)
                env.y_velocity_poi = np.full(env.num_cassie,
                                             command_schedule[command_schedule_lookup[episode_length]]['y'],
                                             dtype=float)
                env.turn_rate_poi = np.full(env.num_cassie,
                                            command_schedule[command_schedule_lookup[episode_length]][
                                                't'] * np.pi / 180.0, dtype=float)
                env.height_base = np.full(env.num_cassie,
                                          command_schedule[command_schedule_lookup[episode_length]]['h'], dtype=float)

            if single_windowed_force:
                if episode_length == perturbation_force_start:
                    apply_linear_perturbation(env, args.perturbation_force)
                elif episode_length == perturbation_force_end:
                    apply_linear_perturbation(env, 0)

            state, reward, done, info = env.step(action)

            # for k, v in env.state_dict.items():
            #     print(k, v.shape)
            # print()
            # # print('state:', env.state_dict)

            # Override done without height check
            # done, _ = env.compute_done()

            total_reward += reward

            # print('turn_rate:', env.turn_rate_poi,'orient_add:', np.degrees(env.orient_add))

            for i in range(env.num_cassie):
                if 'reward_raw' in info:
                    for reward_key, reward_val in info['reward_raw'][i].items():
                        episode_reward_raw[reward_key] += reward_val

                if 'reward' in info:
                    for reward_key, reward_val in info['reward'][i].items():
                        episode_reward[reward_key] += reward_val

            # r = env.compute_reward(action)

            # assert np.allclose(reward, r), f"Reward mismatch: {reward} != {r}"

            # print('poi site orientation:', np.degrees(quaternion2euler(env.get_poi_orientation())[-1]), 'poi pose:',
            #       env.get_poi_position())
            # print('leader orientation:', np.degrees(quaternion2euler(env.get_base_orientation()[0])[-1]),
            #       'leader pose:', env.get_base_position()[0])

            done_sum += done

            # if episode_length == 100:
            #     poi_position = env.get_poi_position()[:2]
            #     # poi_orientation = quaternion2euler(np.array(env.get_poi_orientation()).reshape(1, -1))[0, -1]
            #     # Update the poi position
            #     env._update_poi_position(poi_position=poi_position, poi_orientation=0)

            if keyboard and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input_char = sys.stdin.read(1)
                match input_char:
                    case "a":
                        env.y_velocity_poi += 0.1
                    case "d":
                        env.y_velocity_poi -= 0.1
                    case "w":
                        env.x_velocity_poi += 0.1
                    case "s":
                        env.x_velocity_poi -= 0.1
                    case "q":
                        env.turn_rate_poi += np.radians(5)
                    case "e":
                        env.turn_rate_poi -= np.radians(5)
                    case "1":
                        env.height_base -= 0.05
                    case "2":
                        env.height_base += 0.05
                    case "c":
                        env.stand = not env.stand
                        if env.stand:
                            env.x_velocity_poi = np.zeros(env.num_cassie, dtype=float)
                            env.y_velocity_poi = np.zeros(env.num_cassie, dtype=float)
                            env.turn_rate_poi = np.zeros(env.num_cassie, dtype=float)
                    case "-":
                        swing_ratio -= 0.1
                        for i in range(env.num_cassie):
                            env.clock[i].set_swing_ratios([swing_ratio, swing_ratio])
                        print('swing_ratio:', swing_ratio)
                    case "[":
                        period_shifts[1] -= 0.1
                        for i in range(env.num_cassie):
                            env.clock[i].set_period_shifts(period_shifts)
                        print('period_shifts:', period_shifts)
                    case ";":
                        cycle_time -= 0.1
                        for i in range(env.num_cassie):
                            env.clock[i].set_cycle_time(cycle_time)
                        print('cycle_time:', cycle_time)
                    case "=":
                        swing_ratio += 0.1
                        for i in range(env.num_cassie):
                            env.clock[i].set_swing_ratios([swing_ratio, swing_ratio])
                        print('swing_ratio:', swing_ratio)
                    case "]":
                        period_shifts[1] += 0.1
                        for i in range(env.num_cassie):
                            env.clock[i].set_period_shifts(period_shifts)
                        print('period_shifts:', period_shifts)
                    case "'":
                        cycle_time += 0.1
                        for i in range(env.num_cassie):
                            env.clock[i].set_cycle_time(cycle_time)
                        print('cycle_time:', cycle_time)
                    case 'h':
                        for actor in actors:
                            if hasattr(actor, 'init_hidden_state'):
                                actor.init_hidden_state(device=device,
                                                        batch_size=batch_size * (1 + compute_mirror_loss))
                    case input_char if input_char in ['i', 'm', 'j', 'k', 'l', 'u', 'o']:

                        if input_char in ['u', 'o']:
                            delta_rot = {'u': -0.1, 'o': 0.1}[input_char]
                            delta_pos = [0, 0]
                        else:
                            delta_pos = {'i': [0.1, 0],
                                         'm': [-0.1, 0],
                                         'j': [0, 0.1],
                                         'k': [0, -0.1],
                                         'l': [0, 0]}[input_char]
                            delta_rot = 0

                        # Get local cassie positions. We cannot use sensor as it gives global position
                        local_base_position = []
                        for i in range(env.num_cassie):
                            i = '' if i == 0 else f'c{i + 1}_'
                            local_base_position.append(env.sim.model.body(f'{i}cassie-pelvis').pos[:2] + delta_pos)
                        local_base_position = np.array(local_base_position)

                        # Get local poi position and orientation. We cannot use sensor as it gives global position
                        if input_char == 'l':
                            poi_position = local_base_position[poi_idx]
                            poi_idx = (poi_idx + 1) % env.num_cassie
                        else:
                            poi_position = env.sim.model.site('poi_site').pos[:2] + delta_pos
                        poi_orientation = \
                            quaternion2euler(np.array(env.sim.model.site('poi_site').quat).reshape(1, -1))[0, -1] \
                            + delta_rot

                        # Update the poi position
                        env._update_poi_position(poi_position=poi_position, poi_orientation=poi_orientation)

                        # Re compute encoding
                        env._compute_encoding(poi_position=poi_position, poi_orientation=poi_orientation,
                                              base_positions=local_base_position)

                        plotter_data['clear'] = True

                    case "r":
                        reset = True
                        # done = np.ones(env.num_cassie, dtype=bool)

            # env.x_velocity_poi[1] = 0
            # env.y_velocity_poi[1] = 0
            # env.turn_rate_poi[1] = 0

            if vis_reward_plot or vis_clock_plot:
                if episode_length >= rewards.shape[0]:
                    rewards = np.roll(rewards, -1, axis=0)
                    rewards[-1] = reward
                    clocks = np.roll(clocks, -1, axis=0)
                    clocks[-1] = state['clocks'].T
                else:
                    rewards[episode_length] = reward
                    clocks[episode_length] = state['clocks'].T

            if vis_reward_plot:
                plt.clf()
                plt.plot(rewards[:min(episode_length + 1, rewards.shape[0])])
                plt.pause(0.0001)

            if vis_clock_plot:
                plt.clf()
                plt.plot(clocks[:min(episode_length + 1, clocks.shape[0])][..., 0])
                plt.plot(clocks[:min(episode_length + 1, clocks.shape[0])][..., 1])
                plt.legend(np.arange(8))
                plt.pause(0.0001)

            if vis_position_plot:
                plotter_data['vis_position_plot'] = {'poi_position': env.get_poi_position(),
                                                     'base_positions': env.get_base_position()}

            # print('poi_position:', env.get_poi_position(), 'base_positions:', env.get_base_position(), env.get_poi_orientation())

            if vis_encoding_plot:
                plotter_data['vis_encoding_plot'] = {'encoding': env.state_dict['encoding'],
                                                     'num_cassie': env.num_cassie,
                                                     'base_yaws_poi': env.get_encoder_yaw(),
                                                     'poi_position': env.get_poi_position(),
                                                     'poi_orientation': env.get_poi_orientation(),
                                                     'base_positions': env.get_base_position(),
                                                     'base_orientations': env.get_base_orientation(),
                                                     'base_orient_cmd': quaternion2euler(
                                                         env.rotate_to_heading(env.get_base_orientation()))[:, -1],
                                                     'orient_add': env.orient_add}

            if vis_reward_dict_plot:
                plotter_data['vis_reward_dict_plot'] = {'num_cassie': env.num_cassie,
                                                        'reward': info['reward']}

            if vis_individual_reward:
                if 'vis_individual_reward' not in plotter_data:
                    plotter_data['vis_individual_reward'] = [[info['reward'][i].get(vis_individual_reward, 0) for i in
                                                              range(env.num_cassie)]]
                else:
                    plotter_data['vis_individual_reward'] = plotter_data['vis_individual_reward'][-100:] + [
                        [info['reward'][i].get(vis_individual_reward, 0) for i in range(env.num_cassie)]]

            if vis_power_plot:
                power = np.abs(env.sim.get_motor_velocity() * env.sim.get_torque().reshape(env.num_cassie,
                                                                                           -1)).sum(axis=1)
                if 'vis_power_plot' not in plotter_data:
                    plotter_data['vis_power_plot'] = [power]
                else:
                    plotter_data['vis_power_plot'] = plotter_data['vis_power_plot'][-500:] + [power]

                total_power += power

            if vis_force_plot:
                force = np.linalg.norm(env.get_base_force()[..., :2], axis=-1)

                if 'vis_force_plot' not in plotter_data:
                    plotter_data['vis_force_plot'] = [force]
                else:
                    plotter_data['vis_force_plot'] = plotter_data['vis_force_plot'][-500:] + [force]

            # if vis_motion_plot:
            #     # motion_data.append(env.get_base_position()[0])
            #     # plotter_data['motion'] = {'base_positions': motion_data}
            #
            #     stable_base = env.running_pelvis_position(env.get_base_position()[0, :3])
            #
            #     motion_data.append(stable_base)
            #
            #     plotter_data['vis_motion_plot'] = {'base_positions': motion_data}

            # base_acc = env.sim.get_body_acceleration(env.sim.base_body_name[0])[2]
            #
            # base_accs.append(base_acc)
            #
            # plotter_data['base_acc'] = base_accs

            episode_length += 1

            # print(info['reward'][0]['feet_air_time'], info['reward_raw'][0]['feet_air_time'])

            # print('Encoding:', env.encoding.T)

            # print('reward_raw:', info['reward_raw'])

            # logging.info('Orientation reward', info['reward'][0]['orientation'], info['reward'][1]['orientation'])

            # logging.info(
            #     f'Loop frequency:{1 / (time.time() - start_time)}, Reward: {reward}, Info: {info}, Done: {done}, Episode length: {episode_length}, Command: x_vel={env.x_velocity_poi}, y_vel={env.y_velocity_poi}, turn_rate={env.turn_rate_poi}')

        # logging.info(
        #     f'orientation error: {np.degrees(quaternion2euler(np.array([env.get_poi_orientation()]))[0, -1] + np.pi - initial_poi_orientation)}')

        if not offscreen:
            render_state = env.sim.viewer_render()

            delaytime = max(0, env.default_policy_rate / 2000 - (time.time() - start_time))

            time.sleep(delaytime)

        if done.any():
            print('done', done, info.get('done_info', None))

        if (done.any() and done_enable) or reset or episode_length >= args.time_horizon:
            reset = False
            # plotter_data.clear()
            plotter_data['clear'] = True
            poi_idx = 0

            logging.info(
                f'Total reward dict:{episode_reward}\n'
                f'Total reward raw:{episode_reward_raw}\n'
                f'Total reward (all cassie): {np.sum(list(episode_reward.values()))}\n'
                f'Total reward:{total_reward}\nEpisode length:{episode_length}\n'
                f'Encoding: {env.encoding}\n'
                f'force_vector={(env.force_vector, np.linalg.norm(env.force_vector)) if hasattr(env, "force_vector") else None}\n'
                f'torque_vector={(env.torque_vector, np.linalg.norm(env.torque_vector)) if hasattr(env, "force_vector") else None}\n'
                f'Total power per step={total_power / episode_length}\n'
                f'done_info={info.get("done_info", None)}\n'
                f'pert_force_start={perturbation_force_start if single_windowed_force else None}\n'
                f'pert_force_end={perturbation_force_end if single_windowed_force else None}\n')

            logging.info(f'odometry: {env.get_poi_position() - initial_poi_position}')

            rem_rotation = (args.time_horizon - episode_length) * env.turn_rate_poi[0] / env.default_policy_rate

            logging.info(
                f'orientation error: {np.degrees((quaternion2euler(np.array([env.get_poi_orientation()]))[0, -1] - env.orient_add[0] - rem_rotation + np.pi) % (2 * np.pi) - np.pi)}, rem_rotation: {np.degrees(rem_rotation)} env.orient_add[0]={np.degrees(env.orient_add[0])}, current_orientation={np.degrees(quaternion2euler(np.array([env.get_poi_orientation()]))[0, -1])}')

            if args.evaluation_mode == 'navigation':
                navigation = next(navigations)

                command_schedule = navigation['command_schedule']

                command_schedule_lookup = np.repeat(np.arange(len(command_schedule)),
                                                    [command['duration'] * env.default_policy_rate for command in
                                                     command_schedule],
                                                    axis=0)

                args.time_horizon = len(command_schedule_lookup)

                print(
                    f"Running navigation {navigation['idx']}:{navigation['name']} with {len(command_schedule)} for total of {args.time_horizon} steps / {args.time_horizon / env.default_policy_rate} seconds.")

            episode_reward.clear()
            episode_reward_raw.clear()

            time.sleep(0.5)

            total_reward = 0

            total_power = np.zeros(env.num_cassie, dtype=float)

            done = np.zeros(env.num_cassie, dtype=bool)
            done_sum = np.zeros(env.num_cassie)

            state = env.reset(interactive_evaluation=args.evaluation_mode == 'interactive')

            initial_poi_position = env.get_poi_position().copy()

            if single_windowed_force:
                perturbation_force_start = np.random.randint(50, args.time_horizon)
                perturbation_force_end = np.random.randint(perturbation_force_start, args.time_horizon) + 1

            if args.evaluation_mode == 'interactive':
                env.x_velocity_poi = np.zeros(env.num_cassie, dtype=float)
                env.y_velocity_poi = np.zeros(env.num_cassie, dtype=float)
                env.turn_rate_poi = np.zeros(env.num_cassie, dtype=float)
                env.stand = np.zeros(env.num_cassie, dtype=bool)
                env.height_base = np.full(env.num_cassie, 0.7, dtype=float)

                if env.clock_type:
                    swing_ratio = 0.5
                    period_shifts = [0, 0.5]
                    cycle_time = 0.6

                    for i in range(env.num_cassie):
                        env.clock[i].set_swing_ratios([swing_ratio, swing_ratio])
                        env.clock[i].set_period_shifts(period_shifts)
                        env.clock[i].set_cycle_time(cycle_time)

            print('commands:', env.x_velocity_poi, env.y_velocity_poi, env.turn_rate_poi,
                  env.stand if hasattr(env, 'stand') else None)

            episode_length = 0
            episode_reward.clear()

            for actor in actors:
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state(device=device,
                                            batch_size=batch_size * (1 + compute_mirror_loss))

            if not offscreen and hasattr(env.sim, 'renderer'):
                if env.sim.renderer is not None:
                    print("re-init non-primary screen renderer")
                    env.sim.renderer.close()
                    env.sim.init_renderer(offscreen=env.offscreen,
                                          width=env.depth_image_dim[0], height=env.depth_image_dim[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")

    parser.add_argument('--time_horizon', type=int, default=1e5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=4)
    parser.add_argument('--set_adam_eps', action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--std', type=float, default=0.13)
    parser.add_argument('--use_orthogonal_init', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='roadrunner_cassiepede')
    parser.add_argument('--runs_name', type=str, nargs='+', default=["2024-02-16 02:41:50.685098"])
    parser.add_argument('--num_cassie_prob', nargs='+', type=float, default=[0.0, 1.0, 0.0])
    parser.add_argument('--model_checkpoint', type=str, default="latest")
    parser.add_argument('--reward_name', type=str, default='locomotion_cassiepede')
    parser.add_argument('--clock_type', type=str, required=False)
    parser.add_argument("--position_offset", type=float, default=0.2, help="Cassiepede position offset")
    parser.add_argument("--poi_heading_range", type=float, default=0.0, help="Poi heading range")
    parser.add_argument("--poi_position_offset", type=float, default=0.0, help="Poi offset from cassie")
    parser.add_argument("--perturbation_force", type=float, help="Force to apply to the deck", default=0)
    parser.add_argument("--force_prob", type=float, help="Prob of force to apply to the deck", default=0.0)
    parser.add_argument("--cmd_noise", type=float,
                        help="Noise to cmd for each cassie. Tuple of 3 (x_vel, y_vel, turn_rate (deg/t))", nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--mask_tarsus_input", action='store_true', help="Mask tarsus input with zeros")
    parser.add_argument("--cmd_noise_prob", type=float, help="Prob of noise added to cmd for each cassie", default=0.0)
    parser.add_argument('--redownload_checkpoint', action='store_true')
    parser.add_argument('--evaluation_mode', type=str, default='interactive',
                        choices=['interactive', 'navigation', 'random'])
    parser.add_argument('--navigation', type=str, default=None)

    args = parser.parse_args()

    # Either supply one run name of all cassie or supply run name for each cassie
    assert len(args.runs_name) == 1 or len(args.num_cassie_prob) == len(args.runs_name)

    main()

# Example run:
# export PYTHONPATH=.
# python algo/cassiepede/evaluation.py \
#   --hidden_dim 64 \
#   --lstm_hidden_dim 64 \
#   --lstm_num_layers 2 \
#   --set_adam_eps \
#   --eps 1e-5 \
#   --std 0.13 \
#   --use_orthogonal_init \
#   --seed 0 \
#   --project_name roadrunner_cassiepede \
#   --num_cassie_prob 0 1 \
#   --position_offset 1.0 \
#   --model_checkpoint latest \
#   --redownload_checkpoint \
#   --run_name "2024-03-03 11:33:08.428965" \
#   --project_name roadrunner_cassiepede \
#   --reward_name locomotion_cassiepede \
#   --clock_type von_mises
