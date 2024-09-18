import datetime
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse

from env.cassie.cassiepedeHL.cassiepedeHL import CassiepedeHL
from env.util.quaternion import quaternion2euler

plt.ion()

from env.cassie.cassiepede.cassiepede import Cassiepede
from algo.common.network import *

import sys
import tty
from algo.common.utils import *
import oyaml as yaml
import tqdm

logging.basicConfig(level=logging.INFO)


@ray.remote
def eval(env_fn, actor, args, benchmark, device, offscreen, num_episodes):
    env = env_fn()
    env.eval(True)

    # Override termination condition for the benchmark
    env._height_bounds[0] = 0.75

    torch.set_num_threads(1)

    episode = 1

    done = np.zeros(env.num_cassie, dtype=bool)
    episode_length = 0
    # episode_reward = defaultdict(lambda: 0)
    # episode_reward_raw = defaultdict(lambda: 0)

    if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state(device=device, batch_size=1 if env._merge_states else env.num_cassie)
    state = env.reset(interactive_evaluation=False)

    if 'command_schedule' in benchmark:
        command_schedule = benchmark['command_schedule']

        command_schedule_lookup = np.repeat(np.arange(len(command_schedule)),
                                            [command['duration'] * env.default_policy_rate for command in
                                             command_schedule],
                                            axis=0)

        env.x_velocity_poi = np.full(env.num_cassie,
                                     command_schedule[command_schedule_lookup[episode_length]]['x'],
                                     dtype=float)
        env.y_velocity_poi = np.full(env.num_cassie,
                                     command_schedule[command_schedule_lookup[episode_length]]['y'],
                                     dtype=float)
        env.turn_rate_poi = np.full(env.num_cassie,
                                    command_schedule[command_schedule_lookup[episode_length]][
                                        't'] * np.pi / 180.0,
                                    dtype=float)
        env.height_base = np.full(env.num_cassie,
                                  command_schedule[command_schedule_lookup[episode_length]]['h'],
                                  dtype=float)

        args.time_horizon = len(command_schedule_lookup)
    elif 'command' in benchmark:
        env.x_velocity_poi = np.full(env.num_cassie, benchmark['command']['x'], dtype=float)
        env.y_velocity_poi = np.full(env.num_cassie, benchmark['command']['y'], dtype=float)
        env.turn_rate_poi = np.full(env.num_cassie, benchmark['command']['t'] * np.pi / 180.0, dtype=float)
        env.height_base = np.full(env.num_cassie, benchmark['command']['h'], dtype=float)

        args.time_horizon = benchmark['env']['time_horizon']
    else:
        raise ValueError("Either command_schedule or command should be provided in the benchmark")

    perturbation_force_start = np.random.randint(50, args.time_horizon)
    perturbation_force_end = np.random.randint(perturbation_force_start, args.time_horizon) + 1

    render_state = offscreen

    if not offscreen:
        env.sim.viewer_init()
        render_state = env.sim.viewer_render()

    reset = False

    data = defaultdict(lambda: [])

    poi_position = env.get_poi_position().copy()

    poi_linear_acc = 0

    total_episode_reward = 0

    power = np.zeros(env.num_cassie, dtype=float)

    positive_power = np.zeros(env.num_cassie, dtype=float)

    joint_forces = np.zeros(env.num_cassie, dtype=float)

    joint_forces_xyz = []

    height_diff = np.zeros(env.num_cassie, dtype=float)

    robot_positions = []
    poi_positions = []

    while render_state:
        start_time = time.time()
        if offscreen or not env.sim.viewer_paused():

            if actor is None:
                action = np.random.uniform(-0.2, 0.2, args.action_dim)
            else:
                # Numpy to tensor
                for k in state.keys():
                    state[k] = torch.tensor(state[k], dtype=torch.float32, device=device)

                # Add seq dimension
                for k in state.keys():
                    state[k] = state[k].unsqueeze(1)

                with torch.no_grad():
                    action, _ = actor.forward(state)
                action = action.cpu().numpy().squeeze(1)

            if benchmark['env']['force'] is not None:
                if perturbation_force_start == episode_length:
                    apply_linear_perturbation(env, benchmark['env']['force'])
                elif perturbation_force_end == episode_length:
                    apply_linear_perturbation(env, 0)

            state, reward, done, info = env.step(action)

            # This policy is not trained with non-zero base yaw poi and encoding
            if args.run_name == '2024-04-13 21:56:36.012260':
                state['encoding'].fill(0.0)
                state['base_yaw_poi'].fill(0.0)

            total_episode_reward += np.mean(reward)

            if 'command_schedule' in benchmark:
                env.x_velocity_poi = np.full(env.num_cassie,
                                             command_schedule[command_schedule_lookup[episode_length]]['x'],
                                             dtype=float)
                env.y_velocity_poi = np.full(env.num_cassie,
                                             command_schedule[command_schedule_lookup[episode_length]]['y'],
                                             dtype=float)
                env.turn_rate_poi = np.full(env.num_cassie,
                                            command_schedule[command_schedule_lookup[episode_length]][
                                                't'] * np.pi / 180.0,
                                            dtype=float)
                env.height_base = np.full(env.num_cassie,
                                          command_schedule[command_schedule_lookup[episode_length]]['h'],
                                          dtype=float)

                robot_positions.append(env.get_base_position())
                poi_positions.append(env.get_poi_position().copy())

            elif 'command' in benchmark:
                env.x_velocity_poi = np.full(env.num_cassie, benchmark['command']['x'], dtype=float)
                env.y_velocity_poi = np.full(env.num_cassie, benchmark['command']['y'], dtype=float)
                env.turn_rate_poi = np.full(env.num_cassie, benchmark['command']['t'] * np.pi / 180.0, dtype=float)
                env.height_base = np.full(env.num_cassie, benchmark['command']['h'], dtype=float)
            else:
                raise ValueError("Either command_schedule or command should be provided in the benchmark")

            poi_linear_acc += np.linalg.norm(env.get_poi_linear_acceleration())

            # for i in range(env.num_cassie):
            #     for reward_key, reward_val in info['reward_raw'][i].items():
            #         episode_reward_raw[reward_key] += reward_val
            #
            #     for reward_key, reward_val in info['reward'][i].items():
            #         episode_reward[reward_key] += reward_val

            episode_length += 1

            power_per_step = env.sim.get_motor_velocity() * env.sim.get_torque().reshape(env.num_cassie, -1)

            power += power_per_step.sum(axis=1)

            positive_power += power_per_step.clip(0).sum(axis=1)

            joint_forces += np.linalg.norm(env.get_base_force()[..., :2], axis=-1)

            joint_forces_xyz.append(env.get_base_force())

            height_diff += np.abs(env.get_base_position()[..., 2] - env.height_base)

        if not offscreen:
            render_state = env.sim.viewer_render()

            delaytime = max(0, env.default_policy_rate / 2000 - (time.time() - start_time))

            time.sleep(delaytime)

        if done.any() or reset or episode_length >= args.time_horizon:
            reset = False

            # print(
            #     f'Total reward: {total_episode_reward}, Episode length:{episode_length}')
            #
            # logging.info(
            #     f'Episode: {episode}, Total reward: {total_episode_reward}, Episode length:{episode_length}')

            # # Bookkeeping
            # for k in episode_reward.keys():
            #     data[k].append(episode_reward[k])
            #
            # for k in episode_reward_raw.keys():
            #     data[k + '_raw'].append(episode_reward_raw[k])

            data['episode_length'].append(episode_length)
            data['total_reward'].append(total_episode_reward)
            data['num_cassie'].append(env.num_cassie)
            data['power'].append(power)
            data['positive_power'].append(positive_power)
            data['height_diff'].append(height_diff)

            for i, dir in enumerate('xyz'):
                data[f'odometry_{dir}'].append(env.get_poi_position()[i] - poi_position[i])

            data[f'poi_linear_acceleration'].append(poi_linear_acc)

            data['joint_forces'].append(joint_forces)

            data['joint_forces_xyz'].append(joint_forces_xyz)

            data['perturbation_force_bound'].append((perturbation_force_start, perturbation_force_end))

            # doesn't account for full rotation
            data['orientation_error'] \
                = np.degrees(
                (quaternion2euler(np.array([env.get_poi_orientation()]))[0, -1] - env.orient_add[0] + np.pi) % (
                        2 * np.pi) - np.pi)

            data['robot_positions'].append(robot_positions)
            data['poi_positions'].append(poi_positions)

            state = env.reset(interactive_evaluation=False)

            perturbation_force_start = np.random.randint(50, args.time_horizon)
            perturbation_force_end = np.random.randint(perturbation_force_start, args.time_horizon) + 1

            episode_length = 0
            poi_linear_acc = 0

            total_episode_reward = 0
            done = np.zeros(env.num_cassie, dtype=bool)
            poi_position = env.get_poi_position().copy()
            power = np.zeros(env.num_cassie, dtype=float)
            joint_forces = np.zeros(env.num_cassie, dtype=float)
            joint_forces_xyz = []

            robot_positions = []
            poi_positions = []

            if 'command_schedule' in benchmark:
                env.x_velocity_poi = np.full(env.num_cassie,
                                             command_schedule[command_schedule_lookup[episode_length]]['x'],
                                             dtype=float)
                env.y_velocity_poi = np.full(env.num_cassie,
                                             command_schedule[command_schedule_lookup[episode_length]]['y'],
                                             dtype=float)
                env.turn_rate_poi = np.full(env.num_cassie,
                                            command_schedule[command_schedule_lookup[episode_length]][
                                                't'] * np.pi / 180.0,
                                            dtype=float)
                env.height_base = np.full(env.num_cassie,
                                          command_schedule[command_schedule_lookup[episode_length]]['h'],
                                          dtype=float)

            elif 'command' in benchmark:
                env.x_velocity_poi = np.full(env.num_cassie, benchmark['command']['x'], dtype=float)
                env.y_velocity_poi = np.full(env.num_cassie, benchmark['command']['y'], dtype=float)
                env.turn_rate_poi = np.full(env.num_cassie, benchmark['command']['t'] * np.pi / 180.0, dtype=float)
                env.height_base = np.full(env.num_cassie, benchmark['command']['h'], dtype=float)
            else:
                raise ValueError("Either command_schedule or command should be provided in the benchmark")

            # episode_reward.clear()

            if hasattr(actor, 'init_hidden_state'):
                actor.init_hidden_state(device=device, batch_size=1 if env._merge_states else env.num_cassie)

            if not offscreen and hasattr(env.sim, 'renderer'):
                if env.sim.renderer is not None:
                    print("re-init non-primary screen renderer")
                    env.sim.renderer.close()
                    env.sim.init_renderer(offscreen=env.offscreen,
                                          width=env.depth_image_dim[0], height=env.depth_image_dim[1])

            if episode >= num_episodes:
                break

            episode += 1

    del env

    return pd.DataFrame(data)


def create_range(ids, size):
    if ids is None:
        return range(size)

    ids = [_id.split('-') for _id in ids]

    for i in range(len(ids)):
        if len(ids[i]) == 1:
            # Only specific benchmark is given
            ids[i] = [int(ids[i][0])]
        elif len(ids[i]) == 2:
            # Range of benchmarks is given
            if ids[i][0] == '':
                # Start of the range is not given
                ids[i] = range(int(ids[i][1]) + 1)
            elif ids[i][1] == '':
                # End of the range is not given
                ids[i] = range(int(ids[i][0]), size)
            else:
                # Both start and end of the range are given
                ids[i] = range(int(ids[i][0]), int(ids[i][1]) + 1)
        else:
            raise ValueError(
                f"Invalid range specification: {ids[i]}. Should be either 'start-end' or 'start-' or '-end' or 'single'")

    ids = sorted(list(set([i for _id in ids for i in _id if 0 <= i < size])))

    return ids


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
    device = torch.device('cpu')

    logging.info(f'Args: {args.__dict__}')

    keyboard = False
    offscreen = True
    dummy_actor = False
    actor_model = Actor_LSTM_v2

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    ids = create_range(args.ids, len(config['benchmarks']))

    args.num_workers = min(args.num_workers, config['n_traj'])

    ray.init(num_cpus=args.num_workers)

    config_name = args.save_dir or args.config_path.split('/')[-1].split('.')[0]

    save_dir = f"algo/cassiepede/benchmark/data/{config_name}"

    os.makedirs(save_dir, exist_ok=True)

    tq = tqdm.tqdm(total=len(ids))
    for i, id_ in enumerate(ids):
        benchmark = config['benchmarks'][id_]

        ENV_CLASS = CassiepedeHL if benchmark['env']['policy'] == 'hierarchical' else Cassiepede

        env_fn = lambda: ENV_CLASS(
            clock_type=args.clock_type,
            reward_name=args.reward_name,
            simulator_type='mujoco',
            policy_rate=50,
            dynamics_randomization=True,
            state_noise=[0.05, 0.1, 0.01, 0.2, 0.01, 0.2, 0.05, 0.05, 0.05],
            # state_noise=0,
            velocity_noise=0.0,
            state_est=False,
            full_clock=True,
            full_gait=False,
            integral_action=False,
            com_vis=False,
            depth_input=False,
            num_cassie=benchmark['env']['num_cassie'],
            custom_terrain=benchmark['env']['terrain'],
            poi_position_offset=args.poi_position_offset,
            position_offset=args.position_offset,
            # Manually set logic for force perturbation
            perturbation_force=args.perturbation_force if benchmark['env']['force'] is None else 0,

            force_prob=args.force_prob if benchmark['env']['force'] is None else 0,

            merge_states='centralized' in benchmark['env']['policy'],
            only_deck_force=False if benchmark['env']['force'] is None else True,
            height_control=True,
            poi_heading_range=args.poi_heading_range,
            cmd_noise=args.cmd_noise,
            cmd_noise_prob=args.cmd_noise_prob,
            offscreen=offscreen)

        env = env_fn()

        args.run_name = benchmark['env']['run_name']
        args.model_checkpoint = benchmark['env']['checkpoint']

        args.state_dim = env.observation_size
        args.action_dim = env.action_size

        if keyboard:
            tty.setcbreak(sys.stdin.fileno())

        if dummy_actor:
            actor = actor_model(args)
        else:
            actor = load_actor(args, device, actor_model)

        # logging.info(f'Benchmark: {benchmark["idx"]}-{benchmark["name"]} ({i + 1}/{len(ids)})')
        tq.set_description(f'Benchmark: {benchmark["idx"]}')
        tq.update(1)

        eval_workers = [eval.remote(env_fn, actor, args, benchmark, device, offscreen,
                                    config['n_traj'] // args.num_workers) for _ in range(args.num_workers)]

        workers_data = ray.get(eval_workers)

        data = None
        for i in range(args.num_workers):
            if data is None:
                data = workers_data[i]
            else:
                data = pd.concat([data, workers_data[i]], ignore_index=True)

        data['run_args'] = str({**args.__dict__, **benchmark})

        # save as pickle (enable for maneuveribilty plot)
        data.to_pickle(os.path.join(save_dir, f"{benchmark['idx']}-{benchmark['name']}.pkl"))
        # data.to_csv(os.path.join(save_dir, f"{benchmark['idx']}-{benchmark['name']}.csv"))

        logging.info(f"Done {benchmark['idx']}-{benchmark['name']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")

    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=4)
    parser.add_argument('--set_adam_eps', action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--std', type=float, default=0.13)
    parser.add_argument('--use_orthogonal_init', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='roadrunner_cassiepede')
    parser.add_argument('--reward_name', type=str, default='locomotion_cassiepede')
    parser.add_argument('--clock_type', type=str, required=False)
    parser.add_argument("--position_offset", type=float, default=0.0, help="Cassiepede position offset")
    parser.add_argument("--poi_heading_range", type=float, default=0.0, help="Poi heading range")
    parser.add_argument("--poi_position_offset", type=float, default=0.0, help="Poi offset from cassie")
    parser.add_argument("--force_prob", type=float, help="Prob of force to apply to the deck", default=0.0)
    parser.add_argument("--perturbation_force", type=float, help="Force to apply to the deck", default=0)
    parser.add_argument("--cmd_noise", type=float,
                        help="Noise to cmd for each cassie. Tuple of 3 (x_vel, y_vel, turn_rate (deg/t))", nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--cmd_noise_prob", type=float, help="Prob of noise added to cmd for each cassie", default=0.0)
    parser.add_argument('--redownload_checkpoint', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config_path', type=str, help='Path to config file for benchmark', required=True)
    parser.add_argument('--ids', type=str, nargs='+', help='Ids of the benchmarks to run', required=False)
    parser.add_argument('--save_dir', type=str, help='Path to save the data', required=False)

    args = parser.parse_args()

    main()

# Example run:
# export PYTHONPATH=.
# export WANDB_API_KEY=
# python algo/cassiepede/benchmark/collect_data.py \
#   --hidden_dim 64 \
#   --lstm_hidden_dim 64 \
#   --lstm_num_layers 2 \
#   --set_adam_eps \
#   --eps 1e-5 \
#   --use_orthogonal_init \
#   --seed 0 \
#   --project_name roadrunner_cassiepede \
#   --std 0.13 \
#   --reward_name locomotion_cassiepede_feetairtime_modified \
#   --redownload_checkpoint \
#   --model_checkpoint latest \
#   --config_path "algo/cassiepede/benchmark/configs/varying_cassie_count.yaml" \
#   --save_dir "varying_cassie_count_new" \
#   --num_workers 100
