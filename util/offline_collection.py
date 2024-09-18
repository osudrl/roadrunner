import glob
import logging
import shutil

import cv2
import numpy as np
import ray
import torch
import mediapy
import pickle
import os
import time
import tqdm
import pandas as pd

from algo.util.worker import AlgoWorker
from algo.util.sampling import Buffer
from util.colors import OKGREEN, ENDC
from env.util.quaternion import euler2quat

logging.basicConfig(level=logging.INFO)


def collect_data(env_fn,
                 actor,
                 critic,
                 args,
                 data_path: str = None,
                 num_workers: int = 4,
                 max_traj_len: int = 512,
                 max_trajs: int = 60000,
                 save_video: bool = False,
                 mode: str = 'dynamic',
                 high_rate_data: bool = False):
    discount = args.discount

    # Mannually set these here to allow flexibility during data collection
    TERRAIN_NAME = 'bump'
    DIFFICULTY = 1
    # NOTE: options to randomize the data collection that cannot be done in the post process
    # We use stochastic actor to collect data for heightmap training
    # For standard evaluation, we use determinstic actor
    use_deterministic_actor = True

    if data_path is None:
        commit_hash = os.popen('git rev-parse HEAD').read().strip()[:6]
        data_path = os.path.join("offline_data", args.env_name, args.run_name, commit_hash,
                                 TERRAIN_NAME + f'-diff-{DIFFICULTY}', time.strftime("%Y%m%d-%H%M%S"))
        print(f"{OKGREEN}Data will be saved to {data_path}{ENDC}")
        os.makedirs(data_path)

    # Save project files to data_path
    cwd = os.getcwd()

    save_path = os.path.join(data_path, "project_files")

    os.makedirs(save_path, exist_ok=True)

    exclude = {'checkpoints', 'saved_models', 'wandb', '.idea', '.git', 'pretrained_models', 'trained_models',
               'offline_data', 'offline_data_1', 'offline_data_2', 'old_files'}

    dirs = [d for d in os.listdir(cwd) if d not in exclude and os.path.isdir(os.path.join(cwd, d))]

    for d in dirs:
        base_paths = [os.path.join(cwd, d, '**', ext) for ext in ['*.py', '*.yaml', '*.yml', '*.sh']]
        for base_path in base_paths:
            for file in glob.glob(base_path, recursive=True):
                src_file = os.path.relpath(file, start=cwd)
                trg_dir = os.path.join(save_path, os.path.dirname(src_file))
                trg_file = os.path.join(trg_dir, os.path.basename(src_file))
                os.makedirs(trg_dir, exist_ok=True)
                logging.info(f'Copying file {src_file} to {trg_file}')
                shutil.copyfile(src_file, trg_file)

    pd.DataFrame(list(args.__dict__.items()), columns=['Name', 'Value']).to_csv(os.path.join(data_path, "args.csv"),
                                                                                index=False)

    # Initialize ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers)
    # Initialize workers
    workers = [AlgoSampler.remote(actor, critic, env_fn, discount,
                                  data_path, high_rate_data, i) for i in range(num_workers)]
    # Sync policy into each worker
    actor_param_id  = ray.put(list(actor.parameters()))
    critic_param_id = ray.put(list(critic.parameters()))
    norm_id = ray.put([actor.welford_state_mean, actor.welford_state_mean_diff, \
                       actor.welford_state_n])
    for w in workers:
        w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
    # Start sampling
    # num_traj = 0
    num_steps = 0
    if mode == 'dynamic':
        sample_jobs = [w.sample_traj.remote(data_stamp=i,
                                            max_traj_len=max_traj_len,
                                            do_eval=use_deterministic_actor,
                                            terrain_name=TERRAIN_NAME,
                                            difficulty=DIFFICULTY) for i,w in enumerate(workers)]
    elif mode == 'kinematic':
        sample_jobs = [w.sample_traj_kinematic.remote(data_stamp=i,
                                                      max_traj_len=max_traj_len) for i,w in enumerate(workers)]
    print(f"Collecting data using {len(workers)} workers for {max_trajs} episodes... ...")

    for num_traj in tqdm.tqdm(range(max_trajs)):
        done_id, _ = ray.wait(sample_jobs, num_returns=1)
        buffer_worker, efficiency, worker_id = ray.get(done_id)[0]
        num_steps += len(buffer_worker)
        if mode == 'dynamic':
            sample_jobs[worker_id] = workers[worker_id].sample_traj.remote(data_stamp=num_traj,
                                                                           max_traj_len=max_traj_len,
                                                                           do_eval=use_deterministic_actor,
                                                                           terrain_name=TERRAIN_NAME,
                                                                           difficulty=DIFFICULTY)
        elif mode == 'kinematic':
            sample_jobs[worker_id] = workers[worker_id].sample_traj_kinematic.remote(data_stamp=num_traj,
                                                                                     max_traj_len=max_traj_len)
    map(ray.cancel, sample_jobs)
    print(f"Total number of trajectories: {num_traj+1}, total number of steps: {num_steps}")
    ray.shutdown()


@ray.remote
class AlgoSampler(AlgoWorker):
    def __init__(self, actor, critic, env_fn, discount, data_path, high_rata_data, worker_id: int):
        self.discount  = discount
        self.env    = env_fn()
        self.worker_id = worker_id
        self.data_path = data_path
        self.env.dynamics_randomization = True
        self.high_rata_data = high_rata_data
        if self.high_rata_data:
            self.env.add_high_rate_trackers()
        self.env.offline_collection = True

        AlgoWorker.__init__(self, actor, critic)

    def sample_traj(self,
                    data_stamp,
                    max_traj_len,
                    do_eval: bool = False,
                    update_normalization_param: bool=False,
                    save_video: bool = False,
                    terrain_name: str = None,
                    difficulty: float = None):
        """
        Function to sample experience and save the data to desired path.

        Args:
            max_traj_len: maximum trajectory length of an episode
            do_eval: if True, use deterministic policy, if False, use stochastic policy (default)
            update_normalization_param: if True, update normalization parameters
        """
        start_t = time.time()
        torch.set_num_threads(1)
        # Toggle models to eval mode
        self.actor.eval()
        self.critic.eval()
        memory = Buffer(self.discount)
        # Info dictionary to store data and push to buffer
        # NOTE: User can add any dict keys from env/sim
        info_dict = {}

        position_shift = 0.01
        angular_shift = 1.0
        fovy_shift = 1.0

        xshift = np.arange(3)
        start, end = .1856 - position_shift, .1856 + position_shift
        xshift = xshift / xshift.ptp() * (end - start) + start

        yshift = np.arange(3)
        start, end = 0.0 - position_shift, 0.0 + position_shift
        yshift = yshift / yshift.ptp() * (end - start) + start

        zshift = np.arange(3)
        start, end = 0.24 - position_shift, 0.24 + position_shift
        zshift = zshift / zshift.ptp() * (end - start) + start

        fovys = np.arange(3)
        start, end = 58 - fovy_shift, 58 + fovy_shift
        fovys = fovys / fovys.ptp() * (end - start) + start
        fovys = fovys.astype(int)

        tilts = np.arange(3)
        start, end = 30 - angular_shift, 30 + angular_shift
        tilts = tilts / tilts.ptp() * (end - start) + start

        with torch.no_grad():
            # Toggle terrain type if specified
            if terrain_name is not None:
                self.env.mix_terrain = False
                self.env.hfield_name = terrain_name
            state = torch.Tensor(self.env.reset(difficulty=difficulty))
            done = False
            value = 0
            traj_len = 0
            frames=[]

            x_idx = np.random.randint(len(xshift))
            y_idx = np.random.randint(len(yshift))
            z_idx = np.random.randint(len(zshift))
            fovy_idx = np.random.randint(len(fovys))
            tilt_idx = np.random.randint(len(tilts))

            # x_idx, y_idx, z_idx, fovy_idx, tilt_idx = 0, 0, 0, 0, 0

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state()
            if hasattr(self.critic, 'init_hidden_state'):
                self.critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                state = torch.Tensor(state)
                if hasattr(self.env, 'get_privilege_state') and self.critic.use_privilege_critic:
                    privilege_state = self.env.get_privilege_state()
                    critic_state = privilege_state
                    # info_dict['privilege_states'] = privilege_state
                else:
                    critic_state = state
                action = self.actor(state,
                                    deterministic=do_eval,
                                    update_normalization_param=update_normalization_param)

                # Get value based on the current critic state (s)
                if do_eval:
                    # If is evaluation, don't need critic value
                    value = 0.0
                else:
                    value = self.critic(torch.Tensor(critic_state)).numpy()

                # Step the environment
                next_state, reward, done, _ = self.env.step(action.numpy())
                memory.push(state.numpy(), action.numpy(), np.array([reward]), value)

                # Get additional information from env into buffer via info_dict
                if hasattr(self.env, 'depth_image'):
                    depth = self.env.sim.get_depth_image(
                        f'xshift-{x_idx}-yshift-{y_idx}-zshift-{z_idx}-fovy-{fovy_idx}-tilt-{tilt_idx}')

                    # Preprocessing depth image like in the realsense
                    hcrop = (depth.shape[1] - depth.shape[0]) // 2
                    depth = depth[:, hcrop:-hcrop]
                    depth = depth[20:, 10:-10]

                    info_dict['depth'] = cv2.resize(depth, (128, 128))

                    # info_dict['depth'] = self.env.sim.get_depth_image(f"height-{height_idx}")
                    info_dict['camera_xshift'] = xshift[x_idx]
                    info_dict['camera_yshift'] = yshift[y_idx]
                    info_dict['camera_zshift'] = zshift[z_idx]
                    info_dict['camera_fovy'] = fovys[fovy_idx]
                    info_dict['camera_tilt'] = tilts[tilt_idx]
                    frames.append(self.env.depth_image.astype(np.uint8))
                if hasattr(self.env, 'hfield_map'):
                    # NOTE: non-delayed and no noise added in z direction.
                    info_dict['hfield'] = self.env.hfield_ground_truth
                    # NOTE: for replay purposes, need to save the heightmap in robot frame/heading
                    # Can comment out if replay is good
                    info_dict['local_grid'] = self.env.local_grid_rotated
                    info_dict['hfield_big'] = self.env.hfield_map_big
                if hasattr(self.env, 'get_robot_height'):
                    info_dict['robot_height'] = self.env.get_robot_height()
                if hasattr(self.env, 'get_imu_state'):
                    info_dict['imu'] = self.env.get_imu_state(high_rate_data=self.high_rata_data)
                if hasattr(self.env, 'get_feet_state'):
                    info_dict['feet'] = self.env.get_feet_state(high_rate_data=self.high_rata_data)
                if hasattr(self.env, 'get_motor_state'):
                    info_dict['motor'] = self.env.get_motor_state() # no high rate data option
                info_dict['touch'] = np.array([self.env.feet_touch_sensor_tracker_avg['left-toe'],
                                               self.env.feet_touch_sensor_tracker_avg['right-toe']])
                info_dict['scuff'] = np.array([self.env.foot_scuffed])
                info_dict['cmd'] = np.array([self.env.x_velocity,
                                             self.env.y_velocity,
                                             self.env.turn_rate])

                # Fetch simulation data for replay purposes
                info_dict['qpos'] = self.env.sim.data.qpos.copy()
                info_dict['qvel'] = self.env.sim.data.qvel.copy()
                info_dict['done'] = done

                # Push additional info to buffer
                if info_dict:
                    memory.push_additional_info(info_dict)

                state = next_state
                if hasattr(self.env, 'get_privilege_state') and self.critic.use_privilege_critic:
                    critic_state = self.env.get_privilege_state()
                else:
                    critic_state = state
                traj_len += 1

            # Compute the terminal value based on the state (s')
            value = (not done) * self.critic(torch.Tensor(critic_state)).numpy()
            memory.end_trajectory(terminal_value=value)

        # Save video if needed
        if save_video and hasattr(self.env, 'depth_image'):
            filename = os.path.join(self.data_path, f"vid_{data_stamp}.mp4")
            mediapy.write_video(filename, frames, fps=self.env.default_policy_rate)
        # Convert buffer to dictionary and save to disk
        batch = memory.get_buffer_in_dict()
        # Save the entire hfield data is needed
        batch['entire_hfield'] = self.env.sim.hfield_data * self.env.sim.hfield_max_z
        filename = os.path.join(self.data_path, f"{data_stamp}.pkl")
        pickle.dump(batch, open(filename, 'wb'))
        return memory, traj_len / (time.time() - start_t), self.worker_id

    def sample_traj_kinematic(self, data_stamp, max_traj_len):
        """
        Function to sample experience and save the data to desired path.

        Args:
            max_traj_len: maximum trajectory length of an episode
        """
        start_t = time.time()
        torch.set_num_threads(1)
        memory = Buffer(self.discount)
        # Info dictionary to store data and push to buffer
        # NOTE: User can add any dict keys from env/sim
        info_dict = {}
        with torch.no_grad():
            state = torch.Tensor(self.env.reset())
            traj_len = 0
            frames=[]

            while traj_len < max_traj_len:
                # NOTE: Hard set base pose, and keep base flat
                base_position = np.random.uniform(low=[-5, -5, 0.5], high=[5.0, 5.0, 1.5])
                base_yaw = np.random.uniform(low=-np.pi, high=np.pi)
                base_orientation = euler2quat(z=base_yaw, x=0, y=0)
                self.env.sim.set_base_position(base_position)
                self.env.sim.set_base_orientation(base_orientation)
                self.env.update_depth_image()
                self.env.update_hfield_map()

                # Get additional information from env into buffer via info_dict
                if hasattr(self.env, 'depth_image'):
                    info_dict['depth'] = self.env.depth_image
                    frames.append(self.env.depth_image.astype(np.uint8))
                if hasattr(self.env, 'hfield_map'):
                    info_dict['hfield'] = self.env.hfield_map
                    # NOTE: for replay purposes, need to save the heightmap in robot frame/heading
                    # Can comment out if replay is good
                    info_dict['local_grid'] = self.env.local_grid_rotated
                if hasattr(self.env, 'get_robot_height'):
                    info_dict['robot_height'] = self.env.get_robot_height()
                if hasattr(self.env, 'get_feet_position'):
                    info_dict['feet_position'] = self.env.get_feet_position()
                if hasattr(self.env, 'get_proprioceptive_state'):
                    info_dict['proprioceptive_state'] = self.env.get_proprioceptive_state(include_joint=True)

                # Push additional info to buffer
                if info_dict:
                    memory.push_additional_info(info_dict)

                # Fetch simulation data for replay purposes
                info_dict['qpos'] = self.env.sim.data.qpos.copy()

                traj_len += 1

            # Compute the terminal value based on the state (s')
            memory.end_trajectory(terminal_value=0)

        # Convert buffer to dictionary and save to disk
        batch = memory.get_buffer_in_dict()
        # Save the entire hfield data is needed
        batch['entire_hfield'] = self.env.sim.hfield_data * self.env.sim.hfield_max_z
        filename = os.path.join(self.data_path, f"{data_stamp}.pkl")
        pickle.dump(batch, open(filename, 'wb'))
        return memory, traj_len / (time.time() - start_t), self.worker_id
