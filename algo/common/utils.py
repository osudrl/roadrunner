import glob
import logging
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import ray
import torch
import wandb
from torch.nn.utils.rnn import pad_sequence

from algo.common.normalization import RewardScaling
from algo.common.replaybuffer import ReplayBuffer

logging.basicConfig(level=logging.INFO)


def init_logger(args, agent):
    iterations = 0
    total_steps = 0
    trajectory_count = 0

    # 4 ways to initialize wandb
    # 1. Parent run is not given, previous run is not given -> Start a new run from scratch
    # 2. Parent run is given, previous run is not given -> Create a new run resumed but detached from parent
    # 3. Parent run is not given, previous run is given -> Resume previous run attached to same parent
    # 4. Parent run is given, previous run is given -> Start a new run from previous run attached to same parent

    if args.parent_run is None and args.previous_run is None:
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            mode=args.wandb_mode,
            config={**args.__dict__, 'parent_run': args.run_name},
            id=args.run_name.replace(':', '_'),
        )
    elif args.previous_run is None:
        wandb.login()

        run = wandb.Api().run(os.path.join(args.project_name, args.parent_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {args.parent_run}')

        if args.previous_checkpoint:
            checkpoint_name = f'checkpoints/checkpoint-{args.parent_run}-{args.previous_checkpoint}.pt'
        else:
            checkpoint_name = f'checkpoints/checkpoint-{args.parent_run}.pt'

        run.file(name=checkpoint_name).download(replace=True)

        with open(checkpoint_name, 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**args.__dict__, 'parent_run': args.parent_run},
            id=args.run_name.replace(':', '_'),
        )

        # Since we start a new run detached from parent, we don't load run state
        total_steps = 0
        trajectory_count = 0
        iterations = 0
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
    elif args.parent_run is None:
        run_ = wandb.Api().run(os.path.join(args.project_name, args.previous_run.replace(':', '_')))

        previous_config = run_.config

        all_config = previous_config.get(['previous_configs'], []) + [previous_config]

        run = wandb.init(
            project=args.project_name,
            resume='allow',
            config={**args.__dict__, 'parent_run': previous_config['run_name'], 'previous_configs': all_config},
            id=args.previous_run.replace(':', '_'),
        )

        if run.resumed:
            logging.info(f'Checkpoint loaded from: {args.previous_run}')

            if args.previous_checkpoint:
                checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}-{args.previous_checkpoint}.pt'
            else:
                checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}.pt'

            run_.file(name=checkpoint_name).download(replace=True)

            with open(checkpoint_name, 'rb') as r:
                checkpoint = torch.load(r, map_location=agent.device)

            logging.info(f'Resuming from the run: {run.name} ({run.id})')
            total_steps = checkpoint['total_steps']
            trajectory_count = checkpoint['trajectory_count']
            iterations = checkpoint.get('iterations', 0)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        else:
            logging.error(f'Run: {args.previous_run} did not resume')
            raise Exception(f'Run: {args.previous_run} did not resume')
    else:
        wandb.login()

        run = wandb.Api().run(os.path.join(args.project_name, args.previous_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {args.previous_run}')

        if args.previous_checkpoint:
            checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}-{args.previous_checkpoint}.pt'
        else:
            checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}.pt'

        run.file(name=checkpoint_name).download(replace=True)

        with open(checkpoint_name, 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**args.__dict__, 'parent_run': args.parent_run},
            id=args.run_name.replace(':', '_'),
        )

        total_steps = checkpoint['total_steps']
        trajectory_count = checkpoint['trajectory_count']
        iterations = checkpoint.get('iterations', 0)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    # Save files to wandb
    cwd = os.getcwd()

    exclude = {'checkpoints', 'saved_models', 'wandb', '.idea', '.git', 'pretrained_models', 'trained_models',
               'offline_data', 'old_files'}

    inc_exts = ['*.py', '*.yaml', '*.yml', '*.json', '*.xml', '*.sh']

    dirs = [d for d in os.listdir(cwd) if d not in exclude and os.path.isdir(os.path.join(cwd, d))]

    # Process files in subdirectories recursively
    for d in dirs:

        base_paths = [os.path.join(cwd, d, '**', ext) for ext in inc_exts]

        for base_path in base_paths:
            for file in glob.glob(base_path, recursive=True):
                file_path = os.path.relpath(file, start=cwd)
                run.save(file_path, policy='now')

    # Process files in current directory
    base_paths = [os.path.join(cwd, ext) for ext in inc_exts]
    for base_path in base_paths:
        for file in glob.glob(base_path):
            file_path = os.path.relpath(file, start=cwd)
            run.save(file_path, policy='now')

    return run, iterations, total_steps, trajectory_count


def mirror_tensor(t, indices):
    sign = torch.sign(indices)
    indices = indices.long().abs()
    mirror_t = sign * torch.index_select(t, -1, indices)
    return mirror_t


def load_legacy_actor(args, device, model_fn):
    model = model_fn(args)

    checkpoint = './pretrained_models/LocomotionEnv/cassie-LocomotionEnv/10-27-17-03/actor.pt'

    checkpoint = torch.load(checkpoint, map_location=device)['model_state_dict']

    model.load_state_dict(checkpoint, strict=False)

    return model


def load_actor(args, device, model_fn):
    model = model_fn(args)
    model.to(device)

    wandb.login()

    run = wandb.Api().run(os.path.join(args.project_name, args.run_name.replace(':', '_')))

    logging.info(f'Checkpoint loading from: {args.run_name}')

    if args.model_checkpoint == 'latest':
        checkpoint_path = f'checkpoints/checkpoint-{args.run_name}.pt'

        run.file(name=checkpoint_path).download(replace=args.redownload_checkpoint, exist_ok=True)

        with open(checkpoint_path, 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        model.load_state_dict(checkpoint['actor_state_dict'], strict=False)

        logging.info(
            f'Loaded checkpoint: {checkpoint.get("epoch", 0)}, {checkpoint.get("total_steps", 0), {checkpoint.get("trajectory_count", 0)} }')
    else:
        if args.model_checkpoint == 'best':
            model_path = f'saved_models/agent-{args.run_name}.pth'
        else:
            model_path = f'checkpoints/checkpoint-{args.run_name}-{args.model_checkpoint}.pt'

        run.file(name=model_path).download(replace=args.redownload_checkpoint, exist_ok=True)

        with open(model_path, 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        if args.model_checkpoint == 'best':
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint['actor_state_dict'], strict=False)

        logging.info(f'Loaded model: {args.model_checkpoint}')

    model.eval()

    wandb.finish()

    return model


def load_tarsus_predictor(args, device, model_fn):
    model = model_fn(args)
    model.to(device)

    wandb.login()

    run = wandb.Api().run(os.path.join(args.project_name, args.run_name.replace(':', '_')))

    logging.info(f'Checkpoint loading from: {args.run_name}')

    if args.model_checkpoint == 'latest':
        checkpoint_path = f'checkpoints/checkpoint-{args.run_name}.pt'

        run.file(name=checkpoint_path).download(replace=args.redownload_checkpoint, exist_ok=True)

        with open(checkpoint_path, 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info(
            f'Loaded checkpoint: {checkpoint.get("epoch", 0)}, {checkpoint.get("total_steps", 0), {checkpoint.get("trajectory_count", 0)} }')
    else:
        if args.model_checkpoint == 'best':
            model_path = f'saved_models/model-{args.run_name}.pth'
        else:
            model_path = f'saved_models/model-{args.run_name}-{args.model_checkpoint}.pth'

        run.file(name=model_path).download(replace=args.redownload_checkpoint, exist_ok=True)

        with open(model_path, 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        model.load_state_dict(checkpoint)

        logging.info(f'Loaded model: {args.model_checkpoint}')

    model.eval()

    wandb.finish()

    return model


@ray.remote
class Dispatcher:
    def __init__(self):
        self.collecting = defaultdict(lambda: False)
        self.evaluating = False

    def is_collecting(self, num_cassie):
        return self.collecting[num_cassie]

    def is_evaluating(self):
        return self.evaluating

    def set_collecting(self, num_cassie, val):
        self.collecting[num_cassie] = val

    def set_evaluating(self, val):
        self.evaluating = val


def get_device(args):
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
        return torch.device("cpu"), torch.device(args.device)
    elif torch.cuda.is_available():
        return torch.device("cpu"), torch.device(args.device)
    else:
        return torch.device("cpu"), torch.device("cpu")


def update_model(model, new_model_params):
    for p, new_p in zip(model.parameters(), new_model_params):
        p.data.copy_(new_p)


def get_sliding_windows(ep_lens, buffer_size):
    active_seq = torch.ones(buffer_size, dtype=torch.bool).split(ep_lens)
    windows = torch.arange(buffer_size).split(ep_lens)

    active_seq = pad_sequence(active_seq, batch_first=True, padding_value=False)
    windows = pad_sequence(windows, batch_first=True, padding_value=0)

    return windows, active_seq


@ray.remote
class Worker:
    def __init__(self, env, actor, args, device, worker_id):

        # Normalize probability by number of cassies
        num_cassie_prob_norm = args.num_cassie_prob * np.arange(1, len(args.num_cassie_prob) + 1)
        num_cassie_prob_norm /= num_cassie_prob_norm.sum()

        num_cassie = np.random.choice(np.arange(1, len(num_cassie_prob_norm) + 1), p=num_cassie_prob_norm)

        self.env = env(num_cassie)

        self.env.eval(False)

        # self.dispatcher = dispatcher
        if args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        self.args = args
        self.device = device
        self.actor = deepcopy(actor).to(device)
        self.actor.eval()
        self.worker_id = worker_id

        if self.env._merge_states:
            self.num_cassie = 1
        else:
            self.num_cassie = self.env.num_cassie

    def get_num_cassie(self):
        return self.num_cassie

    def get_buffer_size(self):
        return self.args.time_horizon

    def update_model(self, new_actor_params):
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

    def get_action(self, s, deterministic=False):
        if deterministic:
            a, _ = self.actor.forward(s)
            # Get output from last observation

            return a
        else:
            dist = self.actor.pdf(s)
            a = dist.sample()
            # a: [1, seq_len, action_dim]

            return a

    def collect(self, max_ep_len, render=False):
        torch.set_num_threads(1)

        with torch.inference_mode():

            replay_buffer = ReplayBuffer(self.args, buffer_size=max_ep_len, num_cassie=self.num_cassie)

            episode_reward = np.zeros(self.num_cassie)

            s = self.env.reset()

            if render:
                self.env.sim.viewer_init()
                self.env.sim.viewer_render()

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state(self.device, batch_size=self.num_cassie)

            if self.args.use_reward_scaling:
                self.reward_scaling.reset()

            active = torch.ones(self.num_cassie, dtype=torch.bool, device=self.device)

            for step in range(max_ep_len):
                # Numpy to tensor
                for k in s.keys():
                    s[k] = torch.tensor(s[k], dtype=torch.float32).unsqueeze(1)

                a = self.get_action(s, deterministic=False).squeeze(1)

                s_, r, done, _ = self.env.step(a.numpy())
                done = torch.tensor(done, dtype=torch.bool, device=self.device)

                episode_reward += r

                if render:
                    self.env.sim.viewer_render()

                dw = done & (step != self.args.time_horizon - 1)

                if self.args.use_reward_scaling:
                    r = self.reward_scaling(r)

                r = torch.tensor(r, dtype=torch.float32, device=self.device) * active

                # Remove seq dimension from state
                for k in s.keys():
                    s[k] = s[k].squeeze(1)

                replay_buffer.store_transition(s, a, r, dw, active)

                active &= ~done

                s = s_

                if done.all():
                    break

                # if not ray.get(self.dispatcher.is_collecting.remote(self.get_num_cassie())):
                #     return replay_buffer, episode_reward, step + 1, self.worker_id, self.env.num_cassie

            for k in s.keys():
                s[k] = torch.tensor(s[k], dtype=torch.float32)

            replay_buffer.store_last_state(s)

            if render and hasattr(self.env.sim, 'renderer'):
                if self.env.sim.renderer is not None:
                    print("re-init non-primary screen renderer")
                    self.env.sim.renderer.close()
                    self.env.sim.init_renderer(offscreen=self.env.offscreen,
                                               width=self.env.depth_image_dim[0], height=self.env.depth_image_dim[1])

            return replay_buffer, episode_reward, step + 1, self.worker_id, self.num_cassie

    def evaluate(self, max_ep_len, render=False):
        torch.set_num_threads(1)

        with torch.inference_mode():
            s = self.env.reset()

            if render:
                self.env.sim.viewer_init()
                self.env.sim.viewer_render()

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state(self.device, batch_size=self.num_cassie)

            episode_reward = 0

            active = np.ones(self.num_cassie, dtype=bool)

            for step in range(max_ep_len):
                # Numpy to tensor
                for k in s.keys():
                    s[k] = torch.tensor(s[k], dtype=torch.float32).unsqueeze(1)

                a = self.get_action(s, deterministic=True).squeeze(1)

                s, r, done, _ = self.env.step(a.numpy())

                if render:
                    self.env.sim.viewer_render()

                episode_reward += r * active

                active &= ~done

                if done.all():
                    break

                # if not ray.get(self.dispatcher.is_evaluating.remote()):
                #     break

            if render and hasattr(self.env.sim, 'renderer'):
                if self.env.sim.renderer is not None:
                    print("re-init non-primary screen renderer")
                    self.env.sim.renderer.close()
                    self.env.sim.init_renderer(offscreen=self.env.offscreen,
                                               width=self.env.depth_image_dim[0], height=self.env.depth_image_dim[1])

            return None, episode_reward, step + 1, self.worker_id, self.num_cassie
