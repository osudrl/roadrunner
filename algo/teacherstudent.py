"""Proximal Policy Optimization (clip objective)."""
import argparse
import numpy as np
import os
import ray
import wandb
import torch
import torch.optim as optim
import glob
import pickle

from copy import deepcopy
from functools import reduce
from operator import add
import time
from torch.distributions import kl_divergence
from types import SimpleNamespace

from algo.util.sampling import AlgoSampler
from algo.util.worker import AlgoWorker
from util.mirror import mirror_tensor
from algo.util.teacherstudent import DistillOptim


class Distill(AlgoWorker):
    def __init__(self, actor, critic, env_fn, args):

        self.actor = actor
        self.critic = critic
        AlgoWorker.__init__(self, actor=actor, critic=critic)

        if actor.is_recurrent or critic.is_recurrent:
            self.recurrent = True
        else:
            self.recurrent = False

        self.env_fn        = env_fn
        self.discount      = args.discount
        self.entropy_coeff = args.entropy_coeff
        self.grad_clip     = args.grad_clip
        self.mirror        = args.mirror
        self.env           = env_fn()

        if not ray.is_initialized():
            if args.redis is not None:
                ray.init(redis_address=args.redis)
            else:
                ray.init(num_cpus=args.workers)

        mirror_dict = {}
        if self.mirror:
            mirror_dict['state_mirror_indices'] = self.env.get_observation_mirror_indices()
            mirror_dict['action_mirror_indices'] = self.env.get_action_mirror_indices()
        else:
            mirror_dict['state_mirror_indices'] = None
            mirror_dict['action_mirror_indices'] = None

        self.use_offline_data = False
        if args.offline_data_path != "":
            self.offline_data_path = args.offline_data_path
            assert os.path.exists(os.path.join(os.getcwd(), self.offline_data_path)),\
                f"Offline data path {self.offline_data_path} does not exist."
            files = glob.glob(self.offline_data_path+"/*.pkl")
            self.num_offine_data = len(files) - 1
            self.use_offline_data = True
            print(f"Loaded {self.num_offine_data} offline data.")

        self.workers = [AlgoSampler.remote(actor, critic, env_fn, args.discount, i) for i in \
                        range(args.workers)]

        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        num_gpus = 1 if torch.cuda.is_available() else 0
        self.optimizer = DistillOptim.options(num_gpus=num_gpus).remote(actor, critic, mirror_dict,
                                                                        self.use_offline_data,
                                                                        **vars(args))
        # self.optimizer = DistillOptim(actor, critic, mirror_dict, self.use_offline_data, **vars(args))
        # self.optimizer = DistillOptim.remote(actor, critic, mirror_dict,
        #                                                                 self.use_offline_data,
        #                                                                 **vars(args))

    def sample_offline_data(self, batch_size):
        idx = np.random.choice(self.num_offine_data, batch_size)
        files = [os.path.join(self.offline_data_path, f"{i}.pkl") for i in idx]
        batch = {}
        # Randomly select files based on batch size
        for f in sorted(files):
            data = pickle.load(open(f, "rb"))
            if not batch:
                for k,v in data.items():
                    if k == 'depth':
                        batch[k] = [torch.tensor(v)/255.0]
                    else:
                        batch[k] = [torch.tensor(v)]
            else:
                for k,v in data.items():
                    if k == 'depth':
                        batch[k] += [torch.tensor(v)/255.0]
                    else:
                        batch[k] += [torch.tensor(v)]
        # Pad the sequence
        # reshape into [num_steps_per_traj, num_trajs (batch size), num_states_per_step]
        for k,v in batch.items():
            batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=False).to(dtype=torch.float16)
        return batch

    def do_iteration(self,
                     num_steps,
                     max_traj_len,
                     epochs,
                     kl_thresh=0.02,
                     verbose=True,
                     batch_size=64):
        """
        Function to do a single iteration of PPO

        Args:
            max_traj_len (int): maximum trajectory length of an episode
            num_steps (int): number of steps to collect experience for
            epochs (int): optimzation epochs
            batch_size (int): optimzation batch size
            kl_thresh (float): threshold for max kl divergence
            verbose (bool): verbose logging output
        """
        # Output dicts for logging
        time_results = {}
        test_results = {}
        train_results = {}
        optimizer_results = {}

        # Sync up network parameters from main thread to each worker
        copy_start = time.time()
        actor_param_id  = ray.put(list(self.actor.parameters()))
        critic_param_id = ray.put(list(self.critic.parameters()))
        norm_id = ray.put([self.actor.welford_state_mean, self.actor.welford_state_mean_diff, \
                           self.actor.welford_state_n])
        for w in self.workers:
            w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
        if verbose:
            print("\t{:5.4f}s to sync up networks params to workers.".format(time.time() - copy_start))

        # Evaluation
        torch.set_num_threads(1)
        eval_buffers, _, _ = zip(*ray.get([w.sample_traj.remote(max_traj_len=max_traj_len, \
                                           do_eval=True) for w in self.workers]))
        eval_memory = reduce(add, eval_buffers)
        test_results["Return"] = np.mean(eval_memory.ep_returns)
        test_results["Episode Length"] = np.mean(eval_memory.ep_lens)

        # Sampling for optimization
        print("Sampling for optimization...")
        sampling_start = time.time()
        if self.use_offline_data:
            memory = self.sample_offline_data(batch_size)
            # print(f"Using offline data of size {batch_size}, time to sample {time.time() - sampling_start:3.2f}s")
        else:
            sampled_steps = 0
            avg_efficiency = 0
            num_traj = 0
            memory = None
            sample_jobs = [w.sample_traj.remote(max_traj_len) for w in self.workers]
            while sampled_steps < num_steps:
                done_id, remain_id = ray.wait(sample_jobs, num_returns = 1)
                buf, efficiency, work_id = ray.get(done_id)[0]
                if memory is None:
                    memory = buf
                else:
                    memory += buf
                num_traj += 1
                sampled_steps += len(buf)
                avg_efficiency += (efficiency - avg_efficiency) / num_traj
                sample_jobs[work_id] = self.workers[work_id].sample_traj.remote(max_traj_len)

            # Cancel leftover unneeded jobs
            map(ray.cancel, sample_jobs)

            # Collect timing results
            total_steps = len(memory)
            sampling_elapsed = time.time() - sampling_start
            sample_rate = (total_steps / 1000) / sampling_elapsed
            ideal_efficiency = avg_efficiency * len(self.workers)
            train_results["Return"] = np.mean(memory.ep_returns)
            train_results["Episode Length"] = np.mean(memory.ep_lens)
            time_results["Sample Time"] = sampling_elapsed
            time_results["Sample Rate"] = sample_rate
            time_results["Ideal Sample Rate"] = ideal_efficiency / 1000
            time_results["Overhead Loss"] = sampling_elapsed - total_steps / ideal_efficiency
            time_results["Timesteps per Iteration"] = total_steps
            if verbose:
                print(f"\t{sampling_elapsed:3.2f}s to collect {total_steps:6n} timesteps | " \
                    f"{sample_rate:3.2}k/s.")
                print(f"\tIdealized efficiency {time_results['Ideal Sample Rate']:3.2f}k/s \t | Time lost to " \
                    f"overhead {time_results['Overhead Loss']:.2f}s")

        # Optimization
        optim_start = time.time()
        # # Sync up network parameters from main thread to optimizer
        # self.optimizer.sync_policy(list(self.actor.parameters()),
        #                            list(self.critic.parameters()),
        #                            [self.actor.welford_state_mean, self.actor.welford_state_mean_diff, \
        #                             self.actor.welford_state_n])
        # losses = self.optimizer.optimize(memory,
        #                                 epochs=epochs,
        #                                 batch_size=batch_size,
        #                                 kl_thresh=kl_thresh,
        #                                 recurrent=self.recurrent,
        #                                 verbose=verbose)
        # del memory
        # actor_params, critic_params = self.optimizer.retrieve_parameters()
        self.optimizer.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
        losses = ray.get(self.optimizer.optimize.remote(memory,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    kl_thresh=kl_thresh,
                                                    recurrent=self.recurrent,
                                                    verbose=verbose))
        del memory
        actor_params, critic_params = ray.get(self.optimizer.retrieve_parameters.remote())
        a_loss, c_loss, m_loss, kls = losses
        time_results["Optimize Time"] = time.time() - optim_start
        optimizer_results["Actor Loss"] = np.mean(a_loss)
        optimizer_results["Critic Loss"] = np.mean(c_loss)
        optimizer_results["Mirror Loss"] = np.mean(m_loss)
        optimizer_results["KL"] = np.mean(kls)

        # Update network parameters for the main thread after optimization
        self.sync_policy(actor_params, critic_params)

        if verbose:
            print(f"\t{time_results['Optimize Time']:3.2f}s to update policy.")

        return {"Test": test_results, "Train": train_results, "Optimizer": optimizer_results, \
                "Time": time_results}

def add_algo_args(parser):
    default_values = {
        "prenormalize-steps" : (100, "Number of steps to use in prenormlization"),
        "prenorm"            : (False, "Whether to do prenormalization or not"),
        "update-norm"        : (False, "Update input normalization during training."),
        "num-steps"          : (2000, "Number of steps to sample each iteration"),
        "discount"           : (0.99, "Discount factor when calculating returns"),
        "a-lr"               : (1e-4, "Actor policy learning rate"),
        "c-lr"               : (1e-4, "Critic learning rate"),
        "eps"                : (1e-6, "Adam optimizer eps value"),
        "kl"                 : (0.02, "KL divergence threshold"),
        "entropy-coeff"      : (0.0, "Coefficient of entropy loss in optimization"),
        "clip"               : (0.2, "Log prob clamping value (1 +- clip)"),
        "grad-clip"          : (0.05, "Gradient clip value (maximum allowed gradient norm)"),
        "batch-size"         : (64, "Minibatch size to use during optimization"),
        "epochs"             : (3, "Number of epochs to optimize for each iteration"),
        "mirror"             : (1, "Mirror loss coefficient"),
        "workers"            : (4, "Number of parallel workers to use for sampling"),
        "redis"              : (None, "Ray redis address"),
        "previous"           : ("", "Previous model to bootstrap from"),
        "teacher-actor"      : ("", "Path to teacher actor"),
        "offline-data-path"  : ("", "Path to offline data"),
    }
    if isinstance(parser, argparse.ArgumentParser):
        ppo_group = parser.add_argument_group("PPO arguments")
        for arg, (default, help_str) in default_values.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                ppo_group.add_argument("--" + arg, default = default, action = "store_" + \
                                    str(not default).lower(), help = help_str)
            else:
                ppo_group.add_argument("--" + arg, default = default, type = type(default),
                                      help = help_str)
    elif isinstance(parser, (SimpleNamespace, argparse.Namespace)):
        for arg, (default, help_str) in default_values.items():
            arg = arg.replace("-", "_")
            if not hasattr(parser, arg):
                setattr(parser, arg, default)

    return parser


def run_experiment(parser, env_name):
    """
    Function to run a PPO experiment.

    Args:
        parser: argparse object
    """
    from algo.util.normalization import train_normalizer
    from algo.util.log import create_logger
    from util.env_factory import env_factory, add_env_parser
    from util.nn_factory import nn_factory, load_checkpoint, save_checkpoint, add_nn_parser
    from util.colors import FAIL, ENDC, WARNING

    import pickle
    import locale
    locale.setlocale(locale.LC_ALL, '')

    # Add parser arguments from env/nn/algo, then can finally parse args
    if isinstance(parser, argparse.ArgumentParser):
        add_env_parser(env_name, parser)
        add_nn_parser(parser)
        args = parser.parse_args()
        for arg_group in parser._action_groups:
            if arg_group.title == "PPO arguments":
                ppo_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                ppo_args = argparse.Namespace(**ppo_dict)
            elif arg_group.title == "Env arguments":
                env_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                env_args = argparse.Namespace(**env_dict)
            elif arg_group.title == "NN arguments":
                nn_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                nn_args = argparse.Namespace(**nn_dict)
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace):
        env_args = SimpleNamespace()
        nn_args = SimpleNamespace()
        ppo_args = parser
        add_env_parser(env_name, env_args)
        add_nn_parser(nn_args)
        args = parser
        for arg in args.__dict__:
            if hasattr(env_args, arg):
                setattr(env_args, arg, getattr(args, arg))
            if hasattr(nn_args, arg):
                setattr(nn_args, arg, getattr(args, arg))
            if hasattr(ppo_args, arg):
                setattr(ppo_args, arg, getattr(args, arg))
    else:
        raise RuntimeError(f"{FAIL}ppo.py run_experiment got invalid object type for arguments. " \
                           f"Input object should be either an ArgumentParser or a " \
                           f"SimpleNamespace.{ENDC}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create callable env_fn for parallelized envs
    env_fn = env_factory(env_name, env_args)
    env = env_fn()

    # Create nn modules. Append more in here and add_nn_parser() if needed
    nn_args.obs_dim = env.observation_size
    nn_args.action_dim = env.action_size
    policy, critic = nn_factory(args=nn_args, env=env)
    # print(policy)
    # print(critic)

    # Resolve any requirements of envs that have to be resolved with nn arch
    if nn_args.use_privilege_critic:
        assert hasattr(env, 'get_privilege_state'), "Env does not have get_privilege_state() method"

    # Guard any inter-dependencies between args
    if ppo_args.mirror:
        assert hasattr(env_fn(), 'get_observation_mirror_indices'), \
            "Env does not have get_observation_mirror_indices() method for mirror"
        assert hasattr(env_fn(), 'get_action_mirror_indices'), \
            "Env does not have get_action_mirror_indices() method for mirror"

    # Load model attributes if args.previous exists
    if hasattr(args, "previous") and args.previous != "":
        # Load and compare if any arg has been changed (add/remove/update), compared to prev_args
        prev_args_dict = pickle.load(open(os.path.join(args.previous, "experiment.pkl"), "rb"))
        for a in vars(args):
            if hasattr(prev_args_dict['all_args'], a):
                try:
                    if getattr(args, a) != getattr(prev_args_dict['all_args'], a):
                        print(f"{WARNING}Argument {a} is set to a new value {getattr(args, a)}, "
                              f"old one is {getattr(prev_args_dict['all_args'], a)}.{ENDC}")
                except:
                    if getattr(args, a).any() != getattr(prev_args_dict['all_args'], a).any():
                        print(f"{WARNING}Argument {a} is set to a new value {getattr(args, a)}, "
                              f"old one is {getattr(prev_args_dict['all_args'], a)}.{ENDC}")
            else:
                print(f"{WARNING}Added a new argument: {a}.{ENDC}")

        # Load nn modules from checkpoints
        actor_dict = torch.load(os.path.join(args.previous, "actor.pt"))
        critic_dict = torch.load(os.path.join(args.previous, "critic.pt"))
        load_checkpoint(model_dict=actor_dict, model=policy)
        load_checkpoint(model_dict=critic_dict, model=critic)

    # Prenormalization only on new training
    if args.prenorm and args.previous == "":
        print("Collecting normalization statistics with {} states...".format(args.prenormalize_steps))
        train_normalizer(env_fn, policy, args.prenormalize_steps, max_traj_len=args.traj_len, noise=1)
        critic.copy_normalizer_stats(policy)

    # Create actor/critic dict to include model_state_dict and other class attributes
    actor_dict = {'model_class_name': policy._get_name()}
    critic_dict = {'model_class_name': critic._get_name()}

    # Create a tensorboard logging object
    # before create logger files, double check that all args are updated in case any other of
    # ppo_args, env_args, nn_args changed
    for arg in ppo_args.__dict__:
        setattr(args, arg, getattr(ppo_args, arg))
    for arg in env_args.__dict__:
        setattr(args, arg, getattr(env_args, arg))
    for arg in nn_args.__dict__:
        setattr(args, arg, getattr(nn_args, arg))
    logger = create_logger(args, ppo_args, env_args, nn_args)
    args.save_actor_path = os.path.join(logger.dir, 'actor.pt')
    args.save_critic_path = os.path.join(logger.dir, 'critic.pt')
    args.save_path = logger.dir

    # Create algo class
    policy.train(True)
    critic.train(True)
    # If teacher actor exist, then create and overwrite ppo_args.teacher_actor
    if ppo_args.teacher_actor != "":
        teacher_args_dict = pickle.load(open(os.path.join(ppo_args.teacher_actor, "experiment.pkl"), "rb"))
        teacher_actor_checkpoint = torch.load(os.path.join(ppo_args.teacher_actor, 'actor.pt'), map_location='cpu')
        teacher_env = env_factory(teacher_args_dict['all_args'].env_name, teacher_args_dict['env_args'])()
        teacher_actor, _ = nn_factory(args=teacher_args_dict['nn_args'], env=teacher_env)
        load_checkpoint(model=teacher_actor, model_dict=teacher_actor_checkpoint)
        ppo_args.teacher_actor = teacher_actor

    algo = Distill(policy, critic, env_fn, ppo_args)
    print("Proximal Policy Optimization:")
    for key, val in sorted(args.__dict__.items()):
        print(f"\t{key} = {val}")
    # Count nume of params of the trainable params
    num_params = 0
    for param in policy.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print(f"Number of trainable parameters: {num_params}")

    itr = 0
    total_timesteps = 0
    best_reward = None
    past500_reward = -1
    while total_timesteps < args.timesteps:
        ret = algo.do_iteration(args.num_steps,
                                args.traj_len,
                                args.epochs,
                                batch_size=args.batch_size,
                                kl_thresh=args.kl)

        print(f"iter {itr:4d} | return: {ret['Test']['Return']:5.2f} | " \
              f"Actor loss {ret['Optimizer']['Actor Loss']:5.4f} | " \
              f"Critic loss {ret['Optimizer']['Critic Loss']:5.4f} | ", end='\n')
        total_timesteps += ret["Time"]["Timesteps per Iteration"]
        print(f"\tTotal timesteps so far {total_timesteps:n}")

        if algo.use_offline_data:
            save_checkpoint(algo.actor, actor_dict, args.save_actor_path)
            save_checkpoint(algo.critic, critic_dict, args.save_critic_path)
        else:
            # Saving checkpoints for best reward
            if best_reward is None or ret["Test"]["Return"] > best_reward:
                print(f"\tbest policy so far! saving checkpoint to {args.save_actor_path}")
                best_reward = ret["Test"]["Return"]
                save_checkpoint(algo.actor, actor_dict, args.save_actor_path)
                save_checkpoint(algo.critic, critic_dict, args.save_critic_path)

            # Intermitent saving
            if itr % 500 == 0:
                past500_reward = -1
            if ret["Test"]["Return"] > past500_reward:
                past500_reward = ret["Test"]["Return"]
                save_checkpoint(algo.actor, actor_dict, args.save_actor_path[:-3] + "_past500.pt")
                save_checkpoint(algo.critic, critic_dict, args.save_critic_path[:-3] + "_past500.pt")

        if logger is not None:
            for key, val in ret.items():
                for k, v in val.items():
                    logger.add_scalar(f"{key}/{k}", v, itr)
            logger.add_scalar("Time/Total Timesteps", total_timesteps, itr)

        itr += 1
    print(f"Finished ({total_timesteps} of {args.timesteps}).")

    if args.wandb:
        wandb.finish()
