"""Proximal Policy Optimization (clip objective)."""
import argparse
import numpy as np
import os
import ray
import wandb
import torch
import torch.optim as optim

from copy import deepcopy
from time import time, monotonic
from torch.distributions import kl_divergence
from types import SimpleNamespace

from algo.util.sampling import AlgoSampler
from algo.util.worker import AlgoWorker
from algo.util.sampling import Buffer
from util.mirror import mirror_tensor

class PPOOptim(AlgoWorker):
    """
        Worker for doing optimization step of PPO.

        Args:
            actor: actor pytorch network
            critic: critic pytorch network
            a_lr (float): actor learning rate
            c_lr (float): critic learning rate
            eps (float): adam epsilon
            entropy_coeff (float): entropy regularizaiton coefficient
            grad_clip (float): Value to clip gradients at.
            mirror (int or float): scalar multiple of mirror loss
            clip (float): Clipping parameter for PPO surrogate loss

        Attributes:
            actor: actor pytorch network
            critic: critic pytorch network
    """
    def __init__(self,
                 actor,
                 critic,
                 mirror_dict,
                 a_lr=1e-4,
                 c_lr=1e-4,
                 eps=1e-6,
                 entropy_coeff=0,
                 grad_clip=0.01,
                 mirror=0,
                 clip=0.2,
                 save_path=None,
                 **kwargs):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using cuda:1 GPU for optimization.")
                self.device = torch.device("cuda:1")
            else:
                print(f"Using cuda:0 GPU for optimization.")
                self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        AlgoWorker.__init__(self, actor=actor, critic=critic, device=self.device)
        self.old_actor = deepcopy(actor).to(self.device)
        self.actor_optim   = optim.Adam(self.actor.parameters(), lr=a_lr, eps=eps)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=c_lr, eps=eps)
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.mirror    = mirror
        self.clip = clip
        self.save_path = save_path
        # Unpack mirror dict
        self.state_mirror_indices  = mirror_dict['state_mirror_indices']
        self.action_mirror_indices = mirror_dict['action_mirror_indices']
        if self.state_mirror_indices is not None:
            self.state_mirror_indices = torch.tensor(self.state_mirror_indices).to(self.device)
        if self.action_mirror_indices is not None:
            self.action_mirror_indices = torch.tensor(self.action_mirror_indices).to(self.device)

        if kwargs['backprop_workers'] <= 0:
            self.backprop_cpu_count = self._auto_optimize_backprop(kwargs)
        else:
            self.backprop_cpu_count = kwargs['backprop_workers']
        torch.set_num_threads(self.backprop_cpu_count)

    def _auto_optimize_backprop(self, kwargs):
        """
        Auto detects the fastest settings for backprop on the current machine
        """
        print("Auto optimizing backprop settings...")

        # store models to reset after
        actor = self.actor
        self.actor = deepcopy(self.actor)
        critic = self.critic
        self.critic = deepcopy(self.critic)

        # create buffer with random data
        memory = Buffer(kwargs['discount'])
        info = {}
        while len(memory) < kwargs['num_steps']:
            for _ in range(300): # TODO maybe randomize this or get it live
                fake_state = np.random.random((kwargs['obs_dim']))
                fake_privilege_state = np.random.random((kwargs['privilege_obs_size']))
                fake_action = np.random.random((kwargs['action_dim']))
                fake_reward = np.random.random((1))
                fake_value = np.random.random((1))
                memory.push(fake_state, fake_action, fake_reward, fake_value)
                info['privilege_states'] = fake_privilege_state
                memory.push_additional_info(info)
            memory.end_trajectory()

        # run backprop for a few cpu counts and return fastest setting
        times = []
        num_cpus = [1,2,4,6,8,10,12,14,16,18,20]
        for n in num_cpus:
            torch.set_num_threads(n)
            start = monotonic()
            for _ in range(3):
                self.optimize(
                    memory,
                    actor_epochs=kwargs['actor_epochs'],
                    critic_epochs=kwargs['critic_epochs'],
                    batch_size=kwargs['batch_size'],
                    kl_thresh=99999999,
                    recurrent=kwargs['recurrent'],
                    verbose=False
                )
            end = monotonic()
            times.append(end-start)

        optimal_cpu_count = num_cpus[times.index(min(times))]

        print("Backprop times: ")
        for (n,t) in zip(num_cpus, times):
            print(f"{n} cpus: {t:.2f} s")
        print(f"Optimal CPU cores for backprop on this machine is: {optimal_cpu_count}")

        # reset models
        self.actor = actor
        self.critic = critic
        return optimal_cpu_count

    def optimize(self,
                 memory,
                 actor_epochs=4,
                 critic_epochs=4,
                 batch_size=32,
                 kl_thresh=0.02,
                 recurrent=False,
                 verbose=False):
        """
        Does a single optimization step given buffer info

        Args:
            memory (Buffer): Buffer object of rollouts from experience collection phase of PPO
            epochs (int): optimization epochs
            batch_size (int): optimization batch size
            kl_thresh (float): threshold for max kl divergence
            recurrent (bool): Buffer samples for recurrent policy or not
            state_mirror_indices(list): environment-specific list of mirroring information
            state_mirror_indices(list): environment-specific list of mirroring information
            verbose (bool): verbose logger output
        """
        self.old_actor.load_state_dict(self.actor.state_dict())
        kls, a_loss, c_loss, m_loss = [], [], [], []
        done = False
        actor_epoch_idx = 0
        critic_epoch_idx = 0
        for epoch in range(max(actor_epochs, critic_epochs)):
            epoch_start = time()
            for batch in memory.sample(batch_size=batch_size,
                                       recurrent=recurrent):
                # Patch for using encoder predictor
                # self.actor.encoder_patch.reset_hidden_state(batch_size=batch_size)
                # self.old_actor.encoder_patch.reset_hidden_state(batch_size=batch_size)
                # Transfer batch to GPU
                if self.device != 'cpu':
                    for k,v in batch.items():
                        batch[k] = v.to(device=self.device, non_blocking=True)
                if actor_epoch_idx < actor_epochs:
                    kl, losses = self._update_policy(batch['states'],
                                                    batch['privilege_states'],
                                                    batch['actions'],
                                                    batch['returns'],
                                                    batch['advantages'],
                                                    batch['mask'])
                    actor_epoch_idx += 1
                if critic_epoch_idx < critic_epochs:
                    critic_losses = self._update_critic(batch['states'],
                                                        batch['privilege_states'],
                                                        batch['actions'],
                                                        batch['returns'],
                                                        batch['mask'])
                    critic_epoch_idx += 1

                kls    += [kl]
                a_loss += [losses[0]]
                c_loss += [critic_losses]
                m_loss += [losses[2]]

                if max(kls) > kl_thresh:
                    print(f"\t\tbatch had kl of {max(kls)} (threshold {kl_thresh}), stopping " \
                          f"optimization early.")
                    done = True
                    break

            if verbose:
                print(f"\t\tepoch {epoch+1:2d} in {(time() - epoch_start):3.2f}s, " \
                      f"kl {np.mean(kls):6.5f}, actor loss {np.mean(a_loss):6.3f}, " \
                      f"critic loss {np.mean(c_loss):6.3f}")

            if done:
                break
        return np.mean(a_loss), np.mean(c_loss), np.mean(m_loss), np.mean(kls)

    def retrieve_parameters(self):
        """
        Function to return parameters for optimizer copies of actor and critic
        """
        return list(self.actor.parameters()), list(self.critic.parameters())

    def _update_policy(self,
                       states,
                       privilege_states,
                       actions,
                       returns,
                       advantages,
                       mask):
        with torch.no_grad():
            self.old_actor.eval()
            old_pdf       = self.old_actor.pdf(states)
            old_log_probs = old_pdf.log_prob(actions).sum(-1, keepdim=True)

        # active_sum is the summation of trajectory length (non-padded) over episodes in a batch
        active_sum = mask.sum()

        self.actor.train(True)
        # get new action distribution and log probabilities
        pdf       = self.actor.pdf(states)
        log_probs = pdf.log_prob(actions).sum(-1, keepdim=True)

        ratio      = ((log_probs - old_log_probs) * mask).exp()
        cpi_loss   = ratio * advantages
        clip_loss  = ratio.clamp(1.0 - self.clip, 1 + self.clip) * advantages
        actor_loss = -(torch.min(cpi_loss, clip_loss) * mask).sum() / active_sum

        # Mean is computed by averaging critic loss over non-padded trajectory
        # critic_states = privilege_states if self.critic.use_privilege_critic else states
        # critic_loss = 0.5 * ((returns - self.critic(critic_states)) * mask).pow(2).sum() / active_sum

        # The dimension of pdf.entropy() is (num_steps_per_traj, num_trajs, action_dim), so to apply mask,
        # we need to average over action_dim first.
        entropy_penalty = -(self.entropy_coeff * pdf.entropy().mean(-1, keepdim=True) * mask).sum() / active_sum

        # Mirror operations and loss
        if self.mirror and self.state_mirror_indices is not None and self.action_mirror_indices is not None:
            mirrored_actions = self._mirror(states)
            unmirrored_actions = pdf.mean
            # The dimension of mirrored_actions is (num_steps_per_traj, num_trajs, action_dim), so to apply mask,
            # we need to average over action_dim first.
            mirror_loss = self.mirror * 4 * (unmirrored_actions - mirrored_actions).pow(2)\
                .mean(-1, keepdim=True).sum() / active_sum
        else:
            mirror_loss = torch.zeros(1).to(device=self.device)

        if not (torch.isfinite(states).all() and torch.isfinite(actions).all() \
                and torch.isfinite(returns).all() and torch.isfinite(advantages).all() \
                and torch.isfinite(log_probs).all() and torch.isfinite(old_log_probs).all() \
                and torch.isfinite(actor_loss).all() \
                and torch.isfinite(mirror_loss).all()):
            torch.save({"states": states,
                        "actions": actions,
                        "returns": returns,
                        "advantages": advantages,
                        "active_sum": active_sum,
                        "mask": mask,
                        "log_probs": log_probs,
                        "old_log_probs": old_log_probs,
                        "actor_loss": actor_loss,
                        # "critic_loss": critic_loss,
                        "mirror_loss": mirror_loss,
                        "pdf": pdf,
                        "old pdf": old_pdf}, os.path.join(self.save_path, "training_error.pt"))
            raise RuntimeError(f"Optimization experiences non-finite values, please check locally"
                               f" saved file at training_error.pt for further diagonose.")

        self.actor_optim.zero_grad()
        # self.critic_optim.zero_grad()

        (actor_loss + entropy_penalty + mirror_loss).backward()
        # critic_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_optim.step()
        # self.critic_optim.step()

        with torch.no_grad():
          # The dimension of pdf is (num_steps_per_traj, num_trajs, action_dim), so to apply mask,
          # we need to average over action_dim first.
          kl = (kl_divergence(pdf, old_pdf) * mask).mean(-1, keepdim=True).sum() / active_sum

          return kl.item(), ((actor_loss + entropy_penalty).item(), 0, mirror_loss.item())

    def _update_critic(self,
                       states,
                       privilege_states,
                       actions,
                       returns,
                       mask):

        # active_sum is the summation of trajectory length (non-padded) over episodes in a batch
        active_sum = mask.sum()

        # Mean is computed by averaging critic loss over non-padded trajectory
        critic_states = privilege_states if self.critic.use_privilege_critic else states
        critic_loss = 0.5 * ((returns - self.critic(critic_states)) * mask).pow(2).sum() / active_sum

        if not (torch.isfinite(states).all() and torch.isfinite(actions).all() \
                and torch.isfinite(returns).all()):
            torch.save({"states": states,
                        "actions": actions,
                        "returns": returns,
                        "active_sum": active_sum,
                        "mask": mask,
                        "critic_loss": critic_loss}, os.path.join(self.save_path, "training_error_critic.pt"))
            raise RuntimeError(f"Optimization experiences non-finite values, please check locally"
                               f" saved file at training_error.pt for further diagonose.")

        self.critic_optim.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
        self.critic_optim.step()

        return critic_loss.item()

    def _mirror(self, states):
        """Mirror states and actions to get mirrored states and actions. Indices are lists.

        Returns:
            torch.tensor: mirrored states and mirrored actions
        """
        mirrored_states = mirror_tensor(states, self.state_mirror_indices)
        with torch.no_grad():
            mirrored_actions = mirror_tensor(self.actor(mirrored_states), self.action_mirror_indices)
        return mirrored_actions

class PPO(AlgoWorker):
    """
    Worker for sampling experience for PPO

    Args:
        actor: actor pytorch network
        critic: critic pytorch network
        env_fn: environment constructor function
        args: argparse namespace

    Attributes:
        actor: actor pytorch network
        critic: critic pytorch network
        recurrent: recurrent policies or not
        env_fn: environment constructor function
        discount: discount factor
        entropy_coeff: entropy regularization coeff

        grad_clip: value to clip gradients at
        mirror: scalar multiple of mirror loss. No mirror loss if this equals 0
        env: instance of environment
        state_mirror_indices (func): environment-specific function for mirroring state for mirror loss
        action_mirror_indices (func): environment-specific function for mirroring action for mirror loss
        workers (list): list of ray worker IDs for sampling experience
        optim: ray woker ID for optimizing

    """
    def __init__(self, actor, critic, env_fn, args):

        self.actor = actor
        self.critic = critic
        AlgoWorker.__init__(self, actor=actor, critic=critic)

        if actor.is_recurrent or critic.is_recurrent:
            self.recurrent = True
        else:
            self.recurrent = False
        args.recurrent = self.recurrent

        self.env_fn        = env_fn
        self.discount      = args.discount
        self.gae_lambda    = args.gae_lambda
        self.entropy_coeff = args.entropy_coeff
        self.grad_clip     = args.grad_clip
        self.mirror        = args.mirror
        self.env           = env_fn()
        self.eval_freq     = args.eval_freq

        args.privilege_obs_size = self.env.privilege_obs_size

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

        self.workers = [AlgoSampler.remote(actor, critic, env_fn, args.discount, i) for i in \
                        range(args.workers)]
        self.optimizer = PPOOptim(actor, critic, mirror_dict, **vars(args))

    def do_iteration(self,
                     num_steps,
                     max_traj_len,
                     num_eval_eps,
                     actor_epochs,
                     critic_epochs,
                     itr,
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
        copy_start = time()
        actor_param_id  = ray.put(list(self.actor.parameters()))
        critic_param_id = ray.put(list(self.critic.parameters()))
        norm_id = ray.put([self.actor.welford_state_mean, self.actor.welford_state_mean_diff, \
                           self.actor.welford_state_n])
        for w in self.workers:
            w.sync_policy.remote(actor_param_id, critic_param_id, input_norm=norm_id)
        if verbose:
            print("\t{:5.4f}s to sync up networks params to workers.".format(time() - copy_start))

        sampling_start = time()

        # Sampling for evaluation
        if itr % self.eval_freq == 0:
            # start only num_eval_eps eval workers asynchronously
            eval_jobs = [w.sample_traj.remote(max_traj_len=max_traj_len, do_eval=True) for w in self.workers[:num_eval_eps]]
            eval_memory = None
        else:
            num_eval_eps = 0
            eval_jobs = []

        # Sampling for optimization
        sampled_steps = 0
        avg_efficiency = 0
        num_traj = 0
        sample_memory = None
        sample_jobs = [w.sample_traj.remote(max_traj_len) for w in self.workers[num_eval_eps:]]
        jobs = eval_jobs + sample_jobs
        while sampled_steps < num_steps:
            done_id, remain_id = ray.wait(jobs, num_returns = 1)
            buf, efficiency, work_id = ray.get(done_id)[0]
            if done_id[0] in eval_jobs:
                if eval_memory is None:
                    eval_memory = buf
                else:
                    eval_memory += buf
                eval_jobs.remove(done_id[0]) # Remove this job from the list, recycle worker for sampling
            else:
                if sample_memory is None:
                    sample_memory = buf
                else:
                    sample_memory += buf
                num_traj += 1
                sampled_steps += len(buf)
                avg_efficiency += (efficiency - avg_efficiency) / num_traj
            jobs[work_id] = self.workers[work_id].sample_traj.remote(max_traj_len)

        map(ray.cancel, sample_jobs) # Cancel leftover unneeded jobs

        if itr % self.eval_freq == 0:
            # Collect eval results
            test_results["Return"] = np.mean(eval_memory.ep_returns)
            test_results["Episode Length"] = np.mean(eval_memory.ep_lens)

        # Collect timing results
        total_steps = len(sample_memory)
        sampling_elapsed = time() - sampling_start
        sample_rate = (total_steps / 1000) / sampling_elapsed
        ideal_efficiency = avg_efficiency * len(self.workers)
        train_results["Return"] = np.mean(sample_memory.ep_returns)
        train_results["Episode Length"] = np.mean(sample_memory.ep_lens)
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
        optim_start = time()
        losses = self.optimizer.optimize(sample_memory,
                                     actor_epochs=actor_epochs,
                                     critic_epochs=critic_epochs,
                                     batch_size=batch_size,
                                     kl_thresh=kl_thresh,
                                     recurrent=self.recurrent,
                                     verbose=verbose)

        a_loss, c_loss, m_loss, kls = losses
        time_results["Optimize Time"] = time() - optim_start
        optimizer_results["Actor Loss"] = np.mean(a_loss)
        optimizer_results["Critic Loss"] = np.mean(c_loss)
        optimizer_results["Mirror Loss"] = np.mean(m_loss)
        optimizer_results["KL"] = np.mean(kls)

        # Update network parameters by explicitly copying from optimizer
        actor_params, critic_params = self.optimizer.retrieve_parameters()
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
        "num-eval-eps"       : (50, "Number of episodes collected for computing test metrics"),
        "eval-freq"          : (200, "Will compute test metrics once every eval-freq iterations"),
        "discount"           : (0.99, "Discount factor when calculating returns"),
        "gae-lambda"         : (1.0, "Bias-variance tradeoff factor for GAE"),
        "a-lr"               : (1e-4, "Actor policy learning rate"),
        "c-lr"               : (1e-4, "Critic learning rate"),
        "eps"                : (1e-6, "Adam optimizer eps value"),
        "kl"                 : (0.02, "KL divergence threshold"),
        "entropy-coeff"      : (0.0, "Coefficient of entropy loss in optimization"),
        "clip"               : (0.2, "Log prob clamping value (1 +- clip)"),
        "grad-clip"          : (0.05, "Gradient clip value (maximum allowed gradient norm)"),
        "batch-size"         : (64, "Minibatch size to use during optimization"),
        "actor-epochs"       : (3, "Number of epochs to optimize for each iteration"),
        "critic-epochs"      : (3, "Number of epochs to optimize for each iteration"),
        "mirror"             : (1, "Mirror loss coefficient"),
        "workers"            : (4, "Number of parallel workers to use for sampling"),
        "backprop-workers"   : (-1, "Number of parallel workers to use for backprop. -1 for auto."),
        "redis"              : (None, "Ray redis address"),
        "previous"           : ("", "Previous model to bootstrap from"),
        "save-freq"          : (-1, "Save model once every save-freq iterations. -1 for no saving. Does not affect saving of best models."),
        "use-encoder-patch"  : (False, "Use encoder predictor for left tarsus"),
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
    print(policy)
    print(critic)

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
        # load_checkpoint(model_dict=critic_dict, model=critic)

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

    # set number of eval episodes
    num_eval_eps = min(args.num_eval_eps, args.workers)
    if args.num_eval_eps > args.workers:
        print(f"WARNING: only using {args.workers} test episodes for eval because args.workers < args.num_eval_eps")

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

    # Patch for left tarsus broken
    if args.use_encoder_patch:
        policy.load_encoder_patch()
        print("Adding encoder patch into actor\n", policy)

    algo = PPO(policy, critic, env_fn, ppo_args)
    print("Proximal Policy Optimization:")
    for key, val in sorted(args.__dict__.items()):
        print(f"\t{key} = {val}")

    itr = 0
    total_timesteps = 0
    best_reward = None
    past500_reward = -1
    while total_timesteps < args.timesteps:
        start = monotonic()
        ret = algo.do_iteration(num_steps=args.num_steps,
                                max_traj_len=args.traj_len,
                                num_eval_eps=num_eval_eps,
                                actor_epochs=args.actor_epochs,
                                critic_epochs=args.critic_epochs,
                                batch_size=args.batch_size,
                                kl_thresh=args.kl,
                                itr=itr)
        end = monotonic()
        ret["Time"]["Timesteps per Second (FULL)"] = round(args.num_steps / (end - start))

        print(f"iter {itr:4d} | return: {ret['Train']['Return']:5.2f} | " \
              f"KL {ret['Optimizer']['KL']:5.4f} | " \
              f"Actor loss {ret['Optimizer']['Actor Loss']:5.4f} | " \
              f"Critic loss {ret['Optimizer']['Critic Loss']:5.4f} | "\
              f"Mirror {ret['Optimizer']['Mirror Loss']:6.5f}", end='\n')
        total_timesteps += ret["Time"]["Timesteps per Iteration"]
        print(f"\tTotal timesteps so far {total_timesteps:n}")

        # Saving checkpoints for best reward
        if "Return" in ret["Test"].keys():
            if best_reward is None or ret["Test"]["Return"] > best_reward:
                print(f"\tbest policy so far! saving checkpoint to {args.save_actor_path}")
                best_reward = ret["Test"]["Return"]
                save_checkpoint(algo.actor, actor_dict, args.save_actor_path)
                save_checkpoint(algo.critic, critic_dict, args.save_critic_path)

        if args.save_freq > 0 and itr % args.save_freq == 0:
            print(f"saving policy at iteration {itr} to {args.save_actor_path[:-3] + f'_{itr}.pt'}")
            save_checkpoint(algo.actor, actor_dict, args.save_actor_path[:-3] + f"_{itr}.pt")
            save_checkpoint(algo.critic, critic_dict, args.save_critic_path[:-3] + f"_{itr}.pt")

        # always save latest policies
        save_checkpoint(algo.actor, actor_dict, args.save_actor_path[:-3] + "_latest.pt")
        save_checkpoint(algo.critic, critic_dict, args.save_critic_path[:-3] + "_latest.pt")

        if logger is not None:
            for key, val in ret.items():
                for k, v in val.items():
                    logger.add_scalar(f"{key}/{k}", v, itr)
            logger.add_scalar("Time/Total Timesteps", total_timesteps, itr)

        itr += 1
    print(f"Finished ({total_timesteps} of {args.timesteps}).")

    if args.wandb:
        wandb.finish()
