import argparse
import datetime

import numpy as np
import ray
import tqdm

from algo.common.ppo_algo import PPO_algo
from algo.common.utils import *
from env.cassie.cassiepede.cassiepede import Cassiepede
from env.cassie.cassiepedeHL.cassiepedeHL import CassiepedeHL


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Normalize the probability of number of cassie
    args.num_cassie_prob = np.array(args.num_cassie_prob, dtype=float) / np.sum(args.num_cassie_prob)

    dev_cpu, dev_gpu = get_device(args)

    args.device = dev_gpu.__str__()

    custom_terrain = None
    # custom_terrain = 'cassiepede_bar'
    # custom_terrain = 'cassiepede_equilateral'

    env_fn = lambda num_cassie: Cassiepede(
        clock_type=args.clock_type,
        reward_name=args.reward_name,
        simulator_type='mujoco',
        policy_rate=50,
        dynamics_randomization=True,
        state_noise=[0.05, 0.1, 0.01, 0.2, 0.01, 0.2, 0.05, 0.05, 0.05],
        velocity_noise=0.0,
        state_est=False,
        full_clock=True,
        full_gait=False,
        integral_action=False,
        depth_input=False,
        num_cassie=num_cassie,
        custom_terrain=custom_terrain,
        only_deck_force=False,
        height_control=True,
        merge_states=False,
        poi_position_offset=args.poi_position_offset,
        perturbation_force=args.perturbation_force,
        force_prob=args.force_prob,
        position_offset=args.position_offset,
        poi_heading_range=args.poi_heading_range,
        cmd_noise=args.cmd_noise,
        cmd_noise_prob=args.cmd_noise_prob,
        mask_tarsus_input=args.mask_tarsus_input,
        offscreen=False)

    env = env_fn(num_cassie=np.nonzero(args.num_cassie_prob)[0][0] + 1)

    """Training code"""
    max_reward = float('-inf')
    time_now = datetime.datetime.now()

    args.env_name = env.__class__.__name__
    args.state_dim = env.observation_size
    args.action_dim = env.action_size
    args.reward_name = env.reward_name

    args.run_name = str(time_now)

    if args.use_mirror_loss:
        mirror_dict = env.get_mirror_dict()

        for k in mirror_dict['state_mirror_indices'].keys():
            mirror_dict['state_mirror_indices'][k] = torch.tensor(mirror_dict['state_mirror_indices'][k],
                                                                  dtype=torch.float32,
                                                                  device=dev_gpu)

        mirror_dict['action_mirror_indices'] = torch.tensor(mirror_dict['action_mirror_indices'],
                                                            dtype=torch.float32,
                                                            device=dev_gpu)
    else:
        mirror_dict = None

    agent = PPO_algo(args, device=dev_gpu, mirror_dict=mirror_dict)

    args.actor_name = agent.actor.__class__.__name__
    args.critic_name = agent.critic.__class__.__name__

    logging.info(args)

    logging.info(f'Using device:{dev_cpu}(inference), {dev_gpu}(optimization)\n')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

    run, iterations, total_steps, trajectory_count = init_logger(args, agent)

    prev_save_steps = 0

    render = False

    wandb.watch(models=agent.actor, log_freq=1)

    actor_global = deepcopy(agent.actor).to(dev_cpu)
    critic_global = deepcopy(agent.critic).to(dev_cpu)

    # dispatcher = Dispatcher.remote()

    collectors = [Worker.remote(env_fn, actor_global, args, dev_cpu, i) for i in range(args.n_collectors)]

    evaluators = [Worker.remote(env_fn, actor_global, args, dev_cpu, i) for i in range(args.n_evaluators)]

    # Store worker id for each num_cassie worker. This is useful to cancel all workers for a particular num_cassie
    num_cassie_worker_map = defaultdict(lambda: [])
    for wid, num_cassie in enumerate(ray.get([collector.get_num_cassie.remote() for collector in collectors])):
        num_cassie_worker_map[num_cassie].append(wid)

    # Initializes replay buffer for each type of environment (number of cassie in it)
    replay_buffers = {}
    for num_cassie in np.nonzero(args.num_cassie_prob)[0]:
        replay_buffers[num_cassie + 1] \
            = ReplayBuffer(args, buffer_size=int(np.ceil(args.buffer_size * args.num_cassie_prob[num_cassie])))

    pbar_total_steps = tqdm.tqdm(total=args.max_steps, desc='Total steps', position=0, colour='cyan')

    pbar_evaluator = tqdm.tqdm(total=args.eval_buffer_size, desc='Evaluating', position=1, colour='yellow')

    pbar_collector = {
        k: tqdm.tqdm(total=replay_buffers[k].buffer_size,
                     desc=f'Collecting [num_cassie={k},workers={len(num_cassie_worker_map[k])}]', position=i + 2,
                     colour='blue')
        for i, k in enumerate(replay_buffers.keys())
    }

    while total_steps < args.max_steps:
        actor_param_id = ray.put(list(actor_global.parameters()))

        evaluator_ids = {}
        if iterations > 0 and iterations % args.evaluate_freq == 0:
            """Evaluation"""
            # logging.debug("Evaluating")
            time_evaluating = datetime.datetime.now()

            # Copy the latest actor to all evaluators
            for evaluator in evaluators:
                evaluator.update_model.remote(actor_param_id)

            # Start the evaluators
            evaluator_ids = {
                i: evaluator.evaluate.remote(max_ep_len=min(args.time_horizon, args.eval_buffer_size), render=render)
                for i, evaluator in enumerate(evaluators)}

        """Collect data"""
        # logging.debug("Collecting")
        time_collecting = datetime.datetime.now()

        # Copy the latest actor to all collectors
        for collector in collectors:
            collector.update_model.remote(actor_param_id)

        # Start the collectors
        collector_ids = {i: collector.collect.remote(max_ep_len=min(args.time_horizon, args.buffer_size), render=render)
                         for i, collector in enumerate(collectors)}

        evaluator_steps = 0
        eval_rewards = defaultdict(lambda: [])
        eval_lengths = defaultdict(lambda: [])

        train_rewards = defaultdict(lambda: [])
        train_lengths = defaultdict(lambda: [])

        # Reset the replay buffer for each type of environment
        for replay_buffer in replay_buffers.values():
            replay_buffer.reset_buffer()

        # Reset the progress bar
        for pc in pbar_collector.values():
            pc.reset()
        pbar_evaluator.reset()

        while evaluator_ids or collector_ids:
            done_ids, remain_ids = ray.wait(list(collector_ids.values()) + list(evaluator_ids.values()), num_returns=1)

            _replay_buffer, episode_reward, episode_length, worker_id, num_cassie = ray.get(done_ids)[0]

            if _replay_buffer is None:
                # This worker is evaluator
                eval_rewards[num_cassie].append(episode_reward)
                eval_lengths[num_cassie].append(episode_length)

                evaluator_steps += episode_length

                rem_buffer_size = args.eval_buffer_size - evaluator_steps

                # Update the progress bar
                pbar_evaluator.n = min(pbar_evaluator.total, evaluator_steps)
                pbar_evaluator.refresh()

                if rem_buffer_size > 0:
                    logging.debug(f"{rem_buffer_size} steps remaining to evaluate")
                    evaluator_ids[worker_id] = evaluators[worker_id].evaluate.remote(
                        max_ep_len=min(args.time_horizon, rem_buffer_size), render=render)
                else:
                    time_evaluating = datetime.datetime.now() - time_evaluating
                    logging.debug('Evaluation done. Cancelling stale evaluators')
                    # ray.get(dispatcher.set_evaluating.remote(False))
                    map(ray.cancel, evaluator_ids.values())
                    evaluator_ids.clear()
            else:
                # This worker is collector
                train_rewards[num_cassie].append(episode_reward)
                train_lengths[num_cassie].append(episode_length)
                replay_buffers[num_cassie].merge(_replay_buffer)

                del _replay_buffer

                # Update the progress bar
                pbar_collector[num_cassie].n = replay_buffers[num_cassie].count
                pbar_collector[num_cassie].refresh()

                if not replay_buffers[num_cassie].is_full():
                    logging.debug(
                        f"{args.buffer_size - replay_buffers[num_cassie].count} steps remaining to collect [num_cassie={num_cassie}]")
                    collector_ids[worker_id] = collectors[worker_id].collect.remote(
                        max_ep_len=min(args.time_horizon, args.buffer_size - replay_buffers[num_cassie].count),
                        render=render)
                else:
                    # Prevent collecting for this num_cassie
                    for worker_id in num_cassie_worker_map[num_cassie]:
                        ray.cancel(collector_ids[worker_id])
                        del collector_ids[worker_id]
                        logging.debug(f'Collector done [num_cassie={num_cassie}]. Removing from collectors worker')

                    if len(collector_ids) == 0:
                        time_collecting = datetime.datetime.now() - time_collecting

        # Cancel any remaining collector/evaluator
        # ray.get(dispatcher.set_evaluating.remote(False))
        # ray.get(dispatcher.set_collecting.remote(False))
        map(ray.cancel, list(collector_ids.values()) + list(evaluator_ids.values()))

        if iterations > 0 and iterations % args.evaluate_freq == 0:
            eval_rewards = dict([(f'eval/episode_reward/num_cassie_{k}', np.mean(v)) for k, v in eval_rewards.items()])
            eval_lengths = dict([(f'eval/episode_length/num_cassie_{k}', np.mean(v)) for k, v in eval_lengths.items()])

            reward = np.mean(list(eval_rewards.values()))
            length = np.mean(list(eval_lengths.values()))

            if reward >= max_reward:
                max_reward = reward
                torch.save(agent.actor.state_dict(), f'saved_models/agent-{run.name}.pth')
                run.save(f'saved_models/agent-{run.name}.pth', policy='now')

            log = {'eval/episode_reward': reward,
                   'eval/episode_length': length,
                   **eval_rewards,
                   **eval_lengths,
                   'misc/total_steps': total_steps,
                   'misc/iterations': iterations,
                   'misc/time_evaluating': time_evaluating.total_seconds(),
                   'misc/evaluation_rate': evaluator_steps / time_evaluating.total_seconds(),
                   'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

            logging.debug(log)
            run.log(log, step=total_steps)

        train_rewards = dict([(f'train/episode_reward/num_cassie_{k}', np.mean(v)) for k, v in train_rewards.items()])
        train_lengths = dict([(f'train/episode_length/num_cassie_{k}', np.mean(v)) for k, v in train_lengths.items()])

        mean_train_rewards = np.mean(list(train_rewards.values()))
        mean_train_lens = np.mean(list(train_lengths.values()))

        replay_buffer_size = sum([rb.count for rb in replay_buffers.values()])
        total_steps += replay_buffer_size
        trajectory_count += sum([len(replay_buffer.ep_lens) for replay_buffer in replay_buffers.values()])

        log = {'train/episode_reward': mean_train_rewards,
               'train/episode_length': mean_train_lens,
               **train_rewards,
               **train_lengths,
               'misc/trajectory_count': trajectory_count,
               'misc/total_steps': total_steps,
               'misc/collection_rate': replay_buffer_size / time_collecting.total_seconds(),
               'misc/iterations': iterations,
               'misc/total_steps_rate': total_steps / (datetime.datetime.now() - time_now).total_seconds(),
               'misc/time_collecting': time_collecting.total_seconds(),
               'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.debug(log)
        run.log(log, step=total_steps)

        """Training"""
        logging.debug("Training")
        time_training = datetime.datetime.now()
        actor_loss, entropy_loss, mirror_loss, critic_loss, kl, num_batches, train_epoch = \
            agent.update(replay_buffers, total_steps, check_kl=args.kl_check_min_itr >= iterations)

        pbar_total_steps.update(replay_buffer_size)

        # Copy updated models to global models
        update_model(actor_global, agent.actor.parameters())
        update_model(critic_global, agent.critic.parameters())

        time_training = datetime.datetime.now() - time_training

        log = {'train/actor_loss': actor_loss,
               'train/entropy_loss': entropy_loss,
               'train/mirror_loss': mirror_loss,
               'train/critic_loss': critic_loss,
               'train/kl_divergence': kl,
               'misc/total_steps': total_steps,
               'misc/trajectory_count': trajectory_count,
               'misc/num_batches': num_batches,
               'misc/iterations': iterations,
               'misc/train_epoch': train_epoch,
               'misc/time_training': time_training.total_seconds(),
               'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.debug(log)
        run.log(log, step=total_steps)

        checkpoint = {
            'total_steps': total_steps,
            'iterations': iterations,
            'trajectory_count': trajectory_count,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),

        }
        torch.save(checkpoint, f'checkpoints/checkpoint-{run.name}.pt')
        run.save(f'checkpoints/checkpoint-{run.name}.pt', policy='now')

        if total_steps - prev_save_steps >= args.model_save_steps:
            torch.save(checkpoint, f'checkpoints/checkpoint-{run.name}-{iterations}.pt')

            run.save(f'checkpoints/checkpoint-{run.name}-{iterations}.pt', policy='now')

            prev_save_steps = total_steps

        iterations += 1

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")

    # Training
    parser.add_argument("--model_save_steps", type=int, default=int(5e7), help="Save model steps")
    parser.add_argument("--max_steps", type=int, default=int(100e9), help="Maximum number of training steps")
    parser.add_argument("--num_epoch", type=int, default=10, help="PPO parameter")
    parser.add_argument("--evaluate_freq", type=int, default=2,
                        help="Policy evaluation frequency")
    parser.add_argument("--n_collectors", type=int, default=80, help="Number of collectors")
    parser.add_argument("--n_evaluators", type=int, default=4, help="Number of evaluators")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Buffer size for training")
    parser.add_argument("--eval_buffer_size", type=int, default=3000, help="Buffer size for evaluation")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--empty_cuda_cache", action='store_true', help="Whether to empty cuda cache")
    parser.add_argument("--time_horizon", type=int, default=500,
                        help="The maximum length of the episode")
    parser.add_argument('--num_cassie_prob', type=float, nargs='+', default=[1.0],
                        help='Probability of number of cassie')
    parser.add_argument("--reward_name", type=str, default='feet_air_time', help="Name of the reward function")
    parser.add_argument('--clock_type', type=str, required=False, help="Type of the clock")
    parser.add_argument("--position_offset", type=float, default=0.2, help="Cassiepede position offset")
    parser.add_argument("--poi_heading_range", type=float, default=0.0, help="Poi heading range")
    parser.add_argument("--poi_position_offset", type=float, default=0.0, help="Poi offset from cassie")
    parser.add_argument("--perturbation_force", type=float, help="Force to apply to the deck", default=0)
    parser.add_argument("--force_prob", type=float, help="Prob of force to apply to the deck", default=0.0)
    parser.add_argument("--cmd_noise", type=float,
                        help="Noise to cmd for each cassie. Tuple of 3 (x_vel, y_vel, turn_rate (deg/t))", nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--cmd_noise_prob", type=float, help="Prob of noise added to cmd for each cassie", default=0.0)
    parser.add_argument("--mask_tarsus_input", action='store_true', help="Mask tarsus input with zeros")
    parser.add_argument("--device", type=str, default='cuda', help="Device name")

    # Network
    parser.add_argument("--hidden_dim", type=int, default=64, help="Latent dim of non-proprioception state")
    parser.add_argument("--lstm_hidden_dim", type=int, default=64, help="Number of hidden units in LSTM")
    parser.add_argument('--lstm_num_layers', type=int, default=4, help='Number of layers in transformer encoder')

    # Optimizer
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--set_adam_eps", action='store_true', help="Set Adam epsilon=1e-5")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps of Adam optimizer (default: 1e-5)")
    parser.add_argument("--std", type=float, help="Std for action")
    parser.add_argument("--use_orthogonal_init", action='store_true', help="Orthogonal initialization")
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    # PPO
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=1.0, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--kl_check", action='store_true', help="Whether to check kl divergence")
    parser.add_argument("--kl_threshold", type=float, default=0.2, help="KL threshold of early stopping")
    parser.add_argument("--kl_check_min_itr", type=int, default=2,
                        help="Epoch after which kl check is done")
    parser.add_argument("--use_adv_norm", action='store_true', help="Advantage normalization")
    parser.add_argument("--use_reward_scaling", action='store_true', help="Reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--use_lr_decay", action='store_true', help="Learning rate Decay")
    parser.add_argument("--use_grad_clip", action='store_true', help="Gradient clip")
    parser.add_argument("--grad_clip", type=str, default=0.05, help="Gradient clip value")
    parser.add_argument("--use_mirror_loss", action='store_true', help="Whether to use mirror loss")

    # Wandb
    parser.add_argument("--project_name", type=str, default='roadrunner_cassiepede', help="Name of project")
    parser.add_argument("--previous_run", type=str, default=None, help="Name of previous run")
    parser.add_argument("--parent_run", type=str, default=None, help="Name of parent run")
    parser.add_argument("--previous_checkpoint", type=str, default=None, help="Timestep of bootstrap checkpoint")
    parser.add_argument("--wandb_mode", type=str, default='online', help="Wandb mode")

    args = parser.parse_args()

    ray.init(num_cpus=args.n_collectors + args.n_evaluators)

    main()

# Example Run script
# export PYTHONPATH=.
# export WANDB_API_KEY=
# python algo/cassiepede/training.py \
#   --n_collectors 120 \
#   --n_evaluators 6 \
#   --time_horizon 500 \
#   --buffer_size 60000 \
#   --eval_buffer_size 3000 \
#   --evaluate_freq 4 \
#   --num_epoch 5 \
#   --mini_batch_size 32 \
#   --hidden_dim 64 \
#   --lstm_hidden_dim 64 \
#   --lstm_num_layers 2 \
#   --use_orthogonal_init \
#   --set_adam_eps \
#   --kl_check \
#   --kl_check_min_itr 2 \
#   --use_adv_norm \
#   --use_lr_decay \
#   --use_grad_clip \
#   --reward_name locomotion_cassiepede_feetairtime_modified \
#   --project_name roadrunner_cassiepede \
#   --wandb_mode online \
#   --device cuda:0 \
#   --position_offset 1.0 \
#   --poi_heading_range 1.05 \
#   --gamma 0.95 \
#   --std 0.13 \
#   --entropy_coef 0.01 \
#   --num_cassie_prob 0.2 0.8 \
#   --wandb_mode online \
#   --perturbation_force 30.0 \
#   --force_prob 0.2 \
#   --cmd_noise 0.0 0.0 0.0 \
#   --cmd_noise_prob 0.0
