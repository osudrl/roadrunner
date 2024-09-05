import torch
import argparse
import sys
import pickle
import os

from util.evaluation_factory import simple_eval, interactive_eval, interactive_xbox_eval, simple_eval_offscreen, slowmo_interactive_eval
from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory, add_env_parser

if __name__ == "__main__":

    try:
        evaluation_type = sys.argv[1]
        sys.argv.remove(sys.argv[1])
    except:
        raise RuntimeError("Choose evaluation type from ['simple','interactive', or 'no_vis']. Or add a new one.")

    if evaluation_type == 'test':
        parser = argparse.ArgumentParser()
        parser.add_argument('--env-name', default="LocomotionClockEnv", type=str)
        # Manually handle env-name argument
        try:
            env_name_idx = sys.argv.index("--env-name")
            env_name = sys.argv[env_name_idx + 1]
            if not isinstance(env_name, str):
                print(f"{__file__}: error: argument --env-name received non-string input.")
                sys.exit()
        except ValueError:
            # If env-name not in command line input, use default value
            env_name = parser._option_string_actions["--env-name"].default
        add_env_parser(env_name, parser)
        args = parser.parse_args()
        for arg_group in parser._action_groups:
            if arg_group.title == "Env arguments":
                env_dict = {a.dest: getattr(args, a.dest, None) for a in arg_group._group_actions}
                env_args = argparse.Namespace(**env_dict)
        env_args.simulator_type += "_mesh"
        env = env_factory(env_name, env_args)()
        simple_eval(actor=None, env=env)
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--slow-factor', default=4, type=int)
    parser.add_argument('--traj-len', default=300, type=int)
    parser.add_argument('--plot-rewards', default=False, action='store_true')
    # Manually handle path argument
    try:
        path_idx = sys.argv.index("--path")
        model_path = sys.argv[path_idx + 1]
        if not isinstance(model_path, str):
            print(f"{__file__}: error: argument --path received non-string input.")
            sys.exit()
    except ValueError:
        print(f"No path input given. Usage is 'python eval.py simple --path /path/to/policy'")

    # model_path = args.path
    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
    critic_checkpoint = torch.load(os.path.join(model_path, 'critic.pt'), map_location='cpu')

    add_env_parser(previous_args_dict['all_args'].env_name, parser, is_eval=True)
    args = parser.parse_args()

    # Overwrite previous env args with current input
    for arg, val in vars(args).items():
        if hasattr(previous_args_dict['env_args'], arg):
            setattr(previous_args_dict['env_args'], arg, val)

    if hasattr(previous_args_dict['env_args'], 'offscreen'):
        previous_args_dict['env_args'].offscreen = True if evaluation_type == 'offscreen' else False
    if hasattr(previous_args_dict['env_args'], 'velocity_noise'):
        delattr(previous_args_dict['env_args'], 'velocity_noise')
    if hasattr(previous_args_dict['env_args'], 'state_est'):
        delattr(previous_args_dict['env_args'], 'state_est')

    # Load environment
    previous_args_dict['env_args'].simulator_type += "_mesh"      # Use mesh model
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    load_checkpoint(model=critic, model_dict=critic_checkpoint)
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env=env, episode_length_max=args.traj_len)
    elif evaluation_type == 'interactive':
        if not hasattr(env, 'interactive_control'):
            raise RuntimeError("this environment does not support interactive control")
        interactive_eval(actor=actor, env=env, episode_length_max=args.traj_len, critic=critic, plot_rewards=args.plot_rewards)
    elif evaluation_type == 'xbox':
        if not hasattr(env, 'interactive_control'):
            raise RuntimeError("this environment does not support interactive control")
        interactive_xbox_eval(actor=actor, env=env, episode_length_max=args.traj_len, critic=critic, plot_rewards=args.plot_rewards)
    elif evaluation_type == 'slowmo':
        if not hasattr(env, 'interactive_control'):
            raise RuntimeError("this environment does not support interactive control")
        slowmo_interactive_eval(actor=actor, env=env, episode_length_max=args.traj_len, slowmo=args.slow_factor, critic=critic)
    elif evaluation_type == "offscreen":
        simple_eval_offscreen(actor=actor, env=env, episode_length_max=args.traj_len)
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
