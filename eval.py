
if __name__ == "__main__":

    import argparse
    import sys
    import pickle
    import os

    try:
        evaluation_type = sys.argv[1]
        sys.argv.remove(sys.argv[1])
    except:
        raise RuntimeError("Choose evaluation type from ['simple','interactive', or 'no_vis']. Or add a new one.")

    # Support for headless render so do this before all imports
    if evaluation_type == 'data':
        import os
        gl_option = 'egl'
        os.environ['MUJOCO_GL']=gl_option
        # Check if the env variable is correct
        if "MUJOCO_GL" in os.environ:
            assert os.getenv('MUJOCO_GL') == gl_option,\
                    f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."

    import torch
    from util.evaluation_factory import simple_eval, interactive_eval, simple_eval_offscreen
    from util.nn_factory import load_checkpoint, nn_factory
    from util.env_factory import env_factory, add_env_parser
    from util.colors import OKGREEN, ENDC

    if evaluation_type == 'test':
        parser = argparse.ArgumentParser()
        parser.add_argument('--env-name', default="CassieEnvClock", type=str)
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
        env = env_factory(env_name, env_args)()
        simple_eval(actor=None, env=env)
        exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--traj-len', default=300, type=int)
    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--record', default=False, action='store_true')
    # Args used for offline data collection
    parser.add_argument('--data-path', default=None, type=str,
                        help="path to offline data used for kinematic replay mode")
    parser.add_argument('--collect-mode', default='dynamic', type=str, choices=['dynamic', 'kinematic'],
                        help="mode to collect data, dynamic means the robot is controlled by the policy,"
                        "kinematic means the robot is controlled by setting qpos and qvel directly")
    parser.add_argument('--high-rate-data', default=False, action='store_true',
                        help="whether to collect data at high rate")
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
    add_env_parser(previous_args_dict['all_args'].env_name, parser)
    args = parser.parse_args()

    if hasattr(previous_args_dict['env_args'], 'offscreen'):
        if evaluation_type == 'offscreen' or evaluation_type == 'data':
            previous_args_dict['env_args'].offscreen = True
            if evaluation_type == 'data':
                previous_args_dict['env_args'].depth_vis = True
            print(f"{OKGREEN}Offscreen rendering for evaluation.{ENDC}")
        else:
            previous_args_dict['env_args'].offscreen = False

    keys = ['integral_action']
    previous_args_dict['env_args'].depth_vis = True

    for k in keys:
        if not hasattr(previous_args_dict['env_args'], k):
            setattr(previous_args_dict['env_args'], k, False)
            setattr(previous_args_dict['all_args'], k, False)
        # else:
        #     # remove this key
        #     previous_args_dict['env_args'].__dict__.pop(k)
        #     previous_args_dict['all_args'].__dict__.pop(k)
    try:
        previous_args_dict['env_args'].__dict__.pop('actorfeet')
        previous_args_dict['all_args'].__dict__.pop('actorfeet')
    except:
        pass

    previous_args_dict['env_args'].__dict__.pop('depth_vis')

    # Load environment
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    load_checkpoint(model=critic, model_dict=critic_checkpoint)
    # actor.load_encoder_patch()
    actor.eval()
    actor.training = False

    if evaluation_type == 'simple':
        simple_eval(actor=actor, env=env, episode_length_max=args.traj_len, critic=critic)
    elif evaluation_type == 'i':
        if not hasattr(env, 'interactive_control'):
            raise RuntimeError("this environment does not support interactive control")
        interactive_eval(actor=actor, env=env, episode_length_max=args.traj_len,
                         critic=critic, plot=args.plot, record=args.record)
    elif evaluation_type == "offscreen":
        simple_eval_offscreen(actor=actor, env=env, episode_length_max=args.traj_len)
    elif evaluation_type == 'data':
        from util.offline_collection import collect_data
        env_fn = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])
        collect_data(actor=actor, critic=critic, env_fn=env_fn, args=previous_args_dict['all_args'],
                     mode=args.collect_mode, high_rate_data=args.high_rate_data)
    elif evaluation_type == 'replay':
        from util.evaluation_factory import kinematic_replay
        kinematic_replay(env=env, data_path=args.data_path)
    else:
        raise RuntimeError(f"This evaluation type {evaluation_type} has not been implemented.")
