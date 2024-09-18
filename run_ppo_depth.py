import numpy as np

from algo.ppo import add_algo_args, run_experiment
from types import SimpleNamespace

if __name__ == "__main__":

    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()
    env_args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = np.random.randint(0, 100000)
    args.traj_len = 300
    args.arch = "residual-cnn"
    args.use_privilege_critic = True
    args.link_base_model = True
    args.nonlinearity = 'relu'
    args.layers = "128,128"
    args.num_steps = 10000
    args.batch_size = 10
    args.epochs = 5
    args.discount = 0.95
    args.mirror = 0
    args.timesteps = 4e9
    args.workers = 10
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = 0.13
    args.previous = "./trained_models/CassieHfield/depth_resnet_smaller/05-17-12-25/"
    args.load_actor = "./pretrained_models/CassieEnvClock/libcassie_linear_lstm_dr_se/04-07-19-00/"
    args.teacher_actor = ""

    # Set env and logging args
    args.env_name = "CassieHfield"

    # Set env args
    args.simulator_type = "mujoco"
    args.policy_rate = 40
    args.dynamics_randomization = False
    args.reward_name = "depth"
    args.clock_type = "linear"
    args.offscreen = True
    args.terrain = 'hfield'
    args.depth_input = True
    args.autoclock = True
    args.hfield_name = 'stair'

    args.run_name = f"depth_resnet_ppo"
    args.wandb = True
    args.wandb_project_name = "roadrunner_refactor"
    args.logdir = "./trained_models/"

    if args.offscreen == True:
        import os
        gl_option = 'egl'
        os.environ['MUJOCO_GL']=gl_option
        # Check if the env variable is correct
        if "MUJOCO_GL" in os.environ:
            assert os.getenv('MUJOCO_GL') == gl_option,\
                    f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
    args = add_algo_args(args)
    run_experiment(args, args.env_name)
