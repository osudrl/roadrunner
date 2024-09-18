import numpy as np

from algo.teacherstudent import add_algo_args, run_experiment
from types import SimpleNamespace

if __name__ == "__main__":

    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()
    env_args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = np.random.randint(0, 100000)
    args.traj_len = 300
    args.use_privilege_critic = True
    args.link_base_model = True
    args.nonlinearity = 'relu'
    args.layers = "128,128"
    args.num_steps = 5000
    args.batch_size = 5
    args.epochs = 10
    args.discount = 0.95
    args.mirror = 0
    args.timesteps = 4e9
    args.workers = 10
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = 0.13
    args.previous = ""
    args.arch = "residual-cnn"
    args.load_actor = "./pretrained_models/CassieEnvClock/ready_2/06-06-16-39/"
    args.teacher_actor = "./pretrained_models/CassieHfield/better_block_pelvis_cr_ready2/06-10-19-50/"
    args.offline_data_path = ""

    # Set env and logging args
    args.env_name = "CassieHfield"

    # Set env args
    args.simulator_type = "mujoco"
    args.state_est = False
    args.policy_rate = 50
    args.dynamics_randomization = True
    args.reward_name = "depth"
    args.clock_type = "linear"
    args.offscreen = True
    args.terrain = 'hfield'
    args.depth_input = True
    args.autoclock = True
    args.state_noise = 0.0
    args.state_noise = [0.02, # orient noise (euler in rad)
                        0.03, # ang vel noise
                        0.01, # motor pos
                        0.03, # motor vel
                        0.01, # joint pos
                        0.03, # joint vel
                        ]
    args.velocity_noise = 0.0
    args.hfield_name = 'block'
    args.reverse_heightmap = True
    args.collision_negative_reward = False
    args.full_clock = False
    args.contact_patch = False
    args.mix_terrain = False

    args.run_name = f"depth_resnet_better"
    args.wandb = True
    args.wandb_project_name = "roadrunner_refactor"
    args.logdir = "./trained_models/"

    if args.offscreen == True:
        import os
        gl_option = 'egl'
        os.environ['MUJOCO_GL']=gl_option
        # os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        # Check if the env variable is correct
        if "MUJOCO_GL" in os.environ:
            assert os.getenv('MUJOCO_GL') == gl_option,\
                    f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
    args = add_algo_args(args)
    run_experiment(args, args.env_name)
