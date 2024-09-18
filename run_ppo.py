import numpy as np

from algo.ppo import add_algo_args, run_experiment
from types import SimpleNamespace

def run_ppo():

    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()
    env_args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = np.random.randint(0, 100000)
    args.traj_len = 400
    args.use_privilege_critic = True
    args.link_base_model = True
    args.nonlinearity = 'relu'
    args.layers = "128,128"
    args.num_steps = 50000
    args.batch_size = 32
    args.actor_epochs = 2
    args.critic_epochs = 2
    args.discount = 0.96
    args.mirror = 1
    args.timesteps = 4e9
    args.workers = 80
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = 0.13
    args.previous = ""
    args.arch = "residual"
    args.teacher_actor = ""
    args.save_freq = 1000
    args.use_encoder_patch = False

    args.env_name = "CassieHfield"
    args.load_actor = "./pretrained_models/CassieEnvClock/spring_3_new/07-12-14-27/" # new lstm
    args.previous = "./pretrained_models/CassieHfield/up2date/fast_test_change/08-18-19-41/"
    # args.previous = "./pretrained_models/CassieHfield/up2date/noise_oc_boot_randcmds_torque09/08-11-17-12/"
    # args.previous = "./pretrained_models/CassieHfield/up2date/noise_oc_scratch_randcmds_torque09/08-11-17-09/"

    # args.env_name = "DigitHfield"
    # args.load_actor = "./pretrained_models/DigitEnvClock/tau_6464_lowpd/08-20-18-21/"

    # Set env args
    args.simulator_type = "mujoco"
    args.state_est = False
    args.policy_rate = 50
    args.dynamics_randomization = True
    args.reward_name = "depth"
    args.clock_type = "linear"
    args.offscreen = False
    args.terrain = 'hfield'
    args.depth_input = False
    args.autoclock = True
    args.autoclock_simple = False if args.autoclock else True
    args.state_noise = 0.0
    args.state_noise = [0.05, # orient noise (euler in rad)
                        0.1, # ang vel noise
                        0.01, # motor pos
                        0.1, # motor vel
                        0.01, # joint pos
                        0.1, # joint vel
                        ]
    # args.state_noise = [0.08, # orient noise (euler in rad)
    #                     0.1, # ang vel noise
    #                     0.01, # motor pos
    #                     0.2, # motor vel
    #                     0.01, # joint pos
    #                     0.2, # joint vel
    #                     ]
    args.velocity_noise = 0.0
    args.hfield_name = 'platform'
    args.reverse_heightmap = True
    args.collision_negative_reward = False
    args.full_clock = True
    args.contact_patch = False
    args.mix_terrain = True
    args.feetmap = True
    args.rangesensor = True
    args.hfield_noise = True
    args.integral_action = False
    args.actor_feet = False

    # Set env and logging args
    args.run_name = f"feet"
    args.wandb = True
    args.wandb_project_name = "roadrunner_refactor"# If running on vlab pur your tier1 folder here, ex. "/tier1/osu/username/"
    args.logdir = "/tier2/osu/bikram/trained_models/"
    args.wandb_dir = "/tier2/osu/bikram"   # If running on vlab pur your tier1 folder here, ex. "/tier1/osu/username/"

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

if __name__ == "__main__":
    run_ppo()