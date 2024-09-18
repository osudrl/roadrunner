import torch # first import to prevent ray 1 core bug
import numpy as np
import argparse

from algo.ppo import add_algo_args, run_experiment
from types import SimpleNamespace

def get_ppo_args():
    # Setup arg Namespaces and get default values for algo args
    args = SimpleNamespace()

    # Overwrite with whatever optimization args you want here
    args.seed = np.random.randint(0, 100000)
    args.traj_len = 300
    args.arch = "lstm"
    args.layers = "64,64"
    args.num_steps = 50000
    args.batch_size = 32
    args.epochs = 2
    args.discount = 0.95
    args.gae_lambda = 0.95
    args.mirror = 1
    args.timesteps = 4e9
    args.workers = 80
    args.do_prenorm = False
    args.a_lr = 3e-4
    args.c_lr = 3e-4
    args.std = 0.13

    # Set env and robot
    args.env_name = "LocomotionClockEnv"
    args.robot_name = "cassie"

    # Set env args
    args.simulator_type = "mujoco"
    args.state_est = False
    args.terrain = False
    args.policy_rate = 50
    args.dynamics_randomization = True
    args.reward_name = "locomotion_vonmises_clock_reward"
    args.clock_type = "von_mises"
    # args.state_noise = 0.0
    args.state_noise = [0.02, # orient noise (euler in rad)
                        0.03, # ang vel noise
                        0.01, # motor pos
                        0.03, # motor vel
                        0.01, # joint pos
                        0.03, # joint vel
                        ]
    args.full_clock = True
    args.integral_action = False

    args.wandb = False
    args.run_name = f"{args.robot_name}-{args.env_name}-insert name here"
    args.wandb_group_name = f"insert group name here"
    args.wandb_project_name = "insert project name here"

    #NOTE: If running on vlab pur your tier1 folder here, ex. "/tier1/osu/username/"
    args.logdir = "./trained_models/"
    args.wandb_dir = "./wandb/"

    args = add_algo_args(args)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a small local test of the run.")
    args = parser.parse_args()

    ppo_args = get_ppo_args()

    if args.test:
        ppo_args.backprop_workers = 8
        ppo_args.workers = 8
        ppo_args.wandb = False
        ppo_args.num_steps = 5000
        ppo_args.logdir = "test_logs/"

    run_experiment(ppo_args, ppo_args.env_name)
