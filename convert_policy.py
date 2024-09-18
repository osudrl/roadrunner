import torch
import argparse
import pickle
import os

from util.nn_factory import load_checkpoint, nn_factory, save_checkpoint
from util.env_factory import env_factory
from types import SimpleNamespace
from collections import OrderedDict

"""Convert old roadrunner models into comptabile one.
To use this file, change nn_args, env_args, and all_args for SimpleNamespace.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()
    model_path = args.path
    if model_path[-1] != "/":
        model_path += "/"
    if not os.path.exists(model_path):
        raise RuntimeError("provided path is not valid")
    actor_checkpoint = torch.load(model_path+"actor.pt", map_location='cpu')
    input("Inspect loaded checkpoint information before continuing...\nHit ENTER to continue")
    actor_checkpoint["obs_dim"] = 42

    nn_args = SimpleNamespace(arch="ff",
                            obs_dim=47,
                            action_dim=10,
                            layers="128,128",
                            bounded=False,
                            learn_stddev=False,
                            nonlinearity="relu",
                            std=0.13,
                            std_array="",
    )

    env_args = SimpleNamespace(
        simulator_type = 'mujoco',
        terrain ='',
        policy_rate = 40,
        dynamics_randomization = True,
        reward_name = 'locomotion_linear_clock_reward',
        state_est=False,
    )

    ppo_args = SimpleNamespace(
        prenormalize_steps = 100,
        prenorm = False,
        update_norm        = False,
        num_steps          = 30000,
        discount           = 0.95,
        a_lr               = 3e-4,
        c_lr               = 3e-4,
        eps                = 1e-6,
        kl                 = 0.02,
        entropy_coeff      = 0.0,
        clip               = 0.2,
        grad_clip          = 0.05,
        batch_size         = 32,
        epochs             = 5,
        mirror             = 1.0,
        workers            = 56,
        redis              = None,
        previous           = "",
    )

    args = SimpleNamespace(
        env_name = "CassieEnvClockOldVonMises",
        seed = 10,
        traj_len = 200,
        timesteps = 4e9,
        state_est = False,
    )

    for arg in vars(nn_args):
        setattr(args, arg, getattr(nn_args, arg))
    for arg in vars(env_args):
        setattr(args, arg, getattr(env_args, arg))
    for arg in vars(ppo_args):
        setattr(args, arg, getattr(ppo_args, arg))

    env_fn = env_factory(env_name=args.env_name, env_args=env_args)
    actor, critic = nn_factory(args=nn_args, env=env_fn)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor_dict = {'model_class_name': actor._get_name()}
    critic_dict = {'model_class_name': critic._get_name()}
    model_path_prefix = os.path.split(model_path)[0]
    model_path_suffix = os.path.split(model_path)[1]+"_converted/"
    output_dir = model_path_prefix+model_path_suffix
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_checkpoint(model=actor, model_dict=actor_dict, save_path=os.path.join(output_dir, "actor.pt"))
    save_checkpoint(model=actor, model_dict=critic_dict, save_path=os.path.join(output_dir, "critic.pt"))
    arg_dict = OrderedDict(sorted(args.__dict__.items(), key=lambda t: t[0]))

    info_path = os.path.join(output_dir, "experiment.info")
    pkl_path = os.path.join(output_dir, "experiment.pkl")
    print(f"Converted policy data will be saved to {output_dir}")
    with open(pkl_path, 'wb') as file:
        save_dict = {'all_args': args,
                'algo_args': ppo_args,
                'env_args': env_args,
                'nn_args': nn_args}
        print("save dict", save_dict)
        pickle.dump(save_dict, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')
