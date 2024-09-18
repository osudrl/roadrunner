import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import torch

from env.util.periodicclock import PeriodicClock
from util.env_factory import env_factory
from util.nn_factory import nn_factory, load_checkpoint

def tracker_test():
    model_path = "./pretrained_models/CassieEnvClockOld/"
    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
    previous_args_dict['env_args'].dynamics_randomization = False
    previous_args_dict['env_args'].simulator_type = "mujoco"
    previous_args_dict['env_args'].state_noise = 0.0

    # Load environment
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()
    env.trackers = {env.update_tracker_grf: {"frequency": 2000},
                    env.update_tracker_velocity: {"frequency": 2000},
                    env.update_tracker_torque: {"frequency": 2000},
                   }
    for tracker, tracker_dict in env.trackers.items():
            freq = tracker_dict["frequency"]
            steps = int(env.sim.simulator_rate // freq)
            tracker_dict["num_step"] = steps

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False

    env_steps = 100
    grfs_2khz = []
    foot_vel_2khz = []
    torque_2khz = []
    time_2khz = 0
    state = env.reset()
    env.x_velocity = 1
    env.y_velocity = 0
    env.turn_rate = 0
    env.clock = PeriodicClock(0.8, 1 / env.default_policy_rate, [0.5, 0.5], [0, 0.5])
    np.random.seed(8888)

    with torch.no_grad():
        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        for i in range(env_steps):
            state = torch.Tensor(state).float()
            action = actor(state).numpy()
            # action = np.random.uniform(-0.5, 0.5, env.action_size)
            start_t = time.time()
            state, reward, done, _ = env.step(action)
            time_2khz += time.time() - start_t
            grfs_2khz.append([env.feet_grf_tracker_avg["left-foot"], env.feet_grf_tracker_avg["right-foot"]])
            foot_vel_2khz.append([env.feet_velocity_tracker_avg["left-foot"], env.feet_velocity_tracker_avg["right-foot"]])
            torque_2khz.append(env.torque_tracker_avg)

    grfs_50hz = []
    foot_vel_50hz = []
    torque_50hz = []
    time_50hz = 0
    state = env.reset()
    env.x_velocity = 1
    env.y_velocity = 0
    env.turn_rate = 0
    env.clock = PeriodicClock(0.8, 1 / env.default_policy_rate, [0.5, 0.5], [0, 0.5])
    env.trackers = {env.update_tracker_grf: {"frequency": 50},
                    env.update_tracker_velocity: {"frequency": 50},
                    env.update_tracker_torque: {"frequency": 50},
                   }
    for tracker, tracker_dict in env.trackers.items():
        freq = tracker_dict["frequency"]
        steps = int(env.sim.simulator_rate // freq)
        tracker_dict["num_step"] = steps

    with torch.no_grad():
        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        for i in range(env_steps):
            state = torch.Tensor(state).float()
            action = actor(state).numpy()
            # action = np.random.uniform(-0.5, 0.5, env.action_size)
            start_t = time.time()
            state, reward, done, _ = env.step(action)
            time_50hz += time.time() - start_t
            grfs_50hz.append([env.feet_grf_tracker_avg["left-foot"], env.feet_grf_tracker_avg["right-foot"]])
            foot_vel_50hz.append([env.feet_velocity_tracker_avg["left-foot"], env.feet_velocity_tracker_avg["right-foot"]])
            torque_50hz.append(env.torque_tracker_avg)

    print(f"2khz took {time_2khz / env_steps:.3f} seconds, 50hz took {time_50hz / env_steps:.3f} seconds")
    time_diff = time_2khz / env_steps - time_50hz / env_steps
    print(f"Over 500 million steps, will lose {time_diff * 500000000/60/60/24:.3f} days due to tracker overhead")

    grfs_2khz = np.array(grfs_2khz)
    foot_vel_2khz = np.array(foot_vel_2khz)
    torque_2khz = np.array(torque_2khz)
    norm_foot_vel_2khz = np.linalg.norm(foot_vel_2khz[:, :, 0:3], axis=2)
    grfs_50hz = np.array(grfs_50hz)
    foot_vel_50hz = np.array(foot_vel_50hz)
    torque_50hz = np.array(torque_50hz)
    norm_foot_vel_50hz = np.linalg.norm(foot_vel_50hz[:, :, 0:3], axis=2)
    fig, ax = plt.subplots(2, 1, figsize=(15, 6))
    t = np.linspace(0, env_steps * 1 / env.default_policy_rate, env_steps)
    ax[0].plot(t, grfs_2khz[:, 0, 2], label="2khz left foot")
    ax[0].plot(t, grfs_2khz[:, 1, 2], label="2khz right foot")
    ax[0].plot(t, grfs_50hz[:, 0, 2], label="50hz left foot")
    ax[0].plot(t, grfs_50hz[:, 1, 2], label="50hz right foot")
    ax[1].plot(t, norm_foot_vel_2khz[:, 0], label="2khz left foot")
    ax[1].plot(t, norm_foot_vel_2khz[:, 1], label="2khz right foot")
    ax[1].plot(t, norm_foot_vel_50hz[:, 0], label="50hz left foot")
    ax[1].plot(t, norm_foot_vel_50hz[:, 1], label="50hz right foot")
    ax[0].set_title("GRF Comparison")
    ax[1].set_title("Foot Velocity Comparison")

    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()

    plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    t = np.linspace(0, env_steps * 1 / env.default_policy_rate, env_steps)
    labels = ["hip roll", "hip yaw", "hip pitch", "knee", "foot"]
    for i in range(5):
        ax[0][i].plot(t, torque_2khz[:, i], label="2khz")
        ax[1][i].plot(t, torque_2khz[:, i + 5], label="2khz")
        ax[0][i].plot(t, torque_50hz[:, i], label="50hz")
        ax[1][i].plot(t, torque_50hz[:, i + 5], label="50hz")
        ax[0][i].set_title("Left " + labels[i])
        ax[1][i].set_title("Right " + labels[i])
        ax[0][i].legend()
        ax[1][i].legend()
    plt.suptitle("Torque Comparison")

    plt.tight_layout()
    plt.show()


