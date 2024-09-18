import numpy as np
import sys
import termios
import time
import torch
import time
import mediapy
import cv2
import copy
import glob
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from env.util.quaternion import scipy2us, us2scipy
from collections import defaultdict

# from util.keyboard import Keyboard
from util.colors import OKGREEN, FAIL


def simple_eval(actor, env, episode_length_max=300, critic=None):
    """Simply evaluating policy in visualization window and no user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    with torch.no_grad():
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()

        state = env.reset()
        if critic is not None:
            if hasattr(env, 'get_privilege_state'):
                critic_state = env.get_privilege_state()
            else:
                critic_state = state
        env.sim.viewer_init()
        render_state = env.sim.viewer_render()
        while render_state:
            start_time = time.time()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                if actor is None:
                    action = np.random.uniform(-0.2, 0.2, env.action_size)
                else:
                    action = actor(state).numpy()
                state, reward, done, _ = env.step(action)
                if critic is not None:
                    if hasattr(env, 'get_privilege_state'):
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                if hasattr(env, 'depth_image'):
                    img = copy.deepcopy(env.depth_image)
                    img /= np.max(img)
                    img = np.clip(img, 0, 1) * 255
                    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Depth", cv2.resize(
                        cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_BONE),
                        (500, 500), interpolation = cv2.INTER_NEAREST))
                    cv2.waitKey(1)
                episode_length += 1
                episode_reward.append(reward)
            render_state = env.sim.viewer_render()
            delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
            time.sleep(delaytime)
            if episode_length == episode_length_max or done:
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                done = False
                state = env.reset()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                if hasattr(critic, 'init_hidden_state'):
                    critic.init_hidden_state()
                # Seems like Mujoco only allows a single mjContext(), and it prefers one context
                # with one window when modifying mjModel. So for onscreen dual window, we re-init
                # the non-main window, ie egocentric view here.
                if hasattr(env.sim, 'renderer'):
                    if env.sim.renderer is not None:
                        print("re-init non-primary screen renderer")
                        env.sim.renderer.close()
                        env.sim.init_renderer(offscreen=env.offscreen,
                                              width=env.depth_image_dim[0], height=env.depth_image_dim[1])

def interactive_eval(actor, env, episode_length_max=300, critic=None, plot=False, record=False):
    """Simply evaluating policy in visualization window with user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    # keyboard = Keyboard()
    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print("Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds." + '\033[0m')
    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()

        if critic is not None:
            if hasattr(env, 'get_privilege_state') and critic.use_privilege_critic:
                critic_state = env.get_privilege_state()
            else:
                critic_state = state

        if record:
            recorder_resolution = (1280, 720)
            # recorder_resolution = (1960, 1080)
            env.sim.viewer_init(height=recorder_resolution[1], width=recorder_resolution[0],
                                record_video=True)
        else:
            env.sim.viewer_init()
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()

        plot_dict = defaultdict(list)

        while render_state:
            start_time = time.time()
            # cmd = keyboard.get_input()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                state, reward, done, _ = env.step(action)
                euler = R.from_quat(us2scipy(env.sim.get_base_orientation())).as_euler('xyz')
                quat = R.from_euler('xyz', [0,0,euler[2]])
                actual_vel = quat.apply(env.sim.get_base_linear_velocity(), inverse=True)
                # plot_dict['xvel'].append(actual_vel[0])
                # plot_dict['yvel'].append(actual_vel[1])
                plot_dict['norm vel'].append(np.linalg.norm(actual_vel[0:2]))
                # plot_dict['xvel_target'].append(env.x_velocity)
                plot_dict['target norm vel'].append(np.linalg.norm(np.array([env.x_velocity, env.y_velocity])))
                # plot_dict['roll'].append(euler[0]/np.pi*180)
                # plot_dict['pitch'].append(euler[1]/np.pi*180)
                plt.cla()
                if plot:
                    for k,v in plot_dict.items():
                        plt.plot(v, label=k)
                    plt.legend()
                    plt.pause(1e-10)
                if hasattr(env, 'depth_image'):
                    img = copy.deepcopy(env.depth_image)
                    img /= np.max(img)
                    img = np.core.umath.clip(img, 0, 1) * 255
                    cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("Depth", cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_VIRIDIS))
                    cv2.waitKey(1)
                if critic is not None:
                    if hasattr(env, 'get_privilege_state') and critic.use_privilege_critic:
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                episode_length += 1
                episode_reward.append(reward)
            # if cmd is not None:
            #     env.interactive_control(cmd)
            # if cmd == "quit":
            #     done = True
            # if cmd == "menu":
            #     env.display_control_commands(erase=True)
            #     env.display_controls_menu()
            #     env.display_control_commands()
            render_state = env.sim.viewer_render()
            delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
            time.sleep(delaytime)
            if done:
                state = env.reset(interactive_evaluation=True)
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                env.display_control_commands()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                if hasattr(critic, 'init_hidden_state'):
                    critic.init_hidden_state()
                # Seems like Mujoco only allows a single mjContext(), and it prefers one context
                # with one window when modifying mjModel. So for onscreen dual window, we re-init
                # the non-main window, ie egocentric view here.
                if hasattr(env.sim, 'renderer'):
                    if env.sim.renderer is not None:
                        print("re-init non-primary screen renderer")
                        env.sim.renderer.close()
                        env.sim.init_renderer(offscreen=env.offscreen,
                                              width=env.depth_image_dim[0], height=env.depth_image_dim[1])
                for k,v in plot_dict.items():
                    plot_dict[k] = []

        # clear terminal on ctrl+q
        print(f"\033[{env.num_menu_backspace_lines - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)

def simple_eval_offscreen(actor, env, episode_length_max=300):
    """Simply evaluating policy without visualization

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    with torch.no_grad():
        state = env.reset()
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()

        while True:
            state = torch.Tensor(state).float()
            if actor is None:
                action = np.random.uniform(-0.2, 0.2, env.action_size)
            else:
                action = actor(state).numpy()
            state, reward, done, _ = env.step(action)
            episode_length += 1
            episode_reward.append(reward)
            if episode_length == episode_length_max or done:
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                state = env.reset()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()


def kinematic_replay(env, data_path):
    """Replay of a saved file. Required information in the data file includes:
        'entire_hfield': entire height field
        'qpos': qpos of the trajectory
        'hfield': height field of the trajectory
        'local_grid': local grid of the trajectory for visualization
        'depth': depth image of the trajectory
    """
    files = glob.glob(data_path + '/*.pkl')
    file = np.random.choice(files)
    print(f"Choose data file {file}")
    data = pickle.load(open(file, "rb"))

    # Get some constants
    env.reset()
    # Hard setting for each qpos and iterate them
    env.sim.randomize_hfield(data=data['entire_hfield'])
    traj_idx_max = data['qpos'].shape[0]
    traj_idx = 0
    env.sim.data.qpos = data['qpos'][traj_idx,:]
    env.sim.viewer_init()
    render_state = env.sim.viewer_render()
    while render_state:
        start_time = time.time()
        if not env.sim.viewer_paused():
            # Only set qpos
            env.sim.set_qpos(data['qpos'][traj_idx,:])
            env.sim.viewer.render_hfield(data['hfield'][traj_idx,:], data['local_grid'][traj_idx,:])
            if 'depth' in data:
                img = copy.deepcopy(data['depth'][traj_idx,:])
                img /= np.max(img)
                img = np.clip(img, 0, 1) * 255
                cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Depth", cv2.resize(
                    cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_BONE),
                    (500, 500), interpolation = cv2.INTER_NEAREST))
                cv2.waitKey(1)
            traj_idx += 1
        render_state = env.sim.viewer_render()
        delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
        time.sleep(delaytime)
        if traj_idx == traj_idx_max:
            print(f"Episode length = {traj_idx_max}.")
            # Hard setting for each qpos and iterate them
            file = np.random.choice(files)
            print(f"Choose data file {file}")
            data = pickle.load(open(file, "rb"))
            env.sim.randomize_hfield(data=data['entire_hfield'])
            traj_idx_max = data['qpos'].shape[0]
            traj_idx = 0
            env.sim.data.qpos = data['qpos'][traj_idx,:]


def plot_clock(line0, line1, ax, plt, t, lr_swing_vals, x, y0, y1):
    x.append(t)
    y0.append(1 - lr_swing_vals[0])
    y1.append(1 - lr_swing_vals[1])
    line0.set_data(x, y0)
    line1.set_data(x, y1)
    try:
        ax.set_xlim(x[-80], x[-1])
    except:
        ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.00001)