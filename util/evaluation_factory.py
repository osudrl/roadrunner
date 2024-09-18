import numpy as np
import sys
import termios
import time
import torch

from util.xbox import XboxController
from util.keyboard import Keyboard
from util.colors import OKGREEN, FAIL, WARNING, ENDC
from util.reward_plotter import Plotter

def simple_eval(actor, env, episode_length_max=300):
    """Simply evaluating policy in visualization window and no user input

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

        env.sim.viewer_init(fps = env.default_policy_rate)
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
                episode_length += 1
                episode_reward.append(reward)
                env.viewer_update_cop_marker()
            render_state = env.sim.viewer_render()
            delaytime = max(0, env.default_policy_rate/2000 - (time.time() - start_time))
            time.sleep(delaytime)
            if episode_length == episode_length_max or done:
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward)}.")
                done = False
                state = env.reset()
                episode_length = 0
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                # Seems like Mujoco only allows a single mjContext(), and it prefers one context
                # with one window when modifying mjModel. So for onscreen dual window, we re-init
                # the non-main window, ie egocentric view here.
                if hasattr(env.sim, 'renderer'):
                    if env.sim.renderer is not None:
                        print("re-init non-primary screen renderer")
                        env.sim.renderer.close()
                        env.sim.init_renderer(offscreen=env.offscreen,
                                              width=env.depth_image_dim[0], height=env.depth_image_dim[1])

def interactive_eval(actor, env, episode_length_max=300, critic=None, plot_rewards=False):
    """Simply evaluating policy in visualization window with user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print(f"Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds.{ENDC}")
    keyboard = Keyboard()

    if plot_rewards:
        plotter = Plotter()

    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()
        env.sim.viewer_init(fps = env.default_policy_rate)
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()
        while render_state:
            start_time = time.time()
            cmd = None
            if keyboard.data():
                cmd = keyboard.get_input()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                state, reward, done, infos = env.step(action)
                episode_length += 1
                episode_reward.append(reward)
                if plot_rewards:
                    plotter.add_data(infos, done or cmd == "quit")
                if critic is not None:
                    if hasattr(env, 'get_privilege_state'):
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                    # print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                env.viewer_update_cop_marker()
            if cmd is not None:
                env.interactive_control(cmd)
            if cmd == "r":
                done = True
            if cmd == "menu":
                env.display_control_commands(erase=True)
                env.display_controls_menu()
                env.display_control_commands()
            render_state = env.sim.viewer_render()
            delaytime = max(0, 1/env.default_policy_rate - (time.time() - start_time))
            time.sleep(delaytime)
            if done:
                state = env.reset(interactive_evaluation=True)
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                env.display_control_commands()
                episode_length = 0
                episode_reward = []
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                done = False
        keyboard.restore()
        # clear terminal on ctrl+q
        print(f"\033[{len(env.control_commands_dict) + 3 - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)

def interactive_xbox_eval(actor, env, episode_length_max=300, critic=None, plot_rewards=False):
    """Simply evaluating policy in visualization window with user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    print(f"{OKGREEN}Feeding xbox inputs to policy for interactive eval mode.{ENDC}")
    xbox = XboxController()

    if plot_rewards:
        plotter = Plotter()

    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()
        env.sim.viewer_init(fps = env.default_policy_rate)
        render_state = env.sim.viewer_render()
        env.display_xbox_controls_menu()
        env.display_control_commands()
        while render_state:
            start_time = time.time()
            if not env.sim.viewer_paused():
                state = torch.Tensor(state).float()
                action = actor(state).numpy()
                state, reward, done, infos = env.step(action)
                episode_length += 1
                episode_reward.append(reward)
                if plot_rewards:
                    plotter.add_data(infos, done)
                if critic is not None:
                    if hasattr(env, 'get_privilege_state'):
                        critic_state = env.get_privilege_state()
                    else:
                        critic_state = state
                    # print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                env.viewer_update_cop_marker()
            env.interactive_xbox_control(xbox)
            if xbox.RightBumper == 1 and xbox.LeftBumper == 0:
                if xbox.Start == 1 and not xbox.Start_pressed:
                    break
                elif xbox.Start_pressed and xbox.Start == 0:
                    xbox.Start_pressed = False
                if xbox.Back == 1 and not xbox.Back_pressed:
                    xbox.Back_pressed = True
                    state = env.reset(interactive_evaluation=True)
                    env.display_control_commands(erase=True)
                    print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                    env.display_control_commands()
                    episode_length = 0
                    episode_reward = []
                    if hasattr(actor, 'init_hidden_state'):
                        actor.init_hidden_state()
                    done = False
                elif xbox.Back_pressed and xbox.Back == 0:
                    xbox.Back_pressed = False
            if xbox.RightBumper == 0 and xbox.LeftBumper == 0:
                if xbox.Start == 1 and not xbox.Start_pressed:
                    xbox.Start_pressed = True
                    env.sim.viewer.paused = not env.sim.viewer.paused
                elif xbox.Start_pressed and xbox.Start == 0:
                    xbox.Start_pressed = False
            if xbox.RightBumper == 1 and xbox.LeftBumper == 1:
                if xbox.Start == 1 and not xbox.Start_pressed:
                    xbox.Start_pressed = True
                    env.display_control_commands(erase=True)
                    env.display_xbox_controls_menu()
                    env.display_control_commands()
                elif xbox.Start_pressed and xbox.Start == 0:
                    xbox.Start_pressed = False
               
            render_state = env.sim.viewer_render()
            delaytime = max(0, 1/env.default_policy_rate - (time.time() - start_time))
            time.sleep(delaytime)
            if done:
                state = env.reset(interactive_evaluation=True)
                env.display_control_commands(erase=True)
                print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                env.display_control_commands()
                episode_length = 0
                episode_reward = []
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state()
                done = False
        # clear terminal on ctrl+q
        print(f"\033[{len(env.control_commands_dict) + 3 - 1}B\033[K")
        termios.tcdrain(sys.stdout)
        time.sleep(0.1)
        termios.tcflush(sys.stdout, termios.TCIOFLUSH)

def slowmo_interactive_eval(actor, env, episode_length_max=300, slowmo=4, critic=None):
    """Simply evaluating policy in visualization window with user input

    Args:
        actor: Actor loaded outside this function. If Actor is None, this function will evaluate
            noisy actions without any policy.
        env: Environment instance for actor
        episode_length_max (int, optional): Max length of episode for evaluation. Defaults to 500.
    """
    if actor is None:
        raise RuntimeError(F"{FAIL}Interactive eval requires a non-null actor network for eval")

    hw_step_counter = int(env.sim.simulator_rate / env.default_policy_rate)
    if slowmo > hw_step_counter:
        print(f'{WARNING}The slowmo factor provided exceeds the maximum possible. Rounding down to the max possible: {hw_step_counter}\n')
    elif hw_step_counter % slowmo != 0:
        while hw_step_counter % slowmo != 0:
            slowmo += 1
        print(f'{WARNING}The slowmo factor provided was uneven to the fps value. It has been rounded up to: {slowmo}\n')
    steps_per_save = int(hw_step_counter / slowmo)
    print(f"{OKGREEN}Feeding keyboard inputs to policy for interactive eval mode.")
    print("Type commands into the terminal window to avoid interacting with the mujoco viewer keybinds." + '\033[0m')
    keyboard = Keyboard()

    with torch.no_grad():
        state = env.reset(interactive_evaluation=True)
        done = False
        episode_length = 0
        episode_reward = []

        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state()
        if hasattr(critic, 'init_hidden_state'):
            critic.init_hidden_state()
        env.sim.viewer_init()
        render_state = env.sim.viewer_render()
        env.display_controls_menu()
        env.display_control_commands()
        pol_counter = 0
        render_counter = 0
        state = torch.Tensor(env.get_state()).float()
        action = actor(state).numpy()
        while render_state:
            start_time = time.time()
            cmd = None
            if keyboard.data():
                cmd = keyboard.get_input()
            if cmd is not None:
                 env.interactive_control(cmd)
            if cmd == "menu":
                env.display_control_commands(erase=True)
                env.display_controls_menu()
                env.display_control_commands()
            if cmd == "quit":
                done = True

            if not env.sim.viewer_paused():
                # If we've taken enough 2kHz steps for a hw_step
                if pol_counter == hw_step_counter:
                    pol_counter = 0
                    # Check done here to ensure the keyboard command does not get overridden
                    if not done:
                        state = torch.Tensor(env.get_state()).float()
                        action = actor(state).numpy()
                        done = env.compute_done()
                        env.compute_reward(action)
                        reward = env.reward
                        episode_length += 1
                        episode_reward.append(reward)
                        env.hw_step()
                    if critic is not None:
                        if hasattr(env, 'get_privilege_state'):
                            critic_state = env.get_privilege_state()
                        else:
                            critic_state = state
                        # print(f"Critic value = {critic(torch.Tensor(critic_state)).numpy() if critic is not None else 'N/A'}")
                    if done:
                        state = env.reset(interactive_evaluation=True)
                        env.display_control_commands(erase=True)
                        print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                        env.display_control_commands()
                        episode_length = 0
                        if hasattr(actor, 'init_hidden_state'):
                            actor.init_hidden_state()
                        done = False
                # If we've taken enough 2kHz steps to call render, and a frame is saved here if we're currently recording
                if render_counter == steps_per_save:
                    render_counter = 0
                    render_state = env.sim.viewer_render()
                    env.viewer_update_cop_marker()
                    delaytime = max(0, (steps_per_save * 0.0005) - (time.time() - start_time))
                    time.sleep(delaytime)
                # Take a single 2kHz step and update the counter variables
                env.step_simulation(action, 1)
                render_counter += 1
                pol_counter += 1
            else:
                # If the sim is reset while paused, ensures that's properly handled without unpausing the sim
                if done:
                    state = env.reset(interactive_evaluation=True)
                    env.display_control_commands(erase=True)
                    print(f"Episode length = {episode_length}, Average reward is {np.mean(episode_reward) if episode_reward else 0}.")
                    env.display_control_commands()
                    episode_length = 0
                    if hasattr(actor, 'init_hidden_state'):
                        actor.init_hidden_state()
                    done = False
                    # Ensures we get a new action and the counter is reset
                    pol_counter = hw_step_counter
                render_state = env.sim.viewer_render()
                delaytime = max(0, 1/env.default_policy_rate - (time.time() - start_time))
                time.sleep(delaytime)

        keyboard.restore()
        # clear terminal on ctrl+q
        print(f"\033[{len(env.control_commands_dict) + 3 - 1}B\033[K")
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
