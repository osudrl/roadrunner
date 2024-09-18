import argparse, time, pickle, platform, socket, json
import os, sys, datetime
import select, termios, tty, atexit
from math import floor
# import cv2

import numpy as np
import torch
from multiprocessing import Process, Manager

from algo.common.network import *
from algo.common.tarsus_patch_wrapper import TarsusPatchWrapper
from algo.common.utils import load_actor
from env.cassie.cassiepede.cassiepede import Cassiepede
from nn import LSTMActor
from sim.cassie_sim.cassiemujoco.cassieUDP import *
from sim.cassie_sim.cassiemujoco.cassiemujoco_ctypes import *
from env.util.quaternion import (
    euler2quat,
    inverse_quaternion,
    rotate_by_quaternion,
    quaternion_product,
    quaternion2euler
)

from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory

from util.topic import Topic
import logging
import wandb
import tty
import select


# entry file for run a specified udp setup
# cassie-async (sim), digit-ar-control-async (sim), cassie-real, digit-real
def remap(val, min1, max1, min2, max2):
    span1 = max1 - min1
    span2 = max2 - min2
    scaled = (val - min1) / span1
    return np.clip(min2 + (scaled * span2), min2, max2)


def save_log():
    global log_hf_ind, log_lf_ind, logdir, part_num, sto_num, time_hf_log, output_log, state_log, target_log, speed_log, orient_log, phaseadd_log, hm_log, time_lf_log #, input_log

    filename = "logdata_part" + str(part_num) + "_sto" + str(sto_num) + ".pkl"
    filename = os.path.join(logdir, filename)
    print("Logging to {}".format(filename))
    print("exit at time {}".format(time_hf_log[log_hf_ind - 1]))
    print("save log: log_hf_ind {}".format(log_hf_ind))
    data = {"highfreq": True,
            "time_hf": time_hf_log[:log_hf_ind],
            "time_lf": time_lf_log[:log_lf_ind],
            "output": output_log[:log_hf_ind],
            # "input": input_log[:log_lf_ind],
            "state": state_log[:log_hf_ind],
            "target": target_log[:log_hf_ind],
            "speed": speed_log[:log_hf_ind],
            "orient": orient_log[:log_hf_ind],
            "phase_add": phaseadd_log[:log_hf_ind],
            "simrate": 50}
    with open(filename, "wb") as filep:
        pickle.dump(data, filep)
    part_num += 1


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


# 2 kHz execution : PD control with or without baseline action
def PD_step(cassie_udp, cassie_env, action):
    target = action[:] + cassie_env.offset

    u = pd_in_t()
    for i in range(5):
        u.leftLeg.motorPd.pGain[i] = cassie_env.kp[0, i]
        u.rightLeg.motorPd.pGain[i] = cassie_env.kp[0, i + 5]

        u.leftLeg.motorPd.dGain[i] = cassie_env.kd[0, i]
        u.rightLeg.motorPd.dGain[i] = cassie_env.kd[0, i + 5]

        u.leftLeg.motorPd.torque[i] = 0  # Feedforward torque
        u.rightLeg.motorPd.torque[i] = 0

        u.leftLeg.motorPd.pTarget[i] = target[i]
        u.rightLeg.motorPd.pTarget[i] = target[i + 5]

        u.leftLeg.motorPd.dTarget[i] = 0
        u.rightLeg.motorPd.dTarget[i] = 0

    cassie_udp.send_pd(u)

    # return log data
    return target


def execute(policy, env, args, do_log, exec_rate=1):
    global log_size, log_hf_ind, log_lf_ind, part_num, sto_num, save_dict, time_hf_log, output_log, state_log, target_log, speed_log, orient_log, phaseadd_log, time_lf_log #, input_log

    # Determine whether running in simulation or on the robot
    print('Running on ' + platform.node())
    if "cassie" in platform.node():
        cassieudp = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                              local_addr='10.10.10.100', local_port='25011')
    else:
        cassieudp = CassieUdp()  # local testing
        print('Running on local machine')

    match platform.node():
        case "cassie-harvard":
            my_address = "192.168.2.29", 1234
            plot_client_address = "192.168.2.97", 1234
        case "cassie":
            my_address = "192.168.2.251", 1234
            plot_client_address = "192.168.2.97", 1235
        case "cassie-play":
            my_address = "192.168.2.252", 1234
            plot_client_address = "192.168.2.97", 1236
        case _:
            raise ValueError(f"Unknown platform: {platform.node()}")

    # Get IMU data from POI
    poi_topic = Topic(freq=100)
    poi_topic.subscribe(my_address=my_address)
    poi_yaw_offset = 0

    yaw_locked = np.all(env.encoding == 0)

    print('yaw_locked', yaw_locked)

    # Send data for plotting
    plot_topic = Topic(fetch=False)
    plot_topic.subscribe(my_address=(my_address[0], 1235))

    # # This will override use of radio controller and expect commands from the remote computer
    cmd_topic = Topic(timeout=3)
    cmd_topic.subscribe(my_address=(my_address[0], 1237))

    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    if exec_rate > env.default_policy_rate:
        print("Error: Execution rate can not be greater than simrate")
        exit()
    # Lock exec_rate to even dividend of simrate
    rem = env.default_policy_rate // exec_rate
    exec_rate = env.default_policy_rate // rem
    print("Execution rate: {} ({:.2f} Hz)".format(exec_rate, 2000 / exec_rate))

    # ESTOP position. True means ESTOP enabled and robot is not running.
    STO = False
    logged = False
    part_num = 0
    sto_num = 0
    save_log_p = None
    # env.reset() # Don't even reset env, so we won't use any simulator stuff
    env.hardware_mode = True
    env.turn_rate_poi = 0
    env.y_velocity_poi = 0
    env.x_velocity_poi = 0
    env.height_poi = 0.75

    if args.clock_type:
        from env.util.periodicclock import PeriodicClock
        env.clock = []

        for i in range(env.num_cassie):
            clock = PeriodicClock(cycle_time=0.6,
                                  phase_add=1 / env.default_policy_rate,
                                  swing_ratios=[0.5, 0.5],
                                  period_shifts=[0, 0.5])
            clock._phase = 0
            clock._von_mises_buf = None

            env.clock.append(clock)

    # 0: walking
    # 1: standing
    # 2: damping
    action = None
    operation_mode = 0
    D_mult = 1  # Reaaaaaally bad stability problems if this is pushed higher as a multiplier
    # Might be worth tuning by joint but something else if probably needed

    empty_u = pd_in_t()
    damp_u = pd_in_t()
    for i in range(5):
        empty_u.leftLeg.motorPd.pGain[i] = 0.0
        empty_u.leftLeg.motorPd.dGain[i] = 0.0
        empty_u.rightLeg.motorPd.pGain[i] = 0.0
        empty_u.rightLeg.motorPd.dGain[i] = 0.0
        empty_u.leftLeg.motorPd.pTarget[i] = 0.0
        empty_u.rightLeg.motorPd.pTarget[i] = 0.0

        damp_u.leftLeg.motorPd.pGain[i] = 0.0
        damp_u.leftLeg.motorPd.dGain[i] = D_mult * env.kd[0, i]
        damp_u.rightLeg.motorPd.pGain[i] = 0.0
        damp_u.rightLeg.motorPd.dGain[i] = D_mult * env.kd[0, i + 5]
        damp_u.leftLeg.motorPd.pTarget[i] = 0.0
        damp_u.rightLeg.motorPd.pTarget[i] = 0.0

    old_settings = termios.tcgetattr(sys.stdin)
    count = 0
    pol_time = 0
    state_count = 0

    # Connect to the simulator or robot
    print('Connecting RTOS...')
    state = None
    while state is None:
        cassieudp.send_pd(pd_in_t())
        time.sleep(0.001)
        state = cassieudp.recv_newest_pd()
    print('Connected to RTOS!\n')

    try:
        tty.setcbreak(sys.stdin.fileno())

        t = time.monotonic()
        pol_time = 0
        first = True
        while True:

            # Get newest state
            t = time.monotonic()
            state = cassieudp.recv_newest_pd()
            while state is None:
                state_count += 1
                time.sleep(0.0001 * exec_rate)
                state = cassieudp.recv_newest_pd()

            # No continue
            if 'cassie' in platform.node():

                # for i in range(len(state.radio.channel)):
                #     print(f'Channel {i}: {state.radio.channel[i]:.2f}')
                # print()

                cmd_remote = None
                if cmd_topic is not None:
                    cmd_remote = cmd_topic.get_data()
                    if cmd_remote is not None:
                        cmd_remote = json.loads(cmd_remote.decode('utf-8'))

                # print('cmd_remote', cmd_remote)

                keyboard_override = state.radio.channel[12] != -1

                if not keyboard_override or cmd_remote is None:
                    l_stick_x = state.radio.channel[0]
                    l_stick_y = state.radio.channel[1] + (0.16483516991138458 if 'harvard' in platform.node() else 0)
                    r_stick_y = state.radio.channel[3]
                    height_offset = 0.0
                else:
                    l_stick_x = cmd_remote['l_stick_x']
                    l_stick_y = cmd_remote['l_stick_y']
                    r_stick_y = cmd_remote['r_stick_y']
                    height_offset = cmd_remote['height_offset']

                # print('cmd_remote', cmd_remote)

                # Control with Taranis radio controller
                if state.radio.channel[9] < -0.5 or (
                        cmd_remote is not None and keyboard_override and cmd_remote['operation_mode'] == 2):
                    operation_mode = 2  # down -> damping
                elif state.radio.channel[9] > 0.5:
                    operation_mode = 1  # up -> nothing
                else:
                    operation_mode = 0  # mid -> normal walking

                base_orient = np.array(env.sim.get_base_orientation(state_est=env.state_est)).reshape(env.num_cassie,
                                                                                                      -1)

                # print('state.radio.channel', list(state.radio.channel))

                # Fetch data from POI
                poi_yaw = poi_topic.get_data()

                # print('poi yaw', poi_yaw)
                if poi_yaw is not None:
                    poi_yaw = float(poi_yaw.decode('utf-8'))
                else:
                    poi_yaw = 0.0

                # Reset orientation on STO
                if state.radio.channel[8] < 0 or (
                        cmd_remote is not None and keyboard_override and cmd_remote['STO'] == 1):
                    STO = True
                    env.sim.robot_estimator_state = state

                    # Calibrate
                    env.orient_add = quaternion2euler(base_orient)[:, -1]

                    # print('orient_add', np.degrees(env.orient_add))
                    poi_yaw_offset = env.orient_add[0] - poi_yaw
                else:
                    STO = False
                    logged = False

                if state.radio.channel[-1] < 0:
                    print('Reset')
                    # poi_yaw_offset = quaternion2euler(base_orient)[0, -1] - poi_yaw

                    env.sim.robot_estimator_state = state

                    # Calibrate
                    env.orient_add = quaternion2euler(base_orient)[:, -1]
                    poi_yaw_offset = env.orient_add[0] - poi_yaw

                # print('poi_yaw_offset', poi_yaw_offset)

                # Example of setting things manually instead. Reference to what radio channel corresponds to what joystick/knob:
                # https://github.com/agilityrobotics/cassie-doc/wiki/Radio#user-content-input-configuration
                # Radio control deadzones

                # for i in range(len(state.radio.channel)):
                #     print(f'Channel {i}: {state.radio.channel[12]:.2f}')
                # print()

                if state.radio.channel[13] == -1:
                    env.encoding = np.array([[args.encoding[0], np.radians(args.encoding[1])]])
                else:
                    l_knob = state.radio.channel[6]
                    r_knob = state.radio.channel[7]
                    # default encoding
                    r = remap(l_knob, -1, 1, 0, 1.0)
                    theta = remap(r_knob, -1, 1, -np.pi, np.pi)
                    env.encoding = np.array([[r, theta]])

                # if 'harvard' in platform.node() and cmd_remote is not None:
                #     # Problem with the radio controller, need to add a constant offset
                #     l_stick_y += 0.16483516991138458

                if abs(l_stick_x) < 0.05:
                    l_stick_x = 0
                if abs(l_stick_y) < 0.05:
                    l_stick_y = 0
                if abs(r_stick_y) < 0.05:
                    r_stick_y = 0

                # print('l_stick_x', l_stick_x, 'l_stick_y', l_stick_y, 'r_stick_y', r_stick_y)
                # Orientation control (Do manually instead of turn_rate)
                # X and Y speed control
                env.x_velocity_poi = remap(l_stick_x, -1, 1, -1.0, 1.0)
                env.y_velocity_poi = -remap(r_stick_y, -1, 1, -0.5, 0.5)
                env.turn_rate_poi = -remap(l_stick_y, -1, 1, -0.5, 0.5)
                env.height_base = remap(state.radio.channel[5] + height_offset, -1, 1, 0.5, 0.8)

                # print('env.height_base', env.height_base)

                env.x_velocity_poi = np.clip(env.x_velocity_poi, -0.3, 0.3)
                env.y_velocity_poi = np.clip(env.y_velocity_poi, -0.3, 0.3)
                env.turn_rate_poi = np.clip(env.turn_rate_poi, -np.radians(7.5), np.radians(7.5))
                env.height_base = np.clip(env.height_base, 0.5, 0.8)

                env.x_velocity_poi = np.array([env.x_velocity_poi])
                env.y_velocity_poi = np.array([env.y_velocity_poi])
                env.turn_rate_poi = np.array([env.turn_rate_poi])
                env.height_base = np.array([env.height_base])

                # print('env.y_velocity_poi', env.y_velocity_poi)

                # print('orient_add', np.degrees(env.orient_add))
                # print('l_stick_y', l_stick_y, 'l_stick_x', l_stick_x, 'r_stick_y', r_stick_y)
                # print('x_velocity_poi', env.x_velocity_poi, 'y_velocity_poi', env.y_velocity_poi, 'turn_rate_poi', env.turn_rate_poi)

                # if args.keyboard and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                #     input_char = sys.stdin.read(1)
                #     match input_char:
                #         case "a":
                #             env.y_velocity_poi -= 0.05
                #         case "d":
                #             env.y_velocity_poi += 0.05
                #         case "w":
                #             env.x_velocity_poi += 0.05
                #         case "s":
                #             env.x_velocity_poi -= 0.05
                #         case "q":
                #             env.turn_rate_poi += np.radians(1)
                #         case "e":
                #             env.turn_rate_poi -= np.radians(1)
                #         case abc:
                #             print(f"Invalid input: {abc}")

                # print('turn_rate_poi', env.turn_rate_poi, 'orient_add', np.degrees(env.orient_add))
                # Gait parameters control

                if args.clock_type:
                    if not hasattr(env, 'autoclock'):
                        cycle_time = remap(state.radio.channel[5], -1, 1, 0.6, 1.0)

                        for i in range(env.num_cassie):
                            env.clock[i].set_cycle_time(cycle_time)

            else:
                """
                    Control of the robot in simulation using a keyboard
                """

                if isData():
                    c = sys.stdin.read(1)
                    if c == 'x':
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                    elif c == 't':
                        STO = True
                        print("\nESTOP enabled")
                    else:
                        env.interactive_control(c)

            if STO:
                if not logged:
                    logged = True
                    save_log()
                    sto_num += 1
                    part_num = 0
                    log_hf_ind = 0
                    log_lf_ind = 0

            curr_state = state
            # Continue to update state while sleeping to hit desired script frequency
            while state is None or time.monotonic() - t < exec_rate / 2000:
                state_count += 1
                time.sleep(0.0001 * exec_rate)
                curr_state = cassieudp.recv_newest_pd()
                if curr_state:
                    state = curr_state

            env.sim.robot_estimator_state = state

            # ------------------------------- Normal Walking ---------------------------
            if operation_mode == 0:
                count += 1
                update_time = time.monotonic() - pol_time

                if first or update_time > 1 / env.default_policy_rate:

                    """
                        Low frequency (Policy Rate) Section. Update policy action
                    """
                    # cnt_from_camera = perception_state[0]
                    # print("cnt_from_camera ", cnt_from_camera)
                    # print("time here ", time.process_time())
                    # print("time diff, ", 1000 * (time.process_time() - cnt_from_camera))
                    # print("inference with delay time ", 1e3*(t_state - t_perception))

                    # RL_state = env.get_state()['base']

                    RL_state = env.get_state()

                    # Override the base_yaw (this is at 4th index)
                    poi_yaw += poi_yaw_offset
                    # Normalize poi_yaw to be between -pi and pi
                    poi_yaw = (poi_yaw + np.pi) % (2 * np.pi) - np.pi

                    # print('poi_yaw before', np.degrees(poi_yaw))

                    base_yaw_poi = env.get_encoder_yaw(poi_yaw=poi_yaw)

                    # print('base_yaw_poi after', np.degrees(base_yaw_poi))

                    # print('base yaw',RL_state[:, 4])
                    # Uncomment for more than one cassie

                    if not yaw_locked:
                        RL_state['base_yaw_poi'][:] = base_yaw_poi

                    # print('yaw from state', RL_state['base_yaw_poi'])

                    # print('poi_yaw', np.degrees(poi_yaw), 'orient_add', np.degrees(env.orient_add), 'base_yaw_poi',
                    #       np.degrees(base_yaw_poi))

                    # Publish: r, theta, base_yaw_poi, base_orient_cmd.
                    # All these are in state space, so we will decompose the deck structure in the plot

                    # Base orientation in command frame
                    # base_orient_cmd = quaternion2euler(env.rotate_to_heading(base_orient))[0, -1]

                    # turn_rate_poi += env.turn_rate_poi
                    # print('env.y_velocity_poi', env.turn_rate_poi)
                    data_to_send = np.array([

                        RL_state['cmd'][:, 0].item(),
                        RL_state['cmd'][:, 1].item(),
                        RL_state['cmd'][:, 2].item(),
                        RL_state['cmd'][:, 3].item(),
                        RL_state['encoding'][:, 0].item(),
                        RL_state['encoding'][:, 1].item(),
                        RL_state['base_yaw_poi'].item(),
                        quaternion2euler(RL_state['base_orient_cmd'])[:, -1].item(),
                        quaternion2euler(base_orient)[:, -1].item()

                    ]

                    ).astype(np.float16)

                    # print('data_to_send', data_to_send)

                    # print(
                    #     f'r: {data_to_send[0]:.2f}\ntheta: {data_to_send[1]:.2f}\nbase_yaw_poi: {data_to_send[2]:.2f}\nbase_orient_cmd: {data_to_send[3]:.2f}')

                    plot_topic.publish(data_to_send.tobytes(), plot_client_address)

                    # # bart
                    # RL_state = RL_state[0]

                    for k in RL_state.keys():
                        RL_state[k] = torch.tensor(RL_state[k], dtype=torch.float32, device='cpu')

                    with torch.no_grad():
                        # bikram, ashutosh
                        action, _ = policy(RL_state)
                        action = action.numpy().squeeze(0)

                        # # bart
                        # action = policy(torch.tensor(RL_state).float(), deterministic=True).numpy()

                    # print('action', action)

                    # action = np.zeros(10)
                    target = PD_step(cassieudp, env, action)
                    pol_time = time.monotonic()
                    # Update env quantities
                    env.orient_add += env.turn_rate_poi / env.default_policy_rate
                    # Gait parameters control
                    if args.clock_type:
                        if hasattr(env, 'autoclock'):
                            if env.autoclock:
                                env.update_clock(action[10:env.action_size])

                        for i in range(env.num_cassie):
                            env.clock[i].increment()

                    if do_log:
                        time_lf_log[log_lf_ind] = time.time()
                        # input_log[log_lf_ind] = RL_state
                        target_log[log_lf_ind] = target
                        log_lf_ind += 1

                    # hm = env.hardware_perception_state.reshape(20, 30).T[::-1, ::-1]
                    # hm = np.mean(hm, axis=1)
                    # hm = np.around(hm, decimals=2)
                    # print(hm)
                    # print(f"Max height diff: {max(hm) - min(hm): .2f}")
                    # Measure delay
                    measured_delay = (update_time - 1 / env.default_policy_rate) * 1000
                    # print(env.sim.robot_estimator_state.joint.position[:])
                    # print(env.sim.robot_estimator_state.pelvis.orientation[:])
                    # print(env.sim.robot_estimator_state.leftFoot.position[:], env.sim.robot_estimator_state.rightFoot.position[:])
                    if not first:
                        sys.stdout.write(f"Speed: {env.x_velocity_poi[0]:1.2f}\t, "
                                         f"Policy Inference delay = {measured_delay: 2.2f}ms, \t"
                                         #  f"Max height diff: {np.max(env.hardware_perception_state) - np.min(env.hardware_perception_state): .2f}, \t"
                                         f"xvel={env.sim.robot_estimator_state.pelvis.translationalVelocity[:][0]: 1.1f}\r")
                        sys.stdout.flush()
                    first = False
                    count = 0
                    # pol_time = new_time

                """
                    High frequency (2000 Hz) Section
                """

                if do_log:
                    time_hf_log[log_hf_ind] = time.time()
                    output_log[log_hf_ind] = action
                    state_log[log_hf_ind] = state
                    speed_log[log_hf_ind] = env.x_velocity_poi
                    orient_log[log_hf_ind] = env.orient_add
                    if args.clock_type:
                        phaseadd_log[log_hf_ind] = env.clock[0]._cycle_time
                    log_hf_ind += 1

                if log_hf_ind == log_size and do_log:
                    if save_log_p is not None:
                        save_log_p.join()
                    save_log_p = Process(target=save_log)
                    save_log_p.start()
                    part_num += 1
                    log_hf_ind = 0
                    log_lf_ind = 0

                # runs faster up to 2khz, ideally policy rate, but need to fetch state in a faster rate
                delaytime = 1 / 1000 - (time.monotonic() - t)
                while delaytime > 0:
                    t0 = time.monotonic()
                    time.sleep(1 / 10000)
                    delaytime -= time.monotonic() - t0
                # print(f"(Expected) Running at = {1/(time.monotonic() - t)}")

            # ------------------------------- Empty Action ---------------------------
            elif operation_mode == 1:
                print('Applying no action')
                # Do nothing
                cassieudp.send_pd(empty_u)

            # ------------------------------- Shutdown Damping ---------------------------
            elif operation_mode == 2:
                # print('Shutdown Damping. Multiplier = ' + str(D_mult))
                cassieudp.send_pd(damp_u)

            # ---------------------------- Other, should not happen -----------------------
            else:
                print('Error, In bad operation_mode with value: ' + str(operation_mode))


    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")

    parser.add_argument('--action_dim', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=4)
    parser.add_argument('--set_adam_eps', action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--std', type=float, default=0.13)
    parser.add_argument('--use_orthogonal_init', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='roadrunner_cassiepede')
    parser.add_argument('--run_name', type=str, default="2024-02-16 02:41:50.685098")
    parser.add_argument('--model_checkpoint', type=str, default="latest")
    parser.add_argument('--reward_name', type=str, default='locomotion_cassiepede')
    parser.add_argument('--clock_type', type=str, required=False)
    parser.add_argument('--redownload_checkpoint', action='store_true')
    parser.add_argument("--mask_tarsus_input", action='store_true', help="Mask tarsus input with zeros")
    parser.add_argument('--do_log', action='store_true')
    parser.add_argument('--keyboard', action='store_true')
    parser.add_argument('--encoding', type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--exec_rate", default=1, type=int,
                        help="Controls the execution rate of the script. Is 1 (full 2kHz) be default")

    args = parser.parse_args()

    env = Cassiepede(
        clock_type=args.clock_type,
        reward_name=args.reward_name,
        simulator_type="libcassie",
        policy_rate=50,
        dynamics_randomization=False,
        state_noise=0,
        velocity_noise=0.0,
        state_est=True,
        full_clock=True,
        full_gait=False,
        integral_action=False,
        com_vis=False,
        depth_input=False,
        num_cassie=1,
        custom_terrain=None,
        poi_position_offset=0.0,
        perturbation_force=0.0,
        only_deck_force=False,
        height_control=True,
        merge_states=False,
        position_offset=0.0,
        poi_heading_range=0.0,
        mask_tarsus_input=args.mask_tarsus_input,
        offscreen=False)

    args.state_dim = env.observation_size

    # env._compute_encoding(poi_position=np.array([0, 0]),
    #                       poi_orientation=0,
    #                       base_positions=np.array([[1.0, 0]]))
    #
    # env.encoding = np.array([[1.0, np.pi / 4]])
    env.encoding = np.array([[args.encoding[0], np.radians(args.encoding[1])]])

    if args.reward_name == 'locomotion_cassiepede_clock_stand':
        env.stand = False

    env.eval(True)
    # #
    # bikram
    actor = load_actor(args, device=torch.device('cpu'), model_fn=Actor_LSTM_v2)

    # # bart
    # previous_args_dict = pickle.load(open(os.path.join('./pretrained_models/LocomotionEnv/cassie-LocomotionEnv/10-27-17-03/', "experiment.pkl"), "rb"))
    # actor_checkpoint = torch.load(os.path.join('./pretrained_models/LocomotionEnv/cassie-LocomotionEnv/10-27-17-03/', 'actor.pt'), map_location='cpu')
    # actor, _ = nn_factory(args=previous_args_dict['nn_args'], env=None)
    # load_checkpoint(model=actor, model_dict=actor_checkpoint)

    # # ashutosh
    # actor = Actor_LSTM_v2(args)
    # actor.load_state_dict(torch.load('cmb_333m.pt'))

    # wrap actor in tarsus predictor:
    # if platform.node() == "cassie":
    #     # This is osu cassie and need tarsus predictor
    #     actor = TarsusPatchWrapper(actor)

    print("Model name: ", actor.__class__.__name__)
    print('Encoding: ', env.encoding)

    actor.eval()
    actor.training = False
    # LOG_NAME = args.path.rsplit('/', 3)[1] + "/"
    LOG_NAME = args.run_name + "/"
    directory = os.path.dirname(os.path.realpath(__file__)) + "/hardware_logs/"
    filename = "logdata"
    timestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    if not os.path.exists(directory + timestr + LOG_NAME + "/"):
        os.makedirs(directory + LOG_NAME + timestr + "/")
    logdir = directory + LOG_NAME + timestr + "/"
    filename = directory + LOG_NAME + timestr + "/" + filename + ".pkl"

    # Global data for logging
    log_size = 100000
    log_lf_ind = 0
    log_hf_ind = 0
    time_lf_log = [time.time()] * log_size  # time stamp
    time_hf_log = [time.time()] * log_size  # time stamp
    # input_log = [np.ones(args.state_dim)] * log_size  # network inputs
    output_log = [np.ones(args.action_dim)] * log_size  # network outputs
    state_log = [state_out_t()] * log_size  # cassie state
    target_log = [np.ones(10)] * log_size  # PD target log
    speed_log = [0.0] * log_size  # speed input commands
    orient_log = [0.0] * log_size  # orient input commands
    phaseadd_log = [0.0] * log_size  # frequency input commands

    part_num = 0
    sto_num = 0

    if args.do_log:
        atexit.register(save_log)
    execute(actor, env, args, args.do_log, args.exec_rate)
