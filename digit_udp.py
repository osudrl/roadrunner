import agility
import agility.messages as msg
import argparse
import asyncio
import datetime
import numpy as np
import sys
import os
import platform
import pickle
import select
import signal
import sys
import termios
import time
import torch
import tty

from env.util.quaternion import us2scipy
from multiprocessing import Process
from scipy.spatial.transform import Rotation as R
from sim.digit_sim.digit_ar_sim.digit_udp import DigitUdp
from sim.digit_sim.digit_ar_sim.interface_ctypes import *
from testing.common import (
    DIGIT_JOINT_MJ2LLAPI_INDEX,
    DIGIT_MOTOR_MJ2LLAPI_INDEX,
    DIGIT_JOINT_LLAPI2MJ_INDEX,
    DIGIT_MOTOR_LLAPI2MJ_INDEX,
    MOTOR_POSITION_SET
)
from util.digit_topic import DigitStateTopic
from util.env_factory import env_factory
from util.nn_factory import load_checkpoint, nn_factory

MOTOR_NAMES = ["left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-foot",
               "left-shoulder-roll", "left-shoulder-pitch", "left-shoulder-yaw", "left-elbow",
               "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-foot",
               "right-shoulder-roll", "right-shoulder-pitch", "right-shoulder-yaw", "right-elbow"]
ROBOT_STATE_NAMES = ["base-orientation-w", "base-orientation-x", "base-orientation-y", "base-orientation-z",
                     "base-roll-velocity", "base-pitch-velocity", "base-yaw-velocity",
                     "left-hip-roll-pos", "left-hip-yaw-pos", "left-hip-pitch-pos", "left-knee-pos", "left-foot-a-pos", "left-foot-b-pos",
                     "left-shoulder-roll-pos", "left-shoulder-pitch-pos", "left-shoulder-yaw-pos", "left-elbow-pos",
                     "right-hip-roll-pos", "right-hip-yaw-pos", "right-hip-pitch-pos", "right-knee-pos", "right-foot-a-pos", "right-foot-b-pos",
                     "right-shoulder-roll-pos", "right-shoulder-pitch-pos", "right-shoulder-yaw-pos", "right-elbow-pos",
                     "left-hip-roll-vel", "left-hip-yaw-vel", "left-hip-pitch-vel", "left-knee-vel", "left-foot-a-vel", "left-foot-b-vel",
                     "left-shoulder-roll-vel", "left-shoulder-pitch-vel", "left-shoulder-yaw-vel", "left-elbow-vel",
                     "right-hip-roll-vel", "right-hip-yaw-vel", "right-hip-pitch-vel", "right-knee-vel", "right-foot-a-vel", "right-foot-b-vel",
                     "right-shoulder-roll-vel", "right-shoulder-pitch-vel", "right-shoulder-yaw-vel", "right-elbow-vel",
                     "left-shin-pos", "left-tarsus-pos", "left-heel-spring-pos", "left-toe-pitch-pos", "left-toe-roll-pos",
                     "right-shin-pos", "right-tarsus-pos", "right-heel-spring-pos", "right-toe-pitch-pos", "right-toe-roll-pos",
                     "left-shin-vel", "left-tarsus-vel", "left-heel-spring-vel", "left-toe-pitch-vel", "left-toe-roll-vel",
                     "right-shin-vel", "right-tarsus-vel", "right-heel-spring-vel", "right-toe-pitch-vel", "right-toe-roll-vel"]

def save_log():
    global logdir, log_size, part_num, log_ind, time_log, input_log, output_log, phase_log, orient_add_log

    filename = os.path.join(logdir, f"logdata_part{part_num}.pkl")
    print("Logging to {}".format(filename))
    # Truncate log data to actual size
    if log_ind < log_size:
        for key, val in input_log.items():
            input_log[key] = val[:log_ind]
        for key, val in output_log.items():
            output_log[key] = val[:log_ind]

    data = {"time": time_log[:log_ind],
            "output": output_log,
            "input": input_log,
            "phase": phase_log[:log_ind],
            "orient_add": orient_add_log[:log_ind]}
    with open(filename, "wb") as filep:
        pickle.dump(data, filep)
    part_num += 1

def signal_handler(sig, frame):
    global final_save
    # Need extra check here in case need multiple ctrl-c to exit
    if not final_save:
        save_log()
        final_save = True
    sys.exit(0)

async def run(actor, env, do_log = True, pol_name = "test"):

    global log_size, log_ind, part_num, time_log, input_log, output_log, phase_log, orient_add_log

    # Start ar-control
    ar_control_path = os.path.expanduser("~/ar-software-2023.01.13a/ar-software-2023.01.13a/ar-software/ar-control")
    toml_path = os.path.abspath("./sim/digit_sim/digit_ar_sim/llapi/digit-rl.toml")
    print("ar path", ar_control_path, toml_path)
    if os.path.isfile(ar_control_path):
        print("Starting ar-control")
        ar_sim = agility.Simulator(ar_control_path, toml_path)
    else:
        print("Assuming ar-control already running")

    save_log_p = None   # Save log process for async file saving

    if platform.node() == "digit-nuc":
        ROBOT_ADDRESS = '10.10.1.1'
    else:
        ROBOT_ADDRESS = '127.0.0.1'

    api = agility.JsonApi(address=ROBOT_ADDRESS,
                          port=8080,
                          connect_timeout=1)
    try:
        await api.connect()
    except:
        raise Exception(f"Cannot connect api at {ROBOT_ADDRESS} with port {8080}\n"
                        f"Check if simulator is running and port is correct!")

    digit_udp = DigitUdp(robot_address=ROBOT_ADDRESS)
    topic = DigitStateTopic(digit_udp)

    env.turn_rate = 0
    env.x_velocity = 0
    env.y_velocity = 0
    env.orient_add = 0
    env.clock._phase = 0
    env.clock._cycle_time = 0.8
    env.clock._swing_ratios = [0.5, 0.5]
    env.clock._period_shifts = [0.0, 0.5]
    env.clock._von_mises_buf = None
    cmd_policy = llapi_command_pd_t()
    for i in range(NUM_MOTORS):
        cmd_policy.kp[i] = env.kp[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
        cmd_policy.kd[i] = env.kd[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
        cmd_policy.feedforward_torque[i] = 0.0

    first_policy_eval = True
    query_json = True
    api_mode = "locomotion"

    cmd_stand = llapi_command_pd_t()
    for i in range(NUM_MOTORS):
        cmd_stand.kp[i] = 1000
        cmd_stand.kd[i] = 30
        cmd_stand.position[i] = MOTOR_POSITION_SET['stand'][i]

    await api.send(["start-log-session", {"label": pol_name}])

    init_llapi_time = time.perf_counter()
    obs = None
    while obs is None and time.perf_counter() - init_llapi_time < 1.5:
        digit_udp.send_pd(llapi_command_pd_t())
        obs = topic.recv()
        if time.perf_counter() - init_llapi_time > 1:
            raise Exception("Cannot receive observation from simulator!")
    print("Connected Python with LLAPI and Digit!")
    for _ in range(50):
        obs = topic.recv()
        q = np.array([obs.base.orientation.w,
                      obs.base.orientation.x,
                      obs.base.orientation.y,
                      obs.base.orientation.z])
        euler = R.from_quat(us2scipy(q)).as_euler('xyz')
        env.orient_add = euler[2]

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)

    print(f"x vel: {env.x_velocity: .3f}, "
          f"y vel: {env.y_velocity: .3f}, "
          f"turn rate: {env.turn_rate: .3f} "
          f"delay: {0.000:.3f}", end="")
    loop_start_time = time.perf_counter()
    pol_time = time.perf_counter()
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            obs = topic.recv()
            # Dummy json call to keep json alive. Only do once per policy call
            if query_json:
                await api.query(msg.GetRobotInfo())
                query_json = False

            if isData():
                c = sys.stdin.read(1)
                if c == "w":
                    env.x_velocity += 0.05
                elif c == "s":
                    env.x_velocity -= 0.05
                elif c == "a":
                    env.y_velocity += 0.05
                elif c == "d":
                    env.y_velocity -= 0.05
                elif c == "q":
                    env.turn_rate += 0.05
                elif c == "e":
                    env.turn_rate -= 0.05
                elif c == "m":  # Toggle api mode
                    if api_mode == "locomotion":
                        api_mode = "llapi"
                        await api.request_privilege('change-action-command')
                        await api.send(msg.ActionSetOperationMode(mode="low-level-api"))
                        q = np.array([obs.base.orientation.w,
                                      obs.base.orientation.x,
                                      obs.base.orientation.y,
                                      obs.base.orientation.z])
                        euler = R.from_quat(us2scipy(q)).as_euler('xyz')
                        env.orient_add = euler[2]
                    elif api_mode == "llapi":
                        api_mode = "locomotion"
                        await api.request_privilege('change-action-command')
                        await api.send(msg.ActionSetOperationMode(mode="locomotion"))

            update_time = time.perf_counter() - pol_time
            if update_time >= 1 / env.default_policy_rate:

                query_json = True
                input_clock = env.clock.input_full_clock()
                q = np.array([obs.base.orientation.w,
                              obs.base.orientation.x,
                              obs.base.orientation.y,
                              obs.base.orientation.z])
                base_orient = np.array(env.rotate_to_heading(q))
                base_ang_vel = np.array(obs.imu.angular_velocity[:])
                motor_pos = np.array(obs.motor.position[:])[DIGIT_MOTOR_LLAPI2MJ_INDEX]
                motor_vel = np.array(obs.motor.velocity[:])[DIGIT_MOTOR_LLAPI2MJ_INDEX]
                joint_pos = np.array(obs.joint.position[:])[DIGIT_JOINT_LLAPI2MJ_INDEX]
                joint_vel = np.array(obs.joint.velocity[:])[DIGIT_JOINT_LLAPI2MJ_INDEX]

                robot_state = np.concatenate([
                    base_orient,
                    base_ang_vel,
                    motor_pos,
                    motor_vel,
                    joint_pos,
                    joint_vel
                ])
                RL_state = np.concatenate((robot_state,
                                        [env.x_velocity, env.y_velocity, env.turn_rate],
                                        [env.clock.get_swing_ratios()[0], 1 - env.clock.get_swing_ratios()[0]],
                                        env.clock.get_period_shifts(),
                                        input_clock))
                with torch.no_grad():
                    action = actor(torch.tensor(RL_state).float(), deterministic=True).numpy()
                action_sum = env.offset + action
                for i in range(NUM_MOTORS):
                    cmd_policy.position[i] = action_sum[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
                digit_udp.send_pd(cmd_policy)
                env.orient_add += env.turn_rate / env.default_policy_rate
                env.clock.increment()
                print(f"\rx vel: {env.x_velocity: .3f}, "
                      f"y vel: {env.y_velocity: .3f}, "
                      f"turn rate: {env.turn_rate: .3f} "
                      f"delay: {((time.perf_counter() - pol_time) - (1 / env.default_policy_rate)) * 100:.3f} ms", end="")
                pol_time = time.perf_counter()

                if do_log:
                    time_log[log_ind] = pol_time
                    for i in range(len(ROBOT_STATE_NAMES)):
                        input_log[ROBOT_STATE_NAMES[i]][log_ind] = robot_state[i]
                    for i in range(len(EXTRA_INPUT_NAMES)):
                        input_log[EXTRA_INPUT_NAMES[i]][log_ind] = RL_state[len(ROBOT_STATE_NAMES) + i]
                    for i in range(len(MOTOR_NAMES)):
                        output_log[MOTOR_NAMES[i]][log_ind] = action_sum[i]
                    phase_log[log_ind] = env.clock._phase
                    orient_add_log[log_ind] = env.orient_add
                    log_ind += 1
                    if log_ind >= log_size:
                        if save_log_p is not None:
                            save_log_p.join()
                        save_log_p = Process(target=save_log)
                        log_ind = 0
                        part_num += 1

            if do_log:
                signal.signal(signal.SIGINT, signal_handler)

            delaytime = 1/1000 - (time.perf_counter() - loop_start_time)
            while delaytime > 0:
                t0 = time.perf_counter()
                time.sleep(1e-5)
                delaytime -= time.perf_counter() - t0
                loop_start_time = time.perf_counter()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
    parser.add_argument("--no-log", dest='do_log', default=True, action="store_false", help="Whether to log data or not. True by default")

    # Manually handle path argument
    try:
        path_idx = sys.argv.index("--path")
        model_path = sys.argv[path_idx + 1]
        if not isinstance(model_path, str):
            print(f"{__file__}: error: argument --path received non-string input.")
            sys.exit()
    except ValueError:
        print(f"No path input given. Usage is 'python eval.py simple --path /path/to/policy'")

    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')
    args = parser.parse_args()

    # Load environment
    previous_args_dict['env_args'].simulator_type = "mujoco"
    previous_args_dict['env_args'].state_est = False
    previous_args_dict['env_args'].velocity_noise = 0.0
    previous_args_dict['env_args'].state_noise = 0.0
    previous_args_dict['env_args'].dynamics_randomization = False
    previous_args_dict['env_args'].reward_name = "locomotion_vonmises_clock_reward"
    previous_args_dict['env_args'].full_gait = True
    if hasattr(previous_args_dict['env_args'], 'velocity_noise'):
        delattr(previous_args_dict['env_args'], 'velocity_noise')
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False
    if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()

    # Extra names for input logging
    global EXTRA_INPUT_NAMES
    EXTRA_INPUT_NAMES = ['x-velocity', 'y-velocity', 'turn-rate',
                         'swing-ratio-left', 'swing-ratio-right', 'period-shift-left', 'period-shift-right',
                         'clock-sin-left', 'clock-cos-left', 'clock-sin-right', 'clock-cos-right']
                        #  'clock-sin', 'clock-cos']
    # Global data for logging
    global log_size
    global log_ind
    global part_num
    global time_log # time stamp
    global input_log # network inputs
    global output_log # network outputs
    global phase_log # clock phase
    global orient_add_log # heading offset
    global final_save
    log_size = 100000
    log_ind = 0
    part_num = 0
    time_log   = [time.time()] * log_size # time stamp
    input_log = {} # network inputs
    final_save = False
    for name in ROBOT_STATE_NAMES + EXTRA_INPUT_NAMES:
        input_log[name] = [0.0] * log_size
    output_log = {}
    for motor in MOTOR_NAMES:
        output_log[motor] = [0.0] * log_size
    phase_log = [0.0] * log_size # clock phase
    orient_add_log = [0.0] * log_size # heading offset

    global logdir
    LOG_NAME = args.path.rsplit('/', 3)[1]
    directory = os.path.dirname(os.path.realpath(__file__)) + "/hardware_logs/digit/"
    timestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    logdir = os.path.join(directory, LOG_NAME, timestr)
    # Check if output directory already exists. If it does, increment logdir name
    index = ''
    while os.path.exists(logdir + index):
        if index:
            index = '_(' + str(int(index[2:-1]) + 1) + ')'
        else:
            index = '_(1)'
    logdir += index + "/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    asyncio.run(run(actor, env, do_log=args.do_log, pol_name = LOG_NAME))