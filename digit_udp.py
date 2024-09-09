import agility
import agility.messages as msg
import atexit
import argparse
import asyncio
import copy
import datetime
import numpy as np
import sys
import os
import platform
import pickle
import select
import sys
import termios
import time
import torch
import tty

from env.genericenv import GenericEnv
from multiprocessing import Process
from scipy.spatial.transform import Rotation as R
from sim.digit_sim.digit_ar_sim.digit_udp import DigitUdp
from sim.digit_sim.digit_ar_sim.interface_ctypes import *
from testing.common import (
    DIGIT_MOTOR_MJ2LLAPI_INDEX,
    DIGIT_MOTOR_LLAPI2MJ_INDEX,
    MOTOR_POSITION_SET,
    DIGIT_MOTOR_NAME_LLAPI,
    DIGIT_JOINT_NAME_LLAPI
)
from util.colors import WARNING, ENDC
from util.state_topic import StateTopic
from util.env_factory import add_env_parser, env_factory
from util.nn_factory import load_checkpoint, nn_factory
from util.quaternion import mj2scipy
from util.xbox import XboxController, check_xbox_connection

LOGSIZE = 100000

def save_log(log_data):
    global logdir, log_ind, log_hf_ind, part_num
    filename = os.path.join(logdir, f"logdata_part{part_num}.pkl")
    print("Logging to {}".format(filename))
    # Truncate log data to actual size
    for key, val in log_data.items():
        if key != "flags":
            if "llapi" in key or key == "delay":
                actual_ind = log_hf_ind
            else:
                actual_ind = log_ind
            if isinstance(val, dict):
                for key2, val2 in val.items():
                    if isinstance(val2, dict):
                        for key3, val3 in val2.items():
                            log_data[key][key2][key3] = val3[:actual_ind]
                    else:
                        log_data[key][key2] = val2[:actual_ind]
            else:
                log_data[key] = val[:actual_ind]

    with open(filename, "wb") as filep:
        pickle.dump(log_data, filep)

def log_llapi(llapi_log, obs, log_ind):
    llapi_log["time"][log_ind] = obs.time
    llapi_log["battery charge"][log_ind] = obs.battery_charge
    for i in range(len(DIGIT_JOINT_NAME_LLAPI)):
        llapi_log["joint/position"][DIGIT_JOINT_NAME_LLAPI[i]][log_ind] = obs.joint.position[i]
        llapi_log["joint/velocity"][DIGIT_JOINT_NAME_LLAPI[i]][log_ind] = obs.joint.velocity[i]
    for i in range(len(DIGIT_MOTOR_NAME_LLAPI)):
        llapi_log["motor/position"][DIGIT_MOTOR_NAME_LLAPI[i]][log_ind] = obs.motor.position[i]
        llapi_log["motor/velocity"][DIGIT_MOTOR_NAME_LLAPI[i]][log_ind] = obs.motor.velocity[i]
        llapi_log["motor/torque"][DIGIT_MOTOR_NAME_LLAPI[i]][log_ind] = obs.motor.torque[i]
        llapi_log["motor/power"][DIGIT_MOTOR_NAME_LLAPI[i]][log_ind] = obs.motor.torque[i] * obs.motor.velocity[i]
    for i, dim in zip(range(3), ["x", "y", "z"]):
        llapi_log["imu/ang-vel"][dim][log_ind] = obs.imu.angular_velocity[i]
        llapi_log["imu/lin-accel"][dim][log_ind] = obs.imu.linear_acceleration[i]
        llapi_log["imu/mag-field"][dim][log_ind] = obs.imu.magnetic_field[i]
        llapi_log["base/ang-vel"][dim][log_ind] = obs.base.angular_velocity[i]
        llapi_log["base/lin-vel"][dim][log_ind] = obs.base.linear_velocity[i]
        llapi_log["base/translation"][dim][log_ind] = obs.base.translation[i]
    for dim in ["w", "x", "y", "z"]:
        llapi_log["imu/quat"][dim][log_ind] = getattr(obs.imu.orientation, dim)
        llapi_log["base/quat"][dim][log_ind] = getattr(obs.base.orientation, dim)

def close_ar_sim(ar_sim):
    ar_sim.close()

async def run(actor, env: GenericEnv, do_log = True, pol_name = "test"):

    global log_ind, log_hf_ind, part_num

    print("in run")
    # Setup logging
    if do_log:
        log_ind = 0
        log_hf_ind = 0
        part_num = 0
        log_data = {"time": [time.time()] * LOGSIZE,
                    "orient add": [0.0] * LOGSIZE,}
        input_log = {} # network inputs
        for name in env.robot.robot_state_names + env.extra_input_names:
            input_log[name] = [0.0] * LOGSIZE
        log_data["input"] = input_log
        output_log = {}
        for motor in env.robot.output_names:
            output_log[motor] = [0.0] * LOGSIZE
        log_data["output"] = output_log
        llapi_log = {}
        llapi_log["time"] = [0.0] * LOGSIZE
        llapi_log["battery charge"] = [0] * LOGSIZE
        llapi_joint_log = {}
        for joint in DIGIT_JOINT_NAME_LLAPI:
            llapi_joint_log[joint] = [0.0] * LOGSIZE
        llapi_motor_log = {}
        for motor in DIGIT_MOTOR_NAME_LLAPI:
            llapi_motor_log[motor] = [0.0] * LOGSIZE
        xyz_log = {"x": [0.0] * LOGSIZE, "y": [0.0] * LOGSIZE, "z": [0.0] * LOGSIZE}
        quat_log = {"w": [0.0] * LOGSIZE, "x": [0.0] * LOGSIZE, "y": [0.0] * LOGSIZE, "z": [0.0] * LOGSIZE}
        llapi_log["joint/position"] = llapi_joint_log
        llapi_log["joint/velocity"] = copy.deepcopy(llapi_joint_log)
        llapi_log["motor/position"] = llapi_motor_log
        llapi_log["motor/velocity"] = copy.deepcopy(llapi_motor_log)
        llapi_log["motor/torque"] = copy.deepcopy(llapi_motor_log)
        llapi_log["motor/power"] = copy.deepcopy(llapi_motor_log)
        llapi_log["imu/ang-vel"] = xyz_log
        llapi_log["imu/lin-accel"] = copy.deepcopy(xyz_log)
        llapi_log["imu/mag-field"] = copy.deepcopy(xyz_log)
        llapi_log["imu/quat"] = quat_log
        llapi_log["base/ang-vel"] = copy.deepcopy(xyz_log)
        llapi_log["base/lin-vel"] = copy.deepcopy(xyz_log)
        llapi_log["base/translation"] = copy.deepcopy(xyz_log)
        llapi_log["base/quat"] = copy.deepcopy(quat_log)
        log_data["llapi"] = llapi_log
        log_data["delay"] = [0.0] * LOGSIZE
        log_data["flags"] = []
        # Init/allocate custom logs here

    # Check if xbox controller connected. If not default to keyboard control
    print("Checking controller connection. Move joysticks to check connection.")
    if check_xbox_connection():
        print("Xbox controller connected")
        use_xbox = True
        xbox = XboxController()
        env.xbox_scale_factor = 0.005
    else:
        print("No xbox controller connected, using keyboard control")
        use_xbox = False

    # Start ar-control
    ar_control_path = os.path.expanduser("~/ar-software-2023.01.13a/ar-software/ar-control")
    toml_path = os.path.abspath("./sim/digit_sim/digit_ar_sim/llapi/digit-rl.toml")
    if os.path.isfile(ar_control_path):
        print("Starting ar-control")
        ar_sim = agility.Simulator(ar_control_path, toml_path)
        atexit.register(close_ar_sim, ar_sim)
    else:
        print(f"{WARNING}Assuming ar-control already running{ENDC}")

    save_log_p = None   # Save log process for async file saving
    if do_log:
        atexit.register(save_log, log_data)

    if platform.node() == "digit-nuc":
        ROBOT_ADDRESS = '10.10.1.1'
    else:
        ROBOT_ADDRESS = '127.0.0.1'

    api = agility.JsonApi(
        address=ROBOT_ADDRESS,
        port=8080,
        connect_timeout=1
    )
    try:
        await api.connect()
    except:
        raise Exception(f"Cannot connect api at {ROBOT_ADDRESS} with port {8080}\n"
                        f"Check if simulator is running and port is correct!")

    digit_udp = DigitUdp(robot_address=ROBOT_ADDRESS)
    topic = StateTopic(digit_udp)

    cmd_policy = llapi_command_pd_t()
    for i in range(NUM_MOTORS):
        cmd_policy.kp[i] = env.robot.kp[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
        cmd_policy.kd[i] = env.robot.kd[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
        cmd_policy.feedforward_torque[i] = 0.0

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
    prev_obs_time = -1
    print("Connected Python with LLAPI and Digit!")
    for _ in range(50):
        obs = topic.recv()
        q = np.array([obs.base.orientation.w,
                      obs.base.orientation.x,
                      obs.base.orientation.y,
                      obs.base.orientation.z])
        euler = R.from_quat(mj2scipy(q)).as_euler('xyz')
        env.orient_add = euler[2]

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    old_settings = termios.tcgetattr(sys.stdin)

    if use_xbox:
        env.display_xbox_controls_menu
    else:
        env.display_controls_menu()
    env.display_control_commands()
    print(f"\033[{env.num_menu_backspace_lines}B\033[K", end='\r')
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

            # Xbox control
            if use_xbox:
                env.interactive_xbox_control(xbox)
                if xbox.Start == 1 and not xbox.Start_pressed: # Toggle api mode
                    xbox.Start_pressed = True
                    if api_mode == "locomotion":
                        api_mode = "llapi"
                        await api.request_privilege('change-action-command')
                        await api.send(msg.ActionSetOperationMode(mode="low-level-api"))
                        q = np.array([obs.base.orientation.w,
                                    obs.base.orientation.x,
                                    obs.base.orientation.y,
                                    obs.base.orientation.z])
                        euler = R.from_quat(mj2scipy(q)).as_euler('xyz')
                        env.orient_add = euler[2]
                    elif api_mode == "llapi":
                        api_mode = "locomotion"
                        await api.request_privilege('change-action-command')
                        await api.send(msg.ActionSetOperationMode(mode="locomotion"))
                elif xbox.Start_pressed and xbox.Start == 0:
                    xbox.Start_pressed = False
                if xbox.RightBumper == 1:
                    if xbox.A == 1 and not xbox.A_pressed:
                        xbox.A_pressed = True
                        if do_log:
                            log_data["flags"].append([log_ind, log_hf_ind])
                    elif xbox.A_pressed and xbox.A == 0:
                        xbox.A_pressed = False
            # Keyboard control
            else:
                if isData():
                    c = sys.stdin.read(1)
                    print(f"\033[{env.num_menu_backspace_lines}A\033[K", end='\r')
                    env.interactive_control(c)
                    print(f"\033[{env.num_menu_backspace_lines}B\033[K", end='\r')
                    if c == "m":  # Toggle api mode
                        if api_mode == "locomotion":
                            api_mode = "llapi"
                            await api.request_privilege('change-action-command')
                            await api.send(msg.ActionSetOperationMode(mode="low-level-api"))
                            q = np.array([obs.base.orientation.w,
                                        obs.base.orientation.x,
                                        obs.base.orientation.y,
                                        obs.base.orientation.z])
                            euler = R.from_quat(mj2scipy(q)).as_euler('xyz')
                            env.orient_add = euler[2]
                        elif api_mode == "llapi":
                            api_mode = "locomotion"
                            await api.request_privilege('change-action-command')
                            await api.send(msg.ActionSetOperationMode(mode="locomotion"))
                    if c == "f": # Set log flag
                        if do_log:
                            log_data["flags"].append([log_ind, log_hf_ind])

            update_time = time.perf_counter() - pol_time
            if update_time >= 1 / env.default_policy_rate:

                query_json = True
                env.robot.llapi_obs = obs
                robot_state = env.get_robot_state()
                RL_state = env.get_state()

                with torch.no_grad():
                    action = actor(torch.tensor(RL_state).float(), deterministic=True).numpy()
                if env.integral_action:
                    action_sum = action + np.array(obs.motor.position[:])[DIGIT_MOTOR_LLAPI2MJ_INDEX]
                else:
                    action_sum = action + env.robot.offset
                for i in range(NUM_MOTORS):
                    cmd_policy.position[i] = action_sum[DIGIT_MOTOR_MJ2LLAPI_INDEX[i]]
                digit_udp.send_pd(cmd_policy)
                env.hw_step()
                print(f"mode: {api_mode}, "
                      f"delay: {((time.perf_counter() - pol_time) - (1 / env.default_policy_rate)) * 100:.3f} ms", end="\r")
                pol_time = time.perf_counter()

                if do_log:
                    log_data["time"][log_ind] = obs.time
                    log_data["orient add"][log_ind] = env.orient_add
                    for i in range(len(env.robot.robot_state_names)):
                        log_data["input"][env.robot.robot_state_names[i]][log_ind] = robot_state[i]
                    for i in range(len(env.extra_input_names)):
                        log_data["input"][env.extra_input_names[i]][log_ind] = RL_state[len(env.robot.robot_state_names) + i]
                    for i in range(len(env.robot.output_names)):
                        log_data["output"][env.robot.output_names[i]][log_ind] = action_sum[i]
                    # Add custom logs here
                    log_ind += 1
                    if log_ind >= LOGSIZE:
                        if save_log_p is not None:
                            save_log_p.join()
                        save_log_p = Process(target=save_log, args=(log_data,))
                        save_log_p.start()
                        log_data["flags"] = []
                        log_ind = 0
                        log_hf_ind = 0
                        part_num += 1

            if do_log and prev_obs_time != obs.time:
                log_llapi(log_data["llapi"], obs, log_hf_ind)
                if log_hf_ind == 0:
                    log_data["delay"][log_hf_ind] = 0
                else:
                    log_data["delay"][log_hf_ind] = (obs.time - log_data["llapi"]["time"][log_hf_ind - 1]) - 1/2000
                log_hf_ind += 1
                if log_hf_ind >= LOGSIZE:
                    if save_log_p is not None:
                        save_log_p.join()
                    save_log_p = Process(target=save_log, args=(log_data,))
                    save_log_p.start()
                    log_data["flags"] = []
                    log_ind = 0
                    log_hf_ind = 0
                    part_num += 1
            delaytime = 1/2000 - (time.perf_counter() - loop_start_time)
            while delaytime > 0:
                t0 = time.perf_counter()
                time.sleep(1e-5)
                delaytime -= time.perf_counter() - t0
                loop_start_time = time.perf_counter()
            prev_obs_time = obs.time
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
    add_env_parser(previous_args_dict['all_args'].env_name, parser, is_eval=True)
    args = parser.parse_args()

    # Overwrite previous env args with current input
    for arg, val in vars(args).items():
        if hasattr(previous_args_dict['env_args'], arg):
            setattr(previous_args_dict['env_args'], arg, val)

    # Load environment
    previous_args_dict['env_args'].simulator_type = "ar_async"
    if hasattr(previous_args_dict['env_args'], 'velocity_noise'):
        delattr(previous_args_dict['env_args'], 'velocity_noise')
    if hasattr(previous_args_dict['env_args'], 'state_est'):
        delattr(previous_args_dict['env_args'], 'state_est')
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()
    env.trackers = {}

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    actor.eval()
    actor.training = False
    if hasattr(actor, 'init_hidden_state'):
        actor.init_hidden_state()

    # Setup log directory
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