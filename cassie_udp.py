import argparse, atexit, copy, datetime, os, pickle, platform, select, sys, time, termios, torch, tty
import numpy as np

from multiprocessing import Manager, Process


from sim.cassie_sim.cassiemujoco.cassieUDP import *
from sim.cassie_sim.cassiemujoco.cassiemujoco_ctypes import *
from testing.common import (
    CASSIE_MOTOR_LLAPI_NAME,
    CASSIE_JOINT_LLAPI_NAME,
)
from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory
from util.state_topic import StateTopic
from util.tarsus_patch_wrapper import TarsusPatchWrapper
from util.quaternion import scipy2mj, mj2scipy
from scipy.spatial.transform import Rotation as R


LOGSIZE = 100000

def remap(val, min1, max1, min2, max2):
    span1 = max1 - min1
    span2 = max2 - min2
    scaled = (val - min1) / span1
    return np.clip(min2 + (scaled * span2), min2, max2)

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

def log_llapi(llapi_log, state, log_ind):
    llapi_log["time"][log_ind] = time.perf_counter()
    llapi_log["battery current"][log_ind] = state.battery.current
    llapi_log["battery StateOfCharge"][log_ind] = state.battery.stateOfCharge
    for i in range(len(CASSIE_JOINT_LLAPI_NAME)):
        llapi_log["joint/position"][CASSIE_JOINT_LLAPI_NAME[i]][log_ind] = state.joint.position[i]
        llapi_log["joint/velocity"][CASSIE_JOINT_LLAPI_NAME[i]][log_ind] = state.joint.velocity[i]
    for i in range(len(CASSIE_MOTOR_LLAPI_NAME)):
        llapi_log["motor/position"][CASSIE_MOTOR_LLAPI_NAME[i]][log_ind] = state.motor.position[i]
        llapi_log["motor/velocity"][CASSIE_MOTOR_LLAPI_NAME[i]][log_ind] = state.motor.velocity[i]
        llapi_log["motor/torque"][CASSIE_MOTOR_LLAPI_NAME[i]][log_ind] = state.motor.torque[i]
    for i, dim in zip(range(3), ["x", "y", "z"]):
        llapi_log["pelvis/position"][dim][log_ind] = state.pelvis.position[i]
        llapi_log["pelvis/rot-vel"][dim][log_ind] = state.pelvis.rotationalVelocity[i]
        llapi_log["pelvis/lin-vel"][dim][log_ind] = state.pelvis.translationalVelocity[i]
        llapi_log["pelvis/lin-accel"][dim][log_ind] = state.pelvis.translationalAcceleration[i]
    for i, dim in zip(range(4), ["w", "x", "y", "z"]):
        llapi_log["pelvis/quat"][dim][log_ind] = state.pelvis.orientation[i]

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# 2 kHz execution : PD control with or without baseline action
def PD_step(cassie_udp, cassie_env, action):
    target = action[:] + cassie_env.robot.offset

    u = pd_in_t()
    for i in range(5):
        u.leftLeg.motorPd.pGain[i]  = cassie_env.robot.kp[i]
        u.rightLeg.motorPd.pGain[i] = cassie_env.robot.kp[i + 5]

        u.leftLeg.motorPd.dGain[i]  = cassie_env.robot.kd[i]
        u.rightLeg.motorPd.dGain[i] = cassie_env.robot.kd[i + 5]

        u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
        u.rightLeg.motorPd.torque[i] = 0

        u.leftLeg.motorPd.pTarget[i]  = target[i]
        u.rightLeg.motorPd.pTarget[i] = target[i + 5]

        u.leftLeg.motorPd.dTarget[i]  = 0
        u.rightLeg.motorPd.dTarget[i] = 0

    cassie_udp.send_pd(u)

    # return log data
    return target

def execute(policy, env, do_log, exec_rate=1):
    global log_hf_ind, log_ind, part_num

    # Global data for logging
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
    llapi_log["battery current"] = [0] * LOGSIZE
    llapi_log["battery StateOfCharge"] = [0] * LOGSIZE
    llapi_joint_log = {}
    for joint in CASSIE_JOINT_LLAPI_NAME:
        llapi_joint_log[joint] = [0.0] * LOGSIZE
    llapi_motor_log = {}
    for motor in CASSIE_MOTOR_LLAPI_NAME:
        llapi_motor_log[motor] = [0.0] * LOGSIZE
    xyz_log = {"x": [0.0] * LOGSIZE, "y": [0.0] * LOGSIZE, "z": [0.0] * LOGSIZE}
    quat_log = {"w": [0.0] * LOGSIZE, "x": [0.0] * LOGSIZE, "y": [0.0] * LOGSIZE, "z": [0.0] * LOGSIZE}
    llapi_log["joint/position"] = llapi_joint_log
    llapi_log["joint/velocity"] = copy.deepcopy(llapi_joint_log)
    llapi_log["motor/position"] = llapi_motor_log
    llapi_log["motor/velocity"] = copy.deepcopy(llapi_motor_log)
    llapi_log["motor/torque"] = copy.deepcopy(llapi_motor_log)
    llapi_log["pelvis/position"] = xyz_log
    llapi_log["pelvis/rot-vel"] = copy.deepcopy(xyz_log)
    llapi_log["pelvis/lin-vel"] = copy.deepcopy(xyz_log)
    llapi_log["pelvis/lin-accel"] = copy.deepcopy(xyz_log)
    llapi_log["pelvis/quat"] = quat_log
    log_data["llapi"] = llapi_log
    log_data["delay"] = [0.0] * LOGSIZE
    log_data["flags"] = []
    # Init/allocate custom logs here

    # Determine whether running in simulation or on the robot
    if "cassie" in platform.node():
        cassieudp = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                        local_addr='10.10.10.100', local_port='25011')
    else:
        cassieudp = CassieUdp()  # local testing
    topic = StateTopic(cassieudp)

    if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

    if exec_rate > env.default_policy_rate:
        print("Error: Execution rate can not be greater than simrate")
        exit()
    # Lock exec_rate to even dividend of simrate
    rem = env.default_policy_rate // exec_rate
    exec_rate = env.default_policy_rate // rem
    print("Execution rate: {} ({:.2f} Hz)".format(exec_rate, 2000/exec_rate))

    # ESTOP position. True means ESTOP enabled and robot is not running.
    STO = False
    logged = False
    part_num = 0
    sto_num = 0
    save_log_p = None

    # 0: Policy
    # 1: Empty action
    # 2: Damping
    operation_mode = 0
    action = None
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
        damp_u.leftLeg.motorPd.dGain[i] = D_mult*env.robot.kd[i]
        damp_u.rightLeg.motorPd.pGain[i] = 0.0
        damp_u.rightLeg.motorPd.dGain[i] = D_mult*env.robot.kd[i + 5]
        damp_u.leftLeg.motorPd.pTarget[i] = 0.0
        damp_u.rightLeg.motorPd.pTarget[i] = 0.0

    pol_time = 0

    # Connect to the simulator or robot
    print('Connecting...')
    state = None
    while state is None:
        cassieudp.send_pd(pd_in_t())
        time.sleep(0.001)
        state = topic.recv()
    print('Connected!\n')

    save_log_p = None   # Save log process for async file saving
    if do_log:
        atexit.register(save_log, log_data)

    old_settings = termios.tcgetattr(sys.stdin)
    env.display_controls_menu()
    env.display_control_commands()
    print(f"\033[{env.num_menu_backspace_lines}B\033[K", end='\r')
    try:
        tty.setcbreak(sys.stdin.fileno())

        t = time.perf_counter()
        pol_time = 0
        while True:

            # Get newest state
            # print("timestep", time.perf_counter() - t)
            t = time.perf_counter()
            state = topic.recv()

            # Radio/keyboard control
            if "cassie" in platform.node():
                # Control with Taranis radio controller
                if state.radio.channel[9] < -0.5:
                    operation_mode = 2  # down -> damping
                elif state.radio.channel[9] > 0.5:
                    operation_mode = 1  # up -> nothing
                else:
                    operation_mode = 0  # mid -> policy

                # Reset orientation on STO
                if state.radio.channel[8] < 0:
                    STO = True
                    env.robot.robot_estimator_state = state
                    env.orient_add = R.from_quat(mj2scipy(env.robot.robot_estimator_state.pelvis.orientation[:])).as_euler('xyz')[2]
                else:
                    STO = False
                    logged = False

                if state.radio.channel[15] < 0:
                    log_data["flags"].append([log_ind, log_hf_ind])

                # Example of setting things manually instead. Reference to what radio channel corresponds to what joystick/knob:
                # https://github.com/agilityrobotics/cassie-doc/wiki/Radio#user-content-input-configuration
                # Radio control deadzones
                l_stick_x = state.radio.channel[0]
                l_stick_y = state.radio.channel[1]
                r_stick_y = state.radio.channel[3]
                # print("radio", state.radio.channel[0:5])
                if abs(l_stick_x) < 0.05:
                    l_stick_x = 0
                if abs(l_stick_y) < 0.05:
                    l_stick_y = 0
                if abs(r_stick_y) < 0.05:
                    r_stick_y = 0
                # Turn rate control
                env.turn_rate = -remap(r_stick_y, -1, 1, -np.pi/8, np.pi/8)
                env.turn_rate = np.clip(env.turn_rate, -0.3, 0.3)
                env.orient_add += env.turn_rate / env.default_policy_rate
                # X and Y speed control
                env.x_velocity = remap(l_stick_x, -1, 1, -0.6, 0.6)
                env.y_velocity = -remap(l_stick_y, -1, 1, -0.3, 0.3)
                env.x_velocity = np.clip(env.x_velocity, -0.2, 0.6)
                env.y_velocity = np.clip(env.y_velocity, -0.3, 0.3)

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
                    elif c == "f": # Set log flag
                        log_data["flags"].append([log_ind, log_hf_ind])
                    else:
                        env.interactive_control(c)

            #------------------------------- Normal Walking ---------------------------
            if operation_mode == 0:
                update_time = time.perf_counter() - pol_time

                if update_time > 1 / env.default_policy_rate:

                    """
                        Low frequency (Policy Rate) Section. Update policy action
                    """
                    env.robot.robot_estimator_state = state
                    robot_state = env.get_robot_state()
                    RL_state = env.get_state()
                    with torch.no_grad():
                        action = policy(torch.tensor(RL_state).float(), deterministic=True).numpy()
                    target = PD_step(cassieudp, env, action)
                    pol_time = time.perf_counter()
                    # Update env quantities
                    env.hw_step()

                    if do_log:
                        log_data["time"][log_ind] = time.perf_counter()
                        log_data["orient add"][log_ind] = env.orient_add
                        for i in range(len(env.robot.robot_state_names)):
                            log_data["input"][env.robot.robot_state_names[i]][log_ind] = robot_state[i]
                        for i in range(len(env.extra_input_names)):
                            log_data["input"][env.extra_input_names[i]][log_ind] = RL_state[len(env.robot.robot_state_names) + i]
                        for i in range(len(env.robot.output_names)):
                            log_data["output"][env.robot.output_names[i]][log_ind] = target[i]
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

                    measured_delay = (update_time - 1 / env.default_policy_rate) * 1000
                    sys.stdout.write(
                        f"x_vel: {env.x_velocity:.2f}\t"
                        f"y_vel: {env.y_velocity:.2f}\t"
                        f"turn_rate: {env.turn_rate:.2f}  \t"
                        f"inference delay: {measured_delay:.2f} ms\r"
                    )
                    sys.stdout.flush()

                """
                    High frequency (2000 Hz) Section
                """

                if do_log:
                    log_llapi(log_data["llapi"], state, log_hf_ind)
                    if log_hf_ind == 0:
                        log_data["delay"][log_hf_ind] = 0
                    else:
                        log_data["delay"][log_hf_ind] = (log_data["llapi"]["time"][log_hf_ind] - log_data["llapi"]["time"][log_hf_ind - 1]) - exec_rate/2000
                    log_hf_ind += 1
                    if log_hf_ind == LOGSIZE and do_log:
                        if save_log_p is not None:
                            save_log_p.join()
                        save_log_p = Process(target=save_log)
                        save_log_p.start()
                        log_data["flags"] = []
                        log_ind = 0
                        log_hf_ind = 0
                        part_num += 1

            #------------------------------- Empty Action ---------------------------
            elif operation_mode == 1:
                print('Applying no action')
                # Do nothing
                cassieudp.send_pd(empty_u)

            #------------------------------- Shutdown Damping ---------------------------
            elif operation_mode == 2:
                print('Shutdown Damping. Multiplier = ' + str(D_mult))
                cassieudp.send_pd(damp_u)

            #---------------------------- Other, should not happen -----------------------
            else:
                print('Error, In bad operation_mode with value: ' + str(operation_mode))

            delaytime = exec_rate/2000 - (time.perf_counter() - t)
            while delaytime > 0:
                t0 = time.perf_counter()
                time.sleep(1e-5)
                delaytime -= time.perf_counter() - t0


    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
    parser.add_argument("--exec-rate", default=1, type=int, help="Controls the execution rate of the script. Is 1 (full 2kHz) be default")
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
    previous_args_dict['env_args'].simulator_type = "real"
    previous_args_dict['env_args'].state_est = False
    previous_args_dict['env_args'].state_noise = [0, 0, 0, 0, 0, 0]
    previous_args_dict['env_args'].dynamics_randomization = False
    if hasattr(previous_args_dict['env_args'], 'velocity_noise'):
        delattr(previous_args_dict['env_args'], 'velocity_noise')
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()
    env.trackers = {}

    # Load model class and checkpoint
    actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
    load_checkpoint(model=actor, model_dict=actor_checkpoint)
    # wrap actor in tarsus predictor:
    actor = TarsusPatchWrapper(actor)
    actor.eval()
    actor.training = False

    # Setup log directory
    global logdir
    if args.do_log:
        LOG_NAME = args.path.rsplit('/', 3)[-3] + "/"
        directory = os.path.dirname(os.path.realpath(__file__)) + "/hardware_logs/cassie/"
        timestr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        logdir = directory + LOG_NAME + timestr

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
            print("made dir", logdir)

    execute(actor, env, args.do_log, args.exec_rate)
