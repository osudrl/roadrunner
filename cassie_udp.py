import argparse, time, pickle, platform, socket, json
import os, sys, datetime
import select, termios, tty, atexit
from math import floor
# import cv2

import numpy as np
import torch
from multiprocessing import Process, Manager

from algo.common.network import Actor_LSTM_v2
from algo.common.utils import load_actor
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

# entry file for run a specified udp setup
# cassie-async (sim), digit-ar-control-async (sim), cassie-real, digit-real
class StateTopic:
    @staticmethod
    def _fetch(data, socket, socket_to_camera, client_address = ("10.25.25.103", 20003)):
        freq = 5000
        while True:
            t = time.monotonic()
            state = socket.recv_newest_pd()
            if state is not None:
                data['state'] = state
                dt = time.monotonic()
                data['time'] = dt
                if socket_to_camera is not None:
                    # state_to_camera = np.concatenate((state.pelvis.translationalAcceleration[:],
                    #                                 state.pelvis.rotationalVelocity[:],
                    #                                 state.leftFoot.position[:],
                    #                                 state.rightFoot.position[:]))
                    state_to_camera = np.concatenate((state.rightFoot.position[:],
                                                      state.motor.position[:]))
                    # state_to_camera[-1] = dt
                    socket_to_camera.sendto(state_to_camera.tobytes(), client_address)
            delaytime = 1/freq - (time.monotonic() - t)
            while delaytime > 0:
                t0 = time.monotonic()
                time.sleep(0.00001)
                delaytime -= time.monotonic() - t0

    def subscribe(self, socket, socket_to_camera=None):
        self.socket = socket
        self.state = Manager().dict()
        remote_func = Process(target=self._fetch, args=(self.state, self.socket, socket_to_camera))
        remote_func.start()

    def recv(self):
        out = self.state.get('state', None)
        return out

    def get_time(self):
        return self.state.get('time', None)

    def __del__(self):
        if self.socket is not None:
            self.socket.__del__()


def remap(val, min1, max1, min2, max2):
    span1 = max1 - min1
    span2 = max2 - min2
    scaled = (val - min1) / span1
    return np.clip(min2 + (scaled * span2), min2, max2)

def save_log():
    global log_hf_ind, log_lf_ind, logdir, part_num, sto_num, time_hf_log, output_log, state_log, target_log, speed_log, orient_log, phaseadd_log, hm_log, time_lf_log, input_log

    filename = "logdata_part" + str(part_num) + "_sto" + str(sto_num) + ".pkl"
    filename = os.path.join(logdir, filename)
    print("Logging to {}".format(filename))
    print("exit at time {}".format(time_hf_log[log_hf_ind-1]))
    print("save log: log_hf_ind {}".format(log_hf_ind))
    data = {"highfreq": True,
            "time_hf": time_hf_log[:log_hf_ind],
            "time_lf": time_lf_log[:log_lf_ind],
            "output": output_log[:log_hf_ind],
            "input": input_log[:log_lf_ind],
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
        u.leftLeg.motorPd.pGain[i]  = cassie_env.kp[i]
        u.rightLeg.motorPd.pGain[i] = cassie_env.kp[i + 5]

        u.leftLeg.motorPd.dGain[i]  = cassie_env.kd[i]
        u.rightLeg.motorPd.dGain[i] = cassie_env.kd[i + 5]

        u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
        u.rightLeg.motorPd.torque[i] = 0

        u.leftLeg.motorPd.pTarget[i]  = target[i]
        u.rightLeg.motorPd.pTarget[i] = target[i + 5]

        u.leftLeg.motorPd.dTarget[i]  = 0
        u.rightLeg.motorPd.dTarget[i] = 0

    cassie_udp.send_pd(u)

    # return log data
    return target

def execute(policy, env, args, do_log, exec_rate=1):
    global log_size, log_hf_ind, log_lf_ind, part_num, sto_num, save_dict, time_hf_log, output_log, state_log, target_log, speed_log, orient_log, phaseadd_log, time_lf_log, input_log

    # Determine whether running in simulation or on the robot
    if platform.node() == 'cassie':
        cassieudp = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                        local_addr='10.10.10.100', local_port='25011')
    else:
        cassieudp = CassieUdp()  # local testing

    USE_CAMERA = False

    # Initialize robot server for broadcast robot state
    # server_address = ("192.168.2.251", 30001)
    # client_address = ("192.168.2.179", 30003)
    # client_address = ("192.168.2.185", 20003)
    client_address = ("10.25.25.101", 20003)

    # Initialize camera server to receive perception state
    camera_topic = Topic(freq=2000)
    # robot_address = ("192.168.2.251", 20001) # Cassie WiFi address
    # robot_address = ("192.168.2.11", 20001) # WiFi address testing PC
    robot_address = ("10.25.25.100", 20001) # Cassie Wired address
    if USE_CAMERA:
        camera_topic.subscribe(robot_address)

    # Put UDP recv into a seperate process
    rtos_udp = StateTopic()
    if USE_CAMERA:
        rtos_udp.subscribe(cassieudp, camera_topic.soc)
    else:
        rtos_udp.subscribe(cassieudp)

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
    # env.reset() # Don't even reset env, so we won't use any simulator stuff
    env.hardware_mode = True
    env.turn_rate = 0
    env.y_velocity = 0
    env.x_velocity = 0
    env.clock._phase = 0
    env.clock._cycle_time = 0.7
    env.clock._swing_ratios = [0.5, 0.5]
    env.clock._period_shifts = [0, 0.5]
    env.clock._von_mises_buf = None

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
        damp_u.leftLeg.motorPd.dGain[i] = D_mult*env.kd[i]
        damp_u.rightLeg.motorPd.pGain[i] = 0.0
        damp_u.rightLeg.motorPd.dGain[i] = D_mult*env.kd[i + 5]
        damp_u.leftLeg.motorPd.pTarget[i] = 0.0
        damp_u.rightLeg.motorPd.pTarget[i] = 0.0

    old_settings = termios.tcgetattr(sys.stdin)
    count = 0
    pol_time = 0

    # Connect to the simulator or robot
    print('Connecting RTOS...')
    state = None
    while state is None:
        cassieudp.send_pd(pd_in_t())
        time.sleep(0.001)
        state = rtos_udp.recv()
    print('Connected to RTOS!\n')

    # Connect to perception server
    perception_state = None
    if hasattr(env, 'perception_policy'):
        if env.perception_policy:
            # Connect to camera server and check connection in while loop until it's over
            perception_state = np.zeros((env.heightmap_num_points, 1))
    print("Initialized perception server.\n")
    # input("Ready to start? Press Enter to continue...")

    try:
        tty.setcbreak(sys.stdin.fileno())

        t = time.monotonic()
        pol_time = 0
        first = True
        while True:

            # Get newest state
            t = time.monotonic()
            state = rtos_udp.recv()

            # No continue
            if platform.node() == 'cassie':
                # Control with Taranis radio controller
                if state.radio.channel[9] < -0.5:
                    operation_mode = 2  # down -> damping
                elif state.radio.channel[9] > 0.5:
                    operation_mode = 1  # up -> nothing
                else:
                    operation_mode = 0  # mid -> normal walking

                # Reset orientation on STO
                if state.radio.channel[8] < 0:
                    STO = True
                    env.sim.robot_estimator_state = state
                    env.orient_add = quaternion2euler(env.sim.robot_estimator_state.pelvis.orientation[:])[2]
                else:
                    STO = False
                    logged = False

                # Example of setting things manually instead. Reference to what radio channel corresponds to what joystick/knob:
                # https://github.com/agilityrobotics/cassie-doc/wiki/Radio#user-content-input-configuration
                # Radio control deadzones
                l_stick_x = state.radio.channel[0]
                l_stick_y = state.radio.channel[1]
                r_stick_y = state.radio.channel[3]
                if abs(l_stick_x) < 0.05:
                    l_stick_x = 0
                if abs(l_stick_y) < 0.05:
                    l_stick_y = 0
                if abs(r_stick_y) < 0.05:
                    r_stick_y = 0
                # Orientation control (Do manually instead of turn_rate)
                env.orient_add += - r_stick_y / (30.0*env.default_policy_rate)
                # X and Y speed control
                env.x_velocity = remap(l_stick_x, -1, 1, -1.0, 1.0)
                env.y_velocity = -remap(l_stick_y, -1, 1, -0.3, 0.3)
                env.x_velocity = np.clip(env.x_velocity, -0.3, 0.55)
                env.y_velocity = np.clip(env.y_velocity, -0.3, 0.3)
                # Gait parameters control
                if not hasattr(env, 'autoclock'):
                    cycle_time = remap(state.radio.channel[5], -1, 1, 0.6, 1.0)
                    env.clock.set_cycle_time(cycle_time)

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

            t_state = rtos_udp.get_time()
            env.sim.robot_estimator_state = rtos_udp.recv()
            if USE_CAMERA:
                # msg_to_perception = env.get_proprioceptive_state(include_joint=False, include_feet=True)
                # camera_topic.publish(msg_to_perception, client_address)
                perception_state = camera_topic.recv()
                if perception_state is None:
                    print(f"T={t:.0f} - Perception state is None.")
                    perception_state = 0.8 * np.ones((600, 1))
            else:
                perception_state = 0.8 * np.ones((600, 1))
                # p = perception_state.reshape(20,30)
                # p[:,0:10] = 0.7
                # perception_state = p.reshape(600,1)
                # perception_state = env.sim.robot_estimator_state.pelvis.position[2] * np.ones((600, 1))
            env.hardware_perception_state = - perception_state
            t_perception = perception_state[-1]

            #------------------------------- Normal Walking ---------------------------
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
                    RL_state = env.get_state()
                    with torch.no_grad():
                        action = policy(torch.tensor(RL_state).float(), deterministic=True).numpy()
                    # action = np.zeros(10)
                    target = PD_step(cassieudp, env, action[:10])
                    pol_time = time.monotonic()
                    # Update env quantities
                    env.orient_add += env.turn_rate / env.default_policy_rate
                    # Gait parameters control
                    if hasattr(env, 'autoclock'):
                        if env.autoclock:
                            env.update_clock(action[10:env.action_size])
                    env.clock.increment()

                    if do_log:
                        time_lf_log[log_lf_ind] = time.time()
                        input_log[log_lf_ind] = RL_state
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
                        sys.stdout.write(f"Speed: {env.x_velocity:1.2f}\t, "
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
                    speed_log[log_hf_ind] = env.x_velocity
                    orient_log[log_hf_ind] = env.orient_add
                    phaseadd_log[log_hf_ind] = env.clock._cycle_time
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
                delaytime = 1/1000 - (time.monotonic() - t)
                while delaytime > 0:
                    t0 = time.monotonic()
                    time.sleep(1/10000)
                    delaytime -= time.monotonic() - t0
                # print(f"(Expected) Running at = {1/(time.monotonic() - t)}")

            #------------------------------- Empty Action ---------------------------
            elif operation_mode == 1:
                print('Applying no action')
                # Do nothing
                cassieudp.send_pd(empty_u)

            #------------------------------- Shutdown Damping ---------------------------
            elif operation_mode == 2:
                # print('Shutdown Damping. Multiplier = ' + str(D_mult))
                cassieudp.send_pd(damp_u)

            #---------------------------- Other, should not happen -----------------------
            else:
                print('Error, In bad operation_mode with value: ' + str(operation_mode))


    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None, help="path to folder containing policy and run details")
parser.add_argument("--exec_rate", default=1, type=int, help="Controls the execution rate of the script. Is 1 (full 2kHz) be default")
parser.add_argument("--no_log", dest='do_log', default=True, action="store_true", help="Whether to log data or not. True by default")
parser.add_argument("--max_x", default=4.0, type=float, help="Maximum x speed")
parser.add_argument("--min_x", default=0.0, type=float, help="Minimum x speed")
parser.add_argument("--max_y", default=0.5, type=float, help="Maximum y speed")
parser.add_argument("--min_y", default=-0.5, type=float, help="Minimum y speed")

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
previous_args_dict['env_args'].simulator_type = "libcassie"
previous_args_dict['env_args'].state_est = True
previous_args_dict['env_args'].velocity_noise = 0.0
previous_args_dict['env_args'].state_noise = 0.0
previous_args_dict['env_args'].dynamics_randomization = False
if 'actor_feet' in previous_args_dict['env_args'].__dict__.keys():
    # previous_args_dict['env_args'].pop('actor_feet')
    delattr(previous_args_dict['env_args'], 'actor_feet')
if 'actorfeet' in previous_args_dict['env_args'].__dict__.keys():
    # previous_args_dict['env_args'].pop('actorfeet')
    delattr(previous_args_dict['env_args'], 'actorfeet')
env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

# def load_actor(args, device, model_fn):
#     model = model_fn(args)
#     model.to(device)
#
#     wandb.login()
#
#     run = wandb.Api().run(os.path.join(args.project_name, args.run_name.replace(':', '_')))
#
#     logging.info(f'Checkpoint loading from: {args.run_name}')
#
#     if args.model_checkpoint == 'latest':
#         checkpoint_path = f'checkpoints/checkpoint-{args.run_name}.pt'
#
#         run.file(name=checkpoint_path).download(replace=args.redownload_checkpoint, exist_ok=True)
#
#         with open(checkpoint_path, 'rb') as r:
#             checkpoint = torch.load(r, map_location=device)
#
#         model.load_state_dict(checkpoint['actor_state_dict'])
#
#         logging.info(
#             f'Loaded checkpoint: {checkpoint.get("epoch", 0)}, {checkpoint.get("total_steps", 0), {checkpoint.get("trajectory_count", 0)} }')
#     else:
#         if args.model_checkpoint == 'best':
#             model_path = f'saved_models/model-{args.run_name}.pth'
#         else:
#             model_path = f'saved_models/model-{args.run_name}-{args.model_checkpoint}.pth'
#
#         run.file(name=model_path).download(replace=args.redownload_checkpoint, exist_ok=True)
#
#         with open(model_path, 'rb') as r:
#             checkpoint = torch.load(r, map_location=device)
#
#         model.load_state_dict(checkpoint)
#
#         logging.info(f'Loaded model: {args.model_checkpoint}')
#
#     model.eval()
#
#     wandb.finish()
#
#     return model
#

actor = load_actor(args, device=torch.device('cpu'), model_fn=Actor_LSTM_v2)

# # Load model class and checkpoint
# actor, critic = nn_factory(args=previous_args_dict['nn_args'], env=env)
# load_checkpoint(model=actor, model_dict=actor_checkpoint)
#
# actor.load_encoder_patch()

actor.eval()
actor.training = False

LOG_NAME = args.path.rsplit('/', 3)[1] + "/"
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
time_lf_log   = [time.time()] * log_size # time stamp
time_hf_log   = [time.time()] * log_size # time stamp
input_log  = [np.ones(actor.obs_dim)] * log_size # network inputs
output_log = [np.ones(actor.action_dim)] * log_size # network outputs
state_log  = [state_out_t()] * log_size  # cassie state
target_log = [np.ones(10)] * log_size  # PD target log
speed_log  = [0.0] * log_size # speed input commands
orient_log  = [0.0] * log_size # orient input commands
phaseadd_log  = [0.0] * log_size # frequency input commands

part_num = 0
sto_num = 0

if args.do_log:
    atexit.register(save_log)
execute(actor, env, args, args.do_log, args.exec_rate)
