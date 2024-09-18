import pickle
from matplotlib import pyplot as plt
import numpy as np
import time
from tempfile import TemporaryFile
import os, re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./hardware_logs/traj-aslip_aslip_old_2048_12288_seed-10/", help="Path to folder containing policy hardware logs")
args = parser.parse_args()

# POLICY_NAME = args.name
# FILE_PATH = "./hardware_logs/"
# FILE_NAME = "logdata"

# POLICY_NAME = "aslip_unified_no_delta_10_TS_only"
# FILE_PATH = "./hardware_logs/"
# FILE_NAME = "2020-01-26_16:27_logfinal"

# logs = pickle.load(open(FILE_PATH + POLICY_NAME + "/" + FILE_NAME + ".pkl", "rb")) #load in file with cassie data
data_dict = {}
for filename in os.listdir(args.path):
    if ".pkl" in filename:
        name_split = filename.split("_")
        part_num = int(re.search(r'\d+$', name_split[1]).group())
        sto_num = int(re.search(r'\d+$', name_split[2][:-4]).group())
        print("part_num: ", part_num)
        print("sto_num: ", sto_num)
        data_dict[(sto_num, part_num)] = pickle.load(open(os.path.join(args.path, filename), "rb"))

total_pelvis          = None
total_pelvis_vels     = None
total_motors_log      = None
total_joints_log      = None
total_torques_mea_log = None
total_left_foot_forces_log = None
total_right_foot_forces_log = None
total_left_foot_pos_log = None
total_right_foot_pos_log = None
total_states_rl = None
total_nn_output = None
total_speed = None
total_time_hf = None
total_time_lf = None
total_numStates = 0


curr_num = 0
curr_sto = 1
while((curr_sto, curr_num) in data_dict):
    while((curr_sto, curr_num) in data_dict):
        print("processing sto {} part {}".format(curr_sto, curr_num))
        logs = data_dict[(curr_sto, curr_num)]

        high_freq_log = True
        time_lf = logs["time_lf"]
        time_hf = logs["time_hf"]
        states_rl = np.array(logs["input"])
        print(states_rl.shape)
        states_rl = np.stack(states_rl, axis=0)
        states = logs["state"]
        nn_output = logs["output"]
        speeds = logs["speed"]
        # print(len(nn_output))
        # print(nn_output[0].shape)

        if high_freq_log:
            print("High frequency log file found")
            numStates = len(time_hf)
        else:
            print("Low frequency log file found")
            numStates = len(time_lf)

        print("numState: ", numStates)
        total_numStates += numStates
        pelvis          = np.zeros((numStates, 3))
        pelvis_vels     = np.zeros((numStates, 3))
        # pel_accel       = np.zeros((numStates, 3))
        motors_log      = np.zeros((numStates, 10))
        joints_log      = np.zeros((numStates, 6))
        torques_mea_log = np.zeros((numStates, 10))
        left_foot_forces_log = np.zeros((numStates, 6))
        right_foot_forces_log = np.zeros((numStates, 6))
        left_foot_pos_log = np.zeros((numStates, 6))
        right_foot_pos_log = np.zeros((numStates, 6))

        j=0
        for s in states:
            pelvis[j, :] = s.pelvis.position[:]
            pelvis[j, 2] -= s.terrain.height
            pelvis_vels[j, :] = s.pelvis.translationalVelocity[:]
            motors_log[j, :] = s.motor.position[:]
            joints_log[j, :] = s.joint.position[:]
            torques_mea_log[j, :] = s.motor.torque[:]
            left_foot_forces_log[j, :] = np.reshape(np.asarray([s.leftFoot.toeForce[:],s.leftFoot.heelForce[:]]), (6))
            right_foot_forces_log[j, :] = np.reshape(np.asarray([s.rightFoot.toeForce[:],s.rightFoot.heelForce[:]]), (6))
            left_foot_pos_log[j, :] = np.reshape(np.asarray([s.leftFoot.position[:],s.leftFoot.position[:]]), (6))
            right_foot_pos_log[j, :] = np.reshape(np.asarray([s.rightFoot.position[:],s.rightFoot.position[:]]), (6))
            
            j += 1

        if total_pelvis is None:
            total_pelvis          = pelvis
            total_pelvis_vels     = pelvis_vels
            total_motors_log      = motors_log
            total_joints_log      = joints_log
            total_torques_mea_log = torques_mea_log
            total_left_foot_forces_log = left_foot_forces_log
            total_right_foot_forces_log = right_foot_forces_log
            total_left_foot_pos_log = left_foot_pos_log
            total_right_foot_pos_log = right_foot_pos_log
            total_states_rl = states_rl
            total_nn_output = nn_output
            total_speed = speeds
            total_time_hf = time_hf
            total_time_lf = time_lf
        else:
            total_pelvis          = np.concatenate((total_pelvis, pelvis), axis=0)
            total_pelvis_vels     = np.concatenate((total_pelvis_vels, pelvis_vels), axis=0)
            total_motors_log      = np.concatenate((total_motors_log, motors_log), axis=0)
            total_joints_log      = np.concatenate((total_joints_log, joints_log), axis=0)
            total_torques_mea_log = np.concatenate((total_torques_mea_log, torques_mea_log), axis=0)
            total_left_foot_forces_log = np.concatenate((total_left_foot_forces_log, left_foot_forces_log), axis=0)
            total_right_foot_forces_log = np.concatenate((total_right_foot_forces_log, right_foot_forces_log), axis=0)
            total_left_foot_pos_log = np.concatenate((total_left_foot_pos_log, left_foot_pos_log), axis=0)
            total_right_foot_pos_log = np.concatenate((total_right_foot_pos_log, right_foot_pos_log), axis=0)
            total_states_rl = np.concatenate((total_states_rl, states_rl), axis=0)
            total_nn_output = np.concatenate((total_nn_output, nn_output), axis=0)
            total_speed = np.concatenate((total_speed, speeds), axis=0)
            total_time_hf = np.concatenate((total_time_hf, time_hf), axis=0)
            total_time_lf = np.concatenate((total_time_lf, time_lf), axis=0)

        curr_num += 1
    curr_sto += 1
    curr_num = 0

print(total_states_rl.shape)
print("total numStates: ", total_numStates)
print(total_pelvis.shape)
# exit()

# j = 0
# for t in trajectory_steps:
#     trajectory_log[j, :] = t[:]
#     j += 1

SAVE_NAME = args.path + 'logdata.npz'
np.savez(SAVE_NAME, rl_state = total_states_rl, nn_output = total_nn_output, time = total_time_hf, time_lf=total_time_lf, pelvis = total_pelvis, 
    pelvis_vels = total_pelvis_vels, motor = total_motors_log, joint = total_joints_log, torques_measured=total_torques_mea_log, 
    left_foot_force = total_left_foot_forces_log, right_foot_force = total_right_foot_forces_log, left_foot_pos = total_left_foot_pos_log, 
    right_foot_pos = total_right_foot_pos_log, speeds=total_speed)