#File to visualize realtime cassie data
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="./hardware_logs/traj-aslip_aslip_old_2048_12288_seed-10/", help="Path to folder containing policy hardware logs")
args = parser.parse_args()

# logs = pickle.load(open(args.path + "logdata.pkl", "rb")) #load in file with cassie data

data = np.load(args.path + "logdata.npz", allow_pickle=True)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

# Flags for reducing and increasing what to visualize
vis = { "mpos": True,
        "trajpos": False,
        "torques": True,
        "joints": False,
        "footforce": False,
        "footpos": False,
        "nn_output": True,
        "pelvis": True,
        "time": True}

vis_input = True

aslip_vis = {"aslipTS": False,
            "aslipVel": False,
            "aslipFootPos": False}

plotnum = sum(vis.values())

# truncate data to last cycle
for key in data:
    print(key)

time = data['time']
time_lf = data['time_lf']
# print(time_lf)
pelvis = data['pelvis']
pelvis_vels = data['pelvis_vels']
motors = data['motor']
joints = data['joint']
nn_output = data['nn_output']
torques_mea = data['torques_measured']
ff_left = data['left_foot_force']
ff_right = data['right_foot_force']
foot_pos_left = data['left_foot_pos']
foot_pos_right= data['right_foot_pos']
# trajectory = data['trajectory']
# states_rl = data["states_rl"]
speeds = data["speeds"]
rl_state = data["rl_state"]

print("max time: ", np.max(time))
print("last time: ", time[-1])

row = plotnum
col = 1
idx = 0
fig, axs = plt.subplots(row, col,sharex=True)


#Plot Motor Positions
if vis["mpos"]:
    motors = np.rad2deg(motors)
    axs[idx].plot(time[:], motors[:, 0], label='left-hip-roll'  )
    axs[idx].plot(time[:], motors[:, 1], label='left-hip-yaw'   )
    axs[idx].plot(time[:], motors[:, 2], label='left-hip-pitch' )
    axs[idx].plot(time[:], motors[:, 3], label='left-knee'      )
    axs[idx].plot(time[:], motors[:, 4], label='left-foot'      )
    axs[idx].plot(time[:], motors[:, 5], label='right-hip-roll' )
    axs[idx].plot(time[:], motors[:, 6], label='right-hip-yaw'  )
    axs[idx].plot(time[:], motors[:, 7], label='right-hip-pitch')
    axs[idx].plot(time[:], motors[:, 8], label='right-knee'     )
    axs[idx].plot(time[:], motors[:, 9], label='right-foot'     )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Motor Position [deg]')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Motor Position')
    idx += 1


#Plot Trajectory Positions
if vis["trajpos"]:
    trajectory = np.rad2deg(trajectory)
    axs[idx].plot(time[:], trajectory[:, 0], label='left-hip-roll'  )
    # axs[idx].plot(time[:], trajectory[:, 1], label='left-hip-yaw'   )
    axs[idx].plot(time[:], trajectory[:, 2], label='left-hip-pitch' )
    axs[idx].plot(time[:], trajectory[:, 3], label='left-knee'      )
    # axs[idx].plot(time[:], motors[:, 4], label='left-foot'      )
    axs[idx].plot(time[:], trajectory[:, 5], label='right-hip-roll' )
    # axs[idx].plot(time[:], trajectory[:, 6], label='right-hip-yaw'  )
    axs[idx].plot(time[:], trajectory[:, 7], label='right-hip-pitch')
    axs[idx].plot(time[:], trajectory[:, 8], label='right-knee'     )
    # axs[idx].plot(time[:], trajectory[:, 9], label='right-foot'     )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Traj Motor Position [deg]')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Traj Motor Position')
    idx += 1


# measured torques
if vis["torques"]:
    axs[idx].plot(time, torques_mea[:, 0], label='left-hip-roll'  )
    # axs[idx].plot(time, torques_mea[:, 1], label='left-hip-yaw'   )
    axs[idx].plot(time, torques_mea[:, 2], label='left-hip-pitch' )
    axs[idx].plot(time, torques_mea[:, 3], label='left-knee'      )
    # axs[idx].plot(time, torques_mea[:, 4], label='left-foot'      )
    axs[idx].plot(time, torques_mea[:, 5], label='right-hip-roll' )
    # axs[idx].plot(time, torques_mea[:, 6], label='right-hip-yaw'  )
    axs[idx].plot(time, torques_mea[:, 7], label='right-hip-pitch')
    axs[idx].plot(time, torques_mea[:, 8], label='right-knee'     )
    # axs[idx].plot(time, torques_mea[:, 9], label='right-foot'     )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Measured Torques [Nm]')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Measured Torques')
    idx += 1

#Plot Joint Positions
if vis["joints"]:
    joints = np.rad2deg(joints)
    axs[idx].plot(time, joints[:, 0], label='left-knee-spring'  )
    # axs[idx].plot(time, joints[:, 1], label='left-tarsus')
    # axs[idx].plot(time, joints[:, 2], label='left-foot'  )
    axs[idx].plot(time, joints[:, 3], label='right-knee-spring'  )
    # axs[idx].plot(time, joints[:, 4], label='right-tarsus')
    # axs[idx].plot(time, joints[:, 5], label='right-foot'  )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Joint Position [deg]')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Joint Position')
    idx += 1

# foot force
if vis["footforce"]:
    # axs[idx].plot(time, ff_left[:, 0], label='left-X'  )
    # axs[idx].plot(time, ff_left[:, 1], label='left-Y'  )
    axs[idx].plot(time, ff_left[:, 2], label='left-Z'  )
    # axs[idx].plot(time, ff_right[:, 0], label='right-X'  )
    # axs[idx].plot(time, ff_right[:, 1], label='right-Y'  )
    axs[idx].plot(time, ff_right[:, 2], label='right-Z'  )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Foot Force [N]')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Foot Forces')
    idx += 1

# foot pos
if vis["footpos"]:
    axs[idx].plot(time, foot_pos_left[:, 0], label='left-X'  )
    axs[idx].plot(time, foot_pos_left[:, 1], label='left-Y'  )
    axs[idx].plot(time, foot_pos_left[:, 2], label='left-Z'  )
    axs[idx].plot(time, foot_pos_right[:, 0], label='right-X'  )
    axs[idx].plot(time, foot_pos_right[:, 1], label='right-Y'  )
    axs[idx].plot(time, foot_pos_right[:, 2], label='right-Z'  )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Foot Pos [m]')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Foot Pos')
    idx += 1

if vis["nn_output"]:
    axs[idx].plot(time[:], nn_output[:, 0], label='left-hip-roll'  )
    axs[idx].plot(time[:], nn_output[:, 1], label='left-hip-yaw'   )
    axs[idx].plot(time[:], nn_output[:, 2], label='left-hip-pitch' )
    axs[idx].plot(time[:], nn_output[:, 3], label='left-knee'      )
    axs[idx].plot(time[:], nn_output[:, 4], label='left-foot'      )
    axs[idx].plot(time[:], nn_output[:, 5], label='right-hip-roll' )
    axs[idx].plot(time[:], nn_output[:, 6], label='right-hip-yaw'  )
    axs[idx].plot(time[:], nn_output[:, 7], label='right-hip-pitch')
    axs[idx].plot(time[:], nn_output[:, 8], label='right-knee'     )
    axs[idx].plot(time[:], nn_output[:, 9], label='right-foot'     )
    # axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('NN Output')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('NN Output')
    idx += 1

if vis["pelvis"]:
    # axs[idx].plot(time[:], pelvis[:, 0], label="x")
    # axs[idx].plot(time[:], pelvis[:, 1], label="y")
    axs[idx].plot(time[:], pelvis[:, 2], label="z pos")
    axs[idx].set_ylabel('Pelvis Position (m)')
    axs[idx].legend(loc='upper left')
    axs[idx].set_title('Pelvis Position')
    idx += 1

if vis["time"]:
    time_diff = time[1:] - time[:-1]
    # delay_inds = np.where(time_diff > .05)
    delay_inds = time_diff.argsort()[-5:][::-1]

    print("Biggest delays are {}".format(time_diff[delay_inds]))
    print("at times {}".format(time[delay_inds]))
    # axs[idx].scatter(time[1:], time_diff)
    axs[idx].plot(time[1:], time_diff)
    axs[idx].set_title("Compute Time")



if aslip_vis["aslipTS"]:
    fig2 = plt.figure(figsize=(10,10))
    ax = fig2.add_subplot(111, projection='3d')

    # 1 m/s constant x velocity
    x_offset = np.linspace(0, 10, num=pelvis.shape[0])
    x_offset = foot_pos_left[:, 0]

    ax.plot(pelvis[:, 0] + x_offset, pelvis[:, 1], pelvis[:, 2], label='pelvis')
    ax.plot(pelvis[:, 0] + x_offset + foot_pos_left[:, 0], pelvis[:, 1] + foot_pos_left[:, 1], pelvis[:, 2] + foot_pos_left[:, 2], label='true left foot pos')
    ax.plot(pelvis[:, 0] + x_offset + foot_pos_right[:, 0], pelvis[:, 1] + foot_pos_right[:, 1], pelvis[:, 2] + foot_pos_right[:, 2], label='true right foot pos')

    set_axes_equal(ax)

if aslip_vis["aslipVel"]:
    ax1 = plt.subplot(1,1,1)
    ax1.plot(time, speeds, label='speed command')
    ax1.plot(time_lf, states_rl[:,61], label='ROM COM x velocity')
    ax1.plot(time, pelvis_vels[:,0], label='pelvis x velocity')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('m/s')
    ax1.legend(loc='upper right')
    ax1.set_title('Varying Vels')

# plt.tight_layout()
plt.show()

if vis_input:
    fig, ax = plt.subplots(1, 1)
    print(rl_state[0, :].shape)
    ax.plot(time_lf, rl_state[:, 40:46])
    plt.show()
