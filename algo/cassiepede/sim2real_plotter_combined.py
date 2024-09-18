import os
import sys
import time
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import socket

from matplotlib import gridspec
from matplotlib.pyplot import figure
import datetime
import moviepy.video.io.ImageSequenceClip
import matplotlib.lines as mlines
import pickle

from util.topic import Topic


def plot_topic(ax, platform, plot_data):
    ax.cla()

    # Initialize plot
    ax.set_title(platform, fontsize=20)
    ax.axis('equal')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_ylim(-2, 2)
    ax.set_xlim(-2, 2)
    ax.grid(True)

    x_vel, y_vel, turn_rate, height, r, theta, base_yaw_poi, base_orient_cmd, global_orient = plot_data

    x, y = r * np.cos(theta), r * np.sin(theta)

    # Plot a line connecting poi and each cassie
    ax.plot((0, y), (0, -x), '--')

    # Plot poi axis (always fixed)
    ax.quiver(0, 0, 0, 0.4,
              color='red',
              angles="xy",
              scale_units='xy',
              width=0.01,
              scale=0.5)

    # Plot commanded turn rate relative with poi axis
    ax.quiver(0, 0, 0.4 * np.cos(base_yaw_poi - base_orient_cmd + np.pi / 2),
              0.4 * np.sin(base_yaw_poi - base_orient_cmd + np.pi / 2),
              color='green',
              angles="xy",
              scale_units='xy',
              width=0.01,
              scale=0.8)

    # Plot base orientation relative to poi axis
    ax.quiver(y,
              -x, 0.4 * np.cos(base_yaw_poi + np.pi / 2),
              0.4 * np.sin(base_yaw_poi + np.pi / 2),
              color='orange',
              angles="xy",
              scale_units='xy',
              width=0.01,
              scale=1.0)

    # Plot global imu of cassie
    ax.quiver(0, 0, 0.4 * np.cos(global_orient + np.pi / 2),
              0.4 * np.sin(global_orient + np.pi / 2),
              color='blue',
              angles="xy",
              scale_units='xy',
              width=0.01,
              scale=0.8)

    ax.annotate(f'x:{x_vel:.2f}\ny:{y_vel:.2f}\nt:{turn_rate:.2f}\nh:{height:.2f}', (-1, -1.75), fontsize=20)

    ax.scatter(0, 0, c='g')
    ax.scatter(y, -x, c='r')


def plot_legend(ax):
    cmd = mlines.Line2D([], [], color='green', marker='s', linestyle='None',
                        markersize=20, label='Cmd')
    poi = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                        markersize=20, label='POI')
    heading = mlines.Line2D([], [], color='orange', marker='s', linestyle='None',
                            markersize=20, label='Base')

    orient = mlines.Line2D([], [], color='blue', marker='s', linestyle='None',
                           markersize=20, label='Orient')

    ax.legend(handles=[cmd, poi, heading, orient], fontsize=20, loc='lower right')


def main():
    ip = '192.168.2.97'
    # ip = '10.0.0.96'
    platforms = dict(
        cassie_harvard=(ip, 1234),
        cassie=(ip, 1235),
        cassie_play=(ip, 1236)
    )

    width_ratio = [1, 1, 1]

    plt.figure(figsize=(int(sum(8 * np.array(width_ratio))), 8))

    gs = gridspec.GridSpec(nrows=1, ncols=len(platforms), width_ratios=width_ratio)

    axs = [plt.subplot(gs[i]) for i in range(len(platforms))]

    topics = {platform: Topic() for platform in platforms.keys()}

    save_video_freq = 5  # FPS

    for platform, topic in topics.items():
        topic.subscribe(platforms[platform])

    # run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # read from arg
    run_name = f"{sys.argv[1]}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    print('run_name:', run_name)

    if save_video_freq:
        os.makedirs(os.path.join('data', 'videos', run_name), exist_ok=True)
        os.makedirs(os.path.join('data', 'raw', run_name), exist_ok=True)

    last_save = time.monotonic()

    for _ in count():
        plot_data_dict = {}
        for i, (platform, topic) in enumerate(topics.items()):
            plot_data = topic.get_data()
            if plot_data is None:
                # print('No data received from platform:', platform)
                continue
            plot_data = np.frombuffer(topic.get_data(), dtype=np.float16)
            plot_data_dict[platform] = plot_data
            plot_topic(axs[i], platform, plot_data)

        plot_legend(axs[len(axs) // 2])

        if save_video_freq and time.monotonic() - last_save > 1 / save_video_freq:
            print('saving')
            plt.savefig(os.path.join('data', 'videos', run_name, f'{time.time()}.png'), dpi=80)
            with open(os.path.join('data', 'raw', run_name, f'{time.time()}.pkl'), 'wb') as f:
                pickle.dump(plot_data_dict, f)

            last_save = time.monotonic()

        plt.pause(0.00001)


if __name__ == '__main__':
    main()
