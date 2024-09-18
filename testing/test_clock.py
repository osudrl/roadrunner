import matplotlib.pyplot as plt
import numpy as np
import time

from env.tasks.locomotionclockenv.locomotionclockenv import LocomotionClockEnv
from env.util.periodicclock import PeriodicClock
from util.colors import OKGREEN, FAIL, ENDC


def plot_clock(t, lr_swing_vals, title: str):
    # Input should be 2d array of shape [swing values, 2], left will come first ([:, 0]) and right
    # is second ([:, 1])
    fig, ax = plt.subplots(2, 1, figsize=(8,5))
    ax[0].plot(t, lr_swing_vals[:, 0], label="swing")
    ax[0].plot(t, 1 - lr_swing_vals[:, 0], label="stance")
    ax[1].plot(t, lr_swing_vals[:, 1], label="swing")
    ax[1].plot(t, 1 - lr_swing_vals[:, 1], label="stance")

    ax[0].set_title("Left Foot", fontsize=18)
    ax[1].set_title("Right Foot", fontsize=18)
    ax[1].set_xlabel("Time (sec)", fontsize=16)
    ax[0].set_ylabel("Cost Weighting", fontsize=16)
    ax[1].set_ylabel("Cost Weighting", fontsize=16)
    fig.suptitle(title, fontsize=20)

    ax[0].legend(loc=1, prop={"size":16})
    ax[1].legend(loc=1, prop={"size":16})
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def test_all_clocks():
    ### Disable plotting tests show plots don't show up in tests, but keep for debugs later ###
    # test_linear_walk_clock()
    # test_linear_run_clock()
    # test_linear_hop_clock()
    # test_linear_gallop_clock()

    # test_vonmises_walk_clock()
    # test_vonmises_run_clock()
    # test_vonmises_hop_clock()
    # test_vonmises_gallop_clock()

    test_vonmises_precompute()
    test_clockenv()
    print(f"{OKGREEN}Passed all clock tests!{ENDC}")

def test_linear_walk_clock():
    cycle_time = 1
    swing_ratios = [0.4, 0.4]
    period_shifts = [0, 0.5]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Walking Clock")

def test_linear_run_clock():
    cycle_time = 1
    swing_ratios = [0.8, 0.8]
    period_shifts = [0, 0.5]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Running Clock")

def test_linear_hop_clock():
    cycle_time = 1
    swing_ratios = [0.5, 0.5]
    period_shifts = [0, 0]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Hopping Clock")

def test_linear_gallop_clock():
    cycle_time = 1
    swing_ratios = [0.6, 0.6]
    period_shifts = [0, 0.25]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.linear_clock()
    plot_clock(xs, ys, "Uneven Gallop Clock")

def test_vonmises_walk_clock():
    cycle_time = 1.0
    swing_ratios = [0.4, 0.4]
    period_shifts = [0.0, 0.5]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Walk Clock")

def test_vonmises_run_clock():
    cycle_time = 1
    swing_ratios = [0.8, 0.8]
    period_shifts = [0.0, 0.5]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Run Clock")

def test_vonmises_hop_clock():
    cycle_time = 1
    swing_ratios = [0.5, 0.5]
    period_shifts = [0.0, 0.0]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Hop Clock")

def test_vonmises_gallop_clock():
    cycle_time = 1
    swing_ratios = [0.6, 0.6]
    period_shifts = [0.0, 0.25]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    plot_clock(xs, ys, "Von Mises Gallop Clock")

def test_vonmises_precompute():
    cycle_time = 1
    swing_ratios = [0.4, 0.4]
    period_shifts = [0.0, 0.5]
    tracker = PeriodicClock(cycle_time, 1 / 50, swing_ratios, period_shifts)
    xs = np.linspace(0, cycle_time, 1000)
    ys = np.zeros((1000, 2))
    start_t = time.time()
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.von_mises()
    print("Von mises time", time.time() - start_t)
    tracker.precompute_von_mises(num_points = 1000)
    start_t = time.time()
    ys = np.zeros((1000, 2))
    for i in range(len(xs)):
        tracker.set_phase(xs[i])
        ys[i, :] = tracker.get_von_mises_values()
    print("Precompute time", time.time() - start_t)

def test_clockenv():
    clock_type = "linear"
    policy_rate = 50
    env = LocomotionClockEnv(
        robot_name="cassie", # same as digit
        clock_type = clock_type,
        reward_name = "locomotion_linear_clock_reward",
        simulator_type = "mujoco",
        terrain = False,
        policy_rate = policy_rate,
        dynamics_randomization = False,
        state_est=False,
        state_noise=[0,0,0,0,0,0]
    )
    env.reset()
    init_phase = env.clock.get_phase()
    for i in range(int(env.cycle_time * policy_rate)):
        env.step(np.zeros(10))
    assert np.abs(env.clock.get_phase() - init_phase) < env.clock._phase_add, \
        f"Failed CassieClockEnv linear test, after one cycle phase should be 0, but phase is " \
        f"{env.clock.get_phase()}"

    clock_type = "von_mises"
    env = LocomotionClockEnv(
        robot_name="cassie", # same as digit
        clock_type = clock_type,
        reward_name = "locomotion_vonmises_clock_reward",
        simulator_type = "mujoco",
        terrain = False,
        policy_rate = policy_rate,
        dynamics_randomization = False,
        state_est=False,
        state_noise=[0,0,0,0,0,0]
    )
    env.reset()
    init_phase = env.clock.get_phase()
    for i in range(int(env.cycle_time * policy_rate)):
        env.step(np.zeros(10))
    assert np.abs(env.clock.get_phase() - init_phase) < env.clock._phase_add, \
        f"Failed CassieClockEnv von mises test, after one cycle phase should be 0, but phase is " \
        f"{env.clock.get_phase()}"
