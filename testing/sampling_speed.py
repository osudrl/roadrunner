import mujoco as mj
import numpy as np
import time
from types import SimpleNamespace

from sim import (
    MjCassieSim,
    LibCassieSim,
    MjDigitSim
)
from util.env_factory import env_factory

def sampling_speed():
    num_steps = 30000
    # Test raw sampling speeds
    sim_types = [MjCassieSim, LibCassieSim]
    sim_speeds = {}
    for sim_type in sim_types:
        sim = sim_type()
        total_time = 0
        for i in range(num_steps):
            sim.set_torque(np.random.uniform(size=(10,)))
            start_t = time.time()
            sim.sim_forward()
            total_time += time.time() - start_t
        sim_speeds[sim.__class__.__name__] = total_time / num_steps

    for key, val in sim_speeds.items():
        print(f"Sim {key} averaged {val:.4e} seconds per step, running at a rate of {1 / val:.2f} Hz")

    args = SimpleNamespace(
        robot_name="cassie",
        simulator_type = "mujoco",
        clock_type = "linear",
        reward_name = "locomotion_linear_clock_reward",
        dynamics_randomization = False
    )
    mj_env = env_factory("LocomotionClockEnv", args)()
    mj_env.reset()
    args = SimpleNamespace(
        robot_name="cassie",
        simulator_type = "libcassie",
        clock_type = "linear",
        reward_name = "locomotion_linear_clock_reward",
        dynamics_randomization = False
    )
    lib_env = env_factory("LocomotionClockEnv", args)()
    lib_env.reset()

    # Test env sampling speed
    env_steps = 1000
    total_time = 0
    mj_sim_per_step = int(mj_env.sim.simulator_rate / mj_env.default_policy_rate)
    for i in range(env_steps):
        act = np.random.uniform(size=(10,))
        start_t = time.time()
        mj_env.step(act)
        total_time += time.time() - start_t
    mj_avg_time = total_time / (env_steps * mj_sim_per_step)
    mj_overhead = (mj_avg_time - sim_speeds["MjCassieSim"]) * mj_sim_per_step

    total_time = 0
    lib_sim_per_step = int(lib_env.sim.simulator_rate / lib_env.default_policy_rate)
    for i in range(env_steps):
        act = np.random.uniform(size=(10,))
        start_t = time.time()
        lib_env.step(act)
        total_time += time.time() - start_t
    lib_avg_time = total_time / (env_steps * lib_sim_per_step)
    lib_overhead = (lib_avg_time - sim_speeds["LibCassieSim"]) * lib_sim_per_step

    print(f"Env with mujoco averaged {mj_avg_time:.4e}, running at rate of {1 / mj_avg_time:.2f} Hz.")
    print(f"Lose {mj_overhead:.4e} seconds per env step due to env overhead. Over 500 million steps " \
        f"lose {mj_overhead * 500000000/60/60/24:.2f} days")

    print(f"Env with libcassie averaged {lib_avg_time:.4e}, running at rate of {1 / lib_avg_time:.2f} Hz.")
    print(f"Lose {lib_overhead:.4e} seconds per env step due to env overhead. Over 500 million steps " \
        f"lose {lib_overhead * 500000000/60/60/24:.2f} days")

def run_PD_env_compare():
    args = SimpleNamespace(
        robot_name="cassie",
        simulator_type = "mujoco",
        clock_type = "von_mises",
        reward_name = "locomotion_vonmises_clock_reward",
        dynamics_randomization = False
    )
    env1 = env_factory("LocomotionClockEnv", args)()
    args.robot_name = "digit"
    env2 = env_factory("LocomotionClockEnv", args)()
    num_steps = 1000
    env1_time = 0
    env2_time = 0
    for i in range(num_steps):
        act = np.random.uniform(size=(env1.action_size,))
        start_t = time.time()
        env1.step_simulation(act, 40)
        env1_time += time.time() - start_t

        start_t = time.time()
        act = np.random.uniform(size=(env2.action_size,))
        env2.step_simulation(act, 40)
        env2_time += time.time() - start_t

        # if (sim1.data.qpos[:] - sim2.data.qpos[:]).sum() > 0.00001:
        #     print("Different qpos")
    print(f"Digit took {env2_time / num_steps}, {1 / (env2_time / num_steps):.2f} Hz")
    print(f"Cassie loop took {env1_time / num_steps}, {1 / (env1_time / num_steps):.2f} Hz")

def print_mj_profile(data):
    print("Mujoco Interal Profiler, \u03BCs per step")
    total_mj_time = 0
    components = 0
    for i in range(8):
        num = max(1, data.timer[i].number)
        istep = data.timer[i].duration / num
        if i == 0:
            total_mj_time = istep
        if i >= int(mj.mjtTimer.mjTIMER_POSITION):
            components += istep
        timestr = mj.mjTIMERSTRING[i]
        if len(timestr) > 7:
            timestr += "\t"
        else:
            timestr += "\t\t"
        print(f"\t\t{timestr}: {1e6 * istep:.2f}")
    print(f"\t\tother\t\t: {1e6 * (total_mj_time-components):.2f}")
    print()
    print(f"\tposition total\t: {1e6 * data.timer[3].duration / data.timer[3].number:.2f}")
    for i in range(8, 13):
        num = max(1, data.timer[i].number)
        istep = data.timer[i].duration / num
        timestr = mj.mjTIMERSTRING[i][4:]
        if len(timestr) <= 4:
            timestr += "\t"
        timestr += "\t"
        print(f"\t  {timestr}: {1e6 * istep:.2f}")
        if i == int(mj.mjtTimer.mjTIMER_POS_COLLISION):
            for j in range(13, 16):
                num = max(1, data.timer[j].number)
                istep = data.timer[j].duration / num
                timestr = mj.mjTIMERSTRING[j][4:]
                if len(timestr) <= 4:
                    timestr += "\t"
                timestr += "\t"
                print(f"\t    {timestr}: {1e6 * istep:.2f}")

def run_model_compare():
    # Compares the simulation performance between the original model and the optimized "fast" model.
    # Shows runtime statistics with the Mujoco profiler
    def Timer():
        return time.perf_counter()

    mj.set_mjcb_time(Timer)

    num_step = 100000
    ctrl_noise = 0.1

    fast_sim = MjDigitSim(model_name="digit-v3-fast.xml")
    sim = MjDigitSim(model_name="digit-v3.xml")

    ctrl = np.zeros((num_step, sim.model.nu))
    for i in range(num_step):
        for j in range(sim.model.nu):
            center = 0
            radius = 1
            ctrlrange = sim.model.actuator_ctrlrange[j, :]
            if sim.model.actuator_ctrllimited[j]:
                center = (ctrlrange[0] + ctrlrange[1]) / 2
                radius = (ctrlrange[1] - ctrlrange[0]) / 2
            radius *= ctrl_noise
            ctrl[i, j] = center + radius * (2 * mj.mju_Halton(i, j+2) - 1)

    print("Made ctrl sequence")

    start_t = time.time()
    for i in range(num_step):
        sim.set_torque(ctrl[i, :])
        sim.sim_forward()
    elapsed = time.time() - start_t

    print(f"Regular Sim Total simulation time\t: {elapsed:.4f} seconds")
    print(f"Regular Sim Total steps per second\t: {num_step / elapsed:.2f}")
    print(f"Regular Sim Realtime factor\t\t: {num_step * sim.model.opt.timestep / elapsed:.4f} x")
    print()
    print_mj_profile(sim.data)
    print("\n")

    start_t = time.time()
    for i in range(num_step):
        fast_sim.set_torque(ctrl[i, :])
        fast_sim.sim_forward()
    elapsed = time.time() - start_t

    print(f"Fast Sim Total simulation time\t: {elapsed:.4f} seconds")
    print(f"Fast Sim Total steps per second\t: {num_step / elapsed:.2f}")
    print(f"Fast Sim Realtime factor\t\t: {num_step * sim.model.opt.timestep / elapsed:.4f} x")
    print()
    print_mj_profile(fast_sim.data)

