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
    # num_steps = 30000
    # # Test raw sampling speeds
    # sim_types = [MjCassieSim, MjDigitSim]
    # sim_speeds = {}
    # for i, sim_type in enumerate(sim_types):
    #     # sim = sim_type()
    #     sim = sim_type(terrain='hfield')
    #     total_time = 0
    #     for i in range(num_steps):
    #         sim.set_torque(np.random.uniform(size=(sim.num_actuators,)))
    #         start_t = time.time()
    #         sim.sim_forward()
    #         total_time += time.time() - start_t
    #     sim_speeds[sim.__class__.__name__] = total_time / num_steps

    # for key, val in sim_speeds.items():
    #     print(f"Sim {key} averaged {val:.4e} seconds per step, running at a rate of {1 / val:.2f} Hz")

    args = SimpleNamespace(simulator_type = "mujoco",
                        clock_type = "linear",
                        reward_name = "depth",
                        dynamics_randomization = False)
    mj_env = env_factory("CassieHfield", args)()
    mj_env.reset()
    args = SimpleNamespace(simulator_type = "mujoco",
                        clock_type = "von_mises",
                        reward_name = "locomotion_vonmises_clock_reward",
                        dynamics_randomization = False)
    lib_env = env_factory("CassieEnvClock", args)()
    lib_env.reset()

    # Test env sampling speed
    env_steps = 1000
    total_time = 0
    mj_sim_per_step = int(mj_env.sim.simulator_rate / mj_env.default_policy_rate)
    for i in range(env_steps):
        act = np.random.uniform(size=(mj_env.sim.num_actuators,))
        start_t = time.time()
        mj_env.step(act)
        total_time += time.time() - start_t
    mj_avg_time = total_time / (env_steps * mj_sim_per_step)
    # mj_overhead = (mj_avg_time - sim_speeds["MjCassieSim"]) * mj_sim_per_step

    total_time = 0
    lib_sim_per_step = int(lib_env.sim.simulator_rate / lib_env.default_policy_rate)
    for i in range(env_steps):
        act = np.random.uniform(size=(lib_env.sim.num_actuators,))
        start_t = time.time()
        lib_env.step(act)
        total_time += time.time() - start_t
    lib_avg_time = total_time / (env_steps * lib_sim_per_step)
    # lib_overhead = (lib_avg_time - sim_speeds["LibCassieSim"]) * lib_sim_per_step

    print(f"Env with mujoco averaged {mj_avg_time:.4e}, running at rate of {1 / mj_avg_time:.2f} Hz.")
    # print(f"Lose {mj_overhead:.4e} seconds per env step due to env overhead. Over 500 million steps " \
    #     f"lose {mj_overhead * 500000000/60/60/24:.2f} days")

    print(f"Env with libcassie averaged {lib_avg_time:.4e}, running at rate of {1 / lib_avg_time:.2f} Hz.")
    # print(f"Lose {lib_overhead:.4e} seconds per env step due to env overhead. Over 500 million steps " \
    #     f"lose {lib_overhead * 500000000/60/60/24:.2f} days")

def run_PD_env_compare():
    args = SimpleNamespace(simulator_type = "mujoco",
                        clock_type = "von_mises",
                        reward_name = "locomotion_vonmises_clock_reward",
                        dynamics_randomization = False)
    env1 = env_factory("CassieEnvClock", args)()
    env2 = env_factory("DigitEnvClock", args)()
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
    print(f"Env2 took {env2_time / num_steps}, {1 / (env2_time / num_steps):.2f} Hz")
    print(f"Env1 loop took {env1_time / num_steps}, {1 / (env1_time / num_steps):.2f} Hz")


