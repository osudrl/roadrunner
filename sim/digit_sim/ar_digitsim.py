import asyncio
import agility
import agility.messages as msg
import atexit
import numpy as np
import time

from .digit_ar_sim.interface_ctypes import (
    llapi_command_t,
    llapi_observation_t,
    NUM_MOTORS,
    NUM_JOINTS,
    Locomotion
)

from .digit_ar_sim.interface_ctypes import (
    llapi_init,
    llapi_get_limits,
    llapi_get_observation,
    llapi_send_command,
    llapi_connected
)


class ArDigitSim:
    """
    This class will handle simulator initialization and close. The path to ar-control binary file
    should be provided at construction. Conf files (args) can be loaded as well.
    @TODO (helei): best way to sync up ports/ips for sim/api/llapi?
    """
    def __init__(self,
                 args,
                 path_to_ar_control: str,
                 address: str = '127.0.0.1',
                 port: int = 8081,
                 connect_timeout: float = 1.0):
        # Initialize simulator and ar-control stack
        self.sim = agility.Simulator(path_to_ar_control, *args, auto_port=True)
        self.address = address
        self.port = self.sim.port
        # Initialize API to comm with simulator/ar-control
        self.api = agility.JsonApi(address=self.address,
                                   port=self.port,
                                   connect_timeout=connect_timeout)
        # Other parameters
        self.simulator_frequency = 2000 #Hz. This is really up to 400Hz

        # Initialize LLAPI
        self._init_llapi()

        # Register atexit calls
        atexit.register(self._simulator_close)

    async def setup(self):
        """Connect API with simulator. Must call this.

        Raises:
            Exception: When connection failed, check the address and port in constructor.
        """
        try:
            await self.api.connect()
        except:
            raise Exception(f"Cannot connect api at {self.address} with port {self.port}")

    def _init_llapi(self):
        """Initialize LLAPI to intercept commands and update obs

        Raises:
            ConnectionError: 5 seconds to connect with LLAPI via UDP.
        """
        llapi_init(self.address)
        start = time.monotonic()
        while not llapi_get_observation(llapi_observation_t()):
            llapi_send_command(llapi_command_t())
            if time.monotonic() - start > 5:
                raise ConnectionError(f"Cannot connect to LLAPI! Check port and IP address setup! "
                                       "Or increase the waiting time for UDP connection!")
        # Define obs, cmd, and limit after connected
        self._actuator_limit = llapi_get_limits()[0]
        self._obs = llapi_observation_t()
        self._cmd = llapi_command_t()

    async def reset(self):
        """Reset and then pause simulator.
        """
        await self.api.send('simulator-reset')
        await self.api.request_privilege('change-action-command')
        # A bit wait for simulator settles
        await asyncio.sleep(0.5)
        # Pause simulator and record timestamp
        await self.api.send(msg.SimulatorPause())
        self.t = (await self.api.query(msg.GetTimestamp()))['run-time']

    async def sim_forward(self, dt:float, actions:list=None):
        """Step simulator by dt and update/track self.t.
        NOTE: Tries to run step at 2kHz, but tested can only reliably run up to 200Hz before losing
        time accuracy, so max dt that correctly sim forward is 0.005s.
        Keep 2kHz (dt=1/2000) does not affect accuracy.

        Args:
            dt (float): Amount of time to forward simulator
            actions (list[float]): A list of actions from upstream envs
        """
        self.t = (await self.api.query(msg.GetTimestamp()))['run-time']
        time_start = self.t
        while dt >= self.t - time_start:
            # Send commands, step sim, and update obs/time. They are all async calls, and this setup
            # tries to make them sync as much as possible.
            await self.api.send(msg.SimulatorPause())
            self._send_commands(actions=actions)
            await self.api.send(msg.SimulatorStep(dt=1/self.simulator_frequency))
            self._get_observations()
            self.t = (await self.api.query(msg.GetTimestamp()))['run-time']

    def _get_observations(self):
        """Update the observations from LLAPI
        """
        # Since UDP-based comm and loop-rate bottleneck, obs will be as close to 400Hz signals with
        # some delays. We can probabaly never get 2kHz state since UDP and loop-time.
        llapi_get_observation(self._obs)
        # To make sure obs is update-to-date, we use return code to validate obs.
        # TODO: helei, below adds tons of extra time to get fresh obs. Figure out if we need to
        # 1. sim_forward with less dt or 2. actually need to make sure obs is update-to-date.
        # start_time = time.monotonic()
        # while not llapi_get_observation(self._obs):
        #     # print("getting one more time", obs_valid)
        #     self._send_commands()
        #     # obs_valid = llapi_get_observation(self._obs)
        #     print(time.monotonic() - start_time)
        #     # if time.monotonic() - start_time > 2:
        #     #     raise ConnectionError("Cannot get fresh observations in 2s. Lost connection to LLAPI")

    def _send_commands(self, actions:list=None):
        """Send commands to LLAPI

        Args:
            actions (list): Coming from upstream env side
        """
        # TODO: helei, add PD setpoint into LLAPI header so the torques can be updated in c level
        if actions is not None: # Write to cmd struct if actions present
            apply_command = True
            for i in range(NUM_MOTORS):
                self._cmd.motors[i].torque = 150*(actions[i] - self._obs.motor.position[i])
                self._cmd.motors[i].velocity = 0.0
                self._cmd.motors[i].damping = 0.75 * self._actuator_limit.damping_limit[i]
        else:
            apply_command = False
        self._cmd.fallback_opmode = Locomotion
        self._cmd.apply_command = apply_command
        llapi_send_command(self._cmd)

    """Followings are getter functions for observations.
    """
    def get_motor_position(self):
        return np.array(self._obs.motor.position[:NUM_MOTORS])

    def get_motor_velocity(self):
        return np.array(self._obs.motor.velocity[:NUM_MOTORS])

    def get_motor_torque(self):
        return np.array(self._obs.motor.torque[:NUM_MOTORS])

    def get_joint_position(self):
        return np.array(self._obs.joint.position[:NUM_JOINTS])

    def get_joint_velocity(self):
        return np.array(self._obs.joint.velocity[:NUM_JOINTS])

    def get_base_translation(self):
        return np.array(self._obs.base.translation[:3])

    def get_base_orientation(self):
        return np.array(self._obs.base.orientation[:4])

    def get_base_linear_velocity(self):
        return np.array(self._obs.base.linear_velocity[:3])

    def get_base_angular_velocity(self):
        return np.array(self._obs.base.angular_velocity[:3])

    def get_imu_linear_accleration(self):
        return np.array(self._obs.imu.linear_acceleration[:3])

    def get_imu_angular_velocity(self):
        return np.array(self._obs.imu.angular_velocity[:3])

    def get_imu_orientation(self):
        return np.array(self._obs.imu.orientation[:4])

    """Placeholder since ar-control visualize via websockets automatically
    """
    def viewer_init(self):
        pass

    def viewer_render(self):
        pass

    def is_llapi_connected(self):
        return llapi_connected()

    async def _api_close(self):
        await self.api.close()

    def _simulator_close(self):
        self.sim.close()

    async def close_all(self):
        await self._api_close()
        self._simulator_close()
