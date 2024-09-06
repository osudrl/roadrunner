import agility.messages as msg
import numpy as np
import os

from sim.digit_sim.ar_digitsim import ArDigitSim
from .common import (
    SIM_PATH,
    ROBOT_CONFIG,
    MOTOR_POSITION_SET
)

"""
The followings are independent tests for wrappers around Agility's ar-control, pysdk, and LLAPI.
"""

async def test_ar_connect():
    ar = ArDigitSim(path_to_ar_control=os.path.expanduser(SIM_PATH), args=ROBOT_CONFIG)
    # Need to call this to connect API with simulator
    await ar.setup()
    await ar.reset()
    print("Connected with sim/api/llapi", ar.port)
    await ar.api.send(msg.SimulatorStart())
    await ar.close_all()
    print("Closed all")

async def test_ar_api_goto():
    ar = ArDigitSim(path_to_ar_control=os.path.expanduser(SIM_PATH), args=ROBOT_CONFIG)
    # Need to call this to connect API with simulator
    await ar.setup()
    await ar.reset()
    await ar.api.send(msg.SimulatorStart())
    ret = (await ar.api.wait_action(msg.ActionGoto(target={"xy": [2, 0]})))
    print(ret)
    await ar.close_all()
    print("Closed all")

async def test_ar_sim_forward():
    ar = ArDigitSim(path_to_ar_control=os.path.expanduser(SIM_PATH), args=ROBOT_CONFIG)
    # Need to call this to connect API with simulator
    await ar.setup()
    await ar.reset()
    # Send API actions, but this won't move robot, since the followings mannually sim-step
    await ar.api.send(msg.ActionGoto(target={"xy": [1, 0]}))

    # Test sim forward while running ar-control
    for _ in range(10):
        await ar.sim_forward(dt=1)
        # await ar.api.send(msg.SimulatorPause())
        # await asyncio.sleep(0.1)
        print("time={:3.2f}, x={:3.1f}".format(ar.t, ar.get_base_translation()[0]))
    if np.abs(ar.get_base_translation()[0] - 1) < 1e-2:
        print("success")
    await ar.close_all()
    print("Closed all")

async def test_ar_sim_llapi_walking_handover():
    ar = ArDigitSim(path_to_ar_control=os.path.expanduser(SIM_PATH), args=ROBOT_CONFIG)
    # Need to call this to connect API with simulator
    await ar.setup()
    await ar.reset()
    await ar.api.send(msg.ActionMove(velocity={'rpyxyz':[0,0,0,0.5,0,0]}))

    # Test sim forward while running ar-control
    for i in range(2):
        await ar.sim_forward(dt=1)
        print("time={:3.2f}, x={:3.1f}, llapi-connected={}".format(
            ar.t, ar.get_base_translation()[0], ar.is_llapi_connected()))

    # Test LLAPI mode toggle
    # To fully enable LLAPI and take over ar-control. Call sim_forward with actions for step 1 & 2
    # and API call for step 3.
    # 1. Write/Send commands struct
    # 2. Toggle apply_command to be True
    # 3. Enable LLAPI mode via API call
    await ar.sim_forward(dt=1, actions=MOTOR_POSITION_SET['pos1'])
    act = (await ar.api.query(msg.GetActionCommand()))
    if act['action'] != 'action-set-operation-mode':
        await ar.api.send(msg.ActionSetOperationMode(mode="low-level-api"))
    print("LLAPI ON")

    # Since LLAPI mode on, we can keep sending to execute custom commands.
    # Robot will fall in this test case.
    # NOTE: still mystery about when Damping mode will turn on
    for i in range(10):
        if i%3==1:
            await ar.sim_forward(dt=0.5, actions=MOTOR_POSITION_SET['pos1'])
        elif i%3==2:
            await ar.sim_forward(dt=0.5, actions=MOTOR_POSITION_SET['pos2'])
        else:
            await ar.sim_forward(dt=0.5, actions=MOTOR_POSITION_SET['pos3'])
        print("time={: 4.2f}, z-vel={: 3.2f}, llapi-connect={}".format(
            ar.t, ar.get_base_linear_velocity()[2], ar.is_llapi_connected()))

        # Example of applying damping mode
        # TODO: helei, ideally this will be triggered auto by ar-control. Need to try with hardware.
        # if ar.obs.imu.linear_acceleration[2] > 0.6:
        #     await ar.api.send(msg.ActionSetOperationMode(mode="damping"))
        #     break
        # await asyncio.sleep(5)

    await ar.close_all()
    print("Closed all")

async def test_ar_sim_llapi_teststand():
    conf = ["""
# Add robot to world
[[model-list]]
model = "robot"
pose = {xyz = [0, 0, 1.0]}
fixed=true

[simulator]
free-run=true

[planning]
initial-operation-mode = "disabled"

[lowlevelapi]
enable = true
# Time in seconds between Low-level API packets before control program
# disables robot
timeout = 0.05
# Port for incoming and outgoing Low-level API communications.
# Change if these settings interfere with any other programs or network
# Traffic on the payload computer. Must match settings in lowlevelapi.c
# (lines 181 and 183). These should not be changed under normal operation
listen-port = 25500
send-port = 25501
"""]
    ar = ArDigitSim(path_to_ar_control=os.path.expanduser(SIM_PATH), args=conf)
    # Need to call this to connect API with simulator
    await ar.setup()
    await ar.reset()

    # Test LLAPI mode toggle
    # To fully enable LLAPI and take over ar-control. Call sim_forward with actions for step 1 & 2
    # and API call for step 3.
    # 1. Write/Send commands struct
    # 2. Toggle apply_command to be True
    # 3. Enable LLAPI mode via API call
    await ar.sim_forward(dt=1, actions=MOTOR_POSITION_SET['pos1'])
    act = (await ar.api.query(msg.GetActionCommand()))
    if act['action'] != 'action-set-operation-mode':
        await ar.api.send(msg.ActionSetOperationMode(mode="low-level-api"))
    print("LLAPI ON")

    # Since LLAPI mode on, we can keep sending to execute custom commands.
    # Robot will fall in this test case.
    # NOTE: still mystery about when Damping mode will turn on
    for i in range(15):
        if i%3==1:
            await ar.sim_forward(dt=0.5, actions=MOTOR_POSITION_SET['pos1'])
        elif i%3==2:
            await ar.sim_forward(dt=0.5, actions=MOTOR_POSITION_SET['pos2'])
        else:
            await ar.sim_forward(dt=0.5, actions=MOTOR_POSITION_SET['pos3'])
        print("time={: 4.2f}, z-vel={: 3.2f}, llapi-connect={}".format(
            ar.t, ar.get_base_linear_velocity()[2], ar.is_llapi_connected()))
    await ar.close_all()
    print("Closed all")
