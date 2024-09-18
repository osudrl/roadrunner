import asyncio
import agility
import agility.messages as msg
import numpy as np
import time

from sim.digit_sim import ArDigitSim
from testing.common import (
    SIM_PATH,
    ROBOT_CONFIG,
    MOTOR_POSITION_SET
)

kp = [100, 80, 180, 180, 40, 40, 80, 80, 50, 80,
      100, 80, 180, 180, 40, 40, 80, 80, 50, 80]
kd = [10, 9, 14, 14, 6, 6, 9, 9, 7, 9,
      10, 9, 14, 14, 6, 6, 9, 9, 7, 9]

from sim.digit_sim.digit_ar_sim.llapi.llapictypes import *
from util.topic import TopicDigit

ROBOT_ADDRESS = '127.0.0.1'
ROBOT_PORT = 8080
CONNECTION_TIMEOUT = 1

async def test_ar_async():
    # sim = agility.Simulator(SIM_PATH, *ROBOT_CONFIG, auto_port=True)
    api = agility.JsonApi(address=ROBOT_ADDRESS,
                          port=ROBOT_PORT,
                          connect_timeout=CONNECTION_TIMEOUT)
    try:
        await api.connect()
    except:
        raise Exception(f"Cannot connect api at {ROBOT_ADDRESS} with port {ROBOT_PORT}"
                        f"Check if simulator is running and port is correct!")

    # Initialize LLAPI
    llapi_init(ROBOT_ADDRESS)
    # Define obs, cmd, and limit after connected
    topic = TopicDigit(freq=3000)
    topic.subscribe()

    start = time.monotonic()
    while not topic.recv():
        if time.monotonic() - start > 1:
            raise ConnectionError(f"Cannot connect to LLAPI! Check port and IP address setup! ")
    # await api.send(msg.ActionSetOperationMode(mode="low-level-api"))
    print("LLAPI connected!")

    topic.cmd['p'] = MOTOR_POSITION_SET['pos1']
    topic.cmd['kp'] = kp
    topic.cmd['kd'] = kd
    start_time = time.monotonic()
    indices = np.arange(3)
    while True:
        obs, code = topic.recv()
        if code == 1:
            print(f"x={obs.base.translation[0]:1.2f}")
        # pos = MOTOR_POSITION_SET[f"pos{np.random.choice(indices)+1}"]
        # topic.set_pd(setpoint=pos, kp=kp, kd=kd)
        # time.sleep(1)

    # obs = llapi_observation_t()
    # command = llapi_command_t()
    # command.apply_command = False
    while not llapi_get_observation(obs):
        llapi_send_command(command)
        # return_code = llapi_get_observation(obs)
        # if return_code == 0:
        #     logging.info("Receiving None observation")
        if time.monotonic() - start > 1:
            raise ConnectionError(f"Cannot connect to LLAPI! Check port and IP address setup! ")
    # Constants after connection established
    _actuator_limit = llapi_get_limits()[0]
    # await api.send(msg.ActionSetOperationMode(mode="low-level-api"))
    print("LLAPI connected!")
    # exit()
    while True:
        return_code = llapi_get_observation(obs)
        if return_code == 1:
            print(f"valid obs={return_code}, x={obs.base.translation[0]:1.2f}")
        for i in range(NUM_MOTORS):
            command.motors[i].torque = 10.0
        command.apply_command = True
        command.fallback_opmode = Locomotion
        llapi_send_command(command)
        time.sleep(1/2000)


import signal

# async def test_ar_async_2():
#     # # Signal handler function
#     # def sigint_handler(signum, frame):
#     #     # Set a flag or perform necessary cleanup actions
#     #     global running
#     #     running = False

#     # # Set up the signal handler for SIGINT
#     # signal.signal(signal.SIGINT, sigint_handler)

#     api = agility.JsonApi(address=ROBOT_ADDRESS,
#                           port=ROBOT_PORT,
#                           connect_timeout=CONNECTION_TIMEOUT)
#     try:
#         await api.connect()
#     except:
#         raise Exception(f"Cannot connect api at {ROBOT_ADDRESS} with port {ROBOT_PORT}"
#                         f"Check if simulator is running and port is correct!")

#     # Initialize LLAPI
#     # llapi_init(ROBOT_ADDRESS)
#     # pd_command_shared = llapi_command_pd_t()
#     # Update the shared values from Python side
#     # for i in range(NUM_MOTORS):
#     #     pd_command_shared.kp[i] = 200.0
#     #     pd_command_shared.kd[i] = 20.0
#     #     pd_command_shared.position[i] = MOTOR_POSITION_SET['pos1'][i]
#     # Pass the address of the shared memory to the C function
#     # llapi_initialize_shared_pd(ctypes.byref(shared_pd_address))
#     shared_pd_address = llapi_initialize_shared_pd_2()
#     pd_command_shared = ctypes.cast(shared_pd_address, ctypes.POINTER(llapi_command_pd_t()))

#     print("Initialize shared memory successfully!")

#     topic = TopicDigit(freq=3000)
#     topic.subscribe()
#     # llapi_run()
#     print("running llapi")
    
#     # Create the shared memory segment
#     indices = np.arange(3)
#     while True:
#         # obs, code = topic.recv()
#         # if code == 1:
#             # print(f"x={obs.base.translation[0]:1.2f}")
#         idx = np.random.choice(indices)+1
#         pos = MOTOR_POSITION_SET[f"pos{idx}"]
#         print(f"indice={idx}")
#         print(pd_command_shared)
#         # for i in range(NUM_MOTORS):
#         #     pd_command_shared.position[i] = pos[i]
#         # llapi_update_pd(pd_command_shared)
#         time.sleep(1)
        
#     if running == False:
#         llapi_free()
#         api.close()
#         exit()

async def test_ar_async_2():
    import socket
    from util.topic import TopicNew


    # Create socket connection
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    local_address = ('127.0.0.1', 30002) # python
    remote_address = ('127.0.0.1', 30001) # c

    topic = TopicNew(freq=3000)
    topic.subscribe(local_address)
    
    # obs = llapi_observation_t()
    cmd = llapi_command_pd_t()
    send_buffer = ctypes.create_string_buffer(ctypes.sizeof(llapi_command_pd_t))

    for i in range(NUM_MOTORS):
        cmd.kp[i] = 200.0
        cmd.kd[i] = 20.0
        cmd.position[i] = MOTOR_POSITION_SET['pos1'][i]
    indices = 1
    
    obs = topic.recv()
    while not obs:
        topic.publish(llapi_command_pd_t(), remote_address)
        obs = topic.recv()
        print("Receiving None observation")
    
    while True:
        obs = topic.recv()
        # Send data to C
        # message = "Hello from Python!"
        position = MOTOR_POSITION_SET[f"pos{indices}"]
        for i in range(NUM_MOTORS):
            cmd.position[i] = position[i]
        # pack_command_pd(send_buffer, 640, ctypes.byref(cmd))
        # client_socket.sendto(send_buffer, (HOST, PORT))
        # cmd.position = position
        topic.publish(cmd, remote_address)
        # print("Sent to C: ", message)

        # Receive response from C
        # response, addr = client_socket.recvfrom(872)
        # unpack_observation(response, 872, obs)
        # print("Received from C: ", obs.base.translation[0])
        indices += 1
        if indices == 3:
            indices = 1
        time.sleep(1/50)

    # Close socket connection
    client_socket.close()


if __name__ == "__main__":
    asyncio.run(test_ar_async())
