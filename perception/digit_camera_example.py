#!/usr/bin/env python3
# Copyright (c) Agility Robotics

import asyncio
import json
import select
import socket
import sys
import time
from enum import Enum
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import websockets


class StreamType(Enum):
    RGB8 = "RGB8"
    Gray8 = "Gray8"
    Depth16 = "Depth16"
    XYZ = "XYZ"
    XYZI = "XYZI",
    XYZIRT = "XYZIRT"


class ReadState(Enum):
    FIND_JSON = 'FIND_JSON'
    READ_DATA = 'READ_DATA'
    PROCESS = 'PROCESS'

# Used to control the flow control method for the data stream. Currently
# supports: request, framerate, none
flow_control = 'none'

ip = "10.10.1.1"

def check_response_msg(msg, expected: str) -> None:
    if msg[0] != expected:
        raise ValueError(f"Response must be '{expected}'. Got: {msg[0]}")


async def select_stream() -> str:
    """List available perception streams and prompt the user to select one."""
    # Request list of available perception stream and wait for response.
    async with websockets.connect('ws://' + ip +':8080',
                                  subprotocols=['json-v1-agility']) as ws:
        await ws.send(json.dumps(['get-perception-streams', {}]))
        msg = json.loads(await ws.recv())

    # Verify received message is the correct type.
    check_response_msg(msg, 'perception-streams')
    streams = msg[1]['available-streams']

    # Verify streams are available.
    if len(streams) == 0:
        raise RuntimeError("No streams available yet")

    # Have the user select the stream they want to start.
    print("Select a stream to start: ")
    streams.sort()
    for i, stream in enumerate(streams):
        print(f"  [{i}] {stream}")
    while True:
        id = input("Option: ")
        if id and int(id) in range(len(streams)):
            return streams[int(id)]
        print("Enter a valid option")

async def start_stream(name: str) -> int:
    """Request perception stream start by name."""
    # Request perception stream and wait for response.
    async with websockets.connect('ws://' + ip + ':8080',
                                  subprotocols=['json-v1-agility']) as ws:
        await ws.send(json.dumps(['perception-stream-start', {
            'stream': name,
            'flow-control': flow_control}]))
        msg = json.loads(await ws.recv())

    # Verify received message is the correct type.
    check_response_msg(msg, 'perception-stream-response')

    port = msg[1]['port']
    print(f"Successfully started {name} stream at port {port}")
    return port


def find_json(data: bytearray) -> Union[
    Tuple[str, bytearray], Tuple[None, bytearray]
]:
    """Attempts to find the JSON header message in the incoming byte stream."""
    text = data.decode('latin-1')
    start = text.find('["perception-stream-frame"')
    count = 0
    for i, ch in enumerate(text[start:]):
        if ch == '[':
            count += 1
        elif ch == ']':
            count -= 1
            if count == 0:
                end = start + i + 1
                return text[start:end], data[end:]
    return None, data


class FrameInfo:

    """A helper class for parsing the frame info."""

    IMG_TYPES = [StreamType.RGB8, StreamType.Gray8, StreamType.Depth16]
    PT_CLOUD_TYPES = [StreamType.XYZ, StreamType.XYZI, StreamType.XYZIRT]

    def __init__(self, json_msg):
        # Add attributes listed in the json message.
        for k in json_msg[1]:
            setattr(self, k.replace('-', '_'),  json_msg[1][k])

        self.format = StreamType(self.format)

        if self.format == StreamType.RGB8:
            self.channels = 3
            self.bit_depth = np.uint8
        elif self.format == StreamType.Gray8:
            self.channels = 1
            self.bit_depth = np.uint8
        elif self.format == StreamType.Depth16:
            self.channels = 1
            self.bit_depth = np.uint16
        elif self.format == StreamType.XYZ:
            self.channels = 3
            self.bit_depth = np.float32
        elif self.format == StreamType.XYZI:
            self.channels = 4
            self.bit_depth = np.float32
        elif self.format == StreamType.XYZIRT:
            self.channels = 6
            self.bit_depth = np.float32
        else:
            raise ValueError("must be RGB8, Gray8, Depth16, XYZ, XYZI, or XYZIRT, not "
                             f"{self.format}")

        bytes_per_channel = int(np.dtype(self.bit_depth).itemsize)
        if self.format in FrameInfo.IMG_TYPES:
            self.size = self.height * self.width * self.channels * bytes_per_channel
        elif self.format in FrameInfo.PT_CLOUD_TYPES:
            self.size = self.size * self.channels * bytes_per_channel


def on_press(event, close_all):
    print(f'Pressed {event.key}. Exiting...')
    if event.key == 'q':
        close_all[0] = True


def process_stream(name: str, port: int) -> None:
    """Connect to perception stream, process incoming byte stream,
    and display images.
    """
    # Establish socket connection.
    print(f"Establishing connection to {name} stream at port {port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    sock.setblocking(0)

    # Variables for parsing the data.
    data = bytearray()
    state = ReadState.FIND_JSON
    read_chunk_size = 8192
    frame_chunk_size = 10
    frames_remaining = 0
    frames_received = 0
    start_time = time.time()

    while True:
        # Check if we can read anything from the socket.
        readable, writable, _ = select.select([sock], [sock], [], 1)

        if len(writable):
            # Request another set of frames if the current set has been read.
            if flow_control == 'framerate':
                if time.time() - start_time > 1.0:
                    start_time = time.time()
                    print(f"Received {frames_received} frames")
                    writable[0].send(
                        (frames_received).to_bytes(1, byteorder='big'))
                    frames_received = 0

            elif flow_control == 'request':
                if frames_remaining == 0:
                    print(f"Requesting {frame_chunk_size} frames")
                    frames_remaining = frame_chunk_size
                    writable[0].send(
                        (frame_chunk_size).to_bytes(1, byteorder='big'))
                    frames_received = 0

        if len(readable):
            # Append received data chunk to frame data.
            data = data + readable[0].recv(read_chunk_size)

            # Search for the JSON header and advance to next state when found.
            if state == ReadState.FIND_JSON:
                json_msg, data = find_json(data)
                if json_msg:
                    frame_info = FrameInfo(json.loads(json_msg))
                    state = ReadState.READ_DATA

            # Read the exact amount of data specified by frame size and advance.
            # to next state when complete
            elif state == ReadState.READ_DATA:
                if frame_info.size <= len(data):
                    state = ReadState.PROCESS
                elif frame_info.size - len(data) < read_chunk_size:
                    read_chunk_size = frame_info.size - len(data)

            # Process the read frame data using the JSON header frame info.
            elif state == ReadState.PROCESS:
                # Update parsing variables
                state = ReadState.FIND_JSON
                read_chunk_size = 8192
                frames_remaining -= 1
                frames_received += 1

                # Interpret buffer as 1-D array with specified frame datatype.
                buffer = data[:frame_info.size]
                a = np.frombuffer(buffer, dtype=frame_info.bit_depth)

                if frame_info.format in FrameInfo.IMG_TYPES:
                    # Reshape array with specified frame dimensions.
                    image = a.reshape((frame_info.height, frame_info.width,
                                       frame_info.channels))

                    # Swap color channels as OpenCV uses BGR not RGB.
                    if frame_info.format == StreamType.RGB8:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Autoscale depth images to increase contrast.
                    if frame_info.format == StreamType.Depth16:
                        image = image / image.max()

                    # Display the image frame and exit if 'q' key is pressed.
                    cv2.imshow(name, image)

                    txt = (f't_mono={frame_info.timestamp_mono}, '
                           f't_utc={frame_info.timestamp_utc}')
                    cv2.setWindowTitle(title=txt, winname=name)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                elif frame_info.format in FrameInfo.PT_CLOUD_TYPES:
                    # Reshape array with specified frame dimensions.
                    cloud = a.reshape((
                        int(a.size / frame_info.channels),
                        frame_info.channels
                    ))

                    # Regardless of format we want the first 3 columns.
                    cloud = cloud[:, 0:3]

                    # Append a column of ones for the homogeneous transformation.
                    cloud = np.append(cloud, np.ones([len(cloud), 1]), 1)

                    # Apply transform to base frame
                    cloud[:] = cloud.dot(frame_info.T_base_to_stream)

                    # Optional: Apply transform to world frame.
                    cloud[:] = cloud.dot(frame_info.T_world_to_base)

                    # Initialize the figure if it hasn't been launched.
                    if 'fig' not in locals():

                        fig = plt.figure()
                        plt.ion()
                        ax = fig.add_subplot(projection='3d')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_zlabel('z')
                        sc = ax.scatter([], [], [], s=1)
                        ub = 4
                        lb = -4
                        ax.set_xlim(lb, ub)
                        ax.set_ylim(lb, ub)
                        ax.set_zlim(lb, ub)
                        plt.show()

                        # Add event handler that quits the program on 'q' press
                        # and sets the mutable close_all flag. Cannot sys.exit()
                        # inside bc figure event loop runs in a child thread.
                        close_all = [False]
                        fig.canvas.mpl_connect(
                            'key_press_event', lambda e: on_press(e, close_all))
                        print('Press q in figure to exit')

                    # Depending on your machine, the number of points returned
                    # may be too many to practically render in realtime.
                    # Thus, skip most of them.
                    skip = 10
                    cloud = cloud[::skip, :]

                    ax.title._text = (f't_mono={frame_info.timestamp_mono}\n'
                                      f't_utc={frame_info.timestamp_utc}')

                    # Update plot data without creating new figure.
                    sc._offsets3d = (cloud[:, 0], cloud[:, 1], cloud[:, 2])
                    fig.canvas.draw_idle()
                    plt.pause(0.001)

                    if close_all[0]:
                        break


if __name__ == '__main__':
    try:
        name = asyncio.get_event_loop().run_until_complete(select_stream())
        port = asyncio.get_event_loop().run_until_complete(start_stream(name))
        process_stream(name, port)
    except websockets.exceptions.ConnectionClosedError:
        sys.exit("Connection to robot lost")
    except (ConnectionRefusedError, asyncio.exceptions.TimeoutError):
        sys.exit("Could not connect to robot")
    except KeyboardInterrupt:
        sys.exit("Caught user exit")


"""
  [0] backward-pelvis-realsense-d430/depth/image-rect
  [1] backward-pelvis-realsense-d430/depth/points
  [2] backward-pelvis-realsense-d430/left-infrared/image-rect
  [3] backward-pelvis-realsense-d430/right-infrared/image-rect
  [4] downward-pelvis-realsense-d430/depth/image-rect
  [5] downward-pelvis-realsense-d430/depth/points
  [6] downward-pelvis-realsense-d430/left-infrared/image-rect
  [7] downward-pelvis-realsense-d430/right-infrared/image-rect
  [8] forward-chest-realsense-d435/color/image-rect
  [9] forward-chest-realsense-d435/depth/image-rect
  [10] forward-chest-realsense-d435/depth/points
  [11] forward-chest-realsense-d435/left-infrared/image-rect
  [12] forward-chest-realsense-d435/right-infrared/image-rect
  [13] forward-pelvis-realsense-d430/depth/image-rect
  [14] forward-pelvis-realsense-d430/depth/points
  [15] forward-pelvis-realsense-d430/left-infrared/image-rect
  [16] forward-pelvis-realsense-d430/right-infrared/image-rect
  [17] forward-tis-dfm27up/color/image-raw
  [18] upper-velodyne-vlp16/depth/points
"""