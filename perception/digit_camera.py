from digit_camera_example import *
from multiprocessing import Manager, Process
import socket
import cv2
import numpy as np

class DigitCamera:
    def __init__(self, port, ip= "10.10.1.1"):
        # Establish socket connection.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.setblocking(0)

        self.payload = Manager().dict()
        remote_func = Process(target=self._fetch_data, args=(self.sock, self.payload))
        remote_func.start()

    def _fetch_data(self, sock, payload):
        """Connect to perception stream, process incoming byte stream,
        and display images.
        """
        # Variables for parsing the data.
        data = bytearray()
        state = ReadState.FIND_JSON
        read_chunk_size = 8192
        start_time = time.time()

        while True:
            # Check if we can read anything from the socket.
            readable, writable, _ = select.select([sock], [sock], [], 1)
            # print(len(readable))

            if len(readable):
                # print("get readable")
                # Append received data chunk to frame data.
                data = data + readable[0].recv(read_chunk_size)
                # print("get data", len(data))

                # Search for the JSON header and advance to next state when found.
                if state == ReadState.FIND_JSON:# and len(data) > 407000:
                    json_msg, data = find_json(data)
                    if json_msg:
                        # print("time to find json: ", time.time() - start_time)
                        start_time = time.time()
                        frame_info = FrameInfo(json.loads(json_msg))
                        state = ReadState.READ_DATA

                # Read the exact amount of data specified by frame size and advance.
                # to next state when complete
                if state == ReadState.READ_DATA:
                    if frame_info.size <= len(data):
                        state = ReadState.PROCESS
                        # print("time to read data: ", time.time() - start_time)
                        start_time = time.time()
                    elif frame_info.size - len(data) < read_chunk_size:
                        read_chunk_size = frame_info.size - len(data)

                # Process the read frame data using the JSON header frame info.
                if state == ReadState.PROCESS:
                    # Update parsing variables
                    state = ReadState.FIND_JSON
                    read_chunk_size = 8192
                    # Interpret buffer as 1-D array with specified frame datatype.
                    buffer = data[:frame_info.size]
                    a = np.frombuffer(buffer, dtype=frame_info.bit_depth)
                    if frame_info.format in FrameInfo.IMG_TYPES:
                        # Reshape array with specified frame dimensions.
                        image = a.reshape((frame_info.height, frame_info.width,
                                        frame_info.channels))
                        # print("time to get image: ", time.time() - start_time)
                        start_time = time.time()
                        payload['depth_image'] = image
                    else:
                        raise TypeError(f"Got desired frame format: {frame_info.format}")

    def get_depth_frame(self):
        return self.payload.get('depth_image', np.zeros((128, 128), dtype=np.float32))

    def __del__(self):
        self.sock.close()

if __name__ == '__main__':
    try:
        # name = asyncio.get_event_loop().run_until_complete(select_stream())
        # name = "forward-pelvis-realsense-d430/left-infrared/image-rect"
        # name = "forward-pelvis-realsense-d430/depth/image-rect"
        # name = "backward-pelvis-realsense-d430/right-infrared/image-rect"
        # name = "downward-pelvis-realsense-d430/depth/image-rect"
        name = "forward-chest-realsense-d435/depth/image-rect"
        port = asyncio.get_event_loop().run_until_complete(start_stream(name))
        print(f"Connected to {name} stream at port {port}")
        cam = DigitCamera(port=port)
        while True:
            image = cam.get_depth_frame()
            # Autoscale depth images to increase contrast.
            image = np.core.umath.clip(image, 0, 2000)
            image = (image / image.max() * 255).astype(np.uint8)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Display the image frame and exit if 'q' key is pressed.
            image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
            cv2.imshow(name, image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # cam.get_depth_frame()
        # get_depth()
    except websockets.exceptions.ConnectionClosedError:
        sys.exit("Connection to robot lost")
    except (ConnectionRefusedError, asyncio.exceptions.TimeoutError):
        sys.exit("Could not connect to robot")
    except KeyboardInterrupt:
        sys.exit("Caught user exit")