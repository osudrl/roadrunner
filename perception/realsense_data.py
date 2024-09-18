import pyrealsense2 as rs
import numpy as np
import cv2
import time
from util.camera_util import crop_from_center

HEIGHT = 256
WIDTH = 144
FPS = 90

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, HEIGHT, WIDTH, rs.format.z16, FPS)
        # self.config.enable_stream(rs.stream.color, HEIGHT, WIDTH, rs.format.bgr8, FPS)

    def get_rs_data(self):
        self.pipeline.start(self.config)

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                # color_frame = frames.get_color_frame()
                if not depth_frame:# or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data(),dtype=np.float32)
                # color_image = np.asanyarray(color_frame.get_data())

                # Crop the depth image and color image
                depth_crop_image = crop_from_center(depth_image, 128, 128)
                # color_crop_image = crop_from_center(color_image, 128, 128)
                # depth_crop_image = np.random.rand(32,1)
                # color_crop_image = np.random.rand(32,1)

                # cv2.imshow('Cropped', depth_crop_image)
                # cv2.imshow('Original', depth_image)

                # if cv2.waitKey(1) == ord("q"):
                #     break

                # Yield the cropped images
                yield (depth_crop_image)
        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    camera = RealSenseCamera()
    for depth_crop_image, color_crop_image in camera.get_rs_data():
        print(depth_crop_image.shape)
        print(color_crop_image.shape)
