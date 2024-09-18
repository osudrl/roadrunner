import numpy as np
import cv2

# Intel realsense d455
FOCAL_LENGTH = 1.93e-3
BASELINE = 95e-3
INVALID_DEPTH = 0.0

def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation

    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose

def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """
    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv

def bilinear_interpolate(im, x, y):
    """
    Bilinear sampling for pixel coordinates x and y from source image im.
    Taken from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform, stride, camera_height, camera_width):
    """
    Transform pixel coordinates to world coordinates using the depth map and camera-to-world transformation matrix.

    By default, MuJoCo will return a depth map that is normalized in [0, 1]. This
    helper function converts the map so that the entries correspond to actual distances.
    Taken from: https://github.com/ARISE-Initiative/robosuite/blob/6fbfae0227fbafe90911a9c57a8f9ed4623df6f0/robosuite/utils/camera_utils.py

    Args:
        pixels (np.array): Pixel coordinates to be transformed
        depth_map (np.array): Depth map with values normalized in [0, 1] (default depth map returned by MuJoCo)
        camera_to_world_transform (np.array): Transformation matrix from camera coordinates to world coordinates
        stride (int): Stride for resizing the output point cloud
        camera_height (int): Height of the camera frame
        camera_width (int): Width of the camera frame

    Returns:
        points (np.array): 3D world coordinates corresponding to the input pixel coordinates
    """
    # Convert pixels to float
    pixels = pixels.astype(float)
    # Compute the new dimensions for the resized image
    new_height = (camera_height + stride - 1) // stride
    new_width = (camera_width + stride - 1) // stride
    # Get the height and width of the depth map
    im_h, im_w = depth_map.shape[:2]
    # Remove any extra dimensions of the depth_map
    depth_map_reshaped = depth_map.squeeze()
    # Obtain depth (z) values at given pixel locations using bilinear interpolation
    z = bilinear_interpolate(im=depth_map_reshaped, x=pixels[..., 1], y=pixels[..., 0])
    # Create homogenous camera coordinates (4D) by combining x, y, and z values
    # Note: Swap the first 2 dimensions of pixels to convert from pixel indices to camera coordinates
    cam_pts = [pixels[..., 1:2] * z[..., None], pixels[..., 0:1] * z[..., None], z[..., None], np.ones_like(z)[..., None]]
    # Concatenate the 4D camera coordinates
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [H, W, 4]
    # Reshape the camera_to_world_transform matrix to a 4x4 matrix
    mat_reshape = [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [4, 4]
    # Perform matrix multiplication to transform the camera coordinates to world coordinates
    points = np.matmul(cam_trans, cam_pts.reshape(-1, 4).T).T  # shape [H*W, 4]
    # Return the 3D world coordinates (x, y, z) in the desired shape
    points = points[:, :3].reshape(new_height, new_width, 3)
    return points

def transform_from_pixels_to_camera_frame(pixels, depth_map, camera_matrix, stride, camera_height, camera_width):
    """
    Transform pixel coordinates to camera frame coordinates using the depth map and camera intrinsics.

    Args:
        pixels (np.array): Pixel coordinates to be transformed
        depth_map (np.array): Depth map with values normalized in [0, 1]
        stride (int): Stride for resizing the output point cloud
        camera_height (int): Height of the camera frame
        camera_width (int): Width of the camera frame

    Returns:
        points (np.array): 3D camera frame coordinates corresponding to the input pixel coordinates
    """
    # Convert pixels to float
    pixels = pixels.astype(float)
    # Compute the new dimensions for the resized image
    new_height = (camera_height + stride - 1) // stride
    new_width = (camera_width + stride - 1) // stride
    # Get the height and width of the depth map
    im_h, im_w = depth_map.shape[:2]
    # Remove any extra dimensions of the depth_map
    depth_map_reshaped = depth_map.squeeze()
    # Obtain depth (z) values at given pixel locations using bilinear interpolation
    z = bilinear_interpolate(im=depth_map_reshaped, x=pixels[..., 1], y=pixels[..., 0])
    # Convert pixels to homogeneous coordinates	
    hom_pixels = np.concatenate([pixels, np.ones((pixels.shape[0], pixels.shape[1], 1))], axis=-1)
    # Convert homogeneous pixels to normalized camera coordinates using the inverse camera matrix
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    normalized_pixels = np.matmul(hom_pixels, inv_camera_matrix.T)[..., :2]
    # Create homogenous camera coordinates (3D) by combining x, y, and z values
    cam_pts = [normalized_pixels[..., 0:1] * z[..., None], normalized_pixels[..., 1:2] * z[..., None], z[..., None]]
    # Concatenate the 3D camera coordinates
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [H, W, 3]
    # Reshape the points to the desired shape
    points = cam_pts.reshape(new_height, new_width, 3)
    return points

def add_gaussian_noise(depth_map, mean=0.0, std_dev=0.01):
    """
    Add Gaussian noise to a depth map.

    Args:
        depth_map (np.array): The original depth map.
        mean (float): The mean of the Gaussian noise distribution. Default is 0.0.
        std_dev (float): The standard deviation of the Gaussian noise distribution. Default is 0.01.

    Returns:
        noisy_depth_map (np.array): The depth map with added Gaussian noise.
    """
    noise = np.random.normal(mean, std_dev, depth_map.shape)
    noisy_depth_map = depth_map + noise
    return noisy_depth_map

def add_gaussian_shifts(depth, std=1/2.0):
    """Add gaussian shifts to the depth map. Adopted from
    https://github.com/ankurhanda/simkinect/
    """

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp

def crop_from_center(image, w: int, h: int):
    assert image.shape[0] >= h, "image height must be greater or equal to crop height"
    assert image.shape[1] >= w, "image width must be greater or equal to crop width"
    assert len(image.shape) == 2, "image dimension must be 2"
    center = image.shape
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2
    return image[int(y):int(y+h), int(x):int(x+w)]