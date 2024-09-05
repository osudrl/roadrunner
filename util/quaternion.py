import math
import numpy as np

def quaternion_distance(q1: np.ndarray, q2: np.ndarray):
    """
    Returns a distance measure between two quaternions. Returns 0 whenever the quaternions represent
    the same orientation and gives 1 when the two orientations are 180 degrees apart. Note that this
    is NOT a quaternion difference; the difference q_diff implies q1 * q_diff = q2 and is NOT
    commutative. This uses the fact that if q1 and q2 are equal, the q1 * q2 is equal to 1 (or -1 if
    they only differ in sign) and IS commutative.

    Arguments:
    q1 (numpy ndarray): first quaternion to compare
    q2 (numpy ndarray): second quaternion to compare to
    """
    assert q1.shape == (4,), \
        f"quaternion_similarity received quaternion 1 of shape {q1.shape}, but should be of shape (4,)"
    assert q2.shape == (4,), \
        f"quaternion_similarity received quaternion 2 of shape {q2.shape}, but should be of shape (4,)"
    return 1 - np.inner(q1, q2) ** 2

def quaternion2euler(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    ysqr = y * y

    ti = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(ti, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    result = np.zeros(3)
    result[0] = X * np.pi / 180
    result[1] = Y * np.pi / 180
    result[2] = Z * np.pi / 180

    return result

def euler2quat(z=0, y=0, x=0):

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result = np.array([
        cx*cy*cz - sx*sy*sz,
        cx*sy*sz + cy*cz*sx,
        cx*cz*sy - sx*cy*sz,
        cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
        result = -result
    return result

def scipy2mj(q):
    x, y, z, w = q
    # quat to rotation is 2 to 1, so always pick the positive w
    if w < 0:
        return np.array([-w, -x, -y, -z])
    return np.array([w, x, y, z])

def mj2scipy(q):
    w, x, y, z = q
    # quat to rotation is 2 to 1, so always pick the positive w
    if w < 0:
        return np.array([-x, -y, -z, -w])
    return np.array([x, y, z, w])

def euler2so3(z=0, y=0, x=0):

    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)],
    ])

    R_y = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)],
    ])

    R_z = np.array([
        [math.cos(z), -math.sin(z), 0,],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1],
    ])

    return R_z @ R_y @ R_x

# def add_euler(a, b):
#     return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]