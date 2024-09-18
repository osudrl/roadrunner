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
