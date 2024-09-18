import torch
import numpy as np
from env.cassie.cassieenv import CassieEnv
from env.digit.digitenv import DigitEnv
from util.mirror import mirror_tensor

"""
This file tests if the mirror inds (from hand written one) are correct when creating a new set
of mirror inds for Digit/Cassie on robot proprioceptive state.
The test is meant for base env classes, with get_robot_state() and robot_state_mirror_indices.
The test mirrors the state once, then check if the mirrored states are correct.

If anyone wants change the original definition of get_robot_state() and robot_state_mirror_indices,
make sure you throughly test with this file.

Also, Cassie and Digit's joint frames are defined in different way when considering mirror. See the
pictures in README.
"""

def compare_inds(inds1, inds2, name):
    assert np.array_equiv(inds1, inds2), \
        f"{name} mirror inds have duplicate values \ninds1 = {inds1}\ninds2={inds2}."

def test_mirror():
    cassie = CassieEnv(simulator_type='mujoco',
                    terrain=False,
                    policy_rate=50,
                    dynamics_randomization=False,
                    state_est=False,
                    state_noise=0,
                    velocity_noise=0)

    digit = DigitEnv(simulator_type='mujoco',
                    terrain=False,
                    policy_rate=50,
                    dynamics_randomization=False,
                    state_noise=0,
                    velocity_noise=0,
                    state_est=False)

    """
    Check if any mirror ind array has duplicate values by mistake
    """
    # Cassie state
    inds_from_mirror = np.sort(np.floor(np.abs(cassie.robot_state_mirror_indices)))
    inds_from_getstate = np.arange(len(cassie.get_robot_state()))
    compare_inds(inds_from_mirror, inds_from_getstate, "Cassie state")

    # Cassie action
    inds_from_mirror = np.sort(np.floor(np.abs(cassie.motor_mirror_indices)))
    inds_from_getstate = np.arange(cassie.sim.num_actuators)
    compare_inds(inds_from_mirror, inds_from_getstate, "Cassie action")

    # Digit state
    inds_from_mirror = np.sort(np.floor(np.abs(digit.robot_state_mirror_indices)))
    inds_from_getstate = np.arange(len(digit.get_robot_state()))
    compare_inds(inds_from_mirror, inds_from_getstate, "Digit state")

    # Digit action
    inds_from_mirror = np.sort(np.floor(np.abs(digit.motor_mirror_indices)))
    inds_from_getstate = np.arange(digit.sim.num_actuators)
    compare_inds(inds_from_mirror, inds_from_getstate, "Digit action")

    """
    Check mirror inds are applied properly
    """
    cassie.reset_simulation()
    digit.reset_simulation()

    cassie_state = cassie.get_robot_state()
    digit_state = digit.get_robot_state()

    cassie_state_mirror_indices = [0.01, -1, 2, -3,      # base orientation
                                    -4, 5, -6,             # base rotational vel
                                    -12, -13, 14, 15, 16,  # right motor pos
                                    -7,  -8,  9,  10,  11, # left motor pos
                                    -22, -23, 24, 25, 26,  # right motor vel
                                    -17, -18, 19, 20, 21,  # left motor vel
                                    29, 30, 27, 28,        # joint pos
                                    33, 34, 31, 32, ]      # joint vel
    cassie_mirrored_left_motor_pos = mirror_tensor(torch.tensor(cassie_state), [-7,  -8,  9,  10,  11])
    cassie_right_motor_pos = np.take(cassie_state, [12, 13, 14, 15, 16])
    assert np.linalg.norm(cassie_mirrored_left_motor_pos - cassie_right_motor_pos) < 1e-6, \
        "Mirror incorrect"

    # Because of Digit model definition, we need to make every entry of this list to negative
    # to get matched, even if they are in saggital plane.
    digit_state_mirror_indices = [0.01, -1, 2, -3,            # base orientation
                                -4, 5, -6,                    # base rotational vel
                                -17, -18, -19, -20, -21, -22, # right leg motor pos
                                -23, -24, -25, -26,           # right arm motor pos
                                -7,  -8,  -9,  -10, -11, -12, # left leg motor pos
                                -13, -14, -15, -16,           # left arm motor pos
                                -37, -38, -39, -40, -41, -42, # right leg motor vel
                                -43, -44, -45, -46,           # right arm motor vel
                                -27, -28, -29, -30, -31, -32, # left leg motor vel
                                -33, -34, -35, -36,           # left arm motor vel
                                -52, -53, -54, -55, -56,      # right joint pos
                                -47, -48, -49, -50, -51,      # left joint pos
                                -62, -63, -64, -65, -66,      # right joint vel
                                -57, -58, -59, -60, -61,      # left joint vel
                                ]

    # If anyone tests with a new set of get_robot_state and mirror inds, make sure to pick out the
    # sub-list and all the possible combinations of mirroring and test below.
    mirror_vec = [-17, -18, -19, -20, -21, -22, -23, -24, -25, -26]
    mirror_to_vec = [7,  8,  9,  10, 11, 12, 13, 14, 15, 16]
    mirrored_s = mirror_tensor(torch.tensor(digit_state), mirror_vec)

    # Get actual positions
    s = np.take(digit_state, mirror_to_vec)
    print("mirroed s", mirrored_s)
    print("original s", s)
    print("mirroring difference norm", np.linalg.norm(mirrored_s - s))
    assert np.linalg.norm(mirrored_s - s) < 1e-2, \
        "Mirror incorrect"
