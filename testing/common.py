"""
This file includes some constants that are used for mujoco or ar-control.
"""

# Testing utils for Digit ar-control
SIM_PATH = '~/ar-software-2023.01.13a/ar-software/ar-control'

ROBOT_CONFIG = ["""
# Add robot to world
[[model-list]]
model = "robot"
pose = {xyz = [0, 0, 1.0]}

[simulator]
free-run=true

[planning]
initial-operation-mode = "locomotion"

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

# Motor set for testing for Digit ar-control
MOTOR_POSITION_SET={}
MOTOR_POSITION_SET['pos1'] = [
-0.0462933, -0.0265814, 0.19299, -0.3, -0.0235182, -0.0571617,
0.0462933, 0.0265814, -0.19299, 0.3, -0.0235182, 0.0571617,
-0.3, 0.943845, 0.0, 0.3633,
0.3, -0.943845, 0.0, -0.3633,
]

MOTOR_POSITION_SET['pos2'] = [
0.332, -0.0265814, 0.19299, 0.218647, 0.0235182, -0.0571617,
-0.332, 0.0265814, -0.19299, -0.218647, -0.0235182, 0.0571617,
-0.106437, 0.89488, -0.00867663, 0.344684,
0.106339, -0.894918, 0.00889888, -0.344627
]

MOTOR_POSITION_SET['pos3'] = [
-0.0462933, -0.5265814, 0.19299, -0.3, -0.0235182, -0.0571617,
0.0462933, 0.5265814, -0.19299, 0.3, -0.0235182, 0.0571617,
-0.3, 0.0943845, 0.0,-0.3633,
0.3, -0.0943845, 0.0, 0.3633,
]

MOTOR_POSITION_SET['stand'] = \
[ 3.74746713e-01, -1.76407791e-04,  3.11254289e-01,  3.44019160e-01,
 -1.23124968e-01,  1.22818172e-01, -3.75034334e-01,  2.83786446e-04,
 -3.11326195e-01, -3.44019160e-01,  1.23347395e-01, -1.22756813e-01,
 -7.73270128e-02,  1.14505434e+00,  1.32689338e-03, -4.25919353e-02,
  7.73701560e-02, -1.14534196e+00, -1.21184482e-03,  4.25487921e-02,
]

# Readable strings for motor and joint definitions. These are also used for Mujoco.
# The order of each list is aligned with XML not hardware header file.
DIGIT_MOTOR_NAME = [
  'left-leg/hip-roll', 'left-leg/hip-yaw', 'left-leg/hip-pitch', 'left-leg/knee',
  'left-leg/toe-a', 'left-leg/toe-b',
  'left-arm/shoulder-roll','left-arm/shoulder-pitch', 'left-arm/shoulder-yaw', 'left-arm/elbow',
  'right-leg/hip-roll', 'right-leg/hip-yaw', 'right-leg/hip-pitch', 'right-leg/knee',
  'right-leg/toe-a', 'right-leg/toe-b',
  'right-arm/shoulder-roll','right-arm/shoulder-pitch', 'right-arm/shoulder-yaw', 'right-arm/elbow'
]

DIGIT_JOINT_NAME = [
  'left-leg/shin', 'left-leg/tarsus', 'left-leg/heel-spring',
  'left-leg/toe-pitch', 'left-leg/toe-roll',
  'right-leg/shin', 'right-leg/tarsus', 'right-leg/heel-spring',
  'right-leg/toe-pitch', 'right-leg/toe-roll'
]

CASSIE_MOTOR_NAME = [
  "left-hip-roll", "left-hip-yaw", "left-hip-pitch", "left-knee", "left-foot",
  "right-hip-roll", "right-hip-yaw", "right-hip-pitch", "right-knee", "right-foot"
]

CASSIE_JOINT_NAME = ["left-shin", "left-tarsus", "right-shin", "right-tarsus"]

CASSIE_MOTOR_LLAPI_NAME = [
  "left-leg/hip-roll", "left-leg/hip-yaw", "left-leg/hip-pitch", "left-leg/knee", "left-leg/foot",
  "right-leg/hip-roll", "right-leg/hip-yaw", "right-leg/hip-pitch", "right-leg/knee", "right-leg/foot"
]

CASSIE_JOINT_LLAPI_NAME = ["left-leg/shin", "left-leg/tarsus", "right-leg/shin", "right-leg/tarsus"]


DIGIT_MOTOR_NAME = [
  'left-leg/hip-roll', 'left-leg/hip-yaw', 'left-leg/hip-pitch', 'left-leg/knee',
  'left-leg/toe-a', 'left-leg/toe-b',
  'left-arm/shoulder-roll','left-arm/shoulder-pitch', 'left-arm/shoulder-yaw', 'left-arm/elbow',
  'right-leg/hip-roll', 'right-leg/hip-yaw', 'right-leg/hip-pitch', 'right-leg/knee',
  'right-leg/toe-a', 'right-leg/toe-b',
  'right-arm/shoulder-roll','right-arm/shoulder-pitch', 'right-arm/shoulder-yaw', 'right-arm/elbow'
]

# Motor and joint names ordered for low-level API
DIGIT_MOTOR_NAME_LLAPI = [
  'left-leg/hip-roll', 'left-leg/hip-yaw', 'left-leg/hip-pitch', 'left-leg/knee',
  'left-leg/toe-a', 'left-leg/toe-b',
  'right-leg/hip-roll', 'right-leg/hip-yaw', 'right-leg/hip-pitch', 'right-leg/knee',
  'right-leg/toe-a', 'right-leg/toe-b',
  'left-arm/shoulder-roll','left-arm/shoulder-pitch', 'left-arm/shoulder-yaw', 'left-arm/elbow',
  'right-arm/shoulder-roll','right-arm/shoulder-pitch', 'right-arm/shoulder-yaw', 'right-arm/elbow'
]

DIGIT_JOINT_NAME_LLAPI = [
  'left-leg/shin', 'left-leg/tarsus',
  'left-leg/toe-pitch', 'left-leg/toe-roll',
  'left-leg/heel-spring',
  'right-leg/shin', 'right-leg/tarsus',
  'right-leg/toe-pitch', 'right-leg/toe-roll',
  'right-leg/heel-spring',
]

DIGIT_MOTOR_MJ2LLAPI_INDEX = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 6, 7, 8, 9, 16, 17, 18, 19]
DIGIT_MOTOR_LLAPI2MJ_INDEX = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19]

DIGIT_JOINT_MJ2LLAPI_INDEX = [0, 1, 3, 4, 2, 5, 6, 8, 9, 7]
DIGIT_JOINT_LLAPI2MJ_INDEX = [0, 1, 4, 2, 3, 5, 6, 9, 7, 8]

# Followings are copied from LLAPI, ordered in LLAPI
# Output torque limit is in Nm
output_torque_limit = [126.682458, 79.176536, 216.927898, 231.31695, 41.975942, 41.975942,\
						126.682458, 79.176536, 216.927898, 231.31695, 41.975942, 41.975942,\
						126.682458, 126.682458, 79.176536, 126.682458,\
						126.682458, 126.682458, 79.176536, 126.682458]
# Output damping limit is in Nm/(rad/s)
output_damping_limit = [66.849046, 26.112909, 38.05002, 38.05002, 28.553161, 28.553161,\
						66.849046, 26.112909, 38.05002, 38.05002, 28.553161, 28.553161,\
						66.849046, 66.849046, 26.112909, 66.849046,\
						66.849046, 66.849046, 26.112909, 66.849046]
# Output velocity limit is in rad/s
output_motor_velocity_limit = [4.5814, 7.3303, 8.5084, 8.5084, 11.5191, 11.5191,\
								4.5814, 7.3303, 8.5084, 8.5084, 11.5191, 11.5191,\
								4.5814, 4.5814, 7.3303, 4.5814,\
								4.5814, 4.5814, 7.3303, 4.5814]
