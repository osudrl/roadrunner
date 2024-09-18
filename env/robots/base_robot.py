from abc import ABC, abstractmethod
import copy

import numpy as np
from scipy.spatial.transform import Rotation as R
from util.quaternion import mj2scipy, scipy2mj


class BaseRobot(ABC):
    """
      Serves as a robot class that defines the structure robot classes should adhere to
      All robots should inherit from this class and overwrite abstract methods
    """
    def __init__(
        self,
        robot_name: str,
        simulator_type: str,
    ):
        self.robot_name = robot_name
        self.simulator_type = simulator_type

        # Define params that child classes should overwrite
        self.kp = None
        self.kd = None
        self._offset = None
        self._motor_mirror_indices = None
        self._robot_state_mirror_indices = None
        self._min_base_height = None

    @property
    def sim(self):
        return self._sim

    @property
    def offset(self):
        return self._offset

    @property
    def n_actuators(self):
        return self._sim.num_actuators

    @property
    def n_unactuated_joints(self):
        return self._sim.num_joints

    @property
    def motor_mirror_indices(self):
        return copy.deepcopy(self._motor_mirror_indices)

    @property
    def robot_state_mirror_indices(self):
        return copy.deepcopy(self._robot_state_mirror_indices)

    @property
    def min_base_height(self):
        return self._min_base_height

    def rotate_to_heading(self, orientation: np.ndarray, orient_add):
        """Offset robot heading in world frame by self.orient_add amount

        Args:
            orientation (list): current robot heading in world frame

        Returns:
            new_orient (list): Offset orientation
        """
        orient_add_quat = R.from_euler('xyz',[0, 0 ,orient_add], degrees=False)
        new_quat = orient_add_quat.inv() * R.from_quat(mj2scipy(orientation))
        return scipy2mj(new_quat.as_quat())

    @abstractmethod
    def get_raw_robot_state(self):
        """Get standard robot proprioceptive states. Child class implements this method

        Returns:
            robot_state (np.ndarray): robot state
        """
        pass
