import copy
import numpy as np

class GenericSim(object):
    """
    A base class to define the functions that interact with simulator.
    This class contains a set of getter/setter that unify generic naming conventions for different simulators
    """

    def __init__(self) -> None:
        pass

    def reset(self, qpos:list=None):
        raise NotImplementedError

    def sim_forward(self, dt: float = None):
        raise NotImplementedError

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        raise NotImplementedError

    def hold(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def viewer_init(self):
        raise NotImplementedError

    def viewer_draw(self):
        raise NotImplementedError

    """Dynamics randomization functions
    """
    def randomize_dynamics(self, dr_ranges):
        """
        Applies dynamics randomization according to the inputted dictionary of values. Dictionary
        should contain ranges and indicies for damping/mass/ipos values to be randomized, along with
        the default values for all the values as well. Note that if a joint/mass index is not in the
        dictionary it will not be randomized. This dictionary input should typically come from an
        environment's loaded in json file. Range values should be given as +- percent of the default
        value. So a range of [-0.1, 0.1] will randomize +- 10% of the default value (randomize in
        the range [0.9*default, 1.1*default]). Note that ipos (link center of mass location)
        randomization is given as a range of values (in meters) to randomize in, not a percentage of
        the default. This is due to the fact that most of the time the CoM is (0, 0, 0), and
        randomization should be small anyway regardless of the CoM location.

        Args:
            dr_ranges (dictionary): Dictionary object containing all of the randomization info:
                ranges, indicies, and default values for damping, mass, ipos (body center of mass
                location), and floor friction.
        """
        # Damping randomization
        rand_damp = copy.deepcopy(self.default_dyn_params["damping"])
        rand_scale = 1 + np.random.uniform(dr_ranges["damping"]["ranges"][:, 0],
                                           dr_ranges["damping"]["ranges"][:, 1])
        rand_damp[dr_ranges["damping"]["inds"]] *= rand_scale
        self.set_dof_damping(rand_damp)
        # Mass randomization
        rand_mass = copy.deepcopy(self.default_dyn_params["mass"])
        rand_scale = 1 + np.random.uniform(dr_ranges["mass"]["ranges"][:, 0],
                                           dr_ranges["mass"]["ranges"][:, 1])
        rand_mass[dr_ranges["mass"]["inds"]] *= rand_scale
        self.set_body_mass(rand_mass)
        # Body CoM location randomization
        # rand_ipos = copy.deepcopy(self.default_dyn_params["ipos"])
        # rand_scale = np.random.uniform(dr_ranges["ipos"]["ranges"][:, 0, :],
        #                                    dr_ranges["ipos"]["ranges"][:, 1, :])
        # rand_ipos[dr_ranges["ipos"]["inds"]] += rand_scale
        # self.set_body_ipos(rand_ipos)
        # Floor friction randomization
        self.set_geom_friction(np.multiply(1 + np.random.uniform(
                                   *dr_ranges["friction"]["ranges"], size=3),
                                   self.default_dyn_params["friction"]), name="floor")
        # Spring stiffness
        if "cassie" in self.__class__.__name__.lower():
            k_knee = np.random.uniform(2000, 3000)
            self.set_jnt_stiffness(name='left-shin', stiffness=k_knee)
            self.set_jnt_stiffness(name='right-shin', stiffness=k_knee)
            k_heel = np.random.uniform(1000, 1500)
            self.set_jnt_stiffness(name='left-heel-spring', stiffness=k_heel)
            self.set_jnt_stiffness(name='right-heel-spring', stiffness=k_heel)
        if "digit" in self.__class__.__name__.lower():
            k_knee = np.random.uniform(5500, 6500)
            self.set_jnt_stiffness(name='left-leg/shin', stiffness=k_knee)
            self.set_jnt_stiffness(name='right-leg/shin', stiffness=k_knee)
            k_heel = np.random.uniform(4000, 4750)
            self.set_jnt_stiffness(name='left-leg/heel-spring', stiffness=k_heel)
            self.set_jnt_stiffness(name='right-leg/heel-spring', stiffness=k_heel)

    def default_dynamics(self):
        """
        Resets all dynamics parameters to their default values.
        """
        self.set_dof_damping(self.default_dyn_params["damping"])
        self.set_body_mass(self.default_dyn_params["mass"])
        self.set_body_ipos(self.default_dyn_params["ipos"])
        self.set_geom_friction(self.default_dyn_params["friction"], name="floor")

    """Getter/Setter to unify across simulators
    """
    def get_joint_position(self):
        raise NotImplementedError

    def set_joint_position(self, position: np.ndarray):
        raise NotImplementedError

    def get_joint_velocity(self):
        raise NotImplementedError

    def set_joint_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def set_joint_range(self, range: np.ndarray):
        raise NotImplementedError

    def get_motor_position(self):
        raise NotImplementedError

    def set_motor_position(self, position: np.ndarray):
        raise NotImplementedError

    def get_motor_velocity(self):
        raise NotImplementedError

    def set_motor_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_base_position(self):
        raise NotImplementedError

    def set_base_position(self, position: np.ndarray):
        raise NotImplementedError

    def get_base_linear_velocity(self):
        raise NotImplementedError

    def set_base_linear_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_base_orientation(self):
        raise NotImplementedError

    def set_base_orientation(self, quat: np.ndarray):
        raise NotImplementedError

    def get_base_angular_velocity(self):
        raise NotImplementedError

    def set_base_angular_velocity(self, velocity: np.ndarray):
        raise NotImplementedError

    def get_torque(self):
        raise NotImplementedError

    def set_torque(self, torque: np.ndarray):
        raise NotImplementedError

    def get_joint_qpos_adr(self, name: str):
        raise NotImplementedError()

    def get_joint_dof_adr(self, name: str):
        raise NotImplementedError()

    def get_joint_range(self):
        raise NotImplementedError

    def get_body_pose(self, name: str, relative_to_body_name=False):
        raise NotImplementedError

    def get_body_velocity(self, name: str, local_frame=True):
        raise NotImplementedError

    def get_body_acceleration(self, name: str, local_frame=True):
        raise NotImplementedError

    def get_body_contact_force(self, name: str):
        raise NotImplementedError

    def get_body_mass(self, name: str = None):
        raise NotImplementedError

    def set_body_mass(self, mass: float | int | np.ndarray, name: str = None):
        raise NotImplementedError

    def get_body_ipos(self, name: str = None):
        raise NotImplementedError

    def set_body_ipos(self, ipos: np.ndarray, name: str = None):
        raise NotImplementedError

    def get_dof_damping(self, name: str = None):
        raise NotImplementedError

    def set_dof_damping(self, damp: float | int | np.ndarray, name: str = None):
        raise NotImplementedError

    def get_geom_friction(self, name: str = None):
        raise NotImplementedError

    def set_geom_friction(self, fric: np.ndarray, name: str = None):
        raise NotImplementedError
