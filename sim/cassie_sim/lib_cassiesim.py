import numpy as np
import pathlib
import time

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from sim import GenericSim
from sim.util.geom import Geom
from util.colors import FAIL, ENDC


class LibCassieSim(GenericSim):

    """
    Cassie simulation using Agility compiled C library libcassiemujoco.so. Uses Mujoco under the
    hood, simulation code is contained in `cassiemujoco` folder.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.state_est_size  = 35
        self.num_actuators   = 10
        self.num_joints = 4
        # self.sim = CassieSim(modelfile=kwargs['modelfile'], terrain=kwargs['terrain'], perception=kwargs['perception'])
        # self.sim = CassieSim(terrain=kwargs['terrain'], perception=kwargs['perception'])
        self.sim = CassieSim()
        self.nq = self.sim.nq
        self.nv = self.sim.nv
        self.nu = self.sim.nu
        self.nbody = self.sim.nbody
        self.njnt = self.sim.njnt
        self.ngeom = self.sim.ngeom
        self.viewer = None
        self.sim_dt = 0.0005    # Assume libcassie sim is always 2kHz
        self.simulator_rate = int(1 / self.sim_dt)

        self.motor_position_inds = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.joint_position_inds = [15, 16, 29, 30]
        self.motor_velocity_inds = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.joint_velocity_inds = [13, 14, 26, 27]

        self.base_position_inds = [0, 1, 2]
        self.base_orientation_inds = [3, 4, 5, 6]
        self.base_linear_velocity_inds = [0, 1, 2]
        self.base_angular_velocity_inds = [3, 4, 5]
        self.base_body_name = "cassie-pelvis"
        self.feet_body_name = ["left-foot", "right-foot"] # force purpose
        self.feet_site_name = ["left-foot-mid", "right-foot-mid"] # pose purpose

        self.num_actuators = 10
        self.num_joints = 4
        self.kp            = np.array([100,  100,  88,  96,  50, 100,  100,  88,  96,  50])
        self.kd            = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
        self.offset        = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.joint_limits_high = np.array([ 0.24, 0.25,  1.35, -0.82, -0.68, 0.2,   0.25,  1.35, -0.82, -0.68])
        self.joint_limits_low  = np.array([-0.2, -0.25, -0.8,  -2.0,  -2.0,  -0.24, -0.25, -0.8,  -2.0,  -2.0])
        self.u            = pd_in_t()
        self.robot_estimator_state = self.sim.step_pd(self.u)

        self.reset_qpos = np.array([0, 0, 1.01, 1, 0, 0, 0,
                    0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
                    -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
                    -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968])

        self.default_dyn_params = {"damping": self.get_dof_damping(),
                                   "mass": self.get_body_mass(),
                                   "ipos": self.get_body_ipos(),
                                   "spring": self.get_joint_stiffness(),
                                   "friction": self.get_geom_friction("floor")}

        # List of bodies that cannot (prefer not) collide with environment
        self.body_collision_list = ['left-tarsus', 'left-achilles-rod', 'left-heel-spring', 'left-foot-crank',\
            'left-plantar-rod',\
            'right-tarsus', 'right-achilles-rod', 'right-heel-spring', 'right-foot-crank',\
            'right-plantar-rod']

        # Load geoms/bodies for hfield/box/obstacle/stone/stair
        self.load_fixed_object()

    def load_fixed_object(self, num_geoms_in_xml=20):
        """Load any geoms. can add more types, such as non box types, but is limited at compile.
        """
        try:
            self.box_geoms = [f'box{i}' for i in range(num_geoms_in_xml)]
            self.geom_generator = Geom(self)
        except Exception as e:
            print(f"No box-typed geom listed in XML.\n"
                  f"Or num of geoms is not equal to {num_geoms_in_xml}.")
            print(e)

    def adjust_robot_pose(self, terrain_type='geom'):
        """Adjust robot pose to avoid robot bodies stuck inside hfield or geoms.
        Make sure to call if env is updating the model.
        NOTE: Be careful of initializing robot in a bad pose. This function will not fix that mostly.
        """
        # Make sure all kinematics are updated
        # mj.mj_kinematics(self.model, self.data)
        collision = True
        base_dx = 0
        original_base_position = self.get_base_position()
        while collision:
            base_position = self.get_base_position()
            lfoot_pos  = self.get_site_pose(self.feet_site_name[0])[0:3]
            rfoot_pos  = self.get_site_pose(self.feet_site_name[1])[0:3]
            if terrain_type == 'geom':
                # Check iteratively if robot is colliding with geom in XY and move robot up in Z
                z_deltas = []
                for (x,y,z) in [lfoot_pos, rfoot_pos]:
                    box_id, heel_hgt = self.geom_generator.check_step(x - 0.09, y, 0)
                    box_id, toe_hgt  = self.geom_generator.check_step(x + 0.09, y, 0)
                    z_hgt = max(heel_hgt, toe_hgt)
                    z_deltas.append((z_hgt-z))
                delta = max(z_deltas)
            else:
                raise RuntimeError(f"Please implement type {terrain_type} for adjust robot pose().")
            base_position[2] = original_base_position[2] + delta
            base_position[0] += base_dx
            self.set_base_position(base_position)
            c = []
            for b in self.body_collision_list:
                c.append(self.is_body_collision(b))
            if any(c):
                collision = True
                base_dx = np.random.uniform(-0.1, 0.1)
            else:
                collision = False

    def reset(self, qpos: np.ndarray=None, qvel: np.ndarray = None):
        self.sim.set_const()
        if qpos is not None:
            assert len(qpos) == self.nq, \
                f"{FAIL}reset qpos len={len(qpos)}, but should be {self.nq}.{ENDC}"
            self.sim.set_qpos(qpos)
        else:
            self.sim.set_qpos(self.reset_qpos)
        if qvel is not None:
            assert len(qvel) == self.nv, \
                f"{FAIL}reset qvel len={len(qvel)}, but should be {self.nv}.{ENDC}"
            self.sim.set_qvel(qvel)
        self.robot_estimator_state = self.sim.step_pd(self.u)

    def sim_forward(self, dt: float = None):
        if dt:
            num_steps = int(dt / self.sim_dt)
            WARNING = '\033[93m'
            ENDC = '\033[0m'
            if num_steps * self.sim_dt != dt:
                print(f"{WARNING}Warning: {dt} does not fit evenly within the sim timestep of"
                    f" {self.sim_dt}, simulating forward"
                    f" {num_steps * self.sim_dt}s instead.{ENDC}")
        else:
            num_steps = 1
        for i in range(num_steps):
            self.robot_estimator_state = self.sim.step_pd(self.u)

    def set_torque(self, torque: np.ndarray):
        assert torque.shape == (self.num_actuators,), \
               f"{FAIL}set_torque got array of shape {torque.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        # Only setting self.u, not actually calling step yet
        # Assume that torque order follows qpos order, so left leg and then right leg
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = 0
            self.u.rightLeg.motorPd.pGain[i] = 0

            self.u.leftLeg.motorPd.dGain[i]  = 0
            self.u.rightLeg.motorPd.dGain[i] = 0

            self.u.leftLeg.motorPd.torque[i]  = torque[i]  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = torque[i+5]

            self.u.leftLeg.motorPd.pTarget[i]  = 0
            self.u.rightLeg.motorPd.pTarget[i] = 0

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        args = locals() # This has to be the first line in the function
        for arg in args:
            if arg != "self":
                assert args[arg].shape == (self.num_actuators,), \
                f"{FAIL}set_PD {arg} was not a 1 dimensional array of size {self.num_actuators}.{ENDC}"
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = kp[i]
            self.u.rightLeg.motorPd.pGain[i] = kp[i + 5]

            self.u.leftLeg.motorPd.dGain[i]  = kd[i]
            self.u.rightLeg.motorPd.dGain[i] = kd[i + 5]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = setpoint[i]
            self.u.rightLeg.motorPd.pTarget[i] = setpoint[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = velocity[i]
            self.u.rightLeg.motorPd.dTarget[i] = velocity[i + 5]

    def hold(self):
        self.sim.hold()

    def release(self):
        self.sim.release()

    def is_body_collision(self, body: str):
        self.sim.is_body_collision(body)

    def viewer_init(self):
        self.viewer = CassieVis(self.sim)

    def viewer_render(self):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not render.{ENDC}"
        return self.viewer.draw(self.sim)

    def viewer_paused(self):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not check paused status.{ENDC}"
        return self.viewer.ispaused()

    def viewer_add_marker(self,
                          geom_type: str,
                          name: str,
                          position: list,
                          size: list,
                          rgba: list,
                          so3: np.ndarray):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not add marker status.{ENDC}"
        so3_input = so3.flatten().tolist()
        self.viewer.add_marker(geom_type, name, position, size, rgba, so3_input)

    def viewer_update_marker_type(self, marker_id: int, geom_type: str):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not update marker type status.{ENDC}"
        self.viewer.update_marker_type(marker_id, geom_type)

    def viewer_update_marker_name(self, marker_id: int, name: str):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not update marker name status.{ENDC}"
        self.viewer.update_marker_name(marker_id, name)

    def viewer_update_marker_position(self, marker_id: int, pos: list):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not update marker position status.{ENDC}"
        self.viewer.update_marker_position(marker_id, pos)

    def viewer_update_marker_size(self, marker_id: int, size: list):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not update marker size status.{ENDC}"
        self.viewer.update_marker_size(marker_id, size)

    def viewer_update_marker_rgba(self, marker_id: int, rgba: list):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not update marker rgba status.{ENDC}"
        self.viewer.update_marker_rgba(marker_id, rgba)

    def viewer_update_marker_so3(self, marker_id: int, so3: np.ndarray):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not update marker so3 status.{ENDC}"
        so3_input = so3.flatten().tolist()
        self.viewer.update_marker_so3(marker_id, so3_input)

    def viewer_remove_marker(self, marker_id: int):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not remove marker status.{ENDC}"
        self.viewer.remove_marker(marker_id)

    """
    The followings are getter/setter functions to unify with naming with GenericSim()
    """

    def get_joint_position(self, state_est: bool = False):
        if state_est:
            # remove double-counted joint/motor positions
            joint_pos = self.robot_estimator_state.joint.position[:]
            joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
            return joint_pos
        else:
            return np.array(self.sim.qpos())[self.joint_position_inds]

    def get_joint_velocity(self, state_est: bool = False):
        if state_est:
            # remove double-counted joint/motor positions
            joint_vel = self.robot_estimator_state.joint.velocity[:]
            joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])
            return joint_vel
        else:
            return np.array(self.sim.qvel())[self.joint_velocity_inds]

    def get_motor_position(self, state_est: bool = False):
        if state_est:
            return self.robot_estimator_state.motor.position[:]
        else:
            return np.array(self.sim.qpos())[self.motor_position_inds]

    def get_motor_velocity(self, state_est: bool = False):
        if state_est:
            return self.robot_estimator_state.motor.velocity[:]
        else:
            return np.array(self.sim.qvel())[self.motor_velocity_inds]

    def get_base_position(self, state_est: bool = False):
        if state_est:
            return self.robot_estimator_state.pelvis.position[:]
        else:
            return np.array(self.sim.qpos())[self.base_position_inds]

    def get_base_linear_velocity(self, state_est: bool = False):
        if state_est:
            return self.robot_estimator_state.pelvis.translationalVelocity[:]
        else:
            return np.array(self.sim.qvel())[self.base_linear_velocity_inds]

    def get_base_orientation(self, state_est: bool = False):
        if state_est:
            return self.robot_estimator_state.pelvis.orientation[:]
        else:
            return np.array(self.sim.qpos())[self.base_orientation_inds]

    def get_base_angular_velocity(self, state_est: bool = False):
        if state_est:
            return self.robot_estimator_state.pelvis.rotationalVelocity[:]
        else:
            return np.array(self.sim.qvel())[self.base_angular_velocity_inds]

    def get_feet_position_in_base(self, state_est: bool = False):
        """
        Returns the foot position relative to base position
        """
        if state_est:
            output = np.concatenate([self.robot_estimator_state.leftFoot.position[:],
                                     self.robot_estimator_state.rightFoot.position[:]])
            return output
        else:
            base_pos = self.get_base_position()
            l_foot_pos = self.get_site_pose(self.feet_site_name[0])[:3] - base_pos
            r_foot_pos = self.get_site_pose(self.feet_site_name[1])[:3] - base_pos
            output = np.concatenate([l_foot_pos, r_foot_pos])
            return output

    def get_torque(self, state_est: bool = False):
        # NOTE: The torque this returns may not actually reflect the current command if
        # sim_forward has not been called yet. This function returns the "last applied" torque. So
        # for example, if you run `set_torque(trq)` and then call `get_torque()` right after, it
        # will not return the same `trq` array. Only after you call `sim_forward()` will `get_torque()`
        # return the `trq` array. This is because `set_torque` uses the pd_in_t struct, which
        # doesn't actually write to mjData.ctrl until sim_forward, i.e sim.step_pd, is called.
        if state_est:
            return np.array(self.robot_estimator_state.motor.torque[:])
        else:
            return np.array(self.sim.ctrl())

    def get_joint_qpos_adr(self, name: str):
        jnt_ind = self.sim.mj_name2id("joint", name)
        return self.sim.jnt_qposadr()[self.sim.mj_name2id("joint", name)]

    def get_joint_dof_adr(self, name: str):
        return self.sim.jnt_dofadr()[self.sim.mj_name2id("joint", name)]

    def get_body_adr(self, name: str):
        return self.sim.mj_name2id("body", name)

    def get_joint_adr(self, name: str):
        return self.sim.mj_name2id("joint", name)

    def get_simulation_time(self):
        return self.sim.time()

    def get_body_pose(self, name: str):
        """Get body pose by name

        Args:
            name (str): body name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.sim.xpos(name)
        pose[3:] = self.sim.xquat(name)
        return pose

    def get_site_pose(self, name: str):
        pose = np.zeros(7)
        pose[:3] = self.sim.get_site_xpos(name)
        pose[3:] = self.sim.get_site_quat(name)
        return pose

    def get_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray):
        pose = np.zeros(7)
        self.sim.get_object_relative_pose(pose1, pose2, pose)
        return pose

    def get_body_velocity(self, name: str, local_frame=False):
        """Get body velocity by name

        Args:
            name (str): body name
            local_frame (bool, optional): Defaults to False.

        Returns:
            ndarray: velocity [3xlinear, 3xangular]
        """
        velocity = np.zeros(6)
        self.sim.body_vel(velocity, name)
        tmp = velocity[3:6].copy()
        velocity[3:6] = velocity[0:3]
        velocity[0:3] = tmp
        if local_frame:
            raise NotImplementedError("Not implemented local frame option. Need to add in c code.")
        return velocity

    def get_body_acceleration(self, name: str, local_frame=False):
        """Get body acceleration by name

        Args:
            name (str): body name
            local_frame (bool, optional): Defaults to False.

        Returns:
            ndarray: velocity [3xlinear, 3xangular]
        """
        accel = np.zeros(6)
        self.sim.get_body_acceleration(accel, name)
        tmp = accel[3:6].copy()
        accel[3:6] = accel[0:3]
        accel[0:3] = tmp
        if local_frame:
            raise NotImplementedError("Not implemented local frame option. Need to add in c code.")
        return accel

    def get_body_contact_force(self, name: str):
        """Get sum of contact forces at the named body in global frame

        Args:
            name (str): body name

        Returns:
            ndarray: sum of all wrenches acting on the body
        """
        total_wrench = np.zeros(6)
        self.sim.get_body_contact_force(total_wrench, name)
        return total_wrench[:3]

    def get_body_mass(self, name: str = None):
        return self.sim.get_body_mass(name)

    def get_body_ipos(self, name: str = None):
        return self.sim.get_body_ipos(name)

    def get_dof_damping(self, name: str = None):
        return self.sim.get_dof_damping(name)

    def get_joint_stiffness(self, name: str = None):
        return self.sim.get_joint_stiffness(name)

    def get_geom_friction(self, name: str = None):
        return self.sim.get_geom_friction(name)

    def get_geom_size(self, name: str = None):
        return self.sim.get_geom_size(name)

    def get_geom_pose(self, name: str = None):
        return self.sim.get_geom_pose(name)

    def set_joint_position(self, position: np.ndarray):
        assert position.shape == (self.num_joints,), \
               f"{FAIL}set_joint_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_joints},).{ENDC}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.joint_position_inds] = position
        self.sim.set_qpos(curr_qpos)

    def set_joint_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_joints,), \
               f"{FAIL}set_joint_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_joints},).{ENDC}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.joint_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_motor_position(self, position: np.ndarray):
        assert position.shape == (self.num_actuators,), \
               f"{FAIL}set_motor_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.motor_position_inds] = position
        self.sim.set_qpos(curr_qpos)

    def set_motor_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_actuators,), \
               f"{FAIL}set_motor_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.motor_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_base_position(self, position: np.ndarray):
        assert position.shape == (3,), \
               f"{FAIL}set_base_position got array of shape {position.shape} but " \
               f"should be shape (3,).{ENDC}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.base_position_inds] = position
        self.sim.set_qpos(curr_qpos)

    def set_base_linear_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"{FAIL}set_base_linear_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,).{ENDC}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.base_linear_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_base_orientation(self, quat: np.ndarray):
        assert quat.shape == (4,), \
               f"{FAIL}set_base_orientation got array of shape {quat.shape} but " \
               f"should be shape (4,).{ENDC}"
        curr_qpos = np.array(self.sim.qpos())
        curr_qpos[self.base_orientation_inds] = quat
        self.sim.set_qpos(curr_qpos)

    def set_base_angular_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"{FAIL}set_base_angular_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,).{ENDC}"
        curr_qvel = np.array(self.sim.qvel())
        curr_qvel[self.base_angular_velocity_inds] = velocity
        self.sim.set_qvel(curr_qvel)

    def set_body_mass(self, mass: float | int | np.ndarray, name: str = None):
        if name:
            assert isinstance(mass, (float, int)), \
                f"{FAIL}set_body_mass got a {type(mass)} instead of a single float when setting " \
                f"mass for a single body {name}.{ENDC}"
        else:
            assert mass.shape == (self.nbody,), \
                f"{FAIL}set_body_mass got array of shape {mass.shape} but should be shape " \
                f"({self.nbody},).{ENDC}"
        self.sim.set_body_mass(mass, name)

    def set_body_ipos(self, ipos: np.ndarray, name: str = None):
        if name:
            assert ipos.shape == (3,), \
                f"{FAIL}set_body_ipos got array of shape {ipos.shape} when setting ipos for " \
                f"single body {name} but should be shape (3,).{ENDC}"
        else:
            assert ipos.shape == (self.nbody, 3), \
                f"{FAIL}set_body_mass got array of shape {ipos.shape} but should be shape " \
                f"({self.nbody}, 3).{ENDC}"
            ipos = ipos.flatten()
        self.sim.set_body_ipos(ipos, name)

    def set_dof_damping(self, damp: np.ndarray, name: str = None):
        if name:
            num_dof = self.sim.get_joint_num_dof(name)
            if num_dof == 1:
                assert isinstance(damp, (float, int)), \
                    f"{FAIL}set_dof_damping got a {type(damp)} when setting damping for single dof " \
                    f"{name} but should be a float or int.{ENDC}"
            else:
                assert damp.shape == (num_dof,), \
                    f"{FAIL}set_dof_damping got array of shape {damp.shape} when setting damping " \
                    f"for single dof {name} but should be shape ({num_dof},).{ENDC}"
        else:
            assert damp.shape == (self.nv,), \
                f"{FAIL}set_dof_damping got array of shape {damp.shape} when setting all joint " \
                f"dofs but should be shape ({self.nv},).{ENDC}"
            damp = damp.flatten()
        self.sim.set_dof_damping(damp, name)

    def set_joint_stiffness(self, stiffness: float | int | np.ndarray, name: str = None):
        if name:
            assert isinstance(stiffness, (float, int)), \
                f"{FAIL}set_joint_stiffness got a {type(stiffness)} instead of a single float when setting " \
                f"stiffness for a single joint {name}.{ENDC}"
        else:
            assert stiffness.shape == (self.njnt,), \
                f"{FAIL}set_joint_stiffness got array of shape {stiffness.shape} but should be shape " \
                f"({self.njnt},).{ENDC}"
        self.sim.set_joint_stiffness(stiffness, name)

    def set_geom_friction(self, fric: np.ndarray, name: str = None):
        if name:
            assert fric.shape == (3, ), \
                f"{FAIL}set_geom_friction got array of shape {fric.shape} when setting friction " \
                f"for single geom {name} but should be shape (3,).{ENDC}"
        else:
            assert fric.shape == (self.ngeom, 3), \
                f"{FAIL}set_geom_friction got array of shape {fric.shape} when setting all geom " \
                f"friction but should be shape ({self.ngeom}, 3).{ENDC}"
            fric = fric.flatten()
        self.sim.set_geom_friction(fric, name)

    def set_geom_pose(self, name: str, pose: np.ndarray):
        self.sim.set_geom_pos(pose[0:3], name)
        self.sim.set_geom_quat(pose[3:7], name)

    def set_geom_quat(self, name: str, quat: np.ndarray):
        self.sim.set_geom_quat(quat, name)

    def set_geom_size(self, name: str, size: np.ndarray):
        self.sim.set_geom_size(size, name)

    def set_geom_color(self, name: str, color: np.ndarray):
        self.sim.set_geom_color(color, name)
