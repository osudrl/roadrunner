import copy
import numpy as np
import mujoco as mj
import warnings

from .generic_sim import GenericSim
from .mujoco_viewer import MujocoViewer
from .mujoco_render import MujocoRender
from util.colors import FAIL, WARNING, ENDC
from sim.util.geom import Geom
from sim.util.hfield import Hfield
from util.quaternion import quaternion2euler, euler2quat
from util.camera_util import (
    make_pose, pose_inv, transform_from_pixels_to_world,
    transform_from_pixels_to_camera_frame
)
class MujocoSim(GenericSim):
    """
    A base class to define general useful functions that interact with Mujoco simulator.
    This class explicitly avoids robot-specific names.
    """

    def __init__(self, model_path, terrain=None):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(str(model_path))
        self.data = mj.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nbody = self.model.nbody
        self.njnt = self.model.njnt
        self.ngeom = self.model.ngeom
        self.viewer = None
        self.renderer = None # non-primary viewer or offscreen renderer
        self.terrain = terrain
        # Enforce that necessary constants and index arrays are defined
        check_vars = ["torque_delay_cycles", "torque_efficiency", "motor_position_inds",
            "motor_velocity_inds", "joint_position_inds", "joint_velocity_inds",
            "base_position_inds", "base_orientation_inds", "base_linear_velocity_inds",
            "base_angular_velocity_inds", "num_actuators", "num_joints", "reset_qpos"]
        for var in check_vars:
            assert hasattr(self, var), \
                f"{FAIL}Env {self.__class__.__name__} has not defined self.{var}.{ENDC}"
            assert getattr(self, var) is not None, \
                f"{FAIL}In env {self.__class__.__name__} self.{var} is None.{ENDC}"

        assert self.torque_delay_cycles > 0, \
            f"{FAIL}Env {self.__class__.__name__} must have non-zeron torque_delay_cycles. Note " \
            f"that delay cycle of 1 corresponds to no delay (specifies size of the torque buffer.{ENDC}"
        self.torque_buffer = np.zeros((self.torque_delay_cycles, self.model.nu))
        self.torque_buffer_ind = 0
        # Motor constants for torque limit, compute once to save time later
        self.ctrlrange2 = 2 * self.model.actuator_ctrlrange[:, 1]
        self.twmax = self.ctrlrange2 / (self.model.actuator_user[:, 0] * 2 * np.pi / 60)
        self.gear_ratio = self.model.actuator_gear[:, 0]
        self.torque_limit = self.model.actuator_ctrlrange[:, 1]

        self.default_dyn_params = {"damping": self.get_dof_damping(),
                                   "mass": self.get_body_mass(),
                                   "ipos": self.get_body_ipos(),
                                   "spring": self.get_joint_stiffness(),
                                   "friction": self.get_geom_friction("floor"),
                                   "solref": self.get_geom_solref()}

        # Load geoms/bodies for hfield/box/obstacle/stone/stair
        self.load_fixed_object()
        # self.load_movable_object()

    def load_fixed_object(self, num_geoms_in_xml=20):
        """Load any geoms. can add more types, such as non box types, but is limited at compile.
        """
        try:
            self.box_geoms = [f'box{i}' for i in range(num_geoms_in_xml)]
            self.geom_generator = Geom(self)
        except:
            print(f"No box-typed geom listed in XML.\n"
                  f"Or num of geoms is not equal to {num_geoms_in_xml}.")

    def reset(self, qpos: np.ndarray=None, qvel: np.ndarray = None):
        mj.mj_setConst(self.model, self.data)
        mj.mj_resetData(self.model, self.data)
        self.torque_buffer = np.zeros((self.torque_delay_cycles, self.model.nu))
        self.torque_buffer_ind = 0
        if qpos is not None:
            assert len(qpos) == self.model.nq, \
                f"{FAIL}reset qpos len={len(qpos)}, but should be {self.model.nq}{ENDC}"
            self.data.qpos = qpos
        else:
            self.data.qpos = self.reset_qpos
        if qvel is not None:
            assert len(qvel) == self.model.nv, \
                f"{FAIL}reset qvel len={len(qvel)}, but should be {self.model.nv}.{ENDC}"
            self.data.qvel = qvel
        # To avoid mjWarning on arena memory allocation, init hfield before first mj_forward().
        if self.terrain == 'hfield':
            self.init_hfield()
        mj.mj_forward(self.model, self.data)

    def sim_forward(self, dt: float = None):
        if dt:
            num_steps = int(dt / self.model.opt.timestep)
            if num_steps * self.model.opt.timestep != dt:
                warnings.warn(f"{WARNING}Warning: {dt} does not fit evenly within the sim "
                    f"timestep of {self.model.opt.timestep}, simulating forward "
                    f"{num_steps * self.model.opt.timestep}s instead.{ENDC}")
        else:
            num_steps = 1
        mj.mj_step(self.model, self.data, nstep=num_steps)

    def set_torque(self, output_torque: np.ndarray):
        assert output_torque.shape == (self.num_actuators,), \
               f"{FAIL}set_torque got array of shape {output_torque.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        # Apply next torque command in buffer
        self.data.ctrl[:] = self.torque_buffer[self.torque_buffer_ind, :]

        # Torque limit based on motor speed
        tlim = self.ctrlrange2 - self.twmax * abs(self.data.actuator_velocity[:])
        tlim = np.core.umath.clip(tlim, 0, self.torque_limit)
        motor_torque = self.torque_efficiency * np.minimum(tlim, output_torque / self.gear_ratio)
        self.torque_buffer[self.torque_buffer_ind, :] = motor_torque

        self.torque_buffer_ind += 1
        self.torque_buffer_ind %= self.torque_delay_cycles

    def set_PD(self,
               setpoint: np.ndarray,
               velocity: np.ndarray,
               kp: np.ndarray,
               kd: np.ndarray):
        args = locals() # This has to be the first line in the function
        for arg in args:
            if arg != "self":
                assert args[arg].shape == (self.model.nu,), \
                f"{FAIL}set_PD {arg} was not a 1 dimensional array of size {self.model.nu}{ENDC}"
        torque = kp * (setpoint - self.data.qpos[self.motor_position_inds])
        # NOTE: 2 options for damping control below, implicit is recommended by mujoco doc
        # Explicit damping
        # torque += kd * (0 - self.data.qvel[self.motor_velocity_inds])
        # Implicit damping
        self.model.dof_damping[self.motor_velocity_inds] = kd
        self.set_torque(torque)

    def hold(self):
        """Set stiffness/damping for base 6DOF so base is fixed
        NOTE: There is an old funky stuff when left hip-roll motor is somehow coupled with the base
        joint, so left-hip-roll is not doing things correctly when holding.
        Turns out xml seems need to be defined with 3 slide and 1 ball instead of free joint.
        """
        for i in range(3):
            self.model.jnt_stiffness[i] = 1e5
            self.model.dof_damping[i] = 1e4
            self.model.qpos_spring[i] = self.data.qpos[i]

        for i in range(3, 6):
            self.model.dof_damping[i] = 1e5

    def release(self):
        """Zero stiffness/damping for base 6DOF
        """
        for i in range(3):
            self.model.jnt_stiffness[i] = 0
            self.model.dof_damping[i] = 0

        for i in range(3, 6):
            self.model.dof_damping[i] = 0

    def viewer_init(self, fps=50, width=1280, height=960, camera_id=-1):
        self.viewer = MujocoViewer(self.model, self.data, self.reset_qpos, width=width, \
            height=height, camera_id=camera_id)
        self.viewer._fps = fps

    def viewer_render(self):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not render.{ENDC}"
        if self.viewer.is_alive:
            return self.viewer.render()
        else:
            raise RuntimeError(f"{FAIL}Error: Viewer not alive, can not check paused status. Check "
                f"that viewer has not been destroyed.{ENDC}")

    def viewer_paused(self):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not check paused status.{ENDC}"
        if self.viewer.is_alive:
            return self.viewer.paused
        else:
            raise RuntimeError(f"{FAIL}Error: Viewer not alive, can not check paused status. Check "
                f"that viewer has not been destroyed.{ENDC}")

    def viewer_add_marker(self,
                          geom_type: str,
                          name: str,
                          position: list,
                          size: list,
                          rgba: list,
                          so3: np.ndarray):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not add marker status.{ENDC}"
        return self.viewer.add_marker(geom_type, name, position, size, rgba, so3)

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
        self.viewer.update_marker_so3(marker_id, so3)

    def viewer_remove_marker(self, marker_id: int):
        assert not self.viewer is None, \
               f"{FAIL}viewer has not been initalized yet, can not remove marker status.{ENDC}"
        self.viewer.remove_marker(marker_id)

    def init_renderer(self, offscreen: bool, height: int, width: int):
        """Initialized renderer class for rendering.
        For offscreen render: Need to change os environ before
        importing all librairies (including mujoco). This does not work alongside with MujocoViewer.

        import os
        gl_option = 'egl' or 'glx' or 'osmesa'
        os.environ['MUJOCO_GL']=gl_option

        OSMesa has problems see, https://github.com/deepmind/mujoco/issues/700
        Args: height and width of image size. Once set, cannot change.
        """
        if offscreen:
            self.renderer = mj.Renderer(self.model, height=height, width=width)
            # print("Verify the Gl context object, ", self.renderer._gl_context)
        else:
            self.renderer = MujocoRender(self.model, height=height, width=width)

    def get_render_image(self, camera_name: str, type: str = 'depth'):
        """ Get render image (depth or rgb) given camera name and type.
        To use offscreen rendering, first call init_offscreen_renderer() at reset().
        And then call this function.
        The depth value is in meters. To visualize depth on image with better constrast, use
        depth -= depth.min() or depth /= depth.max().
        """
        if camera_name is None:
            raise RuntimeError("Specify a camera name.")

        if type == 'depth':
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=camera_name)
            depth = copy.deepcopy(self.renderer.render())
            assert depth.shape == (self.renderer._height, self.renderer._width), \
                f"Depth image shape {depth.shape} does not match " \
                f"renderer shape {(self.renderer._height, self.renderer._width)}."
            return depth
        elif type == 'rgb':
            self.renderer.disable_depth_rendering()
            self.renderer.disable_segmentation_rendering()
            self.renderer.update_scene(self.data, camera=camera_name)
            rgb = copy.deepcopy(self.renderer.render())
            assert rgb.shape == (self.renderer._height, self.renderer._width, 3), \
                f"RGB image shape {rgb.shape} does not match " \
                f"renderer shape {(self.renderer._height, self.renderer._width), 3}."
            return rgb
        else:
            raise ValueError("Invalid type specified. Choose 'depth' or 'rgb'.")

    def init_hfield(self):
        """Initialize hfield relevant params from XML, hfield_data, reset hfield to flat ground.
        """
        self.hfield_radius_x, self.hfield_radius_y, \
            self.hfield_max_z, self.hfield_min_z = self.model.hfield('hfield0').size
        self.hfield_nrow, self.hfield_ncol = self.model.hfield_nrow[0], self.model.hfield_ncol[0]
        # Resolution meter/pixel
        self.hfield_res_x = self.hfield_radius_x * 2 / self.hfield_nrow
        self.hfield_res_y = self.hfield_radius_y * 2 / self.hfield_ncol
        self.hfield_data = np.zeros((self.model.hfield_nrow[0], self.model.hfield_ncol[0]))
        self.model.hfield('hfield0').data[:] = self.hfield_data
        self.hfield_generator = Hfield(nrow=self.hfield_nrow, ncol=self.hfield_ncol)

    def upload_hfield(self, hfieldid=0):
        """Sync up MjModel with MjContext, applyting to main viewer or renderers.
        https://mujoco.readthedocs.io/en/stable/programming/simulation.html?highlight=mjr_uploadHField#model-changes
        Mujoco seems only allow a single mjrContext when updating mjModel.
        Thus, only one of the following will actually take an affect.
        For regular single-window mujoco viewer, the first is active.
        For single offscreen rendering, the second is active.
        """
        if self.renderer is not None: # if renderer exists, use it
            mj.mjr_uploadHField(self.model, self.renderer._mjr_context, hfieldid)
        if self.viewer is not None: # the main mujoco_viewer is alive
            mj.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)

    def randomize_hfield(self, hfield_type: str=None, data: np.ndarray=None):
        """Randomize hfield data and reset robot pose. If using viewer to visualize, this function
        needs to be called after viewer_init() so mjContext is initialized.

        Args:
            hfield_type (str, optional): Type of hfield ['flat', 'noisy', 'stone', 'bump', 'stair].
                This is loading precomputed hfield.
            data (np.ndarray, optional): 2D ndarray. For hfield.data [0, 0] is bottom right.
                Rows are world-Y axis, and Cols are world-X axis. Data will be normalized to [0, 1].
        """
        if hfield_type is not None:
            if hfield_type in self.hfield_generator.hfield_names:
                run_func = f"self.hfield_generator.create_{hfield_type}()"
                data = getattr(self.hfield_generator, f"create_{hfield_type}")()
        elif data is not None:
            if data.shape != (self.hfield_nrow, self.hfield_ncol):
                raise TypeError(f"randomize_hfield got not supported data {data}")
        else:
            data = np.zeros((self.hfield_nrow, self.hfield_nrow))
        # Mujoco takes in normalized data and scale it to max_z
        assert np.max(data) <= self.hfield_max_z, \
            f"Max height {np.max(data)} is larger than max_z {self.hfield_max_z}."
        self.hfield_data = data / self.hfield_max_z
        self.model.hfield('hfield0').data[:] = self.hfield_data
        self.upload_hfield()
        self.adjust_robot_pose(terrain_type='hfield')

    def get_hfield_height(self, x: float, y: float):
        """Get the height of hfield given world XY location. Returns a single float for height.
        """
        if x > self.hfield_radius_x or x < -self.hfield_radius_x or \
           y > self.hfield_radius_y or y < -self.hfield_radius_y:
            return -10
        x_pixel = self.hfield_nrow // 2 + int(x / self.hfield_radius_x * self.hfield_nrow / 2)
        y_pixel = self.hfield_ncol // 2 + int(y / self.hfield_radius_y * self.hfield_ncol / 2)
        return self.hfield_max_z * self.hfield_data[y_pixel, x_pixel]

    def get_hfield_map(self, grid_unrotated: np.ndarray):
        """Get the local height map at the robot base frame.

        Args:
            grid_unrotated (np.ndarray): 2D float array of shape (heightmap_num_points, 3).
                This grid represents the local heightmap in the robot base frame.
                The first two columns are the x, y coordinates of the grid points.
                The third column is the z coordinate of the grid. This data format only represents
                where to fetch the heightmap data in 3D space relative to robot, not the actual data.

        Returns:
            hfield_map (np.ndarray): 1D array of shape (heightmap_num_points,). This is the local
                heightmap in the robot base frame and heading. The heightmap can be reshaped into
                a 2D array. And the reshape should follow the same order as the grid_unrotated [x, y].
            gridxy_rotated (np.ndarray): 2D float array of shape (heightmap_num_points, 3). This grid
                is rotated to heading but not offset to base position. This is useful for visualization.
        """
        heightmap_num_points = grid_unrotated.shape[0]
        # Rotate the heightmap to base heading, offset to base position, and offset to hfield center
        gridxy_rotated = np.zeros((heightmap_num_points, 3))
        # TODO: helei, change this to vectorized version
        q = euler2quat(z=quaternion2euler(self.get_base_orientation())[2], y=0, x=0)
        for i in range(heightmap_num_points):
            mj.mju_rotVecQuat(gridxy_rotated[i], grid_unrotated[i], q)
        gridxy_rotated += self.get_base_position()
        gridxy_rotated_global = gridxy_rotated + self.hfield_radius_x
        # Conver into global pixel space to get the local heightmap
        pixels = gridxy_rotated_global / self.hfield_res_x
        px = pixels[:, 0].astype(int)
        py = pixels[:, 1].astype(int)
        # Check if the robot is about to run off the hfield
        # User has to make sure training does not let this happen
        assert np.all(px >= 0) and np.all(px <= self.hfield_nrow) and \
               np.all(py >= 0) and np.all(py <= self.hfield_ncol), \
               f"Pixels are out of bound. Robot is at {self.get_base_position()} m, " \
               f"hfield radius is {self.hfield_radius_x}x{self.hfield_radius_y}."
        # Take minimum of surrounding pixels for each point to avoid ambigious height
        # NOTE: need to swap x and y because raw hfield is stored in [y, x] format
        average_height = []
        increment = [[-1, 0], [1, 0], [0, 0], [0, -1], [0, 1]]
        for i, vec in enumerate(increment):
            average_height.append(self.hfield_max_z * self.hfield_data[py+vec[0], px+vec[1]])
        hfield_map = np.min(average_height, axis=0)
        assert hfield_map.shape == (heightmap_num_points,),\
               f"Expected hfield_map shape {(heightmap_num_points,)}, got {hfield_map.shape}."
        return hfield_map, gridxy_rotated

    def adjust_robot_pose(self, terrain_type='geom'):
        """Adjust robot pose to avoid robot bodies stuck inside hfield or geoms.
        Make sure to call if env is updating the model.
        NOTE: Be careful of initializing robot in a bad pose. This function will not fix that mostly.
        """
        # Make sure all kinematics are updated
        mj.mj_kinematics(self.model, self.data)
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
            elif terrain_type == 'hfield':
                # Check iteratively if robot is colliding with geom in XY and move robot up in Z
                z_deltas = []
                for (x,y,z) in [lfoot_pos, rfoot_pos]:
                    tl_hgt = self.get_hfield_height(x + 0.09, y + 0.05)
                    tr_hgt = self.get_hfield_height(x + 0.09, y - 0.05)
                    bl_hgt = self.get_hfield_height(x - 0.09, y + 0.05)
                    br_hgt = self.get_hfield_height(x - 0.09, y - 0.05)
                    z_hgt = max(tl_hgt, tr_hgt, bl_hgt, br_hgt)
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

    def is_self_collision(self):
        """ Check for self collisions. Returns True if there are self collisions, and False otherwise
        """
        for contact_id, contact_struct in enumerate(self.data.contact):
            if self.model.geom_user[contact_struct.geom1] == 1 and \
               self.model.geom_user[contact_struct.geom2] == 1:
                return True
        return False

    """The followings are getter/setter functions to unify with naming with GenericSim()
    """
    def get_joint_position(self):
        return self.data.qpos[self.joint_position_inds]

    def get_joint_velocity(self):
        return self.data.qvel[self.joint_velocity_inds]

    def get_motor_position(self):
        return self.data.qpos[self.motor_position_inds]

    def get_motor_velocity(self):
        return self.data.qvel[self.motor_velocity_inds]

    def get_base_position(self):
        return self.data.qpos[self.base_position_inds]

    def get_base_linear_velocity(self):
        return self.data.qvel[self.base_linear_velocity_inds]

    def get_base_orientation(self):
        return self.data.qpos[self.base_orientation_inds]

    def get_base_angular_velocity(self):
        return self.data.qvel[self.base_angular_velocity_inds]

    def get_feet_position_in_base(self):
        """
        Returns the foot position relative to base position
        """
        base_pos = self.get_base_position()
        l_foot_pos = self.get_site_pose(self.feet_site_name[0])[:3] - base_pos
        r_foot_pos = self.get_site_pose(self.feet_site_name[1])[:3] - base_pos
        output = np.concatenate([l_foot_pos, r_foot_pos])
        return output

    def get_torque(self):
        return self.data.ctrl[:] * copy.deepcopy(self.model.actuator_gear[:, 0])

    def get_joint_qpos_adr(self, name: str):
        return self.model.jnt_qposadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)]

    def get_joint_dof_adr(self, name: str):
        return self.model.jnt_dofadr[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)]

    def get_body_adr(self, name: str):
        return mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)

    def get_joint_adr(self, name: str):
        return mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)

    def get_geom_adr(self, name: str):
        return mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)

    def get_simulation_time(self):
        return self.data.time

    def get_body_pose(self, name: str):
        """Get object pose by name

        Args:
            name (str): object name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.data.body(name).xpos
        pose[3:] = self.data.body(name).xquat
        return pose

    def get_site_pose(self, name: str):
        """Get site pose by name

        Args:
            name (str): site name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.data.site(name).xpos
        mj.mju_mat2Quat(pose[3:7], self.data.site(name).xmat)
        return pose

    def get_geom_pose(self, name: str):
        """Get geom pose by name

        Args:
            name (str): geom name

        Returns:
            ndarray: pose [3xlinear, 4xquaternion]
        """
        pose = np.zeros(7)
        pose[:3] = self.data.geom(name).xpos
        mj.mju_mat2Quat(pose[3:7], self.data.geom(name).xmat)
        return pose

    def get_geom_size(self, name: str):
        """Get geom size by name

        Args:
            name (str): geom name

        Returns:
            ndarray: size [3x]
        """
        return copy.deepcopy(self.model.geom(name).size)

    def get_relative_pose(self, pose1: np.ndarray, pose2: np.ndarray):
        """Computes relative pose of object2 in the frame of object1.

        Args:
            pose1 (np.ndarray): pose of object1
            pose2 (np.ndarray): pose of object2

        Returns:
            pose2_in_pose1: relative pose
        """
        conjugate_pose1 = np.zeros(7)
        mj.mju_negPose(conjugate_pose1[0:3], conjugate_pose1[3:7],
                       pose1[0:3], pose1[3:7])
        pose2_in_pose1 = np.zeros(7)
        mj.mju_mulPose(pose2_in_pose1[0:3], pose2_in_pose1[3:7],
                       conjugate_pose1[0:3], conjugate_pose1[3:7],
                       pose2[0:3], pose2[3:7])
        return pose2_in_pose1

    def get_body_velocity(self, name: str, local_frame=False):
        """Get body velocity by name.
        # TODO: helei, make this generic to take geom and site

        Args:
            name (str): body name
            local_frame (bool, optional): Defaults to False.

        Returns:
            ndarray: velocity [3xlinear, 3xangular]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        velocity = np.zeros(6)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body_id, velocity, local_frame)
        tmp = velocity[3:6].copy()
        velocity[3:6] = velocity[0:3]
        velocity[0:3] = tmp
        return velocity

    def get_body_acceleration(self, name: str, local_frame=False):
        """Get body acceleration by name

        Args:
            name (str): body name
            local_frame (bool, optional): Defaults to False.

        Returns:
            ndarray: velocity [3xlinear, 3xangular]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        velocity = np.zeros(6)
        mj.mj_objectAcceleration(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body_id, velocity, local_frame)
        tmp = velocity[3:6].copy()
        velocity[3:6] = velocity[0:3]
        velocity[0:3] = tmp
        return velocity

    def get_body_contact_force(self, name: str):
        """Get sum of contact forces at the named body in global frame

        Args:
            name (str): body name

        Returns:
            ndarray: sum of all wrenches acting on the body
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        # Sum over all contact wrenches over possible geoms within the body
        total_wrench = np.zeros(6)
        contact_points = 0
        for contact_id, contact_struct in enumerate(self.data.contact):
            if body_id == self.model.geom_bodyid[contact_struct.geom1] or \
               body_id == self.model.geom_bodyid[contact_struct.geom2]:
                contact_points += 1
                contact_wrench_local = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, contact_id, contact_wrench_local)
                contact_wrench_global = np.zeros(6)
                mj.mju_transformSpatial(contact_wrench_global, contact_wrench_local, True,
                                        self.data.xpos[body_id],
                                        self.data.contact[contact_id].pos,
                                        self.data.contact[contact_id].frame)
                if body_id == self.model.geom_bodyid[contact_struct.geom1]:
                    # This body is exerting forces onto geom2, substract from the sum.
                    total_wrench -= contact_wrench_global
                elif body_id == self.model.geom_bodyid[contact_struct.geom2]:
                    # This body is taking forces from geom1, add into the sum.
                    total_wrench += contact_wrench_global
        # Since condim=3, let's keep XYZ for now
        return total_wrench[:3]

    def get_body_contact_force_multipoint(self, name: str):
        """Get each of the contact forces at the named body in global frame along with their global
        positions

        Args:
            name (str): body name

        Returns:
            list[ndarray]: list of numpy arrays of each wrench acting on the body
            list[ndarray]: list of numpy arrays of corresponding global position of each wrench
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        # Sum over all contact wrenches over possible geoms within the body
        forces = []
        contact_pts = []
        for contact_id, contact_struct in enumerate(self.data.contact):
            if body_id == self.model.geom_bodyid[contact_struct.geom1] or \
               body_id == self.model.geom_bodyid[contact_struct.geom2]:
                contact_wrench_local = np.zeros(6)
                mj.mj_contactForce(self.model, self.data, contact_id, contact_wrench_local)
                contact_wrench_global = np.zeros(6)
                mj.mju_transformSpatial(contact_wrench_global, contact_wrench_local, True,
                                        self.data.xpos[body_id],
                                        self.data.contact[contact_id].pos,
                                        self.data.contact[contact_id].frame)
                contact_pts.append(self.data.contact[contact_id].pos)
                if body_id == self.model.geom_bodyid[contact_struct.geom1]:
                    # This body is exerting forces onto geom2, substract from the sum.
                    # total_wrench -= contact_wrench_global
                    forces.append(-contact_wrench_global[:3])
                elif body_id == self.model.geom_bodyid[contact_struct.geom2]:
                    # This body is taking forces from geom1, add into the sum.
                    # total_wrench += contact_wrench_global
                    forces.append(contact_wrench_global[:3])
        forces = np.array(forces)
        contact_pts = np.array(contact_pts)
        return forces, contact_pts

    def is_body_collision(self, body: str):
        """Get if given body has collision by checking any geoms within the body
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, body)
        for contact_id, contact_struct in enumerate(self.data.contact):
            if body_id == self.model.geom_bodyid[contact_struct.geom1] or \
               body_id == self.model.geom_bodyid[contact_struct.geom2]:
                return True
        return False

    def compute_cop(self):
        """
        Computes the current center of pressure (CoP). Note that the class attribute "feet_body_name"
        must be defined for this to be valid, i.e. the sim model should have feet. If this is not
        the case, this function just returns "None". "None" is also returned if the feet are not in
        contact, i.e. there is no CoP.
        Note that is just an approximation of the true center of pressure since our contacts are
        only 3 dimensional, and thus no contact torque is taken into account.

        Returns:
            cop (np.ndarray or None): A 3 long np array representing the point in global
            world frame of the location of the center of pressure. Is instead None in the case where
            either the sim has no feet or none of the feet are in contact with the ground.
        """
        if not hasattr(self, "feet_body_name"):
            cop = None
        else:
            l_multi_force, l_contact_pts = self.get_body_contact_force_multipoint(self.feet_body_name[0])
            r_multi_force, r_contact_pts = self.get_body_contact_force_multipoint(self.feet_body_name[1])
            total_force = 0
            if len(l_multi_force) > 0:
                l_force = np.apply_along_axis(np.linalg.norm, 1, l_multi_force)
                total_force += np.sum(l_force)
            if len(r_multi_force) > 0:
                r_force = np.apply_along_axis(np.linalg.norm, 1, r_multi_force)
                total_force += np.sum(r_force)
            if total_force > 0:
                cop = np.zeros(3)
                for i in range(l_multi_force.shape[0]):
                    cop += l_force[i] / total_force * l_contact_pts[i, :]
                for i in range(r_multi_force.shape[0]):
                    cop += r_force[i] / total_force * r_contact_pts[i, :]
            else:
                cop = None

        return cop

    def get_body_mass(self, name: str = None):
        # If name is None, return all body masses
        if name:
            return copy.deepcopy(self.model.body(name).mass)
        else:
            return copy.deepcopy(self.model.body_mass)

    def get_body_ipos(self, name: str = None):
        # If name is None, return all body masses
        if name:
            return copy.deepcopy(self.model.body(name).ipos)
        else:
            return copy.deepcopy(self.model.body_ipos)

    def get_body_inertia(self, name: str = None):
        # If name is None, return all body inertias
        if name:
            return copy.deepcopy(self.model.body(name).inertia)
        else:
            return copy.deepcopy(self.model.body_inertia)

    def get_dof_damping(self, name: str = None):
        if name:
            return copy.deepcopy(self.model.joint(name).damping)
        else:
            return copy.deepcopy(self.model.dof_damping)

    def get_joint_stiffness(self, name: str = None):
        if name:
            return copy.deepcopy(self.model.joint(name).stiffness)
        else:
            return copy.deepcopy(self.model.jnt_stiffness)

    def get_geom_friction(self, name: str = None):
        if name:
            return copy.deepcopy(self.model.geom(name).friction)
        else:
            return copy.deepcopy(self.model.geom_friction)

    def get_geom_solref(self, name: str = None):
        if name:
            return copy.deepcopy(self.model.geom(name).solref)
        else:
            return copy.deepcopy(self.model.geom_solref)

    def set_joint_position(self, position: np.ndarray):
        assert position.shape == (self.num_joints,), \
               f"{FAIL}set_joint_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_joints},).{ENDC}"
        self.data.qpos[self.joint_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_joint_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_joints,), \
               f"{FAIL}set_joint_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_joints},).{ENDC}"
        self.data.qvel[self.joint_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_motor_position(self, position: np.ndarray):
        assert position.shape == (self.num_actuators,), \
               f"{FAIL}set_motor_position got array of shape {position.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        self.data.qpos[self.motor_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_motor_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (self.num_actuators,), \
               f"{FAIL}set_motor_velocity got array of shape {velocity.shape} but " \
               f"should be shape ({self.num_actuators},).{ENDC}"
        self.data.qvel[self.motor_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_base_position(self, position: np.ndarray):
        assert position.shape == (3,), \
               f"{FAIL}set_base_position got array of shape {position.shape} but " \
               f"should be shape (3,).{ENDC}"
        self.data.qpos[self.base_position_inds] = position
        mj.mj_forward(self.model, self.data)

    def set_base_linear_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"{FAIL}set_base_linear_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,).{ENDC}"
        self.data.qvel[self.base_linear_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    def set_base_orientation(self, quat: np.ndarray):
        assert quat.shape == (4,), \
               f"{FAIL}set_base_orientation got array of shape {quat.shape} but " \
               f"should be shape (4,).{ENDC}"
        self.data.qpos[self.base_orientation_inds] = quat
        mj.mj_forward(self.model, self.data)

    def set_base_angular_velocity(self, velocity: np.ndarray):
        assert velocity.shape == (3,), \
               f"{FAIL}set_base_angular_velocity got array of shape {velocity.shape} but " \
               f"should be shape (3,).{ENDC}"
        self.data.qvel[self.base_angular_velocity_inds] = velocity
        mj.mj_forward(self.model, self.data)

    """The following setters are meant for during simulation or reset, so should not advance
    simulation by itself.
    """

    def set_body_mass(self, mass: float | int | np.ndarray, name: str = None):
        # If name is None, expect setting all masses
        if name:
            assert isinstance(mass, (float, int)), \
                f"{FAIL}set_body_mass got a {type(mass)} instead of a single float when setting mass " \
                f"for single body {name}.{ENDC}"
            self.model.body(name).mass = mass
        else:
            assert mass.shape == (self.model.nbody,), \
                f"{FAIL}set_body_mass got array of shape {mass.shape} but should be shape " \
                f"({self.model.nbody},).{ENDC}"
            self.model.body_mass = mass

    def set_body_ipos(self, ipos: np.ndarray, name: str = None):
        if name:
            assert ipos.shape == (3,), \
                f"{FAIL}set_body_ipos got array of shape {ipos.shape} when setting ipos for " \
                f"single body {name} but should be shape (3,).{ENDC}"
            self.model.body(name).ipos = ipos
        else:
            assert ipos.shape == (self.model.nbody, 3), \
                f"{FAIL}set_body_mass got array of shape {ipos.shape} but should be shape " \
                f"({self.model.nbody}, 3).{ENDC}"
            self.model.body_ipos = ipos

    def set_body_inertia(self, inertia: np.ndarray, name: str = None):
        if name:
            assert inertia.shape == (3,), \
                f"{FAIL}set_body_ipos got array of shape {inertia.shape} when setting ipos for " \
                f"single body {name} but should be shape (3,).{ENDC}"
            self.model.body(name).inertia = inertia
        else:
            assert inertia.shape == (self.model.nbody, 3), \
                f"{FAIL}set_body_mass got array of shape {inertia.shape} but should be shape " \
                f"({self.model.nbody}, 3).{ENDC}"
            self.model.body_inertia = inertia

    def set_dof_damping(self, damp: float | int | np.ndarray, name: str = None):
        if name:
            num_dof = len(self.model.joint(name).damping)
            if num_dof == 1:
                assert isinstance(damp, (float, int)), \
                    f"{FAIL}set_dof_damping got a {type(damp)} when setting damping for single dof " \
                    f"{name} but should be a float or int.{ENDC}"
            else:
                assert damp.shape == (num_dof,), \
                    f"{FAIL}set_dof_damping got array of shape {damp.shape} when setting damping " \
                    f"for single dof {name} but should be shape ({num_dof},).{ENDC}"
            self.model.joint(name).damping = damp
        else:
            assert damp.shape == (self.model.nv,), \
                f"{FAIL}set_dof_damping got array of shape {damp.shape} when setting all joint " \
                f"dofs but should be shape ({self.model.nv},).{ENDC}"
            self.model.dof_damping = damp

    def set_joint_stiffness(self, stiffness: float | int | np.ndarray, name: str = None):
        if name:
            num_dof = len(self.model.joint(name).stiffness)
            if num_dof == 1:
                assert isinstance(stiffness, (float, int)), \
                    f"{FAIL}set_dof_damping got a {type(stiffness)} when setting damping for single dof " \
                    f"{name} but should be a float or int.{ENDC}"
            else:
                assert stiffness.shape == (num_dof,), \
                    f"{FAIL}set_dof_damping got array of shape {stiffness.shape} when setting damping " \
                    f"for single dof {name} but should be shape ({num_dof},).{ENDC}"
            self.model.joint(name).stiffness = stiffness
        else:
            assert stiffness.shape == (self.model.njnt,), \
                f"{FAIL}set_dof_damping got array of shape {stiffness.shape} when setting all joint " \
                f"dofs but should be shape ({self.model.njnt},).{ENDC}"
            self.model.jnt_stiffness = stiffness

    def set_geom_friction(self, fric: np.ndarray, name: str = None):
        if name:
            assert fric.shape == (3, ), \
                f"{FAIL}set_geom_friction got array of shape {fric.shape} when setting friction " \
                f"for single geom {name} but should be shape (3,).{ENDC}"
            self.model.geom(name).friction = fric
        else:
            assert fric.shape == (self.model.ngeom, 3), \
                f"{FAIL}set_geom_friction got array of shape {fric.shape} when setting all geom " \
                f"friction but should be shape ({self.model.ngeom}, 3).{ENDC}"
            self.model.geom_friction = fric

    def set_geom_solref(self, solref: np.ndarray, name: str = None):
        if name:
            assert solref.shape == (1, 2), \
                f"{FAIL}set_geom_solref got array of shape {solref.shape} when setting solref " \
                f"for single geom {name} but should be shape (1,2).{ENDC}"
            self.model.geom(name).solref = solref
        else:
            assert solref.shape == (self.model.ngeom, 2), \
                f"{FAIL}set_geom_solref got array of shape {solref.shape} when setting all geom " \
                f"solref but should be shape ({self.model.ngeom}, 2).{ENDC}"
            self.model.geom_solref = solref

    def set_geom_pose(self, name: str, pose: np.ndarray):
        self.model.geom(name).pos = pose[0:3]
        self.model.geom(name).quat = pose[3:7]

    def set_geom_quat(self, name: str, quat: np.ndarray):
        self.model.geom(name).quat = quat[:]

    def set_geom_size(self, name: str, size: np.ndarray):
        self.model.geom(name).size = size
        # Update radius of bounding sphere for contact detection. Radius depends on geom type,
        # according to Mujoco source code here:
        # https://github.com/google-deepmind/mujoco/blob/6f8128dc6ac853f1bd8da63e6e43e5d13141aeaf/src/user/user_objects.cc#L1457
        match self.model.geom(name).type:
            case mj.mjtGeom.mjGEOM_SPHERE:
                self.model.geom(name).rbound = size[0]
            case mj.mjtGeom.mjGEOM_CAPSULE:
                self.model.geom(name).rbound = np.sum(size[0:2])
            case mj.mjtGeom.mjGEOM_CYLINDER:
                self.model.geom(name).rbound = np.sqrt(np.sum(np.power(size[0:2])))
            case mj.mjtGeom.mjGEOM_ELLIPSOID:
                self.model.geom(name).rbound = max(max(size[0], size[1]), size[2])
            case mj.mjtGeom.mjGEOM_BOX:
                self.model.geom(name).rbound = np.sqrt(np.sum(np.power(size, 2)))
            case mj.mjtGeom.mjGEOM_HFIELD | mj.mjtGeom.mjGEOM_SDF | mj.mjtGeom.mjGEOM_MESH:
                warnings.warn(f"{WARNING}Warning: `set_geom_size` not compatible with height field,"
                    f" sdf, or mesh type geoms. Use the respective geom type functions instead.{ENDC}")

        geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
        # Update aabb parameters just in case midphase pruning is enabled
        self.model.geom_aabb[geom_id, 3:] = size
        self.model.bvh_aabb[geom_id, 3:] = size

    def set_geom_color(self, name: str, color: np.ndarray):
        self.model.geom(name).rgba = color

    def set_body_pose(self, name: str, pose: np.ndarray):
        self.data.body(name).xpos = pose[0:3]
        self.data.body(name).xquat = pose[3:7]

    def get_camera_intrinsic_matrix(self, camera_name, camera_height, camera_width):
        """
        Obtains camera intrinsic matrix.

        Args:
            camera_name (str): name of camera
            camera_height (int): height of camera images in pixels
            camera_width (int): width of camera images in pixels
        Return:
            cam_intrinsic_mat (np.array): 3x3 camera matrix
        """
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        fovy = self.model.cam_fovy[cam_id]
        f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
        cam_intrinsic_mat = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
        return cam_intrinsic_mat

    def get_camera_extrinsic_matrix(self, camera_name):
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame. MuJoCo has a weird convention for how it sets up the
        camera body axis, so we also apply a correction so that the x and y
        axis are along the camera view and the z axis points along the
        viewpoint.
        Normal camera convention: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

        Args:
            camera_name (str): name of camera
        Return:
            cam_extrinsic_mat (np.array): 4x4 camera extrinsic matrix (also know as rotation matrix of camera)
        """
        cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
        camera_pos = self.data.cam_xpos[cam_id]
        camera_rot = self.data.cam_xmat[cam_id].reshape(3, 3)
        cam_extrinsic_mat = make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        cam_extrinsic_mat = cam_extrinsic_mat @ camera_axis_correction
        return cam_extrinsic_mat

    def get_camera_transform_matrix(self, camera_name, camera_height, camera_width):
        """
        Camera transform matrix to project from world coordinates to pixel coordinates.

        Args:
            camera_name (str): name of camera
            camera_height (int): height of camera images in pixels
            camera_width (int): width of camera images in pixels
        Return:
            cam_intrinsic_mat (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
        """
        cam_extrinsic_mat = self.get_camera_extrinsic_matrix(camera_name=camera_name)
        cam_intrinsic_mat = self.get_camera_intrinsic_matrix(camera_name=camera_name, camera_height=camera_height, camera_width=camera_width)
        cam_intrinsic_mat_exp = np.eye(4)
        cam_intrinsic_mat_exp[:3, :3] = cam_intrinsic_mat

        # Takes a point in world, transforms to camera frame, and then projects onto image plane.
        return cam_intrinsic_mat_exp @ pose_inv(cam_extrinsic_mat)

    def get_camera_segmentation(self, camera_name, camera_height, camera_width):
        """
        Obtains camera segmentation matrix.

        Args:
            camera_name (str): name of camera
            camera_height (int): height of camera images in pixels
            camera_width (int): width of camera images in pixels
        Return:
            im (np.array): 2-channel segmented image where the first contains the
                geom types and the second contains the geom IDs
        """
        return self.render(camera_name=camera_name, height=camera_height, width=camera_width, segmentation=True)[::-1]

    def get_point_cloud(self, camera_name, depth_image, stride):
        """
        Generate a point cloud from a depth image using the specified camera and stride.
        Always use raw_depth=True when calling get_depth_image.
        Args:
            camera_name (str): Name of the camera used to capture the depth image
            depth_image (np.array): Depth image to be converted into a point cloud.
            stride (int): Stride for resizing the output point cloud

        Returns:
            point_cloud (np.array): 3D point cloud generated from the depth image
        """
        # Set camera_name and get dimensions of the depth_image
        camera_height, camera_width = depth_image.shape
        # Generate the pixel coordinates with the given stride
        y_indices, x_indices = np.indices((camera_height, camera_width))
        pixels = np.stack([y_indices[::stride, ::stride], x_indices[::stride, ::stride]], axis=-1)
        # Assign depth_map to the input depth_image
        depth_map = depth_image
        # Get the world-to-camera transformation matrix for the given camera
        world_to_camera_transform = self.get_camera_transform_matrix(camera_name=camera_name, camera_height=camera_height, camera_width=camera_width)
        # Compute the camera-to-world transformation matrix by inverting the world-to-camera matrix
        camera_to_world_transform = np.linalg.inv(world_to_camera_transform)
        # Transform pixel coordinates to world coordinates using the depth_map and camera-to-world transformation matrix
        point_cloud = transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform, stride, camera_height, camera_width)
        # Return the generated point cloud
        return point_cloud

    def get_point_cloud_in_camera_frame(self, camera_name, depth_image, stride):
        """
        Generate a point cloud from a depth image using the specified camera and stride.
        Always use raw_depth=True when calling get_depth_image.
        Args:
            camera_name (str): Name of the camera used to capture the depth image
            depth_image (np.array): Depth image to be converted into a point cloud.
            stride (int): Stride for resizing the output point cloud

        Returns:
            point_cloud (np.array): 3D point cloud in the camera frame generated from the depth image
        """
        # Set camera_name and get dimensions of the depth_image
        camera_height, camera_width = depth_image.shape
        # Generate the pixel coordinates with the given stride
        y_indices, x_indices = np.indices((camera_height, camera_width))
        pixels = np.stack([y_indices[::stride, ::stride], x_indices[::stride, ::stride]], axis=-1)
        # Assign depth_map to the input depth_image
        depth_map = depth_image
        # Get the camera intrinsic matrix
        camera_matrix = self.get_camera_intrinsic_matrix(camera_name, camera_height, camera_width)
        # Transform pixel coordinates to camera frame coordinates using the depth_map
        point_cloud = transform_from_pixels_to_camera_frame(pixels, depth_map, camera_matrix, stride, camera_height, camera_width)
        # Return the generated point cloud
        return point_cloud

