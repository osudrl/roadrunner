import glfw
import mujoco as mj
import numpy as np

class MujocoRender():
    def __init__(self, model, width, height):
        """Class to provide minimal render and visualizer capability given mjModel and mjData.
        visualize a fixed camera view with GLFW onscreen and update rendered pixels.
        Primarily adopted from mj.Renderer() in order to align function names.
        Seems like Mujoco does not allow multiple mjContext using the same MjModel/Data, thus
        when using this class, at reset(), user has to manually close the renderer and 
        initialize a new one, to avoid any inconsistent MjModel.

        Args:
            model (_type_): Mj Model object
            width, height (int): Resolution of renderer
        """
        self._model = model
        self._width = width
        self._height = height

        glfw.init()
        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(
            width, height, "", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width
        self.viewport = mj.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # create options, camera, scene, context
        self._scene = mj.MjvScene(self._model, maxgeom=1000)
        self._scene_option = mj.MjvOption()
        self._mjr_context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Internal buffers.
        self._rgb_buffer = np.empty((height, width, 3), dtype=np.uint8)
        self._depth_buffer = np.empty((height, width), dtype=np.float32)

        # Default render flags.
        self._depth_rendering = False
        self._segmentation_rendering = False

    def update_scene(self, mjdata, camera):
        if not isinstance(camera, mj.MjvCamera):
            camera_id = camera
            if isinstance(camera_id, str):
                camera_id = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_CAMERA, camera_id)
            if camera_id < -1:
                raise ValueError('camera_id cannot be smaller than -1.')
            if camera_id >= self._model.ncam:
                raise ValueError(
                f'model has {self._model.ncam} fixed cameras. '
                f'camera_id={camera_id} is invalid.'
            )

            # Render camera.
            camera = mj.MjvCamera()
            camera.fixedcamid = camera_id

            # Defaults to mjCAMERA_FREE, otherwise mjCAMERA_FIXED refers to a
            # camera explicitly defined in the model.
            if camera_id == -1:
                camera.type = mj.mjtCamera.mjCAMERA_FREE
                mj.mjv_defaultFreeCamera(self._model, camera)
            else:
                camera.type = mj.mjtCamera.mjCAMERA_FIXED

        glfw.make_context_current(self.window)
        mj.mjv_updateScene(
            self._model,
            mjdata,
            self._scene_option,
            None,
            camera,
            mj.mjtCatBit.mjCAT_ALL.value,
            self._scene)

    def render(self):
        if glfw.window_should_close(self.window):
            glfw.destroy_window(self.window)
            self.window = None
            return False
        # Set up for rendering
        mj.mjr_render(self.viewport, self._scene, self._mjr_context)
        mj.mjr_readPixels(self._rgb_buffer, self._depth_buffer, self.viewport, self._mjr_context)

        if self._depth_rendering:
            # Get the distances to the near and far clipping planes.
            extent = self._model.stat.extent
            near = self._model.vis.map.znear * extent
            far = self._model.vis.map.zfar * extent
            # Convert from [0 1] to depth in units of length, see links below:
            # http://stackoverflow.com/a/6657284/1461210
            # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
            pixels = near / (1 - self._depth_buffer * (1 - near / far))
        else:
            pixels = self._rgb_buffer
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return np.flipud(pixels)

    def enable_depth_rendering(self):
        self._segmentation_rendering = False
        self._depth_rendering = True

    def disable_depth_rendering(self):
        self._depth_rendering = False

    def enable_segmentation_rendering(self):
        self._segmentation_rendering = True
        self._depth_rendering = False

    def disable_segmentation_rendering(self):
        self._segmentation_rendering = False

    def close(self):
        self._mjr_context.free()
        glfw.window_should_close(self.window)
        glfw.destroy_window(self.window)
        self.window = None
