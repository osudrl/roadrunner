import glfw
import copy
import imageio
import time
import mujoco as mj
import numpy as np

from threading import Lock
from multiprocessing import Process
from util.colors import FAIL, WARNING, ENDC
from util.quaternion import euler2so3
"""
Class to handle visualization of generic Mujoco models. Adapted from cassie-mujoco-sim
(https://github.com/osudrl/cassie-mujoco-sim) and mujoco-python-viewer
(https://github.com/rohanpsingh/mujoco-python-viewer)
"""

# help strings
# TODO: Fill out rest of help strings for key functionality
HELP_CONTENT = ("Alt mouse button\n"
        "UI right hold\n"
        "UI title double-click\n"
        "Space\n"
        "Esc\n"
        "Right arrow\n"
        "Left arrow\n"
        "Down arrow\n"
        "Up arrow\n"
        "Page Up\n"
        "Double-click\n"
        "Right double-click\n"
        "Ctrl Right double-click\n"
        "Scroll, middle drag\n"
        "Left drag\n"
        "[Shift] right drag\n"
        "Ctrl [Shift] drag\n"
        "Ctrl [Shift] right drag\n"
        "Ctrl v")

HELP_TITLE = ("Swap left-right\n"
        "Show UI shortcuts\n"
        "Expand/collapse all  \n"
        "Pause\n"
        "Free camera\n"
        "Step forward\n"
        "Step back\n"
        "Step forward 100\n"
        "Step back 100\n"
        "Select parent\n"
        "Select\n"
        "Center\n"
        "Track camera\n"
        "Zoom\n"
        "View rotate\n"
        "View translate\n"
        "Object rotate\n"
        "Object translate\n"
        "Toggle video rec")

class MujocoViewer():
    def __init__(self,
                 model,
                 data,
                 reset_qpos,
                 camera_id=-1,
                 width=None,
                 height=None):

        self.model = model
        self.data = data
        self.reset_qpos = reset_qpos
        assert len(self.reset_qpos) == self.model.nq, \
               f"{FAIL}Size of MujocoViewer reset qpos does not match model nq size.{ENDC}"

        self.is_alive = True
        self.paused = True

        glfw.init()
        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        self.window = glfw.create_window(
            width, height, "Mujoco Sim", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width
        self.viewport = mj.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # set callbacks
        glfw.set_cursor_pos_callback(
            self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(
            self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mj.MjvOption()
        self.cam = mj.MjvCamera()
        self.scn = mj.MjvScene(self.model, maxgeom=1000000)
        self.pert = mj.MjvPerturb()
        self.ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        self.fontscale = 150
        # Set up mujoco visualization objects
        if camera_id != -1 and isinstance(camera_id, str):
            # If camera specified, attach camera to fixed view
            camera_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera_id)
            self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
        elif camera_id < -1:
            raise ValueError('camera_id cannot be smaller than -1.')
        elif camera_id >= self.model.ncam:
            raise ValueError(
                f'model has {self._model.ncam} fixed cameras. '
                f'camera_id={camera_id} is invalid.'
            )
        else: # Default to tracking camera
            self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = 0
            self.cam.distance = 3
            self.cam.azimuth = 90
            self.cam.elevation = -20
        self.cam.fixedcamid = camera_id

        # Set interaction ctrl vars
        self.paused = True
        self._lastbutton = glfw.MOUSE_BUTTON_1
        self._lastclicktm = 0.0
        self._showhelp = False
        self._showvishelp = False
        self._showrndhelp = False
        self._showGRF = False
        self._GRFcount = 0
        self._showfullscreen = False
        self._showsensor = False
        self._slowmotion = False
        self._showinfo = True
        self._framenum = 0
        self._lastframenum = 0
        self._marker_num = 0
        self._perturb_body = 1
        self._pipe_video_out = None
        self._perturb_force = np.zeros(6)

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_left_click_time = None
        self._last_right_click_time = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._image_idx = 0
        self._image_path = "/tmp/screenshot_"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menus = False

        # Visual marker infos
        self.num_marker = 0
        self.marker_info = {}
        self.pointcloud_marker_ids = []

        # Video recording
        self._record_video = False
        self._video_frames = []
        self._video_path = "/tmp/video_"
        self._fps = None

    def render(self):
        if glfw.window_should_close(self.window):
            self.close()
            print(f"{WARNING}MujocoViewer window closed.{ENDC}")
            return False
        # # Apply perturbations
        if self.pert.select > 0:
            self.data.xfrc_applied = np.zeros((self.model.nbody, 6))
            mj.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)    # move mocap bodies only
            mj.mjv_applyPerturbForce(self.model, self.data, self.pert)
        with self._gui_lock:
            # Set up for rendering
            glfw.make_context_current(self.window)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            mj.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                self.cam,
                mj.mjtCatBit.mjCAT_ALL.value,
                self.scn)
            self._write_marker_to_scene()
            mj.mjr_render(self.viewport, self.scn, self.ctx)
            if self._showhelp:
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150.value,
                               mj.mjtGridPos.mjGRID_TOPLEFT,
                               self.viewport,
                               HELP_TITLE,
                               HELP_CONTENT,
                               self.ctx)
            if self._showvishelp:
                # Make vis help strings
                key_title = ""
                key_content = ""
                for i in range(mj.mjtVisFlag.mjNVISFLAG):
                    key_title += mj.mjVISSTRING[i][0].replace("&", "") + "\n"
                    key_content += mj.mjVISSTRING[i][2] + "\n"
                key_title = key_title[:-1]
                key_content = key_content[:-1]
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150.value,
                               mj.mjtGridPos.mjGRID_TOPLEFT,
                               self.viewport,
                               key_title,
                               key_content,
                               self.ctx)
            if self._showrndhelp:
                # Make vis help strings
                key_title = ""
                key_content = ""
                for i in range(mj.mjtRndFlag.mjNRNDFLAG):
                    key_title += mj.mjRNDSTRING[i][0].replace("&", "") + "\n"
                    key_content += mj.mjRNDSTRING[i][2] + "\n"
                frame_title = ["" for i in range(mj.mjtFrame.mjNFRAME)]
                for frame_type in dir(mj.mjtFrame):
                    if "mjFRAME" in frame_type:
                        frame_title[getattr(mj.mjtFrame, frame_type)] = frame_type[8:].lower()
                for i in range(len(frame_title)):
                    key_title += f"{frame_title[i]} frame\n"
                    key_content += f"{i}\n"
                key_title = key_title[:-1]
                key_content = key_content[:-1]
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150.value,
                               mj.mjtGridPos.mjGRID_TOPLEFT,
                               self.viewport,
                               key_title,
                               key_content,
                               self.ctx)
            if self._showinfo:
                if self.paused:
                    str_paused = "\nPaused"
                else:
                    str_paused = "\nRunning"
                str_paused += "\nTime:"
                if self._slowmotion:
                    str_slow = "(10x slowdown)"
                else:
                    str_slow = ""
                str_info = str_slow + f"\n\n{self.data.time:.2f}"
                mj.mjr_overlay(mj.mjtFontScale.mjFONTSCALE_150.value,
                               mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                               self.viewport,
                               str_paused,
                               str_info,
                               self.ctx)
            glfw.swap_buffers(self.window)
        glfw.poll_events()

        if self._record_video:
            frame = np.zeros(
                        (glfw.get_framebuffer_size(
                            self.window)[1], glfw.get_framebuffer_size(
                            self.window)[0], 3), dtype=np.uint8)
            mj.mjr_readPixels(frame, None, self.viewport, self.ctx)
            self._video_frames.append(np.flipud(frame))

        return True

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            # Don't do anything if released the key
            return
        elif action == glfw.PRESS:
            # If press 'P' with no mods, then attach camera to center of model
            if key == glfw.KEY_P and mods == 0:
                self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
                self.cam.trackbodyid = 0
                self.cam.fixedcamid = -1
                self.cam.distance = 3
                self.cam.azimuth = 90
                self.cam.elevation = -20
                return
            # Handle control key mod
            if mods == glfw.MOD_CONTROL:
                if key == glfw.KEY_A:
                    self.cam.lookat = self.model.stat.center
                    self.cam.distance = 1.5 * self.model.stat.extent
                    self.cam.type = mj.mjtCamera.mjCAMERA_FREE
                    return
                elif key == glfw.KEY_P: # Print out qpos
                    qpos_str = "qpos: "
                    for i in range(self.model.nq):
                        qpos_str += f"{self.data.qpos[i]:.4f}"
                        if i != self.model.nq - 1:
                            qpos_str += ", "
                    print(qpos_str)
                    return
                elif key == glfw.KEY_V or \
                (key == glfw.KEY_ESCAPE and self._record_video): # Start or stop a recording
                    if self._fps is not None and self._fps > 0:
                        self._record_video = not self._record_video
                        if self._record_video:
                            glfw.set_window_title(self.window, "Mujoco Sim (recording)")
                if not self._record_video and len(self._video_frames) > 0:
                    frames = [f for f in self._video_frames]
                    time_stamp = time.strftime("%Y-%m-%d_%H%M%S")
                    self.save_video_process = Process(target=save_video,
                                  args=(frames, self._video_path + time_stamp + ".mp4", self._fps))
                    self.save_video_process.start()
                    self._video_frames = []
                    glfw.set_window_title(self.window, "Mujoco Sim")
                elif key == glfw.KEY_T: # Save screenshot
                    time_stamp = time.strftime("%Y%m%d_%X")
                    img = np.zeros(
                        (glfw.get_framebuffer_size(
                            self.window)[1], glfw.get_framebuffer_size(
                            self.window)[0], 3), dtype=np.uint8)
                    mj.mjr_readPixels(img, None, self.viewport, self.ctx)
                    imageio.imwrite(self._image_path + time_stamp + ".png", np.flipud(img))
                    print(f"Saved screenshot to {self._image_path + time_stamp}.png")
                    self._image_idx += 1
                    return
                elif key == glfw.KEY_Q:
                    glfw.set_window_should_close(self.window, True)
                    return
            # Toggle geom/site group
            if mods == glfw.MOD_SHIFT:
                for i in range(mj.mjNGROUP):
                    if key == i + 48:
                        self.vopt.geomgroup[i] = 1 - self.vopt.geomgroup[i]
                        return
            # Toggle visualization flags
            for i in range(mj.mjtVisFlag.mjNVISFLAG):
                if glfw.get_key_name(key, scancode) == mj.mjVISSTRING[i][2].lower():
                    self.vopt.flags[i] = 1 - self.vopt.flags[i]
                    # return
                    # Don't return here due to overlapping key in VISSTRING and RNDSTRING. "," key
                    # is in both, so if "," is pressed toggle both flags and return in RNDSTRING check
            # Toggle rendering flags
            for i in range(mj.mjtRndFlag.mjNRNDFLAG):
                if glfw.get_key_name(key, scancode) == mj.mjRNDSTRING[i][2].lower():
                    self.scn.flags[i] = 1 - self.scn.flags[i]
                    return
            # Toggle frame rendering
            for i in range(mj.mjtFrame.mjNFRAME):
                if key == i + 48:
                    if self.vopt.frame == i:
                        self.vopt.frame = 0
                    else:
                        self.vopt.frame = i
        # Handle regular inidividual key presses
        if key == glfw.KEY_F1:              # help
            self._showhelp = not self._showhelp
            # Turn off other overlays if active
            if self._showhelp:
                self._showvishelp = False
                self._showrndhelp = False
        elif key == glfw.KEY_F2:            # visualizer flag help
            self._showvishelp = not self._showvishelp
            # Turn off other overlays if active
            if self._showvishelp:
                self._showhelp = False
                self._showrndhelp = False
        elif key == glfw.KEY_F3:            # render flag help
            self._showrndhelp = not self._showrndhelp
            # Turn off other overlays if active
            if self._showrndhelp:
                self._showvishelp = False
                self._showhelp = False
        elif key == glfw.KEY_F4:            # info
            self._showinfo = not self._showinfo
        elif key == glfw.KEY_F5:            # GRF
            self._showGRF = not self._showGRF
        elif key == glfw.KEY_F6:            # sensor figure
            self._showsensor = not self._showsensor
        elif key == glfw.KEY_F7:            # toggle fullscreen
            self._showfullscreen = not self._showfullscreen
            if self._showfullscreen:
                glfw.maximize_window(self.window)
            else:
                glfw.restore_window(self.window)
        elif key == glfw.KEY_ENTER:         # toggle slow motion
            self._slowmotion = not self._slowmotion
        elif key == glfw.KEY_SPACE:         # pause
            self.paused = not self.paused
        elif key == glfw.KEY_BACKSPACE:     # reset
            mj.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self.reset_qpos
            mj.mj_forward(self.model, self.data)
        elif key == glfw.KEY_RIGHT:         # step forward
            if self.paused:
                mj.mj_step(self.model, self.data)
        elif key == glfw.KEY_DOWN:          # step forward 100
            if self.paused:
                mj.mj_step(self.model, self.data, nstep=100)
        elif key == glfw.KEY_ESCAPE:        # free camera
            self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        elif key == glfw.KEY_EQUAL:         # bigger font
            if self.fontscale < 200:
                self.fontscale += 50
                self.ctx = mj.MjrContext(self.model, self.fontscale)
        elif key == glfw.KEY_MINUS:         # smaller font
            if self.fontscale > 100:
                self.fontscale -= 50
                self.ctx = mj.MjrContext(self.model, self.fontscale)
        elif key == glfw.KEY_LEFT_BRACKET:  # previous fixed camera or free
            if self.model.ncam > 0 and self.cam.type == mj.mjtCamera.mjCAMERA_FIXED:
                if self.cam.fixedcamid > 0:
                    self.cam.fixedcamid -= 1
                else:
                    self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        elif key == glfw.KEY_RIGHT_BRACKET:  # next fixed camera
            if self.model.ncam > 0:
                if self.cam.type != mj.mjtCamera.mjCAMERA_FIXED:
                    self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
                    self.cam.fixedcamid = 0
                elif self.cam.fixedcamid < self.model.ncam - 1:
                    self.cam.fixedcamid += 1

        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        # no buttons down, nothing to do
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # compute mouse displacement
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos
        width, height = glfw.get_framebuffer_size(window)

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # determine action based on mouse button
        if self._button_right_pressed:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        # move perturb or camera
        with self._gui_lock:
            if self.pert.active:
                mj.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mj.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

    def _mouse_button_callback(self, window, button, act, mods):
        # update button state
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        # Alt: swap left and right
        if mods == glfw.MOD_ALT:
            tmp = self._button_left_pressed
            self._button_left_pressed = self._button_right_pressed
            self._button_right_pressed = tmp
            if button == glfw.MOUSE_BUTTON_LEFT:
                button = glfw.MOUSE_BUTTON_RIGHT
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                button = glfw.MOUSE_BUTTON_LEFT

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = x
        self._last_mouse_y = y

        # set perturbation
        newperturb = 0
        if mods == glfw.MOD_CONTROL and self.pert.select > 0:
            if act == glfw.PRESS:
                # Disable vis perturb force when using mouse perturb, only want to vis perturb object
                self.vopt.flags[11] = 0
                # right: translate, left: rotate
                if self._button_right_pressed:
                    newperturb = mj.mjtPertBit.mjPERT_TRANSLATE
                elif self._button_left_pressed:
                    newperturb = mj.mjtPertBit.mjPERT_ROTATE

                # perturbation onste: reset reference
                if newperturb and not self.pert.active:
                    mj.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
            else:
                # Enable vis perturn force again
                self.vopt.flags[11] = 1
        self.pert.active = newperturb

        # detect a left- or right- doubleclick (250 msec)
        curr_time = glfw.get_time()
        if act == glfw.PRESS and (curr_time - self._lastclicktm < 0.25) and (button == self._lastbutton):
            # determine selection mode
            selmode = 2         # Right Click
            if button == glfw.MOUSE_BUTTON_LEFT:
                selmode = 1
            elif mods == glfw.MOD_CONTROL:
                selmode = 3     # CTRL + Right Click

            # find geom and 3D click point, get corresponding body
            width, height = self.viewport.width, self.viewport.height
            aspectratio = width / height
            relx = x / width
            rely = (self.viewport.height - y) / height
            selpnt = np.zeros((3, 1), dtype=np.float64)
            selgeom = np.zeros((1, 1), dtype=np.int32)
            selskin = np.zeros((1, 1), dtype=np.int32)
            selflex = np.zeros((1, 1), dtype=np.int32)
            selbody = mj.mjv_select(self.model, self.data, self.vopt, aspectratio, relx, rely,
                self.scn, selpnt, selgeom, selflex, selskin)

            # set lookat point, start tracking is requested
            if selmode == 2 or selmode == 3:
                # set cam lookat
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                # switch to tracking camera if dynamic body clicked
                if selmode == 3 and selbody > 0:
                    self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            # set body selection
            else:
                if selbody >= 0:
                    # record selection
                    self.pert.select = selbody
                    self.pert.skinselect = selskin
                    # compute localpos
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    mat = self.data.xmat[selbody].reshape(3, 3)
                    self.pert.localpos = mat.transpose().dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            # stop perturbation on select
            self.pert.active = 0

        if act == glfw.PRESS:
            self._lastbutton = button
            self._lastclicktm = glfw.get_time()
        else:   # 3D release
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mj.mjv_moveCamera(self.model,
                              mj.mjtMouse.mjMOUSE_ZOOM,
                              0,
                              -0.05 * y_offset,
                              self.scn,
                              self.cam)

    def add_marker(self,
                   geom_type: str,
                   name: str,
                   position: list,
                   size: list,
                   rgba: list,
                   so3: np.ndarray):
        """
        This function adds a marker to the visualization. The marker is purely visual, and has no
        effect on the dynamics of the simulation. The marker will made according to the argument
        inputs. Note that extra "rendering-only" geom types are allowed here as well, that you can't
        use in normal modelling. Geoms like "arrow", "line", and "label" can be used as markers.
        The function also returns a marker id, that the user is responsible for holding on to so
        they can reference the marker later on in case they want to update its properties or
        remove it. Note that position and so3 are in global frame.

        Args:
            geom_type (string): Type of geom to make the marker
            name (string): Name of the marker. This will show up attached to the marker as well.
            position (list): global xyz position of the marker
            size (list): Size parameters of the marker. Should always be 3 numbers
            rgba (list): rgba values of the marker geom
            so3 (np.ndarray): Full 3x3 global so3 orientation matrix of the marker geom
        """
        if self.scn.ngeom + 1 > self.scn.maxgeom:
            print(f"{FAIL}Vis scn.maxgeom of {self.scn.maxgeom} reached, cannot add new marker.{ENDC}")
        else:
            assert geom_type in ["plane", "sphere", "capsule", "ellipsoid", "cylinder", "box",
                                 "arrow", "arrow1", "arrow2", "line", "label"], \
               f"{FAIL}MujocoViewer add_marker got `geom_type` of {geom_type}, but should be " \
               f"one of ['plane', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box', 'arrow', " \
               f"'arrow1', 'arrow2', 'line', 'label']{ENDC}"
            assert isinstance(name, str), \
               f"{FAIL}MujocoViewer add_marker got `name` of type {type(name)}, but should be " \
               f"a string.{ENDC}"
            assert len(position) == 3, \
               f"{FAIL}MujocoViewer add_marker got `position` of size {len(position)}, but should be " \
               f"of length 3.{ENDC}"
            assert len(rgba) == 4, \
               f"{FAIL}MujocoViewer add_marker got 'rgba' of size {len(rgba)}, but should be " \
               f"of length 4.{ENDC}"
            assert so3.shape == (3, 3), \
               f"{FAIL}MujocoViewer add_marker got 'so3' array of shape {so3.shape}, but should " \
               f"be shape (3, 3).{ENDC}"
            assert len(size) == 3, \
                f"{FAIL}MujocoViewer add_marker got `size` of size {len(size)}, but should be of " \
                f"length 3.{ENDC}"

            marker_dict = {"geom_type": geom_type,
                           "name": name,
                           "position": position,
                           "size": size,
                           "rgba": rgba,
                           "so3": so3}
            self.marker_info[self.num_marker] = marker_dict
            self.num_marker += 1
            return self.num_marker - 1

    def update_marker_type(self, marker_id: int, geom_type: str):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer update_marker_type got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        assert geom_type in ["plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"], \
               f"{FAIL}MujocoViewer add_marker got `geom_type` of {geom_type}, but should be " \
               f"one of ['plane', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box'].{ENDC}"
        if marker_id in self.marker_info.keys():
            self.marker_info[marker_id]["geom_type"] = geom_type
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not update anything!{ENDC}")

    def update_marker_name(self, marker_id: int, name: str):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer update_marker_name got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        assert isinstance(name, str), \
               f"{FAIL}MujocoViewer update_marker_name got `name` of type {type(name)}, but should " \
               f"be a string.{ENDC}"
        if marker_id in self.marker_info.keys():
            self.marker_info[marker_id]["name"] = name
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not update anything!{ENDC}")

    def update_marker_position(self, marker_id: int, pos: list):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer update_marker_position got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        assert len(pos) == 3, \
               f"{FAIL}MujocoViewer add_marker got `pos` of size {len(pos)}, but should be " \
               f"of length 3.{ENDC}"
        if marker_id in self.marker_info.keys():
            self.marker_info[marker_id]["position"] = pos
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not update anything!{ENDC}")

    def update_marker_size(self, marker_id: int, size: list):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer update_marker_size got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        assert len(size) == 3, \
                f"{FAIL}MujocoViewer update_marker_size got `size` of size {len(size)}, but should " \
                f"be of length 3.{ENDC}"
        if marker_id in self.marker_info.keys():
            self.marker_info[marker_id]["size"] = size
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not update anything!{ENDC}")

    def update_marker_rgba(self, marker_id: int, rgba: list):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer update_marker_rgba got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        assert len(rgba) == 4, \
               f"{FAIL}MujocoViewer update_marker_rgba got 'rgba' of size {len(rgba)}, but should be " \
               f"of length 4.{ENDC}"
        if marker_id in self.marker_info.keys():
            self.marker_info[marker_id]["rgba"] = rgba
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not update anything!{ENDC}")

    def update_marker_so3(self, marker_id: int, so3: np.ndarray):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer update_marker_so3 got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        assert so3.shape == (3, 3), \
               f"{FAIL}MujocoViewer add_marker got 'so3' array of shape {so3.shape}, but should " \
               f"be shape (3, 3).{ENDC}"
        if marker_id in self.marker_info.keys():
            self.marker_info[marker_id]["so3"] = so3
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not update anything!{ENDC}")

    def remove_marker(self, marker_id: int):
        assert isinstance(marker_id, int), \
            f"{FAIL}MujocoViewer remove_marker got invalid marker_id of {marker_id}. marker_id " \
            f"should be an int."
        if marker_id in self.marker_info.keys():
            del self.marker_info[marker_id]
        else:
            print(f"{WARNING}Marker with id {marker_id} not found, did not remove anything!{ENDC}")

    def _write_marker_to_scene(self):
        """
        This internal function will write all of the current marker info to the geoms in the scene.
        Is intended to be called inside 'render' to draw the visual markers to the visualization.
        """
        for marker, info in self.marker_info.items():
            self.scn.geoms[self.scn.ngeom].dataid = -1
            self.scn.geoms[self.scn.ngeom].objtype = mj.mjtObj.mjOBJ_UNKNOWN
            self.scn.geoms[self.scn.ngeom].objid = -1
            self.scn.geoms[self.scn.ngeom].category = mj.mjtCatBit.mjCAT_DECOR
            self.scn.geoms[self.scn.ngeom].texid = -1
            self.scn.geoms[self.scn.ngeom].texuniform = 0
            self.scn.geoms[self.scn.ngeom].texrepeat[0] = 1
            self.scn.geoms[self.scn.ngeom].texrepeat[1] = 1
            self.scn.geoms[self.scn.ngeom].emission = 0
            self.scn.geoms[self.scn.ngeom].specular = 0.5
            self.scn.geoms[self.scn.ngeom].shininess = 0.5
            self.scn.geoms[self.scn.ngeom].reflectance = 0
            self.scn.geoms[self.scn.ngeom].label = info["name"]
            self.scn.geoms[self.scn.ngeom].size[:] = info["size"]
            self.scn.geoms[self.scn.ngeom].rgba[:] = info["rgba"]
            self.scn.geoms[self.scn.ngeom].pos[:] = info["position"]
            self.scn.geoms[self.scn.ngeom].mat[:] = info["so3"]
            self.scn.geoms[self.scn.ngeom].type = mj.mjtGeom.__dict__["mjGEOM_" + info["geom_type"].upper()]
            self.scn.ngeom += 1

    def render_point_cloud(self, point_cloud):
        """
        Render a point cloud by adding markers for each point in the simulation viewer and updating existing markers.

        Args:
            point_cloud (np.array): 3D point cloud to be rendered in the simulation viewer.

        Note:
            This function assumes the simulation viewer has an add_marker and an update_marker_position method.
        """
        # Define the rotation, size, and color of the point markers
        so3 = euler2so3(z=0, x=0, y=0)
        size = [0.015, 0.015, 0.015]
        color = [1, 0, 0]
        rgba = np.concatenate((color, np.ones(1)))

        # Add new point markers and store their IDs if they haven't been added yet, otherwise, update their position.
        for idx, pos in enumerate(point_cloud.reshape(-1, 3).tolist()):
            if idx < len(self.pointcloud_marker_ids):
                self.update_marker_position(self.pointcloud_marker_ids[idx], pos)
            else:
                id = self.add_marker("sphere", "", pos, size, rgba, so3)
                self.pointcloud_marker_ids.append(id)

    def close(self):
        # If video recording is still active, save video before closing
        if len(self._video_frames) > 0:
            frames = [f for f in self._video_frames]
            time_stamp = time.strftime("%Y-%m-%d_%H%M%S")
            save_video(frames, self._video_path + time_stamp + ".mp4", self._fps)
        self.ctx.free()
        glfw.destroy_window(self.window)
        self.window = None
        self.is_alive = False
        # If video saving process not finished yet, join process
        if hasattr(self, "save_video_process"):
            self.save_video_process.join()


def save_video(frames, filename, fps):
    """
    Utility function for saving a video
    """
    writer = imageio.get_writer(filename, fps=fps, macro_block_size=None, ffmpeg_log_level="error")
    for f in frames:
        writer.append_data(f)
    writer.close()
