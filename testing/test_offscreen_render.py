"""Test file for rendering from Mujoco.Renderer class with Cassie/Digit model.
1. When OFFSCREEN is True, the rendering is done in a headless environment using EGL.
A render profiling is plotted vs render resolution.
2. When OFFSCREEN is False, the rendering is done in a dual-window using GLX.
Particularly, the first window is the default mujoco viewer, and the second window is a fixed view.
"""

def test_offscreen_rendering():

    OFFSCREEN = input("Offscreen rendering? (y/n): ").lower() == 'y'
    TERRAIN = input("Terrain? (flat/hfield): ").lower()
    gl_option = 'egl' if OFFSCREEN else 'glx'

    # Need to update env variable before import mujoco
    import os
    os.environ['MUJOCO_GL']=gl_option

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import mediapy
    from sim import MjCassieSim
    from util.camera_util import crop_from_center
    import cv2

    # Check if the env variable is correct
    if "MUJOCO_GL" in os.environ:
        assert os.getenv('MUJOCO_GL') == gl_option,\
               f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
        print("PYOPENGL_PLATFORM =",os.getenv('PYOPENGL_PLATFORM'))

    camera_name = "forward-chest-realsense-d435/depth/image-rect"
    size = [25, 50, 100, 150, 200, 250, 300, 350, 400, 480] if OFFSCREEN else [[1280, 720]]
    sim_time_avg_list = []
    render_time_avg_list = []
    time_render_ratio = []
    for s in size:
        print(f"Testing resolution {s}x{s}")
        # Init mujoco sim
        sim = MjCassieSim(terrain='hfield', fast=False) if TERRAIN == 'hfield' else MjCassieSim(fast=False)
        sim.reset()
        if TERRAIN == 'flat':
            sim.geom_generator._create_geom('box0', *[1, 0, 0], rise=0.3, length=0.3, width=0.3)
        elif TERRAIN == 'hfield':
            sim.geom_generator._create_geom('box0', *[1, 0, 0], rise=0.3, length=0.3, width=0.3)
            # Upload hfield to GPU by calling randomize_hfield
            sim.randomize_hfield(hfield_type='bump')
        # A final call on mj_forward to adjust robot pose and update scene
        sim.adjust_robot_pose()

        # Init renderer that reads the same model/data
        if OFFSCREEN:
            # Init renderer that reads the same model/data
            sim.init_renderer(width=s, height=s, offscreen=OFFSCREEN)
            render_state = True
        else:
            sim.viewer_init(height=1024, width=960)
            # Create a second viewer that renders a different view
            sim.init_renderer(width=s[0], height=s[1], offscreen=OFFSCREEN)
            render_state = sim.viewer_render()

        time_raw_sim_list = []
        time_render_depth = []
        depth_frames = []
        rgb_frames = []
        while render_state:
            paused = False if OFFSCREEN else sim.viewer_paused()
            if not paused:
                for _ in range(50):
                    start = time.time()
                    sim.sim_forward()
                    time_raw_sim_list.append(time.time() - start)
            if not OFFSCREEN:
                render_state = sim.viewer_render()
            start_t = time.time()
            depth = sim.get_render_image(camera_name, 'depth')
            rgb = sim.get_render_image(camera_name, 'rgb')
            time_render_depth.append(time.time() - start_t)
            depth_frames.append(depth)
            rgb_frames.append(rgb)
            if not OFFSCREEN:
                depth = crop_from_center(depth, 720, 720)
                # For visualization purpose and scale depth for better contrast
                depth /= depth.max()
                depth = np.clip(depth, 0, 1) * 255
                rgb = rgb.copy()
                cv2.namedWindow('rgb', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('rgb', rgb)
                cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('depth', depth.astype(np.uint8))
                cv2.waitKey(1)
            if sim.get_base_position()[2] < 0.5:
                mean_time_sim = np.mean(np.array(time_raw_sim_list))
                mean_time_render = np.mean(np.array(time_render_depth))
                sim_time_avg_list.append(mean_time_sim)
                render_time_avg_list.append(mean_time_render)
                time_render_ratio.append(100*mean_time_render / (mean_time_sim+mean_time_render))
                if not OFFSCREEN:
                    sim.renderer.close()
                    sim.viewer.close()
                    cv2.destroyAllWindows()
                break
    if OFFSCREEN:
        mediapy.write_video(path="test_depth.gif", images=depth_frames, fps=50, codec='gif')
        mediapy.write_video(path="test_rgb.gif", images=rgb_frames, fps=50, codec='gif')
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2)
        ax1.plot(size, sim_time_avg_list)
        ax1.set_title('sim time [s] per policy step')
        ax2.plot(size, render_time_avg_list)
        ax2.set_title('render time [s] per policy step')
        ax3.plot(size, time_render_ratio)
        ax3.set_title('render ratio [%] per policy step')
        wall_time = [t * 300e6 / 3600 for t in render_time_avg_list]
        ax4.plot(size, wall_time)
        ax4.set_title('additional walk clock time [hours]\nfor 300M samples')
        ax4.set_xlabel('image size')
        fig.tight_layout()
        plt.show()
    print("Passed offscreen rendering test!")


"""Test file for Onscreen pointcloud rendering from Mujoco. Renderer class with Cassie/Digit model.
"""
def test_pointcloud_rendering():

	OFFSCREEN = 0
	gl_option = 'egl' if OFFSCREEN else 'glx'

	# Need to update env variable before import mujoco
	import os
	os.environ['MUJOCO_GL']=gl_option

	import time
	from sim import MjCassieSim
	import cv2

	# Check if the env variable is correct
	if "MUJOCO_GL" in os.environ:
		assert os.getenv('MUJOCO_GL') == gl_option,\
			   f"GL option is {os.getenv('MUJOCO_GL')} but want to load {gl_option}."
		print("PYOPENGL_PLATFORM =",os.getenv('PYOPENGL_PLATFORM'))

	camera_name = "forward-chest-realsense-d435/depth/image-rect"
	size = [[420, 240]]
	for s in size:
		# Init mujoc sim and optional viewer to verify
		sim = MjCassieSim(fast=False)
		sim.reset()
		sim.geom_generator._create_geom('box0', *[1, 0, 0], rise=0.3, length=0.3, width=0.3)
		sim.adjust_robot_pose()
		if OFFSCREEN:
			# Init renderer that reads the same model/data
			sim.init_renderer(width=s, height=s, offscreen=OFFSCREEN)
			render_state = True
		else:
			sim.viewer_init(height=1024, width=960)
			render_state = sim.viewer_render()
			# Create a second viewer that renderes a different view
			sim.init_renderer(width=s[0], height=s[1], offscreen=OFFSCREEN)

		while render_state:
			paused = False if OFFSCREEN else sim.viewer_paused()
			if not paused:
				for _ in range(50):
					sim.sim_forward()
			if not OFFSCREEN:
				render_state = sim.viewer_render()
			depth = sim.get_render_image(camera_name)

			start_time = time.time()
			pcl = sim.get_point_cloud(camera_name, depth, 10)
			# pcl = sim.get_point_cloud_in_camera_frame(camera_name, depth, 10)
			end_time = time.time()
			print("GET POINTCLOUD: ", end_time - start_time)
			start_render = time.time()
			sim.viewer.render_point_cloud(pcl)
			end_render = time.time()
			print("PC RENDER TIME: ", end_render - start_render)

			cv2.namedWindow("Depth", cv2.WINDOW_AUTOSIZE)
			cv2.imshow("Depth", depth)
			cv2.waitKey(1)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	print("Passed ONSCREEN pointcloud rendering test!")
