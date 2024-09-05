import mujoco as mj
import numpy as np
import time

from sim.cassie_sim.lib_cassiesim import LibCassieSim
from sim.cassie_sim.mj_cassiesim import MjCassieSim
from sim.digit_sim.mj_digitsim import MjDigitSim
from sim.mujoco_viewer import MujocoViewer

from .common import (
    DIGIT_MOTOR_NAME,
    DIGIT_JOINT_NAME,
    CASSIE_MOTOR_NAME,
    CASSIE_JOINT_NAME,
)

from util.colors import FAIL, ENDC, OKGREEN
from util.quaternion import mj2scipy, scipy2mj
from scipy.spatial.transform import Rotation as R

def test_all_sim():
    sim_list = [LibCassieSim, MjCassieSim, MjDigitSim]
    num_pass = 0
    failed = False
    for sim in sim_list:
        num_pass = 0
        print(f"Testing {sim.__name__}")
        num_pass += test_sim_init(sim)
        num_pass += test_sim_sim_forward(sim)
        num_pass += test_sim_viewer_marker(sim)
        num_pass += test_sim_viewer(sim)
        num_pass += test_sim_glfw_multiple_viewer(sim)
        num_pass += test_sim_PD(sim)
        num_pass += test_sim_get_set(sim)
        num_pass += test_sim_indexes(sim)
        num_pass += test_sim_body_pose(sim)
        num_pass += test_sim_body_velocity(sim)
        num_pass += test_sim_body_acceleration(sim)
        num_pass += test_sim_body_contact_force(sim)
        num_pass += test_sim_relative_pose(sim)
        num_pass += test_sim_hfield(sim)
        num_pass += test_self_collision(sim)
        if num_pass == 15:
            print(f"{OKGREEN}{sim.__name__} passed all tests.{ENDC}")
        else:
            failed = True
            print(f"{FAIL}{sim.__name__} failed, only passed {num_pass} out of 15 tests.{ENDC}")
        num_pass = 0
    if not failed:
        print(f"{OKGREEN}Passed all sim tests! \u2713{ENDC}")

def test_sim_init(sim):
    print("Making sim")
    test_sim = sim()
    test_sim.reset()
    print("Passed made sim")
    return True

def test_sim_sim_forward(sim):
    print("Testing sim forward")
    test_sim = sim()
    test_sim.reset()
    for i in range(100):
        test_sim.sim_forward()
    test_sim.sim_forward(dt = 0.5)
    print("Passed sim forward")
    return True

def test_sim_viewer(sim):
    print("Testing sim viewer, quit window to continue")
    test_sim = sim()
    test_sim.reset()
    test_sim.viewer_init()
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        if not test_sim.viewer_paused():
            for _ in range(50):
                test_sim.sim_forward()
        render_state = test_sim.viewer_render()
        # Assume 2kHz sim for now
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    print("Passed sim viewer")
    return True

def test_sim_viewer_marker(sim):
    if "lib" in sim.__name__.lower():
        print("Bypass libcassie for viewer marker test.")
        return True
    print("Testing sim viewer marker rendering, don't quit window yet")
    test_sim = sim()
    test_sim.reset()
    test_sim.viewer_init()
    render_state = test_sim.viewer_render()
    so3 = R.from_euler('xyz', [0, 0.4, 0.3]).as_matrix()
    test_sim.viewer.add_marker("sphere", "foo", [1, 0, 1], [0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 1.0], so3)
    test_sim.viewer.add_marker("sphere", "foo2", [1, 0, 1.3], [0.1, 0.1, 0.1], [0.1, 0.8, 0.1, 1.0], so3)
    count = 0
    while render_state:
        start_t = time.time()
        if not test_sim.viewer_paused():
            for _ in range(50):
                test_sim.sim_forward()
            count += 1
            if count == 50:
                test_sim.viewer.update_marker_type(0, "box")
                test_sim.viewer.update_marker_name(0, "new_marker")
                test_sim.viewer.update_marker_position(0, [0.5, 0, 1])
                test_sim.viewer.update_marker_size(0, [0.05, 0.01, 0.1])
                test_sim.viewer.update_marker_rgba(0, [0.1, 0.1, 0.8, 1.0])
                so3 = R.from_euler('xyz', [0.4, 0.3, 0.0]).as_matrix()
                test_sim.viewer.update_marker_so3(0, so3)
                test_sim.viewer.remove_marker(1)
                so3 = R.from_euler('xyz', [0, 0, 0]).as_matrix()
                test_sim.viewer.add_marker("capsule", "foo3", [1, 0, 1.3], [0.1, 0.1, 0.6], [0.1, 0.8, 0.1, 1.0], so3)
                test_sim.viewer.add_marker("arrow", "arrow", [0.7, -0.2, 1.0], [0.03, 0.03, 0.7], [0.1, 0.1, 0.8, 1.0], so3)
                print("Marker changes done, you can quit the window now")
        render_state = test_sim.viewer_render()
        # Assume 2kHz sim for now
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    print("Passed sim viewer marker")
    return True

def test_sim_glfw_multiple_viewer(sim):
    if "lib" in sim.__name__.lower():
        print("Bypass libcassie for dual window render.")
        return True
    print("Testing sim viewer, quit window to continue")
    # TODO: when closing one window, the other window will be black, because we are free mjContext
    # when closing one window in mjViewer.close(). Need to fix this.
    test_sim = sim()
    test_sim.reset()
    test_sim.viewer_init(width=800, height=800)
    vis1 = test_sim.viewer
    # Create a second viewer that reads sim
    vis2 = MujocoViewer(test_sim.model, test_sim.data, test_sim.reset_qpos, \
        camera_id='forward-chest-realsense-d435/depth/image-rect', width=400, height=400)
    rs1 = vis1.render()
    rs2 = vis2.render()
    while rs1 or rs2:
        start_t = time.time()
        if rs1 and rs2:
            if not vis1.paused and not vis2.paused:
                for _ in range(50):
                    test_sim.sim_forward()
            rs1 = vis1.render()
            rs2 = vis2.render()
        elif rs1:
            if not vis1.paused:
                for _ in range(50):
                    test_sim.sim_forward()
            rs1 = vis1.render()
        elif rs2:
            if not vis1.paused:
                for _ in range(50):
                    test_sim.sim_forward()
            rs2 = vis2.render()
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    print("Passed sim viewer")
    return True

def test_sim_PD(sim):
    print("Testing sim PD")
    test_sim = sim()
    test_sim.reset()
    test_sim.set_base_position(np.array([0, 0, 1.5]))
    test_sim.hold()
    for _ in range(1000):
        test_sim.sim_forward()
    test_sim.viewer_init()
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        for _ in range(50):
            test_sim.set_PD(test_sim.offset, np.zeros(test_sim.num_actuators), test_sim.kp, test_sim.kd)
            test_sim.sim_forward()
        render_state = test_sim.viewer_render()
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    test_sim.release()
    if np.any((test_sim.get_motor_position() - test_sim.offset) > 1e-1):
        print(f"{FAIL}Failed sim PD test. Motor positions not close enough to target.{ENDC}")
        return False
    else:
        print("Passed sim PD")
        return True

def test_sim_drop(sim):
    print("Testing sim PD")
    test_sim = sim()
    test_sim.reset()
    test_sim.set_base_position(np.array([0, 0, 1.5]))
    test_sim.viewer_init()
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        if not test_sim.viewer_paused():
            for _ in range(50):
                test_sim.set_PD(test_sim.offset, np.zeros(test_sim.num_actuators), test_sim.kp, test_sim.kd)
                test_sim.sim_forward()
        render_state = test_sim.viewer_render()
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    return True

def test_sim_get_set(sim):
    print("Testing sim getter and setter functions")
    motor_name = CASSIE_MOTOR_NAME if 'cassie' in sim.__name__.lower() else DIGIT_MOTOR_NAME
    test_sim = sim()
    test_sim.reset()
    # Test getters
    test_sim.get_joint_position()
    test_sim.get_joint_velocity()
    test_sim.get_base_position()
    test_sim.get_base_linear_velocity()
    test_sim.get_base_orientation()
    test_sim.get_base_angular_velocity()
    test_sim.get_torque()
    test_sim.get_body_mass()
    test_sim.get_body_mass(name=motor_name[0])
    test_sim.get_dof_damping()
    test_sim.get_dof_damping(name=motor_name[0])
    test_sim.get_geom_friction()
    test_sim.get_geom_friction(name="floor")
    test_sim.get_body_ipos()
    test_sim.get_body_ipos(name=motor_name[0])

    # Test setters
    test_sim.set_joint_position(np.zeros(test_sim.num_joints))
    test_sim.set_joint_velocity(np.zeros(test_sim.num_joints))
    test_sim.set_base_position(np.ones(3))
    test_sim.set_base_linear_velocity(np.zeros(3))
    test_sim.set_base_orientation(np.array([0, 0.6987058, 0.2329019, 0.6764369]))
    test_sim.set_base_angular_velocity(np.zeros(3))
    test_sim.set_torque(np.zeros(test_sim.num_actuators))
    test_sim.set_body_mass(np.zeros(test_sim.nbody))
    test_sim.set_body_mass(0, name=motor_name[0])
    test_sim.set_dof_damping(np.zeros(test_sim.nv))
    test_sim.set_dof_damping(1, name=motor_name[0])
    test_sim.set_geom_friction(np.zeros((test_sim.ngeom, 3)))
    test_sim.set_geom_friction(np.zeros(3), name="floor")
    test_sim.set_body_ipos(np.zeros((test_sim.nbody, 3)))
    test_sim.set_body_ipos(np.zeros(3), name=motor_name[0])

    print("Pass sim getter and setter functions")
    return True

def test_sim_indexes(sim):
    test_sim = sim()
    test_sim.reset()

    motor_name = CASSIE_MOTOR_NAME if 'cassie' in sim.__name__.lower() else DIGIT_MOTOR_NAME
    joint_name = CASSIE_JOINT_NAME if 'cassie' in sim.__name__.lower() else DIGIT_JOINT_NAME

    motor_position_inds=[]
    motor_velocity_inds=[]
    for m in motor_name:
        motor_position_inds.append(test_sim.get_joint_qpos_adr(m))
        motor_velocity_inds.append(test_sim.get_joint_dof_adr(m))

    joint_position_inds=[]
    joint_velocity_inds=[]
    for m in joint_name:
        joint_position_inds.append(test_sim.get_joint_qpos_adr(m))
        joint_velocity_inds.append(test_sim.get_joint_dof_adr(m))

    assert motor_position_inds == test_sim.motor_position_inds, "Mismatch between motor_position_inds!"
    assert motor_velocity_inds == test_sim.motor_velocity_inds, "Mismatch between motor_velocity_inds!"
    assert joint_position_inds == test_sim.joint_position_inds, "Mismatch between joint_position_inds!"
    assert joint_velocity_inds == test_sim.joint_velocity_inds, "Mismatch between joint_velocity_inds!"

    print("Pass indices test")
    return True

def test_sim_body_pose(sim):
    test_sim = sim()
    test_sim.reset()
    x_target = np.array([0, 0, 1.5, 0, 0, 0, 1])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()
    test_sim.sim_forward(dt=0.1)
    x = test_sim.get_body_pose(name=test_sim.base_body_name)
    assert np.linalg.norm(x[:3] - x_target[:3]) < 1e-2, f"get_body_pose returns base at {x[:3]}, but sim sets to {x_target[:3]}."
    assert 1 - np.inner(x[3:], x_target[3:]) < 1e-2, f"get_body_pose returns base at {x[3:]}, but sim sets to {x_target[3:]}."

    test_sim.release()
    test_sim.sim_forward(dt=1)

    x_target = np.array([10, 10, 2, 0, 1, 0, 0])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()
    test_sim.sim_forward(dt=0.1)
    x = test_sim.get_body_pose(name=test_sim.base_body_name)
    assert np.linalg.norm(x[:3] - x_target[:3]) < 1e-2, f"get_body_pose returns base at {x[:3]}, but sim sets to {x_target[:3]}."
    assert 1 - np.inner(x[3:], x_target[3:]) < 1e-2, f"get_body_pose returns base at {x[3:]}, but sim sets to {x_target[3:]}."

    print("Passed sim get body pose")
    return True

def test_sim_body_velocity(sim):
    test_sim = sim()
    test_sim.reset()
    # NOTE: use small velocity here to avoid creating out of axis velocities, ie, if the torso has
    # inertia like Digit, set dtheta_x=1 will cause other axis to have velocities even sim 1 step.
    dx_target = np.array([1, 0, 0, 0, 0, 0])
    test_sim.set_base_linear_velocity(dx_target[:3])
    test_sim.set_base_angular_velocity(dx_target[3:])
    test_sim.sim_forward()
    dx = test_sim.get_body_velocity(name=test_sim.base_body_name)
    assert np.linalg.norm(dx[:3] - dx_target[:3]) < 1e-1, f"get_body_velocity returns base at {dx[:3]}, but sim sets to {dx_target[:3]}."
    assert np.linalg.norm(dx[3:] - dx_target[3:]) < 1e-1, f"get_body_velocity returns base at {dx[3:]}, but sim sets to {dx_target[3:]}."

    test_sim.release()
    test_sim.sim_forward(dt=1)

    dx_target = np.array([0, 0, 0.1, 0, 0, 0])
    test_sim.set_base_linear_velocity(dx_target[:3])
    test_sim.set_base_angular_velocity(dx_target[3:])
    test_sim.sim_forward()
    dx = test_sim.get_body_velocity(name=test_sim.base_body_name)
    assert np.linalg.norm(dx[:3] - dx_target[:3]) < 1e-1, f"get_body_velocity returns base at {dx[:3]}, but sim sets to {dx_target[:3]}."
    assert np.linalg.norm(dx[3:] - dx_target[3:]) < 1e-1, f"get_body_velocity returns base at {dx[3:]}, but sim sets to {dx_target[3:]}."

    print("Passed sim get body velocity")
    return True

def test_sim_body_acceleration(sim):
    test_sim = sim()
    test_sim.reset()
    test_sim.hold()
    test_sim.sim_forward(dt=3)
    ddx = test_sim.get_body_acceleration(name=test_sim.base_body_name)
    assert np.linalg.norm(ddx[:2]) < 1e-1, f"get_body_acceleration: robot should not have XY accelerations."
    assert np.abs(ddx[2] - 9.80665) < 1e-2, f"get_body_acceleration: gravity messed up."
    assert np.linalg.norm(ddx[3:]) < 1e-1, f"get_body_acceleration: robot should not have rotational accelerations."
    print("Passed sim get body acceleration")
    return True

def test_sim_body_contact_force(sim):
    """Hold robot in the air while feet touching ground. Check contact forces from each foot.
    Then drop the robot and check if floating base body gets contact forces.
    """
    test_sim = sim(fast=False)
    test_sim.reset()
    # Slightly tilted down to let base falling to ground
    x_target = np.array([0, 0, 1, 0.9961947, 0, 0.0871557, 0])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()
    test_sim.sim_forward(dt=1)
    force = test_sim.get_body_contact_force(name=test_sim.feet_body_name[0])
    assert np.linalg.norm(force) > 10 , "get_body_contact_force returns wrong forces."
    force = test_sim.get_body_contact_force(name=test_sim.feet_body_name[1])
    assert np.linalg.norm(force) > 10 , "get_body_contact_force returns wrong forces."

    test_sim.release()
    test_sim.sim_forward(dt=2)
    force = test_sim.get_body_contact_force(name=test_sim.base_body_name)
    assert np.linalg.norm(force) > 10 , "get_body_contact_force returns wrong forces."

    print("Passed sim get body contact force")
    return True

def test_sim_relative_pose(sim):
    """Tilt torso/base + 10deg in pitch and measure feet flat (should be -10deg pitch)
    on ground angle diff in base frame.
    """
    test_sim = sim()
    test_sim.reset()
    # Slightly tilted down
    x_target = np.array([0, 0, 1, 0.9961947, 0, 0.0871557, 0])
    test_sim.set_base_position(x_target[:3])
    test_sim.set_base_orientation(x_target[3:])
    test_sim.hold()

    # Digit torso requires onger time to settle after torso/base joint set to 10deg
    dt = 5 if "digit" in sim.__name__.lower() else 1
    test_sim.sim_forward(dt=dt)
    p1 = test_sim.get_body_pose(name=test_sim.base_body_name)
    p2 = test_sim.get_site_pose(name=test_sim.feet_site_name[0])
    p3 = test_sim.get_site_pose(name=test_sim.feet_site_name[1])
    lfoot_in_base = test_sim.get_relative_pose(p1, p2)
    rfoot_in_base = test_sim.get_relative_pose(p1, p3)
    lfoot_euler = R.from_quat(mj2scipy(lfoot_in_base[3:7])).as_euler('xyz', degrees=True)
    rfoot_euler = R.from_quat(mj2scipy(rfoot_in_base[3:7])).as_euler('xyz', degrees=True)
    x_target_euler = R.from_quat(mj2scipy(x_target[3:7])).as_euler('xyz', degrees=True)
    assert lfoot_euler[1] + x_target_euler[1] < 1e-1 , "get_relative_pose returns wrong lfoot angles."
    assert rfoot_euler[1] + x_target_euler[1] < 1e-1 , "get_relative_pose returns wrong rfoot angles."

    # NOTE: left for testing purposes
    # test_sim.viewer_init()
    # render_state = test_sim.viewer_render()
    # while render_state:
    #     start_t = time.time()
    #     for _ in range(50):
    #         x1 = test_sim.get_body_pose(name=test_sim.base_body_name)
    #         x2 = test_sim.get_site_pose(name=test_sim.feet_site_name[0])
    #         x3 = test_sim.get_relative_pose(x1, x2)
    #         print("left foot relative to base angles, ", quaternion2euler(x3[3:7])/np.pi*180)
    #         print("left foot relative to base positions, ", x3[:3])
    #         print()
    #         test_sim.sim_forward()
    #     render_state = test_sim.viewer_render()
    #     delaytime = max(0, 50/2000 - (time.time() - start_t))
    #     time.sleep(delaytime)

    print("Passed sim get_relative_pose")
    return True

def test_sim_hfield(sim):
    if "lib" in sim.__name__.lower():
        print("Bypass libcassie for dual window render.")
        return True
    print("Testing sim hfield, quit window to continue")
    test_sim = sim(terrain='hfield')
    test_sim.viewer_init()
    test_sim.reset()
    test_sim.randomize_hfield(hfield_type='noisy')
    # Define local heightmap constants
    map_x, map_y = 1.5, 1 # m
    map_points_x, map_points_y = 30, 20 # pixels
    map_dim = int(map_points_x*map_points_y)
    x = np.linspace(-map_x/2, map_x/2, num=map_points_x, dtype=float)
    y = np.linspace(-map_y/2, map_y/2, num=map_points_y, dtype=float)
    grid_x, grid_y = np.meshgrid(x, y)
    heightmap_num_points = grid_x.size
    local_grid_unrotated = np.zeros((heightmap_num_points, 3))
    local_grid_unrotated[:, 0] = grid_x.flatten()
    local_grid_unrotated[:, 1] = grid_y.flatten()
    # Initialize heightmap and visualization
    hfield_map, local_grid_rotated = test_sim.get_hfield_map(grid_unrotated=local_grid_unrotated)
    vis_marker_keys = {'heightmap':[]}
    for i in range(map_dim):
        x, y, z = local_grid_rotated[i][0], local_grid_rotated[i][1], hfield_map[i] + 0.02
        so3 = R.from_euler('xyz', [0, 0, 0]).as_matrix()
        id = test_sim.viewer.add_marker("sphere", "", [x,y,z],
            [0.015, 0.015, 0.005], [0.8, 1, 0.8, 1.0], so3)
        vis_marker_keys['heightmap'].append(id)
    render_state = test_sim.viewer_render()
    while render_state:
        start_t = time.time()
        if not test_sim.viewer_paused():
            for _ in range(50):
                test_sim.sim_forward()
        # Update heightmap visualization
        hfield_map, local_grid_rotated = test_sim.get_hfield_map(grid_unrotated=local_grid_unrotated)
        for i in range(map_dim):
            x, y, z = local_grid_rotated[i][0], local_grid_rotated[i][1], hfield_map[i] + 0.02
            test_sim.viewer.update_marker_position(vis_marker_keys['heightmap'][i], [x,y,z])
        render_state = test_sim.viewer_render()
        # Assume 2kHz sim for now
        delaytime = max(0, 50/2000 - (time.time() - start_t))
        time.sleep(delaytime)
    print("Passed sim viewer")
    return True

def test_self_collision(sim):
    if hasattr(sim, 'is_self_collision'):
        test_sim = sim(fast=False)
        test_sim.reset()
        assert test_sim.is_self_collision() == False, f"robot at default pose should be collision-free, is_self_collsion() returned True"
        #expand all geoms to ensure robot is in self collision
        test_sim.model.geom_size = [np.array([1000, 1000, 1000]) for _ in test_sim.model.geom_size]
        test_sim.sim_forward()
        assert test_sim.is_self_collision() == True, f"robot should be in self collision state after increasing size of all geoms, but is_self_collision() returned False"

        print("Passed sim test self collision")
    return True