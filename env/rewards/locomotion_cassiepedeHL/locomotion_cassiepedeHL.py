from scipy.spatial.transform import Rotation as R

from env.util.quaternion import *


def kernel(x):
    return np.exp(-x)


def compute_rewards(self, action):
    rewards = []
    for i in range(self.num_cassie):
        q = {}

        ### Speed rewards ###
        poi_vel = self.get_poi_linear_velocity(local_frame=False)
        # This is in global frame

        target_vel_in_local = np.array([self.x_velocity_poi, self.y_velocity_poi, 0])

        quat = R.from_euler('xyz', [0, 0, self.orient_add[0]])
        target_vel = quat.apply(target_vel_in_local)

        x_vel = np.abs(poi_vel[0] - target_vel[0])
        y_vel = np.abs(poi_vel[1] - target_vel[1])

        ### Orientation rewards
        poi_orient = self.get_poi_orientation()
        target_quat = np.array([1, 0, 0, 0])
        if self.orient_add != 0:
            command_quat = R.from_euler('xyz', [0, 0, self.orient_add[0]])
            target_quat = R.from_quat(us2scipy(target_quat)) * command_quat
            target_quat = scipy2us(target_quat.as_quat())

        orientation_error = quaternion_distance(poi_orient, target_quat)

        if orientation_error < 5e-3:
            orientation_error = 0

        q["orientation"] = orientation_error

        q['x_vel'] = x_vel  # / (1 - orientation_error + 1e-6)
        q['y_vel'] = y_vel  # / (1 - orientation_error + 1e-6)

        rewards.append(q)

    return rewards


# Termination condition: If orientation too far off terminate
def compute_done(self):
    dones = []
    base_positions = self.get_base_position()
    base_orientations = self.get_base_orientation()
    poi_orientation = R.from_quat(us2scipy(self.get_poi_orientation())).as_euler('xyz')

    # Check if the orientation of deck is too far off
    if np.abs(poi_orientation[1]) > 20 / 180 * np.pi or np.abs(poi_orientation[0]) > 20 / 180 * np.pi:
        return np.ones(self.num_cassie, dtype=bool)

    for i in range(self.num_cassie):
        base_position = base_positions[i]
        base_orientation = base_orientations[i]
        base_height = base_position[2]
        base_euler = R.from_quat(us2scipy(base_orientation)).as_euler('xyz')
        height_limit = 0.65
        if np.abs(base_euler[1]) > 20 / 180 * np.pi or np.abs(
                base_euler[0]) > 20 / 180 * np.pi or base_height < height_limit:
            dones.append(True)
        else:
            dones.append(False)
    return np.array(dones, dtype=bool)
