import os
import numpy as np
import matplotlib.pyplot as plt

from perlin_noise import PerlinNoise
from pathlib import Path
from scipy.ndimage import rotate


class Hfield:
    def __init__(self, nrow: int, ncol: int):
        """Prodive utlity functions to generate a range of height fields.
        """
        self.nrow, self.ncol = nrow, ncol
        # Create folder to save hfield files
        self.path = os.path.join(Path(__file__).parent, "hfield_files")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def create_flat(self):
        return np.zeros((self.nrow, self.ncol))

    def block(self):
        height_map = np.zeros((self.nrow, self.ncol))
        height_map[:, 205:250] = 1
        return height_map

    def create_bump(self, difficulty=0, resolution_x=20 / 400):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        # sparsely bump, bump width, bump height
        # closer bumps, bump distance
        # bump with holes
        height_map = np.zeros((self.nrow, self.ncol))
        bump_width = [0.5, 1.0]
        bump_height = [0.05 + 0.45 * difficulty, 0.4 + 0.2 * difficulty]
        bump_gap = [1.5, 2.5]
        x = np.random.uniform(70, 100)
        while x < self.ncol:
            width = int(np.random.uniform(*bump_width) / resolution_x)
            dx = [int(x), int(x + width)]
            height_map[:, dx[0]:dx[1]] = np.random.uniform(*bump_height)
            x += np.random.uniform(*bump_gap) / resolution_x + width
        return height_map

    def create_bump_single(self, difficulty=0, resolution_x=20 / 400):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        # sparsely bump, bump width, bump height
        # closer bumps, bump distance
        # bump with holes
        height_map = np.zeros((self.nrow, self.ncol))
        bump_width = [0.5, 1.0]
        bump_height = [0.3, 0.3]
        bump_gap = [1.5, 2.5]
        x = np.random.uniform(70, 100)
        height_map[:, 150:160] = np.random.uniform(*bump_height)
        # while x < self.ncol:
        #     width = int(np.random.uniform(*bump_width)/resolution_x)
        #     dx = [int(x), int(x+width)]
        #     height_map[:, dx[0]:dx[1]] = np.random.uniform(*bump_height)
        #     x += np.random.uniform(*bump_gap) / resolution_x + width
        return height_map

    def create_noisy(self):
        hgtmaps = []
        for i in range(5):
            fname = os.path.join(self.path, 'noisy-{}.npy'.format(i))
            if os.path.isfile(fname):
                hgtmaps += [np.load(fname)]
            else:
                print("First-time run: generating terrain map", fname)
                fn = PerlinNoise(octaves=np.random.uniform(5, 10), seed=np.random.randint(10000))
                height_map = [[fn([i/self.nrow, j/self.ncol]) for j in range(self.nrow)] for i in range(self.ncol)]
                height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
                height_map += np.random.uniform(0, 0.01, (self.nrow, self.ncol))
                hgtmaps += [height_map]
                np.save(fname, height_map)
        return hgtmaps[np.random.randint(len(hgtmaps))]

    def create_stair_random_up_down(self, resolution_x=20 / 400):
        step_height = [0.1, 0.17]
        step_length = [0.2, 0.5]
        height_map = np.zeros((self.nrow, self.ncol))
        num_steps = [10, 20]  # total number of steps in the staircase

        # define the pattern of going up and down
        pattern = np.random.choice([1, -1], size=num_steps, p=[0.5, 0.5])

        x = np.random.uniform(50, 150)
        h = 0
        for idx in range(np.random.randint(*num_steps)):
            sl = int(np.random.uniform(*step_length) / resolution_x)
            sh = np.random.uniform(*step_height) * pattern[0][idx]
            dx = [int(x), int(x + sl)]
            if sh + h <= 0:
                sh = 0
            height_map[:, dx[0]:dx[1]] = sh + h
            x += sl
            h += sh
        return height_map

    def create_stair_even(self, resolution_x=20 / 400, difficulty=0):
        # Stair height, length, and numebr of stairs in one cluster
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        step_height = [0.05 + 0.05 * difficulty, 0.1 + difficulty * 0.1]
        step_length = [0.4 - 0.15 * difficulty, 0.4]
        height_map = np.zeros((self.nrow, self.ncol))
        # define number of stair clusters on hfield
        num_clusters = np.random.randint(8, 12)

        x = 0
        for cluster in range(num_clusters):
            # total number of steps in each staircase cluster
            num_steps = int(np.random.uniform(4, 16 + difficulty * 12))
            # define the pattern of going up and down for each cluster
            pattern = np.repeat([1, -1], num_steps // 2)
            # find where the pattern change from 1 to -1
            change_idx = np.where(np.diff(pattern) == -2)[0]
            h = 0
            # x += int(np.random.uniform(1, 2) / resolution_x) if cluster > 0 else 0
            for idx in range(len(pattern)):
                sl = int(np.random.uniform(*step_length) / resolution_x)
                if idx == change_idx:  # longer step on peak or bottom
                    sl += np.random.uniform(0, 0.2) / resolution_x
                if pattern[idx] == -1:  # avoid too hard of inverted-knee on the way down
                    sl += 0.1 / resolution_x
                sh = np.random.uniform(*step_height) * pattern[idx]
                dx = [int(x), int(x + sl)]
                if sh + h <= 0:
                    sh = 0
                height_map[:, dx[0]:dx[1]] = sh + h
                x += sl
                h += sh
        return height_map

    def create_stair(self, resolution_x=20 / 400, difficulty=0, mode='up'):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        step_height = [0.1, 0.1 + difficulty * 0.2]
        step_length = [0.25, 0.4]
        height_map = np.zeros((self.nrow, self.ncol))
        x = 0
        if mode == 'up':
            h = 0
            x += 120
            while h < 30:
                sl = int(np.random.uniform(*step_length) / resolution_x)
                if mode == 'down':  # avoid too hard of inverted-knee on the way down
                    sl += int(np.random.uniform(0.1, 0.15) / resolution_x)
                sh = np.random.uniform(*step_height)
                dx = [int(x), int(x + sl)]
                if sh + h <= 0:
                    sh = 0
                height_map[:, dx[0]:dx[1]] = sh + h
                x += sl
                h += sh
        elif mode == 'down':
            h = 5
            x += 120
            height_map[:, 0:x] = h
            for i in range(50):
                sl = int(np.random.uniform(*step_length) / resolution_x)
                if mode == 'down':  # avoid too hard of inverted-knee on the way down
                    sl += int(np.random.uniform(0.1, 0.15) / resolution_x)
                sh = - np.random.uniform(*step_height)
                dx = [int(x), int(x + sl)]
                if sh + h <= 0:
                    sh = 0
                height_map[:, dx[0]:dx[1]] = sh + h
                x += sl
                h += sh
        return height_map

    def create_stone(self, resolution_x=20 / 400, resolution_y=20 / 400, difficulty=0):
        """Create a discrete terrain, paramterized by gap/step size in XY direction, and elevation
        in Z direction.
        """
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        base_height = 1
        init_platform = 90
        step_height = [0, 0.2 + difficulty * 0.2]
        step_length = [0.3 - difficulty * 0.1, 0.4]
        gap_length = [0.1, 0.15 + difficulty * 0.2]
        height_map = np.zeros((self.nrow, self.ncol))
        height_map[:, 0:init_platform] = base_height
        x = init_platform
        sh = 0
        while True:
            sl_x = int(np.random.uniform(*step_length) / resolution_x)
            gap_x = int(np.random.uniform(*gap_length) / resolution_x)
            sh = np.random.uniform(*step_height)
            dx = [int(x), int(x + sl_x)]
            height_map[:, dx[0]:dx[1]] = base_height + sh
            height_map[:, dx[1]:dx[1] + gap_x] = 0.0
            x += sl_x + gap_x
            if x >= self.ncol:
                break
        y = 0
        while True:
            sl_y = int(np.random.uniform(*step_length) / resolution_y)
            gap_y = int(np.random.uniform(*gap_length) / resolution_x)
            dy = [int(y), int(y + gap_y)]
            height_map[dy[0]:dy[1], init_platform:] = 0
            y += sl_y + gap_y
            if y >= self.nrow:
                break
        return height_map

    def create_block(self, base_pos, difficulty=0, resolution_x=20 / 400, resolution_y=20 / 400):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        height_map = np.zeros((self.nrow, self.ncol))
        # width = [0.3, 1]
        width = [0.5, 1 - 0.5 * difficulty]
        # height = [0.0, 0.2 + 0.3 * difficulty]
        height = [0.0, 0.3 * difficulty]
        widthy = int(np.random.uniform(*width) / resolution_y)
        x = 0
        while x < self.ncol:
            widthx = int(np.random.uniform(*width) / resolution_x)
            dx = [int(x), int(x + widthx)]
            y = 0
            while y < self.nrow:
                widthy = int(np.random.uniform(*width) / resolution_y)
                height_map[dx[0]:dx[1], y:y + widthy] = np.random.uniform(*height)
                y += widthy
            x += widthx
        # x_pixel = self.nrow // 2 + int(base_pos[0] / 10 * self.nrow / 2)
        # y_pixel = self.ncol // 2 + int(base_pos[1] / 10 * self.ncol / 2)
        # height_map[y_pixel-10:y_pixel+10, x_pixel-10:x_pixel+10] = 0.0
        return height_map

    def create_platform(self, base_pos, difficulty=0, resolution_x=20 / 400, resolution_y=20 / 400):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        height_map = np.zeros((self.nrow, self.ncol))
        width, height = 400, 400
        width_range = [1 - difficulty * 0.6, 1]
        breath_range = [1 - difficulty * 0.6, 1]
        # used for bootstrap training
        min_height, max_height = 0.05 + 0.15 * difficulty, 0.2 + 0.2 * difficulty
        num_rectangles = 300
        min_rotation, max_rotation = 0, 0

        height_map = np.zeros((height, width))
        for _ in range(num_rectangles):
            breadth = int(np.random.uniform(*breath_range) / resolution_y)
            width = int(np.random.uniform(*width_range) / resolution_x)
            if np.random.uniform() < 1.0:
                height = np.random.uniform(min_height, max_height)
            else:
                height = np.random.uniform(0.2, 0.3)
            rotation = np.random.uniform(min_rotation, max_rotation)

            x = np.random.randint(0, height_map.shape[1] - width)
            y = np.random.randint(0, height_map.shape[0] - breadth)

            rect = np.ones((breadth, width)) * height
            rect = rotate(rect, rotation, reshape=True)
            expand_breadth, expand_width = rect.shape

            if y + expand_breadth < self.nrow and x + expand_width < self.ncol:
                if height_map[y:y + expand_breadth, x:x + expand_width].any() > 0:
                    continue
                else:
                    height_map[y:y + expand_breadth, x:x + expand_width] = rect
            else:
                continue
        return height_map

    def create_single_block(self, base_pos, difficulty=0, resolution_x=20 / 400, resolution_y=20 / 400):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        height_map = np.zeros((self.nrow, self.ncol))
        width = [0.3, 0.8]
        height = [0.1, 0.2]
        x_pixel = self.nrow // 2
        y_pixel = self.ncol // 2
        x_width = int(np.random.uniform(*width) / resolution_x / 2)
        y_width = int(np.random.uniform(*width) / resolution_y / 2)
        height_map[y_pixel - y_width:y_pixel + y_width, x_pixel - x_width:x_pixel + x_width] = np.random.uniform(
            *height)
        return height_map

    def flat_spawn_hfield(self, base_pos, height_map, z_offset):
        x_pixel = self.nrow // 2 + int(base_pos[0] / 10 * self.nrow / 2)
        y_pixel = self.ncol // 2 + int(base_pos[1] / 10 * self.ncol / 2)
        height_map[y_pixel - 8:y_pixel + 8, x_pixel - 8:x_pixel + 8] = z_offset
        return height_map
