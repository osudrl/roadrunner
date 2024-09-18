import os
import numpy as np

from perlin_noise import PerlinNoise
from pathlib import Path

class Hfield:
    def __init__(self, nrow: int, ncol: int):
        """Prodive utlity functions to generate a range of height fields.
        """
        self.nrow, self.ncol = nrow, ncol
        # Double check if these hfields exist in XML asset
        self.hfield_names = ['flat', 'noisy', 'bump', 'stone', 'stair']
        # Create folder to save hfield files
        self.path = os.path.join(Path(__file__).parent, "hfield_files")
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def create_flat(self):
        return np.zeros((self.nrow, self.ncol))

    def create_bump(self, difficulty=0.5):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        height_map = np.zeros((self.nrow, self.ncol))
        num_bumps = [3, 10]
        bump_width = [3, 15]
        bump_height = [0.0, 0.5 * difficulty]
        x = np.random.uniform(70, 100)
        for _ in range(np.random.randint(*num_bumps)):
            width = np.random.uniform(*bump_width)
            dx = [int(x), int(x+width)]
            height_map[:, dx[0]:dx[1]] = np.random.uniform(*bump_height)
            x += np.random.uniform(30, 60)
        return height_map

    def create_noisy(self):
        hgtmaps = []
        for i in range(3):
            fname = os.path.join(self.path, 'noisy-{}.npy'.format(i))
            if os.path.isfile(fname):
                hgtmaps += [np.load(fname)]
            else:
                print("First-time run: generating terrain map", fname)
                fn = PerlinNoise(octaves=np.random.uniform(5, 10), seed=np.random.randint(10000))
                height_map = [[fn([i/self.nrow, j/self.ncol]) for j in range(self.nrow)] for i in range(self.ncol)]
                height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
                hgtmaps += [height_map]
                np.save(fname, height_map)
        return hgtmaps[np.random.randint(len(hgtmaps))]

    def create_stone(self):
        stone_height = [0.95, 1.0] # percent of 1
        stone_width_px = [6, 12]
        num_stones = [15, 20]
        reset_width = 30

        hgtmaps = []
        for i in range(3):
            fname = os.path.join(self.path, 'stones-{}.npy'.format(i))
            if os.path.isfile(fname):
                hgtmaps += [np.load(fname)]
            else:
                print("First-time run: generating terrain map", fname)
                num_stone = int(np.random.uniform(*num_stones))
                height_map = np.zeros((self.nrow, self.ncol))
                idx = np.linspace(0, self.ncol-1, num=num_stone, dtype=int)
                idy = np.linspace(0, self.nrow-1, num=num_stone, dtype=int)
                # print("num of stones", num_stone)
                for x in idx:
                    height = np.random.uniform(*stone_height)
                    width = np.random.randint(*stone_width_px)
                    for y in idy:
                        height_map[x-width:x+width, y-width:y+width] = height
                # center
                height_map[int(self.ncol/2)-reset_width:int(self.ncol/2)+reset_width, int(self.nrow/2)-reset_width:int(self.nrow/2)+reset_width] = 1.0
                # corners
                height_map[0:0+reset_width, 0:0+reset_width] = 1.0
                height_map[0:0+reset_width, self.nrow-reset_width:self.nrow] = 1.0
                height_map[self.ncol-reset_width:self.ncol, 0:0+reset_width] = 1.0
                height_map[self.ncol-reset_width:self.ncol, self.nrow-reset_width:self.nrow] = 1.0

                height_map = np.transpose(height_map)
                height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
                hgtmaps += [height_map]
                np.save(fname, height_map)
        return hgtmaps[np.random.randint(len(hgtmaps))]

    def create_stair_random_up_down(self, resolution_x=20/400):
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
            dx = [int(x), int(x+sl)]
            if sh + h <= 0:
                sh = 0
            height_map[:, dx[0]:dx[1]] = sh + h
            x += sl
            h += sh
        return height_map

    def create_stair(self, resolution_x=20/400, difficulty=0):
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        step_height = [0.1, 0.1 + difficulty * 0.2]
        step_length = [0.25, 0.4]
        height_map = np.zeros((self.nrow, self.ncol))
        # define number of stair clusters on hfield
        num_clusters = np.random.randint(4, 8)

        x = 0
        for cluster in range(num_clusters):
            # total number of steps in each staircase cluster
            num_steps = int(np.random.uniform(4, 4 + difficulty * 10))
            # define the pattern of going up and down for each cluster
            pattern = np.repeat([1, -1], num_steps//2)
            h = 0
            x += int(np.random.uniform(1, 2) / resolution_x)
            for idx in range(len(pattern)):
                sl = int(np.random.uniform(*step_length) / resolution_x)
                sh = np.random.uniform(*step_height) * pattern[idx]
                dx = [int(x), int(x+sl)]
                if sh + h <= 0:
                    sh = 0
                height_map[:, dx[0]:dx[1]] = sh + h
                x += sl
                h += sh
        return height_map

    def create_discrete_terrain(self, base_position=None,
                                resolution_x=20/400, resolution_y=20/400, difficulty=0):
        """Create a discrete terrain, paramterized by gap/step size in XY direction, and elevation
        in Z direction.
        """
        assert difficulty >= 0 and difficulty <= 1, "difficulty should be in [0, 1]."
        base_height = 0.5
        init_platform = 80
        step_height = [base_height, base_height + difficulty * 0.3]
        step_length = [0.5 - difficulty * 0.4, 0.5]
        gap_length = [0.1, 0.1 + difficulty * 0.2]
        height_map = np.zeros((self.nrow, self.ncol))
        height_map[:, 0:init_platform] = base_height
        x = init_platform
        while True:
            sl_x = int(np.random.uniform(*step_length) / resolution_x)
            gap_x = int(np.random.uniform(*gap_length) / resolution_x)
            sh = np.random.uniform(*step_height)
            dx = [int(x), int(x+sl_x)]
            height_map[:, dx[0]:dx[1]] = sh
            height_map[:, dx[1]:dx[1]+gap_x] = 0.0
            x += sl_x + gap_x
            if x >= self.ncol:
                break
        y = 0
        while True:
            sl_y = int(np.random.uniform(*step_length) / resolution_y)
            gap_y = int(np.random.uniform(*gap_length) / resolution_x)
            dy = [int(y), int(y+gap_y)]
            height_map[dy[0]:dy[1], init_platform:] = 0
            y += sl_y + gap_y
            if y >= self.nrow:
                break
        height_map[250:400:, init_platform:] = 0.0
        height_map[0:150:, init_platform:] = 0.0
        return height_map