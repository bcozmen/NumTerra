
from .worldConfig import WorldConfig
from utils.functions import get_grid, get_cell_size

import numpy as np

def compute_cell_size(lim, size, max_size):
    range_x, range_y = lim[1] - lim[0], lim[3] - lim[2]
    cell_size_x, cell_size_y = range_x / (size[0] - 1), range_y / (size[1] - 1)
    return (cell_size_x * max_size, cell_size_y * max_size)

class Area:
    def __init__(self, worldConfig, lim=(0, 1, 0, 1), size=(512, 512)):
        self.worldConfig = worldConfig
        self.lim = lim
        self.size = size
        self.maps = {}

        self.grid = get_grid(lim = self.lim, shape=self.size)
        self.points = np.stack(self.grid, axis=-1).reshape(-1, 2)
        self.cell_size = get_cell_size(self.lim, self.size, worldConfig.max_size)

    def __setitem__(self, key, value):
        self.maps[key] = value
    def __getitem__(self, key):
        return self.maps[key]

    