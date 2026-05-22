import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from terraLod.utils import normalize, get_grid, FastInterpolator, timeit


from ..noise.noiseGenerator import NoiseGenerator
from .numba import detect_sea


class Terrain:
    @timeit(label="Terrain Initialization")
    def __init__(self, worldConfig ):
        self.worldConfig = worldConfig
        self.worldConfig["model_terrain"] = self
        
        self.noise_generator = None
        self.run()  # Run the simulation immediately to initialize maps

    def __call__(self, area=None):
        if area is None: self.run()
        else: self.generate(area)

    #### ========== Simulation & Generation ==========
    
    @timeit(label="Terrain Simulation")
    def run(self):
        if self.noise_generator is not None:
            raise ValueError("Terrain already initialized. To re-run the simulation, create a new instance of Terrain.")
        self.noise_generator = NoiseGenerator(self.worldConfig.seed)
        height_map, sea_mask, sea_level = self._init_maps()
        self.set_maps(height_map, sea_mask, sea_level)
    
    @timeit(label="Terrain Generation")
    def generate(self, area):
        grid, points, size = area.grid, area.points, area.size
        base = self.worldConfig["height"](points).reshape(size)
        noise = self.noise_generator.get_noise(grid, macro = False)
        pass

    ### ========== Map Management ==========
    def set_maps(self, height_map, sea_mask, sea_level):
        self.worldConfig["height"] = FastInterpolator(height_map, order=3)
        self.worldConfig["sea_mask"] = FastInterpolator(sea_mask, order=0)
        self.worldConfig["sea_level"] = sea_level

    def get_maps(self):
        pass

    ## ---- Initialization ----
    
    def _init_maps(self):
        height_map = self._init_height_map()
        sea_mask, sea_level = self._init_sea_map(height_map)
        return height_map, sea_mask, sea_level
    def _init_height_map(self):
        grid = get_grid(shape= self.worldConfig.size)

        ds = self.noise_generator.build_ds(self.worldConfig.size[0])
        noise = self.noise_generator.get_noise(grid, macro = True)

        height_map = ds * noise
        height_map = normalize(height_map)
        #erode = self.erode(height_map)

        height = normalize(gaussian_filter(height_map, sigma=1))
        height_interp = FastInterpolator(height, order=3)

        return height_interp

    def _init_sea_map(self, height_map):
        height = height_map()
        sea_level = np.percentile(height, self.worldConfig.sea_level_percentile * 100)
        sea_mask = detect_sea(height, sea_level)
        sea_mask = FastInterpolator(sea_mask, order=0)
        return sea_mask, sea_level

    ## ---- Generation ----
    def _generate_height_map(self, grid, points, size):
        base = self.worldConfig["height"](points).reshape(size)
        noise = self.noise_generator.get_noise(grid, macro = False)

        height = base + noise
        return height

    def _generate_sea_mask(self, height, grid, points, size):
        valid_region = self.worldConfig["sea_mask"](points).reshape(size)
        under_sea = height < self.worldConfig["sea_level"]
        sea_mask = np.logical_and(valid_region, under_sea)
        return sea_mask

        

        