from scipy.ndimage import gaussian_filter
import numpy as np

from fields import BaseModel
from fields.utils import normalize
from .noise.noiseGenerator import NoiseGenerator
from .numba import detect_sea

map_info = {
    'H' : {
        'interp_order' : 3,
        'requires_grad' : True,
        'normalize_sea_level' : True,
        'unit' : 'm',
        'description' : 'Height map of the terrain',
        'render' : {'cmap': 'terrain'},  # composite: land=terrain, sea=Blues depth
    },
    'M_sea' : {
        'interp_order' : 0,
        'description' : 'Boolean mask indicating sea vs land',
    },
    'sea_level' : {
        'interp_order' : 0,
        'unit' : 'm',
        'description' : 'Height threshold for sea level',
    },
}


class Terrain(BaseModel):
    info = {
        'name':'terrain',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.noise_generator = NoiseGenerator(self.world.seed)
        self.init()  # Run the simulation immediately to initialize maps


    ## ========== Simulation & Generation ==========
    def init(self):
        grid = self.world.area.grid

        ds = self.noise_generator.build_ds(self.world.size)
        noise = self.noise_generator.get_noise(grid, macro = True)

        height_map = ds * noise
        height_map = normalize(height_map)

        height = normalize(gaussian_filter(height_map, sigma=1))

        sea_level = np.percentile(height, self.world.sea_level_percentile * 100)
        sea_mask = detect_sea(height, sea_level)
        self.set_maps({
            'M_sea' : sea_mask.astype(np.bool_) ,
            'sea_level' : sea_level,
            'H' : height,
        })

    def step(self):
        #erosion = + for erosion, - for deposition
        H, M_sea, sea_level, erosion = self.get_maps()
        
        if np.all(erosion == 0):
            return

        new_H = H - erosion
        new_sea_mask = detect_sea(new_H, sea_level)
        
        self.set_maps({
            'H' : new_H,
            'M_sea' : new_sea_mask.astype(np.bool_),
            'sea_level' : sea_level
        })
    
    def generate(self, area):
        grid, points, size = area.grid, area.points, area.size
        pass

    ## ========== Map Management ==========

    def get_maps(self):
        H = self.world.area["H"]()
        M_sea, sea_level = self.world.area["M_sea"](), self.world.area["sea_level"]()
        erosion = self.world.area["erosion"]()
        return H, M_sea, sea_level, erosion

    ## ---- Initialization ----

