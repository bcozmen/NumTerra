from dataclasses import dataclass
import numpy as np

from terraLod.utils import get_water_masks, timeit


class Erosion:
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.worldConfig["model_erosion"] = self
        self.run()

    def __call__(self, area=None):
        if area is None: self.run()
        else: self.generate(area)

    #### ========== Simulation & Generation ==========
    @timeit(name="Erosion Simulation")
    def run(self):
        pass

    def generate(self, area):
        pass

    ### ========== Map Management ==========
    def set_maps(self, erosion_map):
        pass
    
    def get_maps(self):
        pass

