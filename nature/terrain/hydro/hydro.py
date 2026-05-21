from dataclasses import dataclass
import numpy as np

@dataclass
class HydroConfig:
    pass

class Hydro:
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.worldConfig["model_hydro"] = self
        self.config = HydroConfig()

        self.run()

    def __call__(self, area= None):
        if area is None: self.run()
        else: self.generate(area)
    def run(self):
        pass
    def generate(self, area):
        pass

    def compute_flow_accumulation(self):
        rain = self.worldConfig["rain"]()
        flow = self.worldConfig["mfd_weights"]() # shape (xdim, ydim, 8)

        