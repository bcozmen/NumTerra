import numpy as np
from nature.plotter import Plotter
class MockSpec:
    def __init__(self):
        self.overlay_water = True
        self.cmap = "terrain"
        self.vrange = (-0.2, 1.0)
        self.auto_range = False
        self.unit = "norm"
        self.renderer = None
        self.norm_type="linear"
    def resolve_range(self, x): return 0.0, 1.0

class MockConfig:
    solar_vectors = (0.5, 0.5, 0.707)

class MockWorld:
    def __init__(self):
        self.size = (10, 10)
        self.cell_size = (1000, 1000)
        self.maps = {"height": lambda: np.random.rand(10, 10),
                     "sun": lambda: np.random.rand(10, 10),
                     "sea_mask": lambda: np.zeros((10, 10), dtype=bool)}
        self.worldConfig = MockConfig()
        self.season = "spring"
    def __getitem__(self, k): return self.maps[k]
    def __setitem__(self, k, v): pass

w = MockWorld()
p = Plotter(w)
p.specs["height"] = MockSpec()
p.plot_all(["height"])
