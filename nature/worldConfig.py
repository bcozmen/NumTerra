import numpy as np

from dataclasses import dataclass, field
from functools import cached_property
from abc import ABC, abstractmethod

from utils.functions import get_grid, get_slope, FastInterpolator, get_cell_size
import matplotlib.pyplot as plt
@dataclass
class WorldConfig:
    size_exponent: int = 9
    max_altitude : float = 3000.0 #max altitude in meters
    max_size : float = 200_000.0 #world size in meters
    latitude : float = 35 #latitude in degrees, used for temperature gradient and climate
    hour : float = 14 #hour of the day, used for sun position and lighting
    season : str = "summer" #season, used for sun position and lighting

    sea_level_percentile : float = 0.25 #percentile for sea level
    sea_level : float = None #computed sea level based on height map and percentile
    seed : int = 3563
    debug : bool = False



class World:
    def __init__(self, config = None, lim = (0, 1, 0, 1), size = (1024, 1024)):
        # If config is None, we assume this is the "whole world" initialization
        self.whole_world = False  
        if config is None:
            config = WorldConfig()
            self.whole_world = True
            self.__dict__.update(config.__dict__)
            lim = (0, 1, 0, 1)
            size = (2 ** self.size_exponent + 1, 2 ** self.size_exponent + 1)
            
        self.worldConfig = config
        self.lim = lim
        self.size = size

        self.cell_size = get_cell_size(self.lim, self.size, self.worldConfig.max_size)
        self.grid = get_grid(lim = self.lim, shape=self.size)
        self.points = np.stack(self.grid, axis=-1).reshape(-1, 2)
        self.maps = {}
        self.models = {}
        self.init_maps()
    def init_maps(self):
        if not self.whole_world:
            for model in self.worldConfig.models.values():
                model(self)

    def __call__(self):
        _tracked = ['temperature', 'humidity', 'rain', 'soil_moisture']
        _before = {k: self.maps[k]().copy() for k in _tracked if k in self.maps}

        for key, model in self.models.items():
            if key == 'model_terrain': continue  # Ensure terrain runs first for slope calculations
            elif key == 'model_plotter': continue  # Plotter should run last to visualize all maps
            model()

            _after = {k: self.maps[k]() for k in _tracked if k in self.maps}
            diff = {k: np.abs(_after[k] - _before[k]) for k in _after if k in _before}
            changed_mean = {k: d.mean() for k, d in diff.items()}
            changed_max  = {k: d.max()  for k, d in diff.items()}
            if any(v > 1e-6 for v in changed_mean.values()):
                if self.worldConfig.debug:
                    print(f"  [{key}]  " + "  ".join(
                        f"{k}: mean={changed_mean[k]:.4f} max={changed_max[k]:.4f}"
                        for k in changed_mean
                    ))
            _before = {k: _after[k].copy() for k in _after}

    def __setitem__(self, key, value):
        if key.startswith("model_"):
            self.models[key] = value
        else:             
            self._set_map(key, value)

    def _set_map(self, key, value):
        can_call = self.whole_world  # Only allow interpolation if this is the whole world (precomputed maps)
        if isinstance(value, FastInterpolator):
            value.update() # Ensure the interpolator is initialized with the new data
        else:
            if isinstance(value, np.ndarray):
                value =  FastInterpolator(value, order=1, can_call=can_call)
        self.maps[key] = value
        if key == "height":
            slope, grad_i, grad_j = get_slope(value(), self.cell_size, self.worldConfig.max_altitude)
            self.maps["slope"] = FastInterpolator(slope, order=1, can_call=can_call)
            self.maps["grad_i"] = FastInterpolator(grad_i, order=1, can_call=can_call)
            self.maps["grad_j"] = FastInterpolator(grad_j, order=1, can_call=can_call)
            #if can_call:
                #height = value()
                #self.maps["mfd_weights"] = FastInterpolator(compute_mfd_weights(height, self.worldConfig.max_altitude, self.cell_size[0], self.cell_size[1]), order=1, can_call=True)

        if key == "temperature":
            slope, grad_i, grad_j = get_slope(value(), self.cell_size)
            self.maps["temp_grad_i"] = FastInterpolator(grad_i, order=1, can_call=can_call)
            self.maps["temp_grad_j"] = FastInterpolator(grad_j, order=1, can_call=can_call)


    def __getitem__(self, key):
        if key in self.models:
            return self.models[key]
        return self.maps[key]

    




    

