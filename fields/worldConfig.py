import numpy as np

from dataclasses import dataclass, field
from functools import cached_property
from abc import ABC, abstractmethod

from terraLod.utils import get_grid, get_slope, get_cell_size
from terraLod.utils import Interpolator
import matplotlib.pyplot as plt

#from terraLod.nature import Terrain, Thermal, Wind, Humidity, Erosion
#from terraLod import Plotter


from .time import Time
@dataclass
class InterpConfig:
    height : int = 3
    requires_grad : list = field(default_factory=lambda: ["height", "temperature"])

    def __getitem__(self,key):
        if hasattr(self, key):
            return getattr(self, key)
        elif key.endswith("mask"):
            return 0
        return 1

@dataclass
class WorldConfig:
    size_exponent: int = 9
    max_altitude : float = 1000.0 #max altitude in meters
    max_size : float = 200_000.0 #world size in meters
    
    latitude : float = 41. #latitude in degrees, used for temperature gradient and climate
    longitude : float = 29.
    

    sea_level_percentile : float = 0.25 #percentile for sea level
    sea_level : float = None #computed sea level based on height map and percentile
    seed : int = 3563
    debug : bool = False

    init_models : list = field(default_factory=lambda: [Terrain])
    iterative_models : list = field(default_factory=lambda: [Thermal])#, Wind, Humidity, Erosion])
    plotter : object = Plotter
    interp_config : object = field(default_factory=InterpConfig)



class World:
    def __init__(self, config = None, lim = (0, 1, 0, 1), size = (1024, 1024), models = [] ):
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

        self.time = Time(self)
        self._init_models()
        

        for model in self.worldConfig.init_models + self.worldConfig.iterative_models + [self.worldConfig.plotter]:
            model(self)

        

    def plot(self, keys = None, **kwargs):
        plotter = self['model_plotter']
        if keys is None or type(keys) == list:
            plotter.plot_all(keys=keys,**kwargs)
        elif keys == "wind":
            keys = "height"
            kwargs['show_wind'] = True
            plotter.plot(keys, **kwargs)
        else:
            plotter.plot(keys, **kwargs)

    def _init_models(self):
        if not self.whole_world:
            for model in self.worldConfig.models.values():
                m = model(self)
                self.models[model.__name__] = m
    def __call__(self, **kwargs):
        self.time.step(**kwargs)
        for key, model in self.models.items():
            if key == 'model_terrain': continue  # Ensure terrain runs first for slope calculations
            elif key == 'model_plotter': continue  # Plotter should run last to visualize all maps
            model()
        

    def __setitem__(self, key, value):
        if key.startswith("model_"):
            self.models[key] = value
        else:             
            self._set_map(key, value)


    def _set_grad(self, key, value, can_call):
        if not key in self.worldConfig.interp_config.requires_grad:
            return

        if key == "height":
            sea_level = self["sea_level"]()
        else:
            sea_level = None
        slope, grad_i, grad_j = get_slope(value, self.cell_size, sea_level = sea_level, scale_factor = self.worldConfig.max_altitude)
        self.maps[key + "_slope"] = Interpolator(slope, order=self.worldConfig.interp_config[key + "_slope"], can_call=can_call)
        self.maps[key + "_grad_i"] = Interpolator(grad_i, order=self.worldConfig.interp_config[key + "_grad_i"], can_call=can_call)
        self.maps[key + "_grad_j"] = Interpolator(grad_j, order=self.worldConfig.interp_config[key + "_grad_j"], can_call=can_call)
    def _set_map(self, key, value):
        can_call = self.whole_world  # Only allow interpolation if this is the whole world (precomputed maps)
        if key in self.maps:
            self.maps[key].update(value) # Update existing interpolator with new data
        else:
            self.maps[key] = Interpolator(value, order=self.worldConfig.interp_config[key], can_call=can_call)

        self._set_grad(key, value, can_call)


    def __getitem__(self, key):
        if key in self.models:
            return self.models[key]
        if key in self.maps:
            return self.maps[key]

        shape = self.size
        if key == "wind":
            shape = shape + (2,)
        fakeInterp = Interpolator(np.zeros(shape, dtype=np.float32), order=self.worldConfig.interp_config[key], can_call=False)
            
        return fakeInterp  # Default to zero map if not found




    

