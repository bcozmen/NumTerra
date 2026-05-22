import numpy as np

from dataclasses import dataclass, field
from functools import cached_property
from abc import ABC, abstractmethod

from terraLod.utils import get_grid, get_slope, get_cell_size
from terraLod.utils import FastInterpolator, DummyInterpolator
import matplotlib.pyplot as plt

from terraLod.nature import Terrain, Thermal, Wind, Humidity,  Hydro
from terraLod import Plotter
@dataclass
class WorldConfig:
    size_exponent: int = 9
    max_altitude : float = 3000.0 #max altitude in meters
    max_size : float = 200_000.0 #world size in meters
    
    latitude : float = 30 #latitude in degrees, used for temperature gradient and climate
    hour : float = 14 #hour of the day, used for sun position and lighting
    
    season : str = "spring" #season, used for sun position and lighting
    season_to_declination: dict = field(default_factory=lambda: {
        "spring": 10.0, "summer": 23.44, "autumn": -10.0, "winter": -23.44
    })

    sea_level_percentile : float = 0.25 #percentile for sea level
    sea_level : float = None #computed sea level based on height map and percentile
    seed : int = 3563
    debug : bool = False

    init_models : list = field(default_factory=lambda: [Terrain])
    iterative_models : list = field(default_factory=lambda: [Thermal, Wind, Humidity])
    plotter : object = Plotter

class Time():
    seasons = ["spring", "summer", "autumn", "winter"]
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.current_index = self.seasons.index(worldConfig.season)

    def step(self):
        self.current_index = (self.current_index + 1) % len(self.seasons)
        self.worldConfig.season = self.seasons[self.current_index]
    def get_season(self):
        return self.seasons[self.current_index]
    def get_previous_season(self):
        return self.seasons[(self.current_index - 1) % len(self.seasons)]

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
        self._init_maps()
        

        for model in self.worldConfig.init_models + self.worldConfig.iterative_models + [self.worldConfig.plotter]:
            model(self)

        

    @property
    def declination(self):
        return np.radians(self.season_to_declination[self.time.get_season()])

    @property
    def prev_declination(self):
        return np.radians(self.season_to_declination[self.time.get_previous_season()])

    @property
    def solar_vectors(self):
        declination = self.declination
        lat = np.radians(self.latitude)
        hour_angle = np.radians(15.0 * (self.hour - 12.0))

        solar_altitude = np.arcsin(
            np.sin(lat) * np.sin(declination) +
            np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
        )
        solar_azimuth = np.arctan2(
            -np.cos(declination) * np.sin(hour_angle),
            np.cos(lat) * np.sin(declination) - np.sin(lat) * np.cos(declination) * np.cos(hour_angle)
        )

        sx = np.cos(solar_altitude) * np.sin(solar_azimuth)
        sy = np.cos(solar_altitude) * np.cos(solar_azimuth)
        sz = np.sin(solar_altitude)

        return sx, sy, sz

    def plot(self, keys = None, **kwargs):
        plotter = self['model_plotter']
        if keys is None:
            plotter.plot_all(**kwargs)
        else:
            plotter.plot(keys, **kwargs)

    def _init_maps(self):
        if not self.whole_world:
            for model in self.worldConfig.models.values():
                model(self)

    def __call__(self):
        self.time.step()
        for key, model in self.models.items():
            if key == 'model_terrain': continue  # Ensure terrain runs first for slope calculations
            elif key == 'model_plotter': continue  # Plotter should run last to visualize all maps
            model()

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
            value =  FastInterpolator(value, order=1, can_call=can_call)
        self.maps[key] = value
        if key == "height":
            slope, grad_i, grad_j = get_slope(value(), self["sea_level"](), self.cell_size, self.worldConfig.max_altitude)
            self.maps["slope"] = FastInterpolator(slope, order=1, can_call=can_call)
            self.maps["grad_i"] = FastInterpolator(grad_i, order=1, can_call=can_call)
            self.maps["grad_j"] = FastInterpolator(grad_j, order=1, can_call=can_call)

        if key == "temperature":
            slope, grad_i, grad_j = get_slope(value(), self["sea_level"](), self.cell_size, self.worldConfig.max_altitude)
            self.maps["temp_grad_i"] = FastInterpolator(grad_i, order=1, can_call=can_call)
            self.maps["temp_grad_j"] = FastInterpolator(grad_j, order=1, can_call=can_call)


    def __getitem__(self, key):
        if key in self.models:
            return self.models[key]
        if key in self.maps:
            return self.maps[key].copy()

        shape = self.size
        if key == "wind":
            shape = shape + (2,)
            
        return DummyInterpolator(np.zeros(shape, dtype=np.float32))  # Default to zero map if not found




    

