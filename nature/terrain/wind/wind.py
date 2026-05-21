from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from utils import timeit
from utils.functions import FastInterpolator

from .helper import get_lat_grid, get_prevailing_wind, apply_coriolis

@dataclass
class WindConfig:
    temp_strength:      float = 1.0
    terrain_strength:   float = 1.5
    # Global circulation (Hadley / Ferrel / Polar cells) base wind [m/s equivalent weight]
    prevailing_strength: float = 0.25
    # Coriolis deflection fraction: 0 = none, 1 = fully geostrophic (90° rotation).
    # At coriolis_fraction=0.4 and lat=45°N the wind is rotated ~18° rightward.
    coriolis_fraction:  float = 0.4
    blur_sigma:         float = 1.0
    max_wind_speed:     float = 25.0  # m/s — output is scaled to this maximum

class Wind:
    @timeit(label="Wind Initialization")
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.worldConfig["model_wind"] = self
        self.config = WindConfig()

        self.run()

    def __call__(self, area= None):
        if area is None: self.run()
        else: self.generate(area)
    @timeit(label="Wind Simulation")
    def run(self):
        self.worldConfig["wind"] =  FastInterpolator(self.init_wind(), order=1)
    @timeit(label="Wind Generation")
    def generate(self, area):
        grid, points, size = area.grid, area.points, area.size
        area["wind"] = self.worldConfig["wind"](points).reshape(size + (2,))
    

    def init_wind(self):
        lat_rows = get_lat_grid(self.worldConfig)   # degrees, shape (rows,)

        dTdi, dTdj = self.worldConfig["temp_grad_i"](), self.worldConfig["temp_grad_j"]()
        # grad_i/j already in m/m (dimensionless slope) — get_slope applies max_altitude scale
        dHdi = self.worldConfig["grad_i"]()
        dHdj = self.worldConfig["grad_j"]()

        # Thermal wind: air flows down the temperature gradient (simplified pressure-gradient force)
        Tmag = np.sqrt(dTdi**2 + dTdj**2).max() + 1e-8
        dTdi /= Tmag   # now dimensionless, max magnitude = 1
        dTdj /= Tmag

        # Terrain deflection: only deflect wind blowing into the slope
        dot = -dTdi * dHdi - dTdj * dHdj   # dimensionless
        terrain_x = dot * (-dHdj)
        terrain_y = dot * dHdi

        Hmag = np.sqrt(terrain_x**2 + terrain_y**2).max() + 1e-8
        terrain_x /= Hmag
        terrain_y /= Hmag

        # terrain_x = dot * (-dHdj) → i-component of CCW-rotated gradient = meridional deflection
        # terrain_y = dot *   dHdi  → j-component of CCW-rotated gradient = zonal deflection
        u = -self.config.temp_strength * dTdi + self.config.terrain_strength * terrain_x
        v = -self.config.temp_strength * dTdj + self.config.terrain_strength * terrain_y

        # Add latitude-dependent global circulation (three-cell model)
        prevailing = get_prevailing_wind(self.worldConfig, lat_rows)
        pmag = np.sqrt(prevailing[..., 0]**2 + prevailing[..., 1]**2).max() + 1e-8
        prevailing /= pmag  # normalize to dimensionless, max magnitude = 1
        prevailing *= self.config.prevailing_strength
        u += prevailing[..., 0]
        v += prevailing[..., 1]

        # Apply Coriolis deflection (NH rightward, SH leftward)
        u, v = apply_coriolis(self.config, u, v, lat_rows)

        wind = np.stack([u, v], axis=-1)

        wind[..., 0] = gaussian_filter(wind[..., 0], sigma=self.config.blur_sigma)
        wind[..., 1] = gaussian_filter(wind[..., 1], sigma=self.config.blur_sigma)

        # Scale so the maximum wind speed equals max_wind_speed [m/s]
        mag = np.sqrt(wind[..., 0]**2 + wind[..., 1]**2).max() + 1e-8
        wind *= self.config.max_wind_speed / mag
        # Output units: m/s, range ~ [-max_wind_speed, +max_wind_speed] per component

        return wind




