from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from terraLod.utils import timeit, FastInterpolator

from .helper import get_lat_grid, get_prevailing_wind, apply_coriolis

@dataclass
class WindConfig:
    temp_strength:      float = 0.25   # How heavily temperature variations (sea breezes) drive wind. Increase for strong coastal winds. #1.0 -> 7.0m/s
    terrain_strength:   float = 0.5   # How violently mountains block and deflect wind. Increase if wind ignores valleys. #1.0 -> 0.7m/s
    prevailing_strength: float = 1.0 # Base planetary wind speed contribution (e.g. westerlies vs easterlies). #1.0 -> 0.1m/s max

    coriolis_fraction:  float = 0.4   # The twist injected into the wind by planet rotation.
    blur_sigma:         float = 1.0   # Smoothing applied to wind map to prevent rigid 90-degree jagged turns.
    

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
        # We use a base typical temperature gradient (e.g., 0.05 C/km) to normalize things physically
        # rather than always scaling the max gradient to 1.
        T_typical = 0.05 / 1000.0  # 0.05 C per km in m
        dTdi = dTdi / T_typical
        dTdj = dTdj / T_typical

        # Terrain deflection: only deflect wind blowing into the slope
        dot = -dTdi * dHdi - dTdj * dHdj   # dimensionless
        terrain_x = dot * (-dHdj)
        terrain_y = dot * dHdi

        # Add latitude-dependent global circulation (three-cell model)
        prevailing = get_prevailing_wind(self.worldConfig, lat_rows)
        # prevailing is unit vector or similarly bounded. Just use it as is.

        strengths = np.asarray([self.config.temp_strength, self.config.terrain_strength, self.config.prevailing_strength])
        strengths = strengths / np.sum(strengths)  # Normalize to sum to 1
 
        u = -strengths[0] * dTdi + strengths[1] * terrain_x + strengths[2] * prevailing[..., 0] 
        v = -strengths[0] * dTdj + strengths[1] * terrain_y + strengths[2] * prevailing[..., 1]

        # Apply Coriolis deflection (NH rightward, SH leftward)
        u, v = apply_coriolis(self.config, u, v, lat_rows)

        wind = np.stack([u, v], axis=-1)

        wind[..., 0] = gaussian_filter(wind[..., 0], sigma=self.config.blur_sigma)
        wind[..., 1] = gaussian_filter(wind[..., 1], sigma=self.config.blur_sigma)

        return wind




