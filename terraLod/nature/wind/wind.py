from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from terraLod.utils import timeit, FastInterpolator

from .helper import get_lat_grid, get_prevailing_wind, apply_coriolis, normalize_mean_and_cap

@dataclass
class WindConfig:

    max_wind_speed:     float = 20.0   # Cap on maximum wind speed in m/s to prevent extreme values from steep gradients.
    soft_cap_speed:    float = 5.0   # Speed at which the soft cap starts to kick in, in m/s. Should be less than max_wind_speed.
    wind_scale:         float = 0.3   # Overall scaling factor for wind speed.
    
    temp_strength:      float = 3.0   # How heavily temperature variations (sea breezes) drive wind. Increase for strong coastal winds. #1.0 -> 7.0m/s
    terrain_strength:   float = 3.5   # How violently mountains block and deflect wind. Increase if wind ignores valleys. #1.0 -> 0.7m/s
    prevailing_strength: float = 1.0 # Base planetary wind speed contribution (e.g. westerlies vs easterlies). #1.0 -> 0.1m/s max

    coriolis_fraction:  float = 10.0   # The twist injected into the wind by planet rotation.
    blur_sigma:         float = 10.0   # Smoothing applied to wind map to prevent rigid 90-degree jagged turns.
    

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





    def init_wind_a(self):
        lat_rows = get_lat_grid(self.worldConfig)   # degrees, shape (rows,)

        thermal_i, thermal_j = self.worldConfig["temp_grad_i"](), self.worldConfig["temp_grad_j"]()
        height_gradient_i, height_gradient_j = self.worldConfig["grad_i"](), self.worldConfig["grad_j"]()

        prevailing = get_prevailing_wind(self.worldConfig, lat_rows)
        prev_i, prev_j = prevailing[..., 0], prevailing[..., 1]
        
        base_i = -self.config.temp_strength * thermal_i + self.config.prevailing_strength * prev_i
        base_j = self.config.temp_strength * thermal_j + self.config.prevailing_strength * prev_j

        wind_slope_into = (base_i * height_gradient_i) + (base_j * height_gradient_j)
        terrain_i = - wind_slope_into * height_gradient_j * self.config.terrain_strength
        terrain_j = - wind_slope_into * height_gradient_i * self.config.terrain_strength

        i,j = base_i + terrain_i, base_j + terrain_j
        #i,j = apply_coriolis(self.config, i, j, lat_rows)

        wind = np.stack([i, j], axis=-1)
        wind = wind * self.config.wind_scale  # Scale overall wind speed if desired
        if self.config.blur_sigma > 0:
            wind[..., 0] = gaussian_filter(wind[..., 0], sigma=self.config.blur_sigma)
            wind[..., 1] = gaussian_filter(wind[..., 1], sigma=self.config.blur_sigma)

        return wind

    def init_wind_b(self):
        lat_rows = get_lat_grid(self.worldConfig)   # degrees, shape (rows,)
        
        thermal_i, thermal_j = self.worldConfig["temp_grad_i"](), self.worldConfig["temp_grad_j"]()
        height_gradient_i, height_gradient_j = self.worldConfig["grad_i"](), self.worldConfig["grad_j"]()

        prevailing = get_prevailing_wind(self.worldConfig, lat_rows)
        prev_i, prev_j = prevailing[..., 0], prevailing[..., 1]

        thermal_i, thermal_j = normalize_mean_and_cap(thermal_i, thermal_j, self.config.soft_cap_speed, self.config.max_wind_speed)
        base_i = -self.config.temp_strength * thermal_i + self.config.prevailing_strength * prev_i
        base_j = -self.config.temp_strength * thermal_j + self.config.prevailing_strength * prev_j

        base_i, base_j = apply_coriolis(base_i, base_j, lat_rows, self.config.coriolis_fraction)

        wind_slope_into = (base_i * height_gradient_i) + (base_j * height_gradient_j)
        terrain_i = - wind_slope_into * height_gradient_j * self.config.terrain_strength
        terrain_j = - wind_slope_into * height_gradient_i * self.config.terrain_strength

        i,j = base_i + terrain_i, base_j + terrain_j
        #i,j = apply_coriolis(self.config, i, j, lat_rows)

        wind = np.stack([i, j], axis=-1)
        wind = wind * self.config.wind_scale  # Scale overall wind speed if desired
        if self.config.blur_sigma > 0:
            wind[..., 0] = gaussian_filter(wind[..., 0], sigma=self.config.blur_sigma)
            wind[..., 1] = gaussian_filter(wind[..., 1], sigma=self.config.blur_sigma)

        return wind

    
    def init_wind(self):
        lat_rows = get_lat_grid(self.worldConfig)   # degrees, shape (rows,)

        dTdi, dTdj = self.worldConfig["temp_grad_i"](), self.worldConfig["temp_grad_j"]()
        dHdi = self.worldConfig["grad_i"]()
        dHdj = self.worldConfig["grad_j"]()

        # Terrain deflection: only deflect wind blowing into the slope
        dot = -dTdi * dHdi - dTdj * dHdj   # dimensionless
        terrain_x = dot * (-dHdj) * 100
        terrain_y = dot * dHdi * 100

        # Add latitude-dependent global circulation (three-cell model)
        prevailing = get_prevailing_wind(self.worldConfig, lat_rows) 
        prev_u, prev_v = prevailing[..., 0], prevailing[..., 1]
        prev_u, prev_v = apply_coriolis(prev_u, prev_v, lat_rows, self.config.coriolis_fraction)

        dTdi, dTdj = normalize_mean_and_cap(dTdi, dTdj, self.config.soft_cap_speed, self.config.max_wind_speed)
        terrain_x, terrain_y = normalize_mean_and_cap(terrain_x, terrain_y, self.config.soft_cap_speed, self.config.max_wind_speed)

        

        strengths = np.asarray([self.config.temp_strength, self.config.terrain_strength, self.config.prevailing_strength])

        u = -strengths[0] * dTdi + strengths[1] * terrain_x + strengths[2] * prev_u
        v = -strengths[0] * dTdj + strengths[1] * terrain_y + strengths[2] * prev_v

        # Apply Coriolis deflection (NH rightward, SH leftward)

        wind = np.stack([u, v], axis=-1)

        wind = wind * self.config.wind_scale  # Scale overall wind speed if desired
        if self.config.blur_sigma > 0:
            wind[..., 0] = gaussian_filter(wind[..., 0], sigma=self.config.blur_sigma)
            wind[..., 1] = gaussian_filter(wind[..., 1], sigma=self.config.blur_sigma)

        return wind




