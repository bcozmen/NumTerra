from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from terraLod.utils import timeit, get_lat_grid, get_water_masks

from .helper import get_prevailing_wind, apply_rotation, normalize_mean_and_cap, apply_warp, soft_cap
from .numba import advect_numba_multi

@dataclass
class WindConfig:

    max_wind_speed:     float = 8.0   # Cap on maximum wind speed in m/s to prevent extreme values from steep gradients.
    soft_cap_speed:    float = 3.0   # Speed at which the soft cap starts to kick in, in m/s. Should be less than max_wind_speed.
    wind_scale:         float = 0.5   # Overall scaling factor for wind speed.
    
    temp_strength:      float = 1.0   # How heavily temperature variations (sea breezes) drive wind. Increase for strong coastal winds. #1.0 -> 7.0m/s
    terrain_strength:   float = 1.0   # How violently mountains block and deflect wind. Increase if wind ignores valleys. #1.0 -> 0.7m/s
    prevailing_strength: float = 1.0 # Base planetary wind speed contribution (e.g. westerlies vs easterlies). #1.0 -> 0.1m/s max

    blur_sigma:         float = 1.0   # Smoothing applied to wind map to prevent rigid 90-degree jagged turns.
    rotation_sigma:     float = 60.0  # Standard deviation for random rotation of prevailing winds in degrees. Increase for more chaotic global patterns.

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

    #### ========== Simulation & Generation ==========

    @timeit(label="Wind Simulation")
    def run(self):
        wind = self._init_wind()
        self.set_maps(wind)

    @timeit(label="Wind Generation")
    def generate(self, area):
        grid, points, size = area.grid, area.points, area.size
        area["wind"] = self.worldConfig["wind"](points).reshape(size + (2,))

    ### ======== Maps Management ==========
    def set_maps(self, wind_map):
        self.worldConfig["wind"] = wind_map
    
    def get_maps(self):
        sea_mask, lake_mask, river_mask = get_water_masks(self.worldConfig)
        temperature = self.worldConfig["temperature"]()
        
        thermal_i, thermal_j = self.worldConfig["temperature_grad_i"](), self.worldConfig["temperature_grad_j"]()
        height_gradient_i, height_gradient_j = self.worldConfig["height_grad_i"](), self.worldConfig["height_grad_j"]()
        
        return sea_mask, lake_mask, river_mask, temperature, thermal_i, thermal_j, height_gradient_i, height_gradient_j

    ### ======== Wind Map Initialization Core ==========

    def _get_terrain_deflection(self, base_i, base_j, height_gradient_i, height_gradient_j):
        grad_norm = np.sqrt(height_gradient_i**2 + height_gradient_j**2) + 1e-6
        grad_i, grad_j = height_gradient_i / grad_norm, height_gradient_j / grad_norm

        wind_slope_into = (base_i * grad_i) + (base_j * grad_j)
        blocking = np.maximum(wind_slope_into, 0)

        # FIX: True perpendicular tangent vector
        tangent_i = -grad_j
        tangent_j = grad_i
        
        # FIX: Dynamically orient tangent to match the incoming wind's flow
        wind_dot_tangent = (base_i * tangent_i) + (base_j * tangent_j)
        side_sign = np.where(wind_dot_tangent >= 0, 1.0, -1.0)
        tangent_i *= side_sign
        tangent_j *= side_sign

        # Now this cleanly cancels uphill flow and diverts it sideways
        terrain_i = (- blocking * grad_i) + (blocking * 0.6 * tangent_i)
        terrain_j = (- blocking * grad_j) + (blocking * 0.6 * tangent_j)
        
        return normalize_mean_and_cap(terrain_i, terrain_j, self.config.soft_cap_speed, self.config.max_wind_speed)
    def _init_wind(self):
        sea_mask, lake_mask, river_mask, temperature, thermal_i, thermal_j, height_gradient_i, height_gradient_j = self.get_maps()
        lat_rows = get_lat_grid(self.worldConfig.latitude, self.worldConfig.size, self.worldConfig.max_size)   # degrees, shape (rows,)
        
        # 1. Get prevailing wind (Randomly rotated)
        prev_j, prev_i = get_prevailing_wind(self.worldConfig, lat_rows, rotation_sigma=self.config.rotation_sigma)
   
        # 2. Combine thermal and prevailing to get base wind
        thermal_i, thermal_j = normalize_mean_and_cap(thermal_i, thermal_j, self.config.soft_cap_speed, self.config.max_wind_speed)
        base_i = -self.config.temp_strength * thermal_i + self.config.prevailing_strength * prev_i
        base_j = -self.config.temp_strength * thermal_j + self.config.prevailing_strength * prev_j

        # 3. Calculate terrain deflection based on slope and add to base wind
        terrain_i, terrain_j = self._get_terrain_deflection(base_i, base_j, height_gradient_i, height_gradient_j)
        
        # 4. Combine base wind and terrain deflection
        i,j = base_i + terrain_i * self.config.terrain_strength, base_j + terrain_j * self.config.terrain_strength

        # 5. Final adjustments: scale and blur
        i, j = i * self.config.wind_scale, j * self.config.wind_scale
        if self.config.blur_sigma > 0:
            i = gaussian_filter(i, sigma=self.config.blur_sigma)
            j = gaussian_filter(j, sigma=self.config.blur_sigma)

        wind = np.stack([i, j], axis=-1)

        return wind

    def _advect(self, speed_i, speed_j, temperature, humidity):
        maps_combined = np.stack([temperature, humidity], axis=-1)
        advected = advect_numba_multi(maps_combined, speed_i, speed_j, cap)
        return advected[..., 0], advected[..., 1]



