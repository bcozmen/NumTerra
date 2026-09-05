from dataclasses import dataclass
import numpy as np

from .numba import precompute_horizon_angles, lookup_shadow_from_horizon

@dataclass
class SunConfig:
    horizon_n_dirs : int = 16  # Number of azimuth directions for the precomputed horizon shadow map.
                               # Higher = more accurate shadow edges; 16 is a good default.
    # Solar transmission parameters
    cloud_transmission_coef: float = 1.0 # Extinction coeff for clouds in shortwave
    vapor_transmission_coef: float = 0.005 # Extinction coeff for water vapor in shortwave
    base_solar_transmission: float = 1.00 # Global baseline atmospheric shortwave transmission
    diffuse_fraction: float = 0.25 # Fraction of solar flux that is diffuse (scattered by atmosphere) vs direct beam. Diffuse light is less affected by terrain slope and shadows.

class Sun:
    #TODO Sun gradients do not use water surface gradients (instead calculates slope for water as if no water)
    def __init__(self, world):
        self.world = world
        self.config = SunConfig()
        self._horizon_angles = None

    def invalidate_cache(self):
        self._horizon_angles = None

    def init_horizon_cache(self, H, M_sea):
        if self._horizon_angles is None:
            dx, dy = self.world.area.cell_size
            self._horizon_angles = precompute_horizon_angles(
                H, self.config.horizon_n_dirs, self.world.max_altitude, dx, dy
                
            )
    def _get_terrain_normal(self, H, H_grad_i, H_grad_j, M_sea):
        land_mask = 1.0 - M_sea
        nx = -H_grad_j * land_mask
        ny = -H_grad_i * land_mask
        nz = np.ones_like(H)
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        return nx / norm, ny / norm, nz / norm
    def __call__(self, H, H_grad_i, H_grad_j, M_sea, Wa, Wc):
        """Calculates solar flux across the map, accounting for terrain slope incident angles."""
        sx, sy, sz = self.world['time'].solar_vectors
        

        self.init_horizon_cache(H, M_sea)
        shadow_map = lookup_shadow_from_horizon(self._horizon_angles, sx, sy, sz)
        # If nighttime, return zero flux and an unlit shadow map immediately.
        if sz <= 0.0:
            return np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
            
        sz = np.clip(sz, 0.0, 1.0)
        # Approximate the longer optical path at low solar elevation.  The
        # previous value was calculated but not used, which overestimated
        # sunrise/sunset insolation.
        air_mass = 1.0 / max(sz, 0.05)

        cloud_transmission = np.exp(-self.config.cloud_transmission_coef * Wc * air_mass)
        vapor_transmission = np.exp(-self.config.vapor_transmission_coef * Wa * air_mass)
        base_transmission = self.config.base_solar_transmission ** air_mass
        transmission = base_transmission * vapor_transmission * cloud_transmission

        nx, ny, nz = self._get_terrain_normal(H, H_grad_i, H_grad_j, M_sea)
        solar_norm = np.sqrt(sx**2 + sy**2 + sz**2) + 1e-6
        sx, sy, sz = sx / solar_norm, sy / solar_norm, sz / solar_norm

        cos_incidence = np.clip(nx * sx + ny * sy + nz * sz, 0.0, 1.0)
        
        direct_light = cos_incidence * shadow_map * (1 - self.config.diffuse_fraction)
        ambient_light = self.config.diffuse_fraction * sz

        effective_incidence = direct_light + ambient_light

        sun_toa = self.world.constants['S0'] * effective_incidence
        solar_flux = sun_toa * transmission
        # Energy absorbed by the atmosphere (clouds + water vapour + baseline) before reaching the surface.
        # = S0 * effective_incidence * (1 - transmission).  Passed to thermal so it is not discarded.
        sun_atm = sun_toa - solar_flux
        display_shadow = shadow_map * cos_incidence
        return solar_flux, display_shadow, sun_atm