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
    base_solar_transmission: float = 0.85 # Global baseline atmospheric shortwave transmission

class Sun:
    #TODO Sun gradients do not use water surface gradients (instead calculates slope for water as if no water)
    def __init__(self, world):
        self.world = world
        self.config = SunConfig()
        self._horizon_angles = None

    def invalidate_cache(self):
        self._horizon_angles = None

    def __call__(self, H, H_grad_i, H_grad_j, M_sea, Wa, Wc):
        """Calculates solar flux across the map, accounting for terrain slope incident angles."""
        sx, sy, sz = self.world['time'].solar_vectors
        dx, dy = self.world.area.cell_size

        # --- Shadow map via precomputed horizon angles ----------------------------
        # Precompute once (lazily) when real terrain is available; O(1) per step after that.
        if self._horizon_angles is None:
            self._horizon_angles = precompute_horizon_angles(
                H, self.config.horizon_n_dirs, self.world.max_altitude, dx, dy
            )

       
        shadow_map = lookup_shadow_from_horizon(self._horizon_angles, sx, sy, sz)
        # if nighttime return zero flux and empty shadow map immediately to save computation
        if sz <= 0.0:
            return np.zeros_like(H), np.ones_like(H)
            
        sz = np.clip(sz, 0.0, 1.0)

        # Clouds heavily block solar transmission. Water vapor has a moderate effect.
        cloud_transmission = np.exp(-self.config.cloud_transmission_coef * Wc) # Liquid water blocks heavily
        # Water vapor is mostly transparent to visible light (shortwave), so we lower the coefficient
        vapor_transmission = np.exp(-self.config.vapor_transmission_coef * Wa)
        transmission = self.config.base_solar_transmission * vapor_transmission * cloud_transmission

        # With imshow(origin='lower'): row-axis = North, col-axis = East.
        # The outward surface normal in (East, North, Up) order is (-∂H/∂col, -∂H/∂row, 1)
        # = (-H_grad_j, -H_grad_i, 1).  Pairing with solar vector (sx=East, sy=North, sz=Up).
        # Water/ocean surfaces are flat — suppress horizontal normal components on sea cells
        # so their incidence is determined only by the solar elevation angle, not terrain tilt.
        land_mask = 1.0 - M_sea
        nx = -H_grad_j * land_mask
        ny = -H_grad_i * land_mask
        nz = np.ones_like(H)
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        solar_norm = np.sqrt(sx**2 + sy**2 + sz**2) + 1e-6
        sx, sy, sz = sx / solar_norm, sy / solar_norm, sz / solar_norm

        cos_incidence = np.clip(nx * sx + ny * sy + nz * sz, 0.0, 1.0)
        
        # solar_flux: cos_incidence handles slope incidence, shadow_map handles cast shadows
        solar_flux = self.world.constants['S0'] * cos_incidence * transmission * shadow_map

        # Shadow map stored for display: slope shading × cast shadow → continuous [0, 1]
        # 0 = back-facing slope or blocked by terrain; 1 = fully lit, directly facing sun
        display_shadow = shadow_map * cos_incidence
        return solar_flux, display_shadow