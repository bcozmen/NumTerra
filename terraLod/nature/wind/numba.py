"""
Numba-accelerated kernels for the wind simulation.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def apply_coriolis_numba(u, v, lat_rad_col, coriolis_fraction):
    """
    Rotate each row of the wind field by the latitude-dependent Coriolis angle.

    Parameters
    ----------
    u, v            : (H, W) float64   – wind components
    lat_rad_col     : (H,)   float64   – latitude in radians per row
    coriolis_fraction : float

    Returns
    -------
    u_new, v_new : (H, W) float64
    """
    H, W = u.shape
    u_new = np.empty((H, W), dtype=np.float64)
    v_new = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        angle = coriolis_fraction * np.sin(lat_rad_col[i]) * (np.pi / 2.0)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        for j in range(W):
            u_new[i, j] = cos_a * u[i, j] - sin_a * v[i, j]
            v_new[i, j] = sin_a * u[i, j] + cos_a * v[i, j]

    return u_new, v_new
