import numpy as np
from .numba import apply_coriolis_numba

def get_lat_grid(worldConfig):
    """
    Returns latitude [degrees] for each row of the world grid, from south to north,
    matching the convention of _get_latitude_grid() in thermal/helper.py.
    """
    lat0 = np.radians(worldConfig.latitude)
    meters_per_deg = (
        111132.92
        - 559.82  * np.cos(2 * lat0)
        + 1.175   * np.cos(4 * lat0)
        - 0.0023  * np.cos(6 * lat0)
    )
    lat_span = worldConfig.max_size / meters_per_deg
    return np.linspace(worldConfig.latitude - lat_span / 2, worldConfig.latitude + lat_span / 2, worldConfig.size[0])  # degrees, shape (rows,)

def get_prevailing_wind(world_config, lat_rows):
    """
    Three-cell global circulation model (Hadley / Ferrel / Polar).

    Zonal (east-west, j-component):
    -cos(3*lat_rad) produces the correct sign sequence:
        0°: -1  (easterly trades)
    30°:  0  (horse latitudes / ITCZ boundary)
    60°: +1  (westerlies)
    90°:  0  (polar front)
    Symmetric about equator so both hemispheres are handled correctly.

    Meridional (north-south, i-component):
    sin(2*lat_rad) gives weak equatorward flow in tropics, poleward in mid-lats.
    """
    lat_rad = np.radians(lat_rows)
    zonal      = -np.cos(3.0 * lat_rad) 
    meridional =  np.sin(2.0 * lat_rad) * 0.5

    prevailing = np.zeros(world_config.size + (2,))
    prevailing[..., 0] = meridional[:, None]  # broadcast over columns
    prevailing[..., 1] = zonal[:, None]
    return prevailing

def apply_coriolis(config, u, v, lat_rows):
    """
    Rotate the wind field toward the geostrophic direction.

    In the Northern Hemisphere (lat > 0) the geostrophic wind is the
    pressure-gradient wind rotated 90° clockwise (rightward).  In the SH it
    rotates 90° counter-clockwise (leftward).

    The rotation angle is:
    angle = coriolis_fraction * sin(lat) * (pi/2)

    so it is:
    - zero at the equator  (no Coriolis there, correct)
    - coriolis_fraction * 90° at the poles
    - positive (rightward) in NH, negative (leftward) in SH

    With coriolis_fraction=0.4 and lat=45°N:
    angle = 0.4 * sin(45°) * 90° ≈ 25° rightward deflection.
    """
    lat_rad = np.radians(lat_rows)   # (rows,)  — 1-D, no broadcasting needed here
    u_new, v_new = apply_coriolis_numba(
        np.require(u,       dtype=np.float64, requirements='C'),
        np.require(v,       dtype=np.float64, requirements='C'),
        np.require(lat_rad, dtype=np.float64, requirements='C'),
        config.coriolis_fraction,
    )
    return u_new, v_new
