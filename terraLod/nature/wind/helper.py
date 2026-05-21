import numpy as np
from .numba import apply_coriolis_numba

def get_lat_grid(world) -> np.ndarray:
    """Returns latitude [degrees] for each row of the world grid from South to North."""
    lat0 = np.radians(world.latitude)
    # WGS84 ellipsoidal approximation for meters per degree latitude
    meters_per_deg = (
        111132.92
        - 559.82  * np.cos(2 * lat0)
        + 1.175   * np.cos(4 * lat0)
        - 0.0023  * np.cos(6 * lat0)
    )
    lat_span = world.max_size / meters_per_deg
    return np.linspace(world.latitude - lat_span / 2, world.latitude + lat_span / 2, world.size[0])

def get_prevailing_wind(world, lat_rows) -> np.ndarray:
    """Three-cell global circulation model (Hadley / Ferrel / Polar cells)."""
    lat_rad = np.radians(lat_rows)[:, None]  # shape (rows, 1) for broadcasting
    
    # Zonal (East-West): Easterlies near equator/poles, Westerlies in mid-latitudes
    zonal = -np.cos(3.0 * lat_rad)
    # Meridional (North-South): Cells circulation boundaries
    meridional = np.sin(2.0 * lat_rad) * 0.5
    
    # Broadcast across all columns to match world grid size
    u_grid = np.broadcast_to(meridional, world.size)
    v_grid = np.broadcast_to(zonal, world.size)
    
    return np.stack([u_grid, v_grid], axis=-1)

def apply_coriolis(config, u, v, lat_rows):
    """Rotate the wind field rightward in NH and leftward in SH via Numba kernel."""
    lat_rad = np.radians(lat_rows)
    return apply_coriolis_numba(
        np.ascontiguousarray(u, dtype=np.float64),
        np.ascontiguousarray(v, dtype=np.float64),
        np.ascontiguousarray(lat_rad, dtype=np.float64),
        config.coriolis_fraction,
    )