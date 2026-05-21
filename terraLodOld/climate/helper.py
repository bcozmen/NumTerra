import numpy as np

import numpy as np

def get_temperature_grid(rows: int, cols : int, max_size: float, latitude: float) -> np.ndarray:
    lat_grid = _get_latitude_grid(rows, max_size, latitude)
    temp = -25 + 52 * np.cos(lat_grid)
    temp = np.repeat(temp, cols, axis=1)
    return temp

def _get_latitude_grid(rows: int, max_size: float, latitude: float) -> np.ndarray:
    # convert center latitude to radians
    lat0 = np.radians(latitude)
    # WGS84-ish meters per degree latitude (more accurate than 111k constant)
    meters_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * lat0)
        + 1.175 * np.cos(4 * lat0)
        - 0.0023 * np.cos(6 * lat0)
    )
    lat_span_deg = max_size / meters_per_deg_lat

    lat_grid_deg = np.linspace(
        latitude - lat_span_deg / 2.0,
        latitude + lat_span_deg / 2.0,
        rows,
    )
    # convert output to radians (since you requested radians)
    lat_grid_rad = np.radians(lat_grid_deg)
    return lat_grid_rad[:, None]