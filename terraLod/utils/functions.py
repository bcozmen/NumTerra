from scipy.ndimage import distance_transform_edt, distance_transform_cdt
import numpy as np

def get_water_masks(worldConfig):
    sea_mask = worldConfig["sea_mask"]().astype(bool)
    river_mask = worldConfig["river_mask"]().astype(bool)
    lake_mask = worldConfig["lake_mask"]().astype(bool)
    return sea_mask, river_mask, lake_mask


def normalize(arr, axis = None,vmin = None, vmax = None, range = (0, 1)):
    if vmin is None:
        vmin = np.min(arr, axis=axis, keepdims=True)
    if vmax is None:
        vmax = np.max(arr, axis=axis, keepdims=True)
    return range[0] + (arr - vmin) * (range[1] - range[0]) / (vmax - vmin + 1e-8)

def get_grid(lim = (0, 1, 0, 1), shape = (2048, 2048)):
    if type(shape) == int:
        shape = (shape, shape)
    x = np.linspace(lim[0], lim[1], shape[0])
    y = np.linspace(lim[2], lim[3], shape[1])
    return np.meshgrid(x, y, indexing='ij')

def get_slope(height_map, sea_level, cell_sizes, scale_factor=1.0):
    height_map = np.clip(height_map - sea_level, 0, None)  # Treat anything below sea level as sea level for slope purposes
    height_map = height_map * scale_factor


    grad_x, grad_y = np.gradient(height_map, *cell_sizes)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))   # radians
    return slope_rad, grad_x, grad_y

def get_cell_size(lim, size, max_size):
    range_x, range_y = lim[1] - lim[0], lim[3] - lim[2]
    cell_size_x, cell_size_y = range_x / (size[0] - 1), range_y / (size[1] - 1)
    return (cell_size_x * max_size, cell_size_y * max_size)

def get_lat_grid(latitude, size, max_size) -> np.ndarray:
    """Returns latitude [degrees] for each row of the world grid from South to North."""
    lat0 = np.radians(latitude)
    # WGS84 ellipsoidal approximation for meters per degree latitude
    meters_per_deg = (
        111132.92
        - 559.82  * np.cos(2 * lat0)
        + 1.175   * np.cos(4 * lat0)
        - 0.0023  * np.cos(6 * lat0)
    )
    lat_span = max_size / meters_per_deg
    lat_rows = np.linspace(latitude - lat_span / 2, latitude + lat_span / 2, size[0])
    return np.broadcast_to(lat_rows[:, None], size)


def get_normalized_distance_to_mask(mask, cell_size = 1.0, max_distance = None, mode = 'euclidean'):
    #array shape size
    if mode == 'euclidean':
        distance = distance_transform_edt(~mask) * cell_size  # Convert to physical distance in meters
    elif mode == 'chessboard':
        distance = distance_transform_cdt(~mask, metric='chessboard') * cell_size
    elif mode == 'taxicab':
        distance = distance_transform_cdt(~mask, metric='taxicab') * cell_size
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if max_distance is None:
        max_distance = np.max(distance)
    distance = distance / (max_distance)  # Normalize by max possible distance in the grid
    distance[mask] = 0.0  # Ensure water cells have zero distance
    return distance