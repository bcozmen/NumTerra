import numpy as np
from scipy.ndimage import distance_transform_edt
from terraLod.utils import normalize

def get_temperature_grid(size, max_size, latitude, decl) -> np.ndarray:
    rows, cols = size
    lat_grid = _get_latitude_grid(rows, max_size, latitude)
    seasonal_curve = -20 + 45 * np.cos(lat_grid - decl * 0.7)
    # Cosine thermal gradient across the planetary curvature
    temp = np.repeat(seasonal_curve, cols, axis=1)
    return temp

def _get_latitude_grid(rows: int, max_size: float, latitude: float) -> np.ndarray:
    lat0 = np.radians(latitude)
    # WGS84 ellipsoid approximation for meters per degree latitude
    meters_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * lat0)
        + 1.175 * np.cos(4 * lat0)
        - 0.0023 * np.cos(6 * lat0)
    )
    lat_span_deg = max_size / meters_per_deg_lat
    lat_grid_deg = np.linspace(latitude - lat_span_deg / 2.0, latitude + lat_span_deg / 2.0, rows)
    return np.radians(lat_grid_deg)[:, None]

def get_water_masks(worldConfig):
    def extract_mask(name):
        return worldConfig[name]().astype(bool) if name in worldConfig.maps else np.zeros(worldConfig["sea_mask"]().shape, dtype=bool)
    return [extract_mask(name) for name in ['sea_mask', 'river_mask', 'lake_mask']]
        

def get_water_cooling(worldConfig, config):
    #mask true = water, false = land
    masks = get_water_masks(worldConfig)
    cooling_effect = np.zeros(masks[0].shape) # initialize cooling effect map
    continentality = np.zeros(masks[0].shape) # placeholder for future continentality effect
    for mask, key in zip(masks, ['sea', 'river', 'lake']):
        if not np.any(mask):
            continue  # skip if no cells of this type
        # Calculate exact Euclidean distance from water features in meters
        distance = distance_transform_edt(~mask) * worldConfig.cell_size[0] # exact Euclidean distance in meters
        cooling_effect += config.cooling_effects[key][0] * np.exp(-distance / config.cooling_effects[key][1])
        if key == 'sea':
            continentality += config.cooling_effects['continentality'][0] * ( 1 - np.exp( -distance / config.cooling_effects['continentality'][1]))

    
    sea_mask = masks[0]
    if np.any(sea_mask):
        rows, cols = worldConfig.size
        lat_rows = _get_latitude_grid(rows, worldConfig.max_size, worldConfig.latitude)
        lat_deg = np.degrees(lat_rows)

        # Gaussian envelope peaked around mid-latitudes (±30 degrees)
        lat_envelope = np.exp(-((np.abs(lat_deg) - 30.0) / 15.0) ** 2)
        # Exponential decay stretching eastward from the western coastlines (left 20% of domain)
        lon_decay = np.exp(-np.linspace(0.0, 1.0, cols)[None, :] / 0.20)

        cooling_effect += (3.5 * lat_envelope * lon_decay) * sea_mask

    return cooling_effect, continentality

def get_sun_heating(world, declination, solar_vectors):
    di, dj = world["grad_i"](), world["grad_j"]()

    # Build normalized 3D terrain surface normals (East, North, Up)
    norm = np.sqrt(dj**2 + di**2 + 1.0)
    nx, ny, nz = -dj / norm, -di / norm, 1.0 / norm


    # Calculate solar vectors
    sx, sy, sz = solar_vectors
    sun = np.clip(nx * sx + ny * sy + nz * sz, 0.0, 1.0)
    return normalize(sun)

