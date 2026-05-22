import numpy as np
from scipy.ndimage import distance_transform_edt
from terraLod.utils import normalize




def get_temperature_grid(size, max_size, latitude, phase) -> np.ndarray:
    rows, cols = size
    lat_grid = _get_latitude_grid(size, max_size, latitude)

    mean_temp = _latitude_mean_temperature(latitude)
    seasonal_amplitude = _latitude_seasonal_amplitude(latitude)
    hemisphere_sign = np.where(latitude >= 0, 1, -1)  # +1 for Northern Hemisphere, -1 for Southern Hemisphere

    seasonal_offset = phase * seasonal_amplitude * hemisphere_sign

    return mean_temp, seasonal_offset
    

def season_phase(season):
    if season == 'summer':
        return 1.0
    elif season == 'winter':
        return -1.0
    return 0.0

def _latitude_mean_temperature(latitude):
    lat_abs = abs(latitude)
    return 50.0 * np.cos(np.radians(lat_abs)) - 23.0  # Ranges from 17°C at equator to -10°C at poles

def _latitude_seasonal_amplitude(latitude):
    lat_abs = abs(latitude)
    return 2.0 + (27.0 / 90.0) * lat_abs 

def _get_latitude_grid(size, max_size, latitude: float) -> np.ndarray:
    rows, cols = size
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
    return np.broadcast_to(lat_grid_deg[:, np.newaxis], (rows, rows))

def get_water_masks(worldConfig):
    def extract_mask(name):
        return worldConfig[name]().astype(bool) if name in worldConfig.maps else np.zeros(worldConfig["sea_mask"]().shape, dtype=bool)
    return [extract_mask(name) for name in ['sea_mask', 'river_mask', 'lake_mask']]

def get_normalized_distance(mask, cell_size, max_distance):
    #array shape size
    distance = distance_transform_edt(~mask) * cell_size  # Convert to physical distance in meters
    distance = distance / (max_distance)  # Normalize by max possible distance in the grid
    distance[mask] = 0.0  # Ensure water cells have zero distance
    distance = np.exp(-distance)
    return distance

def get_water_cooling(worldConfig, config):
    #mask true = water, false = land
    masks = get_water_masks(worldConfig)
    cooling_effect = np.zeros(masks[0].shape) # initialize cooling effect map
    continentality = np.zeros(masks[0].shape) # placeholder for future continentality effect
    for mask, key in zip(masks, ['sea', 'river', 'lake']):
        if not np.any(mask):
            continue  # skip if no cells of this type

        water_distance = get_normalized_distance(mask, worldConfig.cell_size[0], config.cooling_effects[key][1]) # Normalize distance to a 200km scale
        cooling_effect += config.cooling_effects[key][0] * water_distance
        if key == 'sea':
            water_distance = get_normalized_distance(mask, worldConfig.cell_size[0], config.cooling_effects['continentality'][1]) # Normalize distance to a 200km scale
            continentality = config.cooling_effects['continentality'][0] * (1-water_distance)

    return cooling_effect, continentality

def get_sun_heating(di, dj, latitude, declination, solar_vectors):
    # terrain normal
    norm = np.sqrt(dj**2 + di**2 + 1.0)
    nx, ny, nz = -dj / norm, -di / norm, 1.0 / norm


    # Calculate solar vectors
    sx, sy, sz = solar_vectors

    slope_factor = np.clip(sx * nx + sy * ny + sz * nz, 0.0, 1.0)  # Only consider sun-facing slopes

    solar_factor = _solar_intensity_factor(latitude, declination)  # Base solar intensity based on latitude and season
    solar_factor = np.clip(solar_factor, 0.0, 1.0)  # Clamp to daylight only

    sun = slope_factor * solar_factor
    return normalize(sun)


def _solar_intensity_factor(latitude, declination):
    lat, dec = np.radians(latitude), np.radians(declination)

    # cosine of zenith angle (core physical approximation)
    cos_zenith = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec)

    # clamp to daylight only
    return np.clip(cos_zenith, 0.0, 1.0)