import numpy as np
from scipy.ndimage import distance_transform_edt
from terraLod.utils import normalize, get_lat_grid, get_normalized_distance_to_mask, get_water_masks





def get_temperature_grid(size, max_size, latitude, phase) -> np.ndarray:
    rows, cols = size
    lat_grid = get_lat_grid(latitude, size, max_size)

    mean_temp = _latitude_mean_temperature(lat_grid)
    seasonal_amplitude = _latitude_seasonal_amplitude(lat_grid)
    hemisphere_sign = np.where(lat_grid >= 0, 1, -1)  # +1 for Northern Hemisphere, -1 for Southern Hemisphere

    seasonal_offset = phase * seasonal_amplitude * hemisphere_sign

    return mean_temp, seasonal_offset
    

def season_phase(season):
    if season == 'summer':
        return 1.0
    elif season == 'winter':
        return -1.0
    return 0.0

def _latitude_mean_temperature(latitude):
    lat_abs = np.abs(latitude)
    return 50.0 * np.cos(np.radians(lat_abs)) - 27.0  # Ranges from 17°C at equator to -10°C at poles

def _latitude_seasonal_amplitude(latitude):
    lat_abs = np.abs(latitude)
    return 2.0 + (27.0 / 90.0) * lat_abs 



def get_water_cooling(worldConfig, config):
    #mask true = water, false = land
    masks = get_water_masks(worldConfig)
    cooling_effect = np.zeros(masks[0].shape) # initialize cooling effect map
    continentality = np.zeros(masks[0].shape) # placeholder for future continentality effect
    for mask, key in zip(masks, ['sea', 'river', 'lake']):
        if not np.any(mask):
            continue  # skip if no cells of this type

        water_distance = get_normalized_distance_to_mask(mask, worldConfig.cell_size[0], config.cooling_effects[key][1]) # Normalize distance to a 200km scale
        water_distance = np.exp(-water_distance)  # Exponential decay for smoother transition
        cooling_effect += config.cooling_effects[key][0] * water_distance
        if key == 'sea':
            water_distance = get_normalized_distance_to_mask(mask, worldConfig.cell_size[0], config.cooling_effects['continentality'][1]) # Normalize distance to a 200km scale
            water_distance = np.exp(-water_distance)  # Exponential decay for smoother transition
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