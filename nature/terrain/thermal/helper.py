import numpy as np
from scipy.ndimage import distance_transform_edt


def get_temperature_grid(size, max_size, latitude) -> np.ndarray:
    rows, cols = size
    lat_grid = _get_latitude_grid(rows, max_size, latitude)
    temp = -20 + 52 * np.cos(lat_grid)
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

def get_water_masks(worldConfig):
    sea_mask = worldConfig["sea_mask"]().astype(bool)
    try:
        river_mask = worldConfig["river_mask"]().astype(bool)
    except:
        river_mask = np.full(sea_mask.shape, False, dtype=bool)
    try:
        lake_mask = worldConfig["lake_mask"]().astype(bool)
    except:
        lake_mask = np.full(sea_mask.shape, False, dtype=bool)
    return sea_mask, river_mask, lake_mask

def get_water_cooling(worldConfig, config):
    #mask true = water, false = land
    masks = get_water_masks(worldConfig)
    cooling_effect = np.zeros(masks[0].shape) # initialize cooling effect map
    continentality = np.zeros(masks[0].shape) # placeholder for future continentality effect
    for mask, key in zip(masks, ['sea', 'river', 'lake']):
        distance = distance_transform_edt(~mask) * worldConfig.cell_size[0] # exact Euclidean distance in meters
        cooling_effect += config.cooling_effects[key][0] * np.exp(-distance / config.cooling_effects[key][1])
        if key == 'sea':
            continentality += config.cooling_effects['continentality'][0] * ( 1 - np.exp( -distance / config.cooling_effects['continentality'][1]))

    # ------------------------------------------------------------------
    # Eastern Boundary Current (cold upwelling) approximation.
    # On real Earth, cold currents hug the western shores of continents at
    # mid-latitudes (California, Benguela, Humboldt, Canary currents).
    # These form because trade-winds push surface water offshore on the
    # eastern side of ocean basins (left / west side of domain = column 0).
    #
    # Proxy: sea cells in the left third of the domain at ±15°–50° latitude
    # receive extra cooling that decays eastward.  The latitude envelope uses
    # a Gaussian peaked at ±30° (the horse-latitude upwelling belt).
    # Magnitude: ~2–4 °C cooling, consistent with real SST anomalies.
    # ------------------------------------------------------------------
    sea_mask = masks[0]
    rows, cols = sea_mask.shape
    lat_rows = _get_latitude_grid(rows, worldConfig.max_size, worldConfig.latitude)
    lat_deg  = np.degrees(lat_rows)                       # (rows, 1)

    # Latitude envelope: peaks at |lat| = 30°, negligible at equator and >60°
    lat_envelope = np.exp(-((np.abs(lat_deg) - 30.0) / 15.0) ** 2)   # (rows, 1)

    # Longitude (column) decay: strongest at western edge, ~e-fold over 20% domain
    col_frac    = np.linspace(0.0, 1.0, cols)[None, :]               # (1, cols)
    lon_decay   = np.exp(-col_frac / 0.20)                            # (1, cols)

    cold_current = 3.5 * lat_envelope * lon_decay                     # (rows, cols)
    cooling_effect += cold_current * sea_mask

    return cooling_effect, continentality

def get_sun_heating(worldConfig, config):
    di, dj = worldConfig["grad_i"](), worldConfig["grad_j"]()

    # Build terrain normal vector directly from gradients — avoids an
    # aspect/slope round-trip that can mis-assign axes.
    # Convention: row index increases southward, col index increases eastward.
    #   di = dH/d(row) → positive = terrain slopes downward toward South
    #   dj = dH/d(col) → positive = terrain slopes downward toward East
    # The outward normal of the surface therefore points:
    #   East component  : -dj  (terrain rising eastward → normal tilts East)
    #   North component : -di  (terrain rising northward → normal tilts North)
    #   Up component    :  1   (base upward component before normalisation)
    nx = -dj
    ny = -di
    nz = np.ones_like(nx)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    lat = np.radians(worldConfig.latitude)

    declination = np.radians(config.season_to_declination[worldConfig.season])
    hour_angle = np.radians(15 * (worldConfig.hour - 12))

    solar_altitude = np.arcsin(
        np.sin(lat) * np.sin(declination) +
        np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    )

    # Standard astronomical azimuth: 0 = North, π/2 = East, π = South, 3π/2 = West
    solar_azimuth = np.arctan2(
        -np.cos(declination) * np.sin(hour_angle),
        np.cos(lat) * np.sin(declination) - np.sin(lat) * np.cos(declination) * np.cos(hour_angle)
    )

    # Solar direction vector in (East, North, Up) coordinates — must match the
    # terrain normal convention above.
    #   Eastward  : sin(azimuth)   (azimuth 0=N → sin(0)=0, azimuth π/2=E → sin=1)
    #   Northward : cos(azimuth)   (azimuth 0=N → cos(0)=1)
    #   Upward    : sin(altitude)
    sx = np.cos(solar_altitude) * np.sin(solar_azimuth)   # East
    sy = np.cos(solar_altitude) * np.cos(solar_azimuth)   # North
    sz = np.sin(solar_altitude)                            # Up

    sun = np.clip(nx * sx + ny * sy + nz * sz, 0, 1)
    return sun