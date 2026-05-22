import numpy as np

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

def mean_zonal_wind(lat_deg):
    lat = np.radians(lat_deg)

    trades = -6.0 * np.exp(-(lat / np.radians(20))**2)
    westerlies = 12.0 * np.exp(-((lat_deg - 45)/18)**2)
    westerlies_s = 12.0 * np.exp(-((lat_deg + 45)/18)**2)
    polar = -4.0 * np.exp(-((np.abs(lat_deg) - 75)/12)**2)

    return trades + westerlies + westerlies_s + polar

def mean_meridional_wind(lat_deg):
    return (
        2.0 * np.exp(-(lat_deg/20)**2)         # Hadley northward
        - 1.5 * np.exp(-((lat_deg-45)/15)**2)  # Ferrel southward
        + 1.0 * np.exp(-((lat_deg-70)/10)**2)  # Polar northward
    )



def get_prevailing_wind(world, lat_rows, curve_strength=1.0, num_turns=1) -> np.ndarray:
    lat_deg = lat_rows[:, None]
    H, W = world.size  
    
    lon_phase = np.linspace(0, 2 * np.pi * num_turns, W, endpoint=False)[None, :]

    zonal_base = mean_zonal_wind(lat_deg)
    meridional_base = np.sin(2.0 * np.radians(lat_deg)) * 2.0

    u_curve = np.cos(lon_phase) * curve_strength
    v_curve = np.sin(lon_phase) * curve_strength

    u_grid = np.broadcast_to(zonal_base + u_curve, world.size)
    v_grid = np.broadcast_to(meridional_base + v_curve, world.size)

    return np.stack([u_grid, v_grid], axis=-1)



def apply_coriolis(u, v, lat_grid, coriolis_fraction):
    """Rotate the wind field rightward in NH and leftward in SH."""
    angle = coriolis_fraction * np.sin(np.radians(lat_grid)) * (np.pi / 2.0)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    u_new = cos_a * u - sin_a * v
    v_new = sin_a * u + cos_a * v

    return u_new, v_new



def normalize_mean_and_cap(u, v, soft_cap_speed, max_wind_speed):
    speed = np.hypot(u, v)
    mean_speed = np.mean(speed)
    mean_speed = max(mean_speed, 1e-5)  # Prevent division by zero for very low speeds

    u, v = u / mean_speed, v / mean_speed
    speed = np.hypot(u, v)

    capped_speed = soft_cap(
        speed,
        soft_cap_speed,
        max_wind_speed
    )

    # avoid divide-by-zero
    scale = capped_speed / (speed + 1e-5)

    return u * scale, v * scale

def hard_cap(x):
    return np.minimum(1.0/x, 1.0)
def soft_cap(speed, soft, max_speed, alpha=3.0):
    speed = np.maximum(speed, 0.0)

    denom = max_speed - soft
    denom = max(denom, 1e-6)

    t = (speed - soft) / denom
    t = np.clip(t, 0.0, 1.0)

    shaped = 1.0 - np.exp(-alpha * t)

    return soft + (max_speed - soft) * shaped