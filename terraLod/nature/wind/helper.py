import numpy as np
from scipy.ndimage import map_coordinates


#get prevailing wind with small 
def get_prevailing_wind(world, lat_deg, curve_strength=1.0, num_turns=1, rotation_sigma=30.0) -> np.ndarray:
    H, W = world.size  
    
    lon_phase = np.linspace(0, 2 * np.pi * num_turns, W, endpoint=False)[None, :]

    zonal_base = _mean_zonal_wind(lat_deg)
    meridional_base = _mean_meridional_wind(lat_deg)
    #meridional_base = np.sin(2.0 * np.radians(lat_deg)) * 2.0

    u_curve = np.cos(lon_phase) * curve_strength
    v_curve = np.sin(lon_phase) * curve_strength
    

    u_grid = zonal_base + u_curve
    v_grid = meridional_base + v_curve

    angle = np.random.normal(loc=0.0, scale=rotation_sigma)  # mean 0, stddev rotation_sigma degrees
    u_grid, v_grid = apply_rotation(u_grid, v_grid, angle)
    u_grid, v_grid = apply_warp(u_grid, v_grid)
    
    return u_grid, v_grid


def apply_rotation(u, v, angle_degrees):
    # angle_degrees range [-180, 180], positive for rightward rotation, negative for leftward
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    u_new = cos_a * u - sin_a * v
    v_new = sin_a * u + cos_a * v
    return u_new, v_new


def apply_warp(u, v):
    iters = np.random.randint(1, 5)  # Randomly choose 1 to 3 warps to apply
    for _ in range(iters):
        center = np.random.rand(2) * 1.5 - 0.25
        radius = np.random.rand() * 1.5 + 0.5
        max_angle = np.random.rand() * 50 + 20
        direction = np.random.choice([-1, 1])
        u, v = _apply_warp(u, v, center=center, radius=radius, max_angle=max_angle, direction=direction)
    return u, v
def _apply_warp(u, v, center=(0.5, -1.0), radius=1.5, max_angle=20, direction=1):
    #direction : +1 (left), -1 (right)

    max_angle = np.radians(max_angle)

    H, W = u.shape

    y, x = np.mgrid[0:H, 0:W]
    x = x / (W - 1)
    y = y / (H - 1)

    cy, cx = center

    dx = x - cx
    dy = y - cy

    r2 = dx*dx + dy*dy
    R2 = radius * radius + 1e-12

    # smooth falloff + bounded rotation field
    w = np.exp(-r2 / R2)

    theta = direction * max_angle * w

    c, s = np.cos(theta), np.sin(theta)

    xw = cx + dx * c - dy * s
    yw = cy + dx * s + dy * c

    # back to index space
    xw = xw * (W - 1)
    yw = yw * (H - 1)

    u2 = map_coordinates(u, [yw, xw], order=1, mode="reflect")
    v2 = map_coordinates(v, [yw, xw], order=1, mode="reflect")

    # rotate vectors consistently
    ur = u2 * c - v2 * s
    vr = u2 * s + v2 * c

    return ur, vr


def apply_coriolis(u, v, lat_grid, coriolis_fraction):
    """Rotate the wind field rightward in NH and leftward in SH."""
    angle = coriolis_fraction * np.sin(np.radians(lat_grid)) * (np.pi / 2.0)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    u_new = cos_a * u - sin_a * v
    v_new = sin_a * u + cos_a * v

    return u_new, v_new


def soft_cap(u, v, soft_cap_speed, max_speed):
    speed = np.hypot(u, v)
    capped_speed = _soft_cap(speed, soft_cap_speed, max_speed)
    scale = capped_speed / (speed + 1e-5)  # Avoid division by zero
    return u * scale, v * scale
def normalize_mean_and_cap(u, v, soft_cap_speed, max_wind_speed):
    speed = np.hypot(u, v)
    mean_speed = np.median(speed)
    mean_speed = max(mean_speed, 1e-5)  # Prevent division by zero for very low speeds

    u, v = u / mean_speed, v / mean_speed
    speed = np.hypot(u, v)

    capped_speed = _soft_cap(
        speed,
        soft_cap_speed,
        max_wind_speed
    )

    # avoid divide-by-zero
    scale = capped_speed / (speed + 1e-5)

    return u * scale, v * scale

def _hard_cap(x):
    return np.minimum(1.0/x, 1.0)


def _soft_cap(speed, soft, max_speed, alpha=3.0):
    speed = np.maximum(speed, 0.0)
    result = np.copy(speed)
    
    mask = speed > soft
    if np.any(mask):
        denom = max_speed - soft
        denom = max(denom, 1e-6)
        
        t = (speed[mask] - soft) / denom
        t = np.clip(t, 0.0, 1.0)
        shaped = 1.0 - np.exp(-alpha * t)
        
        result[mask] = soft + (max_speed - soft) * shaped
        
    return result


def _mean_zonal_wind(lat_deg):
    lat = np.radians(lat_deg)

    trades = -6.0 * np.exp(-(lat / np.radians(20))**2)
    westerlies = 12.0 * np.exp(-((lat_deg - 45)/18)**2)
    westerlies_s = 12.0 * np.exp(-((lat_deg + 45)/18)**2)
    polar = -4.0 * np.exp(-((np.abs(lat_deg) - 75)/12)**2)

    return trades + westerlies + westerlies_s + polar

def _mean_meridional_wind(lat_deg):
    return (
        2.0 * np.exp(-(lat_deg/20)**2)         # Hadley northward
        - 1.5 * np.exp(-((lat_deg-45)/15)**2)  # Ferrel southward
        + 1.0 * np.exp(-((lat_deg-70)/10)**2)  # Polar northward
    )
