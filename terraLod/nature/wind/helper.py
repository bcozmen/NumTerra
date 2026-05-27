import numpy as np
from scipy.ndimage import map_coordinates

from .numba import soft_cap_numba, apply_warp_numba, scale_and_cap_numba

def get_prevailing_wind_base(world, lat_deg):
    zonal_base = _mean_zonal_wind(lat_deg)
    meridional_base = _mean_meridional_wind(lat_deg)
    base = (zonal_base, meridional_base)
    return base

#get prevailing wind with small 
def get_prevailing_wind(world, base, curve_strength=1.0, num_turns=1, rotation_sigma=30.0) -> np.ndarray:
    H, W = world.size  
    
    lon_phase = np.linspace(0, 2 * np.pi * num_turns, W, endpoint=False)[None, :]

    zonal_base, meridional_base = base
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
    cy, cx = center
    max_angle_rad = np.radians(max_angle)
    return apply_warp_numba(u, v, cx, cy, radius, max_angle_rad, direction)


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
    mean_speed = np.median(speed)
    mean_speed = max(mean_speed, 1e-5)  # Prevent division by zero for very low speeds

    return scale_and_cap_numba(u, v, mean_speed, soft_cap_speed, max_wind_speed)




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
