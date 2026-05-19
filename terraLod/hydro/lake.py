import numpy as np
from numba import njit

def simulate_step(
    height_map,
    precipitation,
    temperature,
    flow_weights,
    basin_id,
    spill_level,
    basin_area,
    infiltration_rate=0.2,
    evaporation_rate=0.05
):
    """
    One full hydro-geomorphic timestep.
    """

    xdim, ydim = height_map.shape

    # -------------------------
    # 1. initial water input
    # -------------------------
    water = precipitation.astype(np.float32).copy()

    # -------------------------
    # 2. simple atmospheric loss
    # -------------------------
    temp_factor = np.clip((temperature + 10.0) / 35.0, 0.0, 1.0)

    water *= (1.0 - evaporation_rate * temp_factor)
    water *= (1.0 - infiltration_rate * (0.5 + 0.5 * temp_factor))

    # -------------------------
    # 3. lake equilibrium
    # -------------------------
    lake_level = update_lake_state(
        height_map, water, basin_id, spill_level, basin_area
    )

    # -------------------------
    # 4. overflow → rivers
    # -------------------------
    overflow = compute_overflow(
        height_map, lake_level, spill_level, basin_id
    )

    # -------------------------
    # 5. total discharge
    # -------------------------
    flow_acc = overflow  # you can also add MFD routing here

    # -------------------------
    # 6. erosion
    # -------------------------
    erosion = compute_erosion(height_map, flow_acc, temperature)

    height_map = height_map - erosion

    return height_map, flow_acc, lake_level

@njit(cache=True)
def update_lake_state(height_map, water, basin_id, spill_level, basin_area):
    """
    Converts water flux into lake surface levels per basin.
    """

    n = water.size
    nb = spill_level.shape[0]

    lake_mass = np.zeros(nb, dtype=np.float32)
    lake_level = np.zeros(nb, dtype=np.float32)

    h = height_map.ravel()
    w = water.ravel()
    bmap = basin_id.ravel()

    # accumulate water per basin
    for i in range(n):
        b = bmap[i]
        if b >= 0:
            lake_mass[b] += w[i]

    # convert to water height (volume / area)
    for b in range(nb):
        if basin_area[b] > 0:
            lake_level[b] = lake_mass[b] / basin_area[b]

            # clamp at spill elevation
            if lake_level[b] > spill_level[b]:
                lake_level[b] = spill_level[b]

    return lake_level

@njit(cache=True)
def compute_overflow(height_map, lake_level, spill_level, basin_id):
    """
    Water above spill becomes river discharge.
    """

    xdim, ydim = height_map.shape
    n = xdim * ydim

    overflow = np.zeros(n, dtype=np.float32)

    bmap = basin_id.ravel()

    for i in range(n):
        b = bmap[i]
        if b < 0:
            continue

        excess = lake_level[b] - spill_level[b]
        if excess > 0.0:
            overflow[i] = excess

    return overflow.reshape((xdim, ydim))

@njit(cache=True)
def compute_erosion(height_map, flow_acc, temperature):
    xdim, ydim = height_map.shape
    erosion = np.zeros_like(height_map, dtype=np.float32)

    for x in range(xdim):
        for y in range(ydim):

            discharge = flow_acc[x, y]
            T = temperature[x, y]

            # temperature factor
            temp_factor = (T + 10.0) / 35.0
            if temp_factor < 0.0:
                temp_factor = 0.0
            elif temp_factor > 1.0:
                temp_factor = 1.0

            # slope (cheap stencil)
            center = height_map[x, y]
            sx = 0.0
            sy = 0.0

            if x > 0: sx += center - height_map[x-1, y]
            if x < xdim-1: sx += center - height_map[x+1, y]
            if y > 0: sy += center - height_map[x, y-1]
            if y < ydim-1: sy += center - height_map[x, y+1]

            slope = (sx*sx + sy*sy) ** 0.5

            erosion[x, y] = 1e-4 * discharge * slope * temp_factor

    return erosion