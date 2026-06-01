import numpy as np
from numba import njit, prange

@njit(parallel=True)
def d8_water_routing(H, M_sea, Ws, max_altitude, dx, dy, dt, slope_exponent, flow_rate):
    """
    Multi-directional D8-style surface water routing (pull approach, parallel-safe).

    Water drains from each cell to all 8 downhill neighbours, weighted by
    slope^slope_exponent.  Each destination cell independently re-derives how much
    water flows into it from each uphill neighbour, so no atomic writes are needed.

    Parameters
    ----------
    H              : float32 (rows, cols)  — normalised terrain height [0, 1]
    M_sea          : float32 (rows, cols)  — sea mask (1.0 = ocean, 0.0 = land)
    Ws             : float32 (rows, cols)  — surface water [mm]
    max_altitude   : float                 — physical height of H=1 [m]
    dx, dy         : float                 — cell size [m]
    dt             : float                 — timestep [hr]
    slope_exponent : float                 — steeper slopes receive exponentially more flow
                                             (1 = linear, 2 = quadratic, …)
    flow_rate      : float                 — fraction of water drained per hour at full slope

    Returns
    -------
    Ws_out : float32 (rows, cols)
    """
    rows, cols = H.shape
    Ws_out = np.empty_like(Ws)

    # 8-connected neighbour offsets (row-delta, col-delta)
    NDI = np.array((-1, -1, -1,  0,  0,  1,  1,  1), dtype=np.int64)
    NDJ = np.array((-1,  0,  1, -1,  1, -1,  0,  1), dtype=np.int64)
    # reverse mapping: index in neighbour that points back to the centre
    REV = np.array((7, 6, 5, 4, 3, 2, 1, 0), dtype=np.int64)

    # Precompute geometric distances for each neighbour index (avoids sqrt in inner loops)
    dx2 = dx * dx
    dy2 = dy * dy
    diag = np.sqrt(dx2 + dy2)
    DIST = np.empty(8, dtype=np.float64)
    # Matches the (NDI,NDJ) ordering above
    DIST[0] = diag
    DIST[1] = dx
    DIST[2] = diag
    DIST[3] = dy
    DIST[4] = dy
    DIST[5] = diag
    DIST[6] = dx
    DIST[7] = diag

    flow_fraction = min(1.0, flow_rate * dt)   # fraction leaving per timestep (stability cap)

    # Precompute terrain (metres) and effective surface for this timestep.
    # terrain_m can be cached across timesteps externally if H never changes.
    terrain_m = np.empty_like(H)
    for i in range(rows):
        for j in range(cols):
            terrain_m[i, j] = H[i, j] * max_altitude

    surface = np.empty_like(H)
    for i in range(rows):
        for j in range(cols):
            surface[i, j] = terrain_m[i, j] + Ws[i, j] * 0.001

    # First pass: compute routing weights to each of the 8 neighbours and totals.
    weights = np.zeros((rows, cols, 8), dtype=np.float32)
    totals = np.zeros((rows, cols), dtype=np.float32)
    for i in prange(rows):
        for j in range(cols):
            if M_sea[i, j] > 0.5:
                totals[i, j] = 0.0
                continue

            h_c = surface[i, j]
            total_w = 0.0

            for k in range(8):
                ni = i + NDI[k]
                nj = j + NDJ[k]
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue

                h_n = surface[ni, nj]
                slope = (h_c - h_n) / DIST[k]
                if slope > 0.0:
                    # Special-case common exponents to avoid slow generic pow
                    if slope_exponent == 2.0:
                        w = slope * slope
                    elif slope_exponent == 1.0:
                        w = slope
                    else:
                        w = slope ** slope_exponent

                    weights[i, j, k] = np.float32(w)
                    total_w += w

            totals[i, j] = np.float32(total_w)

    # Second pass: pull inflow from neighbours using precomputed neighbour weights.
    for i in prange(rows):
        for j in range(cols):
            net_inflow = 0.0

            for k in range(8):
                ni = i + NDI[k]
                nj = j + NDJ[k]
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue

                total_n = totals[ni, nj]
                if total_n <= 0.0:
                    continue

                # weight of neighbour (ni,nj) towards this cell
                rr = REV[k]
                w_to_me = weights[ni, nj, rr]
                if w_to_me <= 0.0:
                    continue

                own_outflow_n = Ws[ni, nj] * flow_fraction
                net_inflow += own_outflow_n * (w_to_me / total_n)

            if M_sea[i, j] > 0.5:
                Ws_out[i, j] = np.float32(Ws[i, j] + net_inflow)
            else:
                own_outflow = Ws[i, j] * flow_fraction if totals[i, j] > 0.0 else 0.0
                Ws_out[i, j] = max(np.float32(0.0), np.float32(Ws[i, j] - own_outflow + net_inflow))

    return Ws_out
