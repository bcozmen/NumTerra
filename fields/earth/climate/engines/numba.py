import numpy as np
from numba import njit, prange


@njit(parallel=True)
def precompute_horizon_angles(H, N_dirs, max_altitude, dx, dy):
    """
    Precomputes the maximum horizon elevation angle (radians) for every cell in N_dirs
    evenly-spaced azimuth directions.  Call once after terrain is set; the result is
    valid as long as H does not change.

    Returns
    -------
    horizon_angles : float32 array, shape (rows, cols, N_dirs)
        horizon_angles[r, c, k] = max elevation angle (rad) from cell (r,c) looking
        in the direction alpha_k = 2*pi*k/N_dirs  (East=0, North=pi/2).

    Implementation notes
    --------------------
    - Tracks tan(elevation) in the inner loop to avoid expensive arctan2 per step;
      one arctan is computed per (cell, direction) pair at the end.
    - Early-termination: once the ceiling-tan  (max_altitude - h_origin) / d_cumulative
      falls below the running max, no further terrain can improve the horizon angle.
    - max_steps uses ceil(sqrt(2) * max_dim) so diagonal rays reach map corners.
    """
    rows, cols = H.shape
    # Diagonal rays need ~1.42× more steps than axis-aligned ones to reach corners.
    max_steps = int(1.45 * (rows if rows > cols else cols)) + 1
    horizon_angles = np.empty((rows, cols, N_dirs), dtype=np.float32)

    for r in prange(rows):
        for c in range(cols):
            h_origin   = np.float64(H[r, c]) * max_altitude
            h_headroom = max_altitude - h_origin   # max possible height gain

            for k in range(N_dirs):
                alpha = 2.0 * np.pi * k / N_dirs
                cos_a = np.cos(alpha)   # East  component (→ col / j axis)
                sin_a = np.sin(alpha)   # North component (→ row / i axis)

                # Scale so the dominant axis advances exactly 1 cell per step
                abs_cos = cos_a if cos_a >= 0.0 else -cos_a
                abs_sin = sin_a if sin_a >= 0.0 else -sin_a
                scale   = abs_cos if abs_cos > abs_sin else abs_sin
                if scale < 1e-9:
                    scale = 1.0
                # With imshow(origin='lower'): row (i) = North, col (j) = East.
                # alpha=0 → East (+j), alpha=pi/2 → North (+i)
                di = sin_a / scale   # row direction  (North)
                dj = cos_a / scale   # col direction  (East)

                # Physical metres per step
                d_step = np.sqrt((di * dx) * (di * dx) + (dj * dy) * (dj * dy))

                # Track tan(elevation) — avoids arctan2 in the hot inner loop.
                max_tan  = np.float64(-1e10)
                curr_i   = np.float64(r)
                curr_j   = np.float64(c)
                d_cum    = np.float64(0.0)

                for _ in range(max_steps):
                    curr_i += di
                    curr_j += dj
                    d_cum  += d_step

                    # Early exit: ceiling tan (best any future terrain can offer) < current max
                    if h_headroom < max_tan * d_cum:
                        break

                    if curr_i < 0.0 or curr_j < 0.0:
                        break
                    ci = int(curr_i)
                    cj = int(curr_j)
                    if ci >= rows - 1 or cj >= cols - 1:
                        break

                    fi = curr_i - ci
                    fj = curr_j - cj
                    h_t = (H[ci,     cj    ] * (1.0 - fi) * (1.0 - fj) +
                           H[ci + 1, cj    ] * fi          * (1.0 - fj) +
                           H[ci,     cj + 1] * (1.0 - fi) * fj          +
                           H[ci + 1, cj + 1] * fi          * fj         ) * max_altitude

                    tan_e = (h_t - h_origin) / d_cum
                    if tan_e > max_tan:
                        max_tan = tan_e

                # One arctan per (cell, direction) — not in the hot loop
                horizon_angles[r, c, k] = np.float32(np.arctan(max_tan))

    return horizon_angles


@njit(parallel=True)
def lookup_shadow_from_horizon(horizon_angles, sx, sy, sz):
    """
    Derives a shadow map from precomputed horizon angles for the given sun direction.
    Per-step cost is O(rows*cols) — one lookup + comparison per cell.

    Parameters
    ----------
    horizon_angles : float32 (rows, cols, N_dirs) — output of precompute_horizon_angles
    sx, sy, sz     : scalar floats — sun direction vector (East, North, Up)

    Returns
    -------
    shadow_map : float32 (rows, cols),  1.0 = lit,  0.0 = shadowed
    """
    rows, cols, N_dirs = horizon_angles.shape
    shadow_map = np.ones((rows, cols), dtype=np.float32)

    if sz <= 0.0:
        for r in prange(rows):
            for c in range(cols):
                shadow_map[r, c] = 0.0
        return shadow_map

    norm_xy = np.sqrt(sx * sx + sy * sy)
    if norm_xy < 1e-5:
        return shadow_map   # sun directly overhead: no cast shadows

    sun_altitude = np.arctan2(sz, norm_xy)

    # Map sun azimuth to [0, 2π) matching alpha_k = 2π*k/N_dirs convention
    sun_azimuth = np.arctan2(sy, sx)   # angle from East, CCW toward North
    if sun_azimuth < 0.0:
        sun_azimuth += 2.0 * np.pi

    frac = sun_azimuth / (2.0 * np.pi) * N_dirs
    k0   = int(frac) % N_dirs
    k1   = (k0 + 1) % N_dirs
    t    = frac - int(frac)

    for r in prange(rows):
        for c in range(cols):
            h_angle = horizon_angles[r, c, k0] * (1.0 - t) + horizon_angles[r, c, k1] * t
            if sun_altitude <= h_angle:
                shadow_map[r, c] = 0.0

    return shadow_map




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
                if M_sea[ni, nj] > 0.5:
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
            if M_sea[i, j] > 0.5:
                Ws_out[i, j] = Ws[i, j]
                continue

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

            own_outflow = Ws[i, j] * flow_fraction if totals[i, j] > 0.0 else 0.0
            Ws_out[i, j] = max(np.float32(0.0), np.float32(Ws[i, j] - own_outflow + net_inflow))

    return Ws_out
