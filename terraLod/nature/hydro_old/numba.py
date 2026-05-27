import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Shared neighbour offsets used by all kernels (8-connected grid).
# Axis-0 = row (i), axis-1 = column (j).
# ---------------------------------------------------------------------------
#  k:  0       1       2       3       4       5       6       7
# di: -1      -1      -1       0       0      +1      +1      +1
# dj: -1       0      +1      -1      +1      -1       0      +1


@njit(cache=True, parallel=True)
def compute_water_surface_mfd(
    height, water_depth, max_altitude, cell_size_x, cell_size_y, slope_exp=1.7
):
    """
    Multiple-Flow-Direction weights from the *water surface elevation*
    (terrain + water depth) instead of bare terrain.

    This is the key physical mechanism for realistic basin filling:

    * While a basin is not yet full (wse_basin < wse_rim) water continues
      to flow *into* the basin from all higher surrounding cells.
    * When the basin finally fills and wse_basin just exceeds wse_rim, the
      MFD immediately assigns positive weight toward the spillway direction.
    * The spillway cell then carries high discharge, which erodes it
      (see erode_step), lowering the rim and permanently establishing a river.

    Parameters
    ----------
    height       : (H,W) float32/64, normalised [0,1]
    water_depth  : (H,W) float64, metres of surface water
    max_altitude : float, metres corresponding to height=1
    cell_size_x/y: cell dimensions in metres (row / col axes)
    slope_exp    : Freeman (1991) exponent (default 1.7 — channels flow well)

    Returns
    -------
    flow_weights : (H,W,8) float32, normalised so each row sums to ≤1
    """
    xdim, ydim = height.shape
    flow_weights = np.zeros((xdim, ydim, 8), dtype=np.float32)

    DX   = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    DY   = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    diag = np.sqrt(cell_size_x**2 + cell_size_y**2)
    dist = np.array(
        [diag, cell_size_x, diag, cell_size_y, cell_size_y,
         diag, cell_size_x, diag],
        dtype=np.float32,
    )

    for x in prange(xdim):
        for y in range(ydim):
            wse   = height[x, y] * max_altitude + water_depth[x, y]
            total = np.float32(0.0)
            for k in range(8):
                nx = x + DX[k]
                ny = y + DY[k]
                if 0 <= nx < xdim and 0 <= ny < ydim:
                    nwse = height[nx, ny] * max_altitude + water_depth[nx, ny]
                    dh   = wse - nwse
                    if dh > 0.0:
                        s = (dh / dist[k]) ** slope_exp
                        flow_weights[x, y, k] = s
                        total += s
            if total > 0.0:
                for k in range(8):
                    flow_weights[x, y, k] /= total
    return flow_weights


@njit(cache=True)
def topo_sort(mfd_weights):
    """
    Kahn's BFS topological sort on the flow DAG defined by *mfd_weights*.

    Returns ``(order, n_valid)`` where ``order[0:n_valid]`` are flat cell
    indices sorted from upstream (watershed divides / ridge tops) to
    downstream (valley floors / basin pits / sea).

    Because the MFD DAG is acyclic by construction (weights only assigned
    to strictly lower neighbours), Kahn's algorithm is guaranteed to visit
    every cell exactly once.

    Pit cells (no outflow in any direction) have zero downstream neighbours
    so they appear at the END of the ordering — meaning all incoming
    sediment/water has already been routed into them by the time they are
    processed.
    """
    H = mfd_weights.shape[0]
    W = mfd_weights.shape[1]
    N = H * W
    DX = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    DY = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    # in_deg[i] = number of upstream cells that donate flow to cell i
    in_deg = np.zeros(N, dtype=np.int32)
    for x in range(H):
        for y in range(W):
            for k in range(8):
                if mfd_weights[x, y, k] > 0.0:
                    nx = x + DX[k]
                    ny = y + DY[k]
                    if 0 <= nx < H and 0 <= ny < W:
                        in_deg[nx * W + ny] += 1

    # seed queue with source cells (no upstream contributors)
    queue = np.empty(N, dtype=np.int32)
    head  = np.int32(0)
    tail  = np.int32(0)
    for i in range(N):
        if in_deg[i] == 0:
            queue[tail] = i
            tail += 1

    order = np.empty(N, dtype=np.int32)
    n_out = np.int32(0)
    while head < tail:
        idx          = queue[head]; head += 1
        order[n_out] = idx;        n_out += 1
        x = idx // W
        y = idx %  W
        for k in range(8):
            nx = x + DX[k]
            ny = y + DY[k]
            if 0 <= nx < H and 0 <= ny < W and mfd_weights[x, y, k] > 0.0:
                nidx         = nx * W + ny
                in_deg[nidx] -= 1
                if in_deg[nidx] == 0:
                    queue[tail] = nidx
                    tail += 1

    return order, n_out


@njit(cache=True, parallel=True)
def route_water_step(
    water_depth, rain_input, mfd_weights, order, n_valid,
    sea_mask, drain_frac, evap_rate,
):
    """
    One explicit water-routing sub-step in topological order.

    Steps per cell (upstream → downstream):
    1. Add ``rain_input`` [m] to every land cell.
    2. Each cell donates ``drain_frac × water_depth`` to its downstream
       neighbours proportional to mfd_weights.
    3. Sea cells drain completely (ocean boundary condition).
    4. Pit cells (zero outflow weight) accumulate water → lakes form.
    5. Per-step evaporation applied to all standing water (open-water loss).

    The water surface MFD (see compute_water_surface_mfd) handles the
    basin-filling transition automatically: once a pit fills to its rim,
    drain_frac routes the overflow directly toward the lowest gap.

    Parameters
    ----------
    water_depth : (H,W) float64, metres [mutated copy returned]
    rain_input  : (H,W) float64, metres added this sub-step
    mfd_weights : (H,W,8) float32
    order       : flat indices in topo order (from topo_sort)
    n_valid     : number of valid entries in order
    sea_mask    : (H,W) bool
    drain_frac  : fraction of standing water routed per step (≤1, stability)
    evap_rate   : fraction of standing water evaporated per step

    Returns
    -------
    new_water_depth : (H,W) float64
    discharge       : (H,W) float64, metres of water routed *through* each
                      cell this sub-step (proxy for river discharge)
    """
    H  = water_depth.shape[0]
    W  = water_depth.shape[1]
    DX = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    DY = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    wd        = water_depth + rain_input     # working copy (rain already added)
    discharge = np.zeros((H, W), dtype=np.float64)

    for i in range(n_valid):
        idx = order[i]
        x   = idx // W
        y   = idx %  W

        # Ocean cells act as sinks — drain all incoming water.
        if sea_mask[x, y]:
            discharge[x, y] += wd[x, y]
            wd[x, y]         = 0.0
            continue

        # Check whether this cell has any downhill flow direction.
        has_outflow = False
        for k in range(8):
            if mfd_weights[x, y, k] > 0.0:
                has_outflow = True
                break

        if has_outflow:
            outflow      = wd[x, y] * drain_frac
            wd[x, y]    -= outflow
            discharge[x, y] += outflow
            for k in range(8):
                w = mfd_weights[x, y, k]
                if w > 0.0:
                    nx = x + DX[k]
                    ny = y + DY[k]
                    if 0 <= nx < H and 0 <= ny < W:
                        wd[nx, ny] += outflow * w
        # else: pit cell — water stays and accumulates, building a lake.
        # When the lake surface reaches the rim, compute_water_surface_mfd
        # will assign positive weight toward the spillway on the next step.

    # Evaporation from all standing water (lakes, rivers) — parallel over rows.
    for x in prange(H):
        for y in range(W):
            if not sea_mask[x, y] and wd[x, y] > 0.0:
                wd[x, y] *= (1.0 - evap_rate)
            if wd[x, y] < 0.0:
                wd[x, y] = 0.0

    return wd, discharge


@njit(cache=True)
def erode_step(
    height, water_depth, discharge, mfd_weights, order, n_valid, sea_mask,
    cell_size_x, cell_size_y, max_altitude,
    erodibility, deposition_rate, capacity_k, max_erosion_norm,
):
    """
    Transport-limited stream-power erosion and sediment routing.

    Physics
    -------
    Stream power per unit width:
        Ω = discharge [m/step] × slope [m/m]
    Sediment transport capacity:
        C = capacity_k × Ω
    If the sediment load arriving from upstream < C:
        Erode: detach material from the bed.
        height -= E / max_altitude,  E = min(erodibility × Ω, max_erosion_norm)
    If the sediment load > C:
        Deposit: drop the excess fraction.
        height += deposit_m / max_altitude

    Sediment is then forwarded downstream proportional to mfd_weights.
    Pit cells and sea cells deposit all remaining sediment unconditionally
    (alluvial fill of basins, delta deposition at the coast).

    Key consequence for basin dynamics
    ------------------------------------
    The rim / spillway cell experiences the highest discharge concentration
    once the lake overflows.  Its stream power Ω is large → it erodes
    faster than surrounding terrain → the rim is progressively lowered →
    a permanent river channel is incised through it.

    Parameters
    ----------
    height, water_depth : (H,W) float64/32, normalised and metres
    discharge           : (H,W) float64, from route_water_step
    mfd_weights         : (H,W,8) float32
    erodibility         : K  [norm_height / (m²/step)]
    deposition_rate     : fraction of excess sediment deposited per step
    capacity_k          : C = capacity_k × Ω
    max_erosion_norm    : hard cap on erosion per step (normalised height)

    Returns
    -------
    height_delta : (H,W) float64, normalised height change (+ = deposition)
    """
    H  = height.shape[0]
    W  = height.shape[1]
    DX = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    DY = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    diag = np.sqrt(cell_size_x**2 + cell_size_y**2)
    dist = np.array(
        [diag, cell_size_x, diag, cell_size_y, cell_size_y,
         diag, cell_size_x, diag],
        dtype=np.float64,
    )

    height_delta = np.zeros((H, W), dtype=np.float64)
    # sed[x,y]: metres of sediment in transit arriving at cell (x,y) from
    # upstream cells already processed in topo order.
    sed = np.zeros((H, W), dtype=np.float64)

    for i in range(n_valid):
        idx = order[i]
        x   = idx // W
        y   = idx %  W

        # Ocean: deposit all arriving sediment (coastal/delta accumulation).
        if sea_mask[x, y]:
            height_delta[x, y] += sed[x, y] / max_altitude
            sed[x, y]           = 0.0
            continue

        d = discharge[x, y]

        # No flow → deposit any upstream sediment that arrived here.
        if d <= 0.0:
            height_delta[x, y] += sed[x, y] / max_altitude
            sed[x, y]           = 0.0
            continue

        # --- local slope from water surface gradient (MFD-weighted) ---
        wse_here  = height[x, y] * max_altitude + water_depth[x, y]
        slope = 0.0
        for k in range(8):
            w = mfd_weights[x, y, k]
            if w > 0.0:
                nx = x + DX[k]
                ny = y + DY[k]
                if 0 <= nx < H and 0 <= ny < W:
                    wse_nb = height[nx, ny] * max_altitude + water_depth[nx, ny]
                    dh     = wse_here - wse_nb
                    slope += w * (dh / dist[k])

        omega    = d * slope              # stream power [m²/step]
        capacity = capacity_k * omega     # carrying capacity [m of sediment]
        total_sed = sed[x, y]             # load arriving from upstream

        # Detachment: river erodes bed when under-loaded.
        if total_sed < capacity:
            erosion = min(erodibility * omega, max_erosion_norm)
            erosion = min(erosion, max(height[x, y], 0.0))   # can't erode below 0
            height_delta[x, y] -= erosion
            total_sed           += erosion * max_altitude     # norm → metres of sediment

        # Deposition: river drops excess load when over-loaded.
        elif total_sed > capacity:
            excess   = total_sed - capacity
            deposit_m = excess * deposition_rate
            height_delta[x, y] += deposit_m / max_altitude
            total_sed           -= deposit_m

        # Propagate remaining sediment downstream proportional to mfd_weights.
        has_outflow = False
        for k in range(8):
            if mfd_weights[x, y, k] > 0.0:
                has_outflow = True
                break

        if has_outflow:
            for k in range(8):
                w = mfd_weights[x, y, k]
                if w > 0.0:
                    nx = x + DX[k]
                    ny = y + DY[k]
                    if 0 <= nx < H and 0 <= ny < W:
                        sed[nx, ny] += total_sed * w
            sed[x, y] = 0.0
        else:
            # Pit / local basin: deposit all sediment (alluvial fill).
            height_delta[x, y] += total_sed / max_altitude
            sed[x, y]           = 0.0

    return height_delta



@njit(cache=True, parallel=True)
def compute_mfd_weights(height_map, max_altitude, cell_size_x, cell_size_y, slope_exp=1.7):
    """
    For every cell compute normalised flow weights to each of its 8 neighbours.

    A neighbour only receives weight when it is strictly lower than the source
    cell.  Weights are proportional to ``slope ** slope_exp`` (Freeman 1991).
    Using slope_exp > 1 concentrates flow into the steepest paths, producing
    channelised rivers rather than a diffuse wetness haze.  slope_exp = 1 is
    the original linear MFD; slope_exp → ∞ approaches single-flow D8.
    Recommended range: 1.0 (dispersed) – 2.5 (channelised).

    Returns
    -------
    flow_weights : float32 array (xdim, ydim, 8)
        Axis-2 ordering matches _NB_DX / _NB_DY (see module constants).
    """
    xdim, ydim = height_map.shape
    flow_weights = np.zeros((xdim, ydim, 8), dtype=np.float32)

    height_map = height_map * max_altitude  # rescale to actual altitudes in meters for slope calculations

    # local copies required because numba @njit cannot capture module-level
    # numpy arrays as constants — values match _NB_DX / _NB_DY / _NB_DIST
    dx   = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy   = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    diag = np.sqrt(cell_size_x**2 + cell_size_y**2)

    dist = np.array([
        diag,          # (-1,-1)
        cell_size_x,   # (-1, 0)
        diag,          # (-1,+1)
        cell_size_y,   # ( 0,-1)
        cell_size_y,   # ( 0,+1)
        diag,          # (+1,-1)
        cell_size_x,   # (+1, 0)
        diag           # (+1,+1)
    ], dtype=np.float32)

    for x in prange(xdim):
        for y in range(ydim):
            h     = height_map[x, y]
            total = np.float32(0.0)

            for k in range(8):
                nx = x + dx[k]
                ny = y + dy[k]
                if 0 <= nx < xdim and 0 <= ny < ydim:
                    dh = h - height_map[nx, ny]
                    if dh > 0.0:
                        slope = (dh / dist[k]) ** slope_exp
                        flow_weights[x, y, k] = slope
                        total += slope

            if total > 0.0:
                for k in range(8):
                    flow_weights[x, y, k] /= total

    return flow_weights