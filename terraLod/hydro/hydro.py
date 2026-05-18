import numpy as np
from numba import njit, prange

from pathFinding.heap import MinHeap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def river_erosion(grid, accumulation_threshold, carve_strength, valley_width_cells,
                  river_iterations, max_rivers=None, sea_level_percentile=0.0,
                  max_lake_area_fraction=0.01, min_lake_depth=0.0, **kwargs):
    """Identify rivers via flow accumulation and carve valley channels.

    Parameters
    ----------
    grid                    : normalised [0,1] height map (float32/float64).
    accumulation_threshold  : fraction of total cells that must drain through
                              a cell for it to be considered a river.
    carve_strength          : maximum carve depth (in normalised height units).
                              Resolution-independent when derived from
                              ``carve_strength_m / max_altitude``.
    valley_width_cells      : half-width of the valley smoothing kernel (cells).
                              Derived from ``valley_width_m / dx_m``.
    river_iterations        : reserved for future use.
    max_rivers              : if set (int), auto-raise the accumulation threshold
                              so at most this many distinct river networks remain.
    sea_level_percentile    : percentile of the carved height map to use as sea
                              level (e.g. 0.15 → 15th percentile).
    max_lake_area_fraction  : fraction of total cells — basins larger than this
                              are discarded (kills flooded plains).
                              Default 0.01 (1 % of map).  Set to 1.0 to disable.
    min_lake_depth          : minimum normalised depth for a basin to be kept.
                              Derived from ``min_lake_depth_m / max_altitude``.
                              Discards nearly-flat depressions that flood whole
                              valleys.  Default 0.0 (no depth filter).

    Returns
    -------
    carved       : (H, W) float64 — eroded height map with channels carved.
    river_mask   : (H, W) float32 — water coverage in [0,1].
    river_height : (H, W) float64 — water-surface elevation.
    sea_mask     : (H, W) bool
    lake_mask    : (H, W) bool
    sea_level    : float
    """
    filled  = _priority_flood(grid)
    acc     = _flow_accumulation(filled)
    n_cells = grid.shape[0] * grid.shape[1]
    thresh  = accumulation_threshold * n_cells

    if max_rivers is not None and max_rivers > 0:
        thresh = _threshold_for_max_rivers(acc, thresh, max_rivers)

    carved = _carve_rivers(filled, acc, thresh, carve_strength, valley_width_cells)

    sea_level, sea_mask = _compute_sea_mask(carved, sea_level_percentile)
    channel_mask        = (acc >= int(thresh)) & (~sea_mask)

    raw_lake_depth = filled - grid.astype(np.float64)
    lake_mask      = _compute_lake_mask(
        raw_lake_depth, sea_mask, channel_mask, n_cells, max_lake_area_fraction, min_lake_depth
    )

    river_mask   = _build_river_mask(channel_mask, acc, carved, sea_mask, lake_mask, raw_lake_depth)
    river_height = _build_water_heights(grid.shape, sea_mask, lake_mask, channel_mask,
                                        sea_level, filled, carved)

    return carved, river_mask, river_height, sea_mask, lake_mask, sea_level


# ---------------------------------------------------------------------------
# Orchestration helpers (Python-level, called only from river_erosion)
# ---------------------------------------------------------------------------

def _compute_sea_mask(carved, sea_level_percentile):
    """Return (sea_level, sea_mask) for the carved terrain."""
    if sea_level_percentile <= 0.0:
        return 0.0, np.zeros(carved.shape, dtype=np.bool_)
    sea_level = float(np.percentile(carved, sea_level_percentile * 100.0))
    return sea_level, _flood_below_level(carved, sea_level)


def _compute_lake_mask(raw_lake_depth, sea_mask, channel_mask,
                       n_cells, max_lake_area_fraction, min_lake_depth):
    """Derive a filtered inland lake mask."""
    EPS            = 1e-6
    raw_lake_mask  = (raw_lake_depth > EPS) & (~sea_mask)
    min_lake_cells = max(4, int(n_cells * 0.00005))
    max_lake_cells = max(min_lake_cells + 1, int(n_cells * max_lake_area_fraction))
    return _filter_connected_lakes(
        raw_lake_mask, channel_mask, raw_lake_depth,
        min_lake_cells, max_lake_cells, min_lake_depth,
    )


def _build_river_mask(channel_mask, acc, carved, sea_mask, lake_mask, raw_lake_depth):
    """Physically-widened display mask in [0, 1]."""
    from scipy.ndimage import distance_transform_edt

    EPS       = 1e-6
    width_map = _compute_river_widths(channel_mask, acc, carved)
    dist, nearest_idx = distance_transform_edt(~channel_mask, return_indices=True)
    nearest_width     = width_map[nearest_idx[0], nearest_idx[1]]

    river_mask               = np.zeros(carved.shape, dtype=np.float32)
    river_mask[channel_mask] = 1.0

    fringe = (dist < nearest_width) & ~channel_mask & ~sea_mask
    river_mask[fringe] = np.clip(
        1.0 - dist[fringe] / (nearest_width[fringe] + EPS), 0.0, 1.0
    ).astype(np.float32)

    if np.any(lake_mask):
        lake_depth = raw_lake_depth * lake_mask
        max_depth  = float(lake_depth[lake_mask].max()) + EPS
        river_mask[lake_mask] = np.minimum(
            1.0, (lake_depth[lake_mask] / max_depth).astype(np.float32)
        )

    return river_mask


def _build_water_heights(shape, sea_mask, lake_mask, channel_mask,
                         sea_level, filled, carved):
    """Water-surface elevation array (NaN where no water)."""
    river_height               = np.full(shape, np.nan, dtype=np.float64)
    river_height[sea_mask]     = sea_level
    river_height[lake_mask]    = filled[lake_mask]
    river_height[channel_mask] = carved[channel_mask]
    return river_height


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _priority_flood(grid):
    """Barnes (2014) priority-flood pit filling.

    Raises every depression to the minimum pour-point elevation so that D8
    flow can reach the boundary from every interior cell.  Uses the shared
    MinHeap jitclass — key = elevation, node = flat index i*w+j.
    """
    h, w   = grid.shape
    out    = grid.copy()
    closed = np.zeros((h, w), dtype=np.bool_)

    heap = MinHeap(h * w + 8)

    EPS = 1e-7
    for i in range(h):
        for j in range(w):
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                heap.push(i * w + j, out[i, j])
                closed[i, j] = True

    D8_I = np.array([-1, -1, -1, 0, 0,  1, 1, 1], dtype=np.int64)
    D8_J = np.array([-1,  0,  1,-1, 1, -1, 0, 1], dtype=np.int64)

    while heap.size > 0:
        node, e = heap.pop()
        ci = node // w
        cj = node  % w
        for d in range(8):
            ni = ci + D8_I[d]
            nj = cj + D8_J[d]
            if ni < 0 or ni >= h or nj < 0 or nj >= w:
                continue
            if closed[ni, nj]:
                continue
            closed[ni, nj] = True
            if out[ni, nj] < e + EPS:
                out[ni, nj] = e + EPS
            heap.push(ni * w + nj, out[ni, nj])

    return out


@njit(cache=True, parallel=True)
def _flow_accumulation(grid):
    """D8 flow accumulation on a pit-filled grid.

    For each cell the steepest downhill neighbour receives all upstream
    drainage.  Cells are processed in decreasing elevation order so every
    upslope contribution arrives before its receiver is processed.

    Returns
    -------
    acc : int64 (H, W) — cells (including self) draining through each cell.
    """
    h, w = grid.shape

    D8_I = np.array([-1, -1, -1, 0, 0,  1, 1, 1], dtype=np.int64)
    D8_J = np.array([-1,  0,  1,-1, 1, -1, 0, 1], dtype=np.int64)
    D8_W = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0,
                     1.41421356, 1.0, 1.41421356], dtype=np.float64)

    recv_i = np.full((h, w), -1, dtype=np.int64)
    recv_j = np.full((h, w), -1, dtype=np.int64)

    for i in prange(h):
        for j in range(w):
            best_slope = 0.0
            best_d     = -1
            for d in range(8):
                ni = i + D8_I[d]
                nj = j + D8_J[d]
                if ni < 0 or ni >= h or nj < 0 or nj >= w:
                    continue
                slope = (grid[i, j] - grid[ni, nj]) / D8_W[d]
                if slope > best_slope:
                    best_slope = slope
                    best_d     = d
            if best_d >= 0:
                recv_i[i, j] = i + D8_I[best_d]
                recv_j[i, j] = j + D8_J[best_d]

    order = np.argsort(grid.ravel())[::-1]   # high → low
    acc   = np.ones((h, w), dtype=np.int64)

    for k in range(order.shape[0]):
        idx = order[k]
        i   = idx // w
        j   = idx  % w
        ri  = recv_i[i, j]
        rj  = recv_j[i, j]
        if ri >= 0:
            acc[ri, rj] += acc[i, j]

    return acc


@njit(cache=True, parallel=True)
def _carve_rivers(grid, acc, threshold, carve_strength, valley_width):
    """Carve river channels and smooth valley walls.

    Pass 1 — lower channel cells proportionally to log1p(acc/threshold).
    Pass 2 — separable box-blur spreads the carved depression to valley walls,
              then a 0.25× attenuated copy is subtracted (only lowers terrain).
    """
    h, w = grid.shape
    out  = grid.copy()

    log_denom = np.log1p(float(h * w) / threshold) + 1e-9

    for i in prange(h):
        for j in range(w):
            if acc[i, j] >= threshold:
                depth = carve_strength * np.log1p(acc[i, j] / threshold) / log_denom
                out[i, j] -= depth

    if valley_width < 1:
        return out

    delta = grid - out   # positive where carving occurred

    # Horizontal box-blur of delta (rows independent → prange)
    buf = np.zeros((h, w), dtype=np.float64)
    for i in prange(h):
        for j in range(w):
            j0 = max(0, j - valley_width)
            j1 = min(w - 1, j + valley_width)
            s  = 0.0
            for jj in range(j0, j1 + 1):
                s += delta[i, jj]
            buf[i, j] = s / (j1 - j0 + 1)

    # Vertical box-blur of the H-blurred result (columns independent → prange)
    smoothed = np.zeros((h, w), dtype=np.float64)
    for j in prange(w):
        for i in range(h):
            i0 = max(0, i - valley_width)
            i1 = min(h - 1, i + valley_width)
            s  = 0.0
            for ii in range(i0, i1 + 1):
                s += buf[ii, j]
            smoothed[i, j] = s / (i1 - i0 + 1)

    # Apply attenuated valley depression (only lower, never raise)
    for i in prange(h):
        for j in range(w):
            drop = smoothed[i, j] * 0.25
            if drop > 0.0:
                out[i, j] -= drop

    return out


@njit(cache=True, parallel=True)
def _compute_river_widths(channel_mask, acc, carved, max_width=10.0):
    """Hydraulic-geometry river width (display cells) per channel cell.

    Width follows Leopold & Maddock (1953):

        width = max_width × sqrt(log(acc)/log(max_acc)) / (1 + S/S_median)^0.25

    Pass 1 builds an ``idx_map`` so Pass 2 (parallel) has O(1) lookup per
    cell, replacing the original O(h·w)-per-row re-scan.
    """
    h, w      = carved.shape
    width_map = np.zeros((h, w), dtype=np.float32)

    n_ch = 0
    for i in range(h):
        for j in range(w):
            if channel_mask[i, j]:
                n_ch += 1
    if n_ch == 0:
        return width_map

    dz       = 1.0 / max(h, w)
    ch_acc   = np.empty(n_ch, dtype=np.float64)
    ch_slope = np.empty(n_ch, dtype=np.float64)
    idx_map  = np.full((h, w), -1, dtype=np.int64)   # channel cell → sequential index

    # --- Pass 1: stats + index map (sequential — order matters) ---
    k = 0
    for i in range(h):
        for j in range(w):
            if not channel_mask[i, j]:
                continue
            if i > 0 and i < h - 1:
                gy = (carved[i + 1, j] - carved[i - 1, j]) / (2.0 * dz)
            elif i == 0:
                gy = (carved[i + 1, j] - carved[i,     j]) / dz
            else:
                gy = (carved[i,     j] - carved[i - 1, j]) / dz

            if j > 0 and j < w - 1:
                gx = (carved[i, j + 1] - carved[i, j - 1]) / (2.0 * dz)
            elif j == 0:
                gx = (carved[i, j + 1] - carved[i, j    ]) / dz
            else:
                gx = (carved[i, j    ] - carved[i, j - 1]) / dz

            ch_acc[k]     = float(acc[i, j])
            ch_slope[k]   = np.sqrt(gx * gx + gy * gy)
            idx_map[i, j] = k
            k += 1

    max_acc      = ch_acc.max() + 1.0
    median_slope = np.sort(ch_slope)[n_ch // 2] + 1e-8

    # --- Pass 2: assign widths (parallel — O(1) lookup via idx_map) ---
    for i in prange(h):
        for j in range(w):
            k = idx_map[i, j]
            if k < 0:
                continue
            log_ratio    = np.log1p(ch_acc[k]) / np.log1p(max_acc)
            slope_factor = (1.0 + ch_slope[k] / median_slope) ** 0.25
            wv = max_width * (log_ratio ** 0.5) / slope_factor
            if wv < 0.5:
                wv = 0.5
            if wv > max_width:
                wv = max_width
            width_map[i, j] = wv

    return width_map


@njit(cache=True)
def _flood_below_level(grid, level):
    """BFS from border cells: mark every cell reachable below *level* as ocean.

    Uses 4-connectivity to avoid diagonal leakage through narrow land bridges.
    Flat-array queue — zero Python interpreter overhead.
    """
    h, w    = grid.shape
    ocean   = np.zeros((h, w), dtype=np.bool_)
    visited = np.zeros((h, w), dtype=np.bool_)

    q_i  = np.empty(h * w, dtype=np.int64)
    q_j  = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    for i in range(h):
        for j_idx in range(2):
            j = 0 if j_idx == 0 else w - 1
            if grid[i, j] < level and not visited[i, j]:
                visited[i, j] = True
                ocean[i, j]   = True
                q_i[tail] = i; q_j[tail] = j; tail += 1
    for j in range(w):
        for i_idx in range(2):
            i = 0 if i_idx == 0 else h - 1
            if grid[i, j] < level and not visited[i, j]:
                visited[i, j] = True
                ocean[i, j]   = True
                q_i[tail] = i; q_j[tail] = j; tail += 1

    DI = np.array([-1,  1, 0, 0], dtype=np.int64)
    DJ = np.array([ 0,  0,-1, 1], dtype=np.int64)

    while head < tail:
        ci = q_i[head]; cj = q_j[head]; head += 1
        for d in range(4):
            ni = ci + DI[d]; nj = cj + DJ[d]
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                visited[ni, nj] = True
                if grid[ni, nj] < level:
                    ocean[ni, nj]   = True
                    q_i[tail] = ni; q_j[tail] = nj; tail += 1

    return ocean


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------

def _threshold_for_max_rivers(acc, base_thresh, max_rivers):
    """Binary-search the minimum threshold yielding ≤ *max_rivers* networks.

    O(40 × label) — ~40 scipy connected-components calls.
    """
    from scipy.ndimage import label as nd_label

    structure = np.ones((3, 3), dtype=np.int32)

    def _count(t):
        mask = acc >= int(t)
        if not np.any(mask):
            return 0
        _, n = nd_label(mask, structure=structure)
        return n

    lo, hi = base_thresh, float(acc.max())
    if _count(lo) <= max_rivers:
        return lo

    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _count(mid) <= max_rivers:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 1.0:
            break

    return hi


def _filter_connected_lakes(lake_mask, channel_mask, lake_depth_map,
                            min_cells, max_cells, min_depth):
    """Keep only hydrologically meaningful lake basins.

    A basin is retained when ALL of the following hold:
      1. ``min_cells`` ≤ area < ``max_cells``
      2. max basin depth ≥ ``min_depth``  (removes shallow flooded valleys)
      3. Connected to or adjacent to at least one channel cell
    """
    from scipy.ndimage import label as nd_label, binary_dilation

    structure       = np.ones((3, 3), dtype=np.int32)
    labeled, n_lbls = nd_label(lake_mask, structure=structure)
    if n_lbls == 0:
        return lake_mask

    basin_ids   = labeled[lake_mask]
    channel_ids = labeled[channel_mask & lake_mask]

    sizes      = np.bincount(basin_ids, minlength=n_lbls + 1)
    max_depths = np.zeros(n_lbls + 1, dtype=np.float64)
    np.maximum.at(max_depths, labeled[lake_mask], lake_depth_map[lake_mask])

    connected = np.zeros(n_lbls + 1, dtype=np.bool_)
    if channel_ids.size > 0:
        connected[channel_ids] = True
    adj_ids = labeled[binary_dilation(channel_mask, structure=structure) & lake_mask]
    if adj_ids.size > 0:
        connected[adj_ids] = True

    keep = np.zeros(n_lbls + 1, dtype=np.bool_)
    for lbl in range(1, n_lbls + 1):
        if (sizes[lbl] >= min_cells
                and sizes[lbl] < max_cells
                and max_depths[lbl] >= min_depth
                and connected[lbl]):
            keep[lbl] = True

    return keep[labeled] & lake_mask
