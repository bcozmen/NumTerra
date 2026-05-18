import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# River erosion — flow accumulation + channel carving
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

    # --- Optional: limit to top N river networks ---
    if max_rivers is not None and max_rivers > 0:
        thresh = _threshold_for_max_rivers(acc, thresh, max_rivers)

    carved = _carve_rivers(filled, acc, thresh, carve_strength, valley_width_cells)

    EPS = 1e-6

    # --- Sea / ocean mask ---
    sea_level = 0.0
    sea_mask = np.zeros(grid.shape, dtype=np.bool_)
    if sea_level_percentile > 0.0:
        sea_level = float(np.percentile(carved, sea_level_percentile * 100.0))
        sea_mask = _flood_below_level(carved, sea_level)

    # --- River channel skeleton (excluding ocean) ---
    channel_mask = (acc >= int(thresh)) & (~sea_mask)

    # --- Inland lake / basin mask ---
    raw_lake_depth = (filled - grid.astype(np.float64))
    raw_lake_mask  = (raw_lake_depth > EPS) & (~sea_mask)
    min_lake_cells = max(4, int(n_cells * 0.00005))
    max_lake_cells = max(min_lake_cells + 1, int(n_cells * max_lake_area_fraction))
    lake_mask = _filter_connected_lakes(
        raw_lake_mask, channel_mask, raw_lake_depth,
        min_lake_cells, max_lake_cells, min_lake_depth,
    )

    # --- river_mask: physically-widened display mask ---
    # Width is based on log(acc)/log(max_acc) — purely absolute, unaffected by
    # any threshold adjustments — so large rivers are always rendered wide.
    from scipy.ndimage import distance_transform_edt
    width_map = _compute_river_widths(channel_mask, acc, carved)
    dist, nearest_idx = distance_transform_edt(~channel_mask, return_indices=True)
    nearest_width = width_map[nearest_idx[0], nearest_idx[1]]
    display_channel = channel_mask | ((dist < nearest_width) & ~sea_mask)

    river_mask = np.zeros(grid.shape, dtype=np.float32)
    river_mask[channel_mask] = 1.0
    fringe = display_channel & ~channel_mask
    river_mask[fringe] = np.clip(
        1.0 - dist[fringe] / (nearest_width[fringe] + 1e-6), 0.0, 1.0
    ).astype(np.float32)
    if np.any(lake_mask):
        lake_depth = raw_lake_depth * lake_mask
        max_depth  = float(lake_depth[lake_mask].max()) + EPS
        river_mask[lake_mask] = np.minimum(
            1.0, (lake_depth[lake_mask] / max_depth).astype(np.float32)
        )

    # --- Water surface heights ---
    river_height = np.full(grid.shape, np.nan, dtype=np.float64)
    river_height[sea_mask]     = sea_level
    river_height[lake_mask]    = filled[lake_mask]
    river_height[channel_mask] = carved[channel_mask]

    return carved, river_mask, river_height, sea_mask, lake_mask, sea_level


@njit(cache=True, parallel=True)
def _compute_river_widths(channel_mask, acc, carved, max_width=10.0):
    """Hydraulic-geometry river width (display cells) for each channel cell.

    Width is scaled on log(acc) normalised by log(max_acc) so it is
    independent of whatever threshold was used to select channels.
    Steep-slope narrowing follows Leopold & Maddock (1953):

        width = max_width * (log(acc)/log(max_acc))^0.5
                           / (1 + S/S_median)^0.25

    This gives:
        - headwater entry point (acc = thresh): narrow (~1–2 cells)
        - main-stem outlet (acc = max):          max_width cells (~10 km)
        - steep gorge: narrowed by slope factor
        - gentle floodplain: near max_width

    Implemented as an @njit kernel: manual central-difference gradient
    (no np.gradient), two sequential passes over channel cells to gather
    stats then assign widths.
    """
    h, w = carved.shape
    width_map = np.zeros((h, w), dtype=np.float32)

    # Count channel cells for pre-allocation
    n_ch = 0
    for i in range(h):
        for j in range(w):
            if channel_mask[i, j]:
                n_ch += 1
    if n_ch == 0:
        return width_map

    dz = 1.0 / max(h, w)
    ch_acc   = np.empty(n_ch, dtype=np.float64)
    ch_slope = np.empty(n_ch, dtype=np.float64)

    # Pass 1: collect per-channel-cell accumulation and slope magnitude
    k = 0
    for i in range(h):
        for j in range(w):
            if channel_mask[i, j]:
                # Central-difference gradient (falls back to forward/backward at edges)
                if i > 0 and i < h - 1:
                    gy = (carved[i + 1, j] - carved[i - 1, j]) / (2.0 * dz)
                elif i == 0:
                    gy = (carved[i + 1, j] - carved[i, j]) / dz
                else:
                    gy = (carved[i, j] - carved[i - 1, j]) / dz
                if j > 0 and j < w - 1:
                    gx = (carved[i, j + 1] - carved[i, j - 1]) / (2.0 * dz)
                elif j == 0:
                    gx = (carved[i, j + 1] - carved[i, j]) / dz
                else:
                    gx = (carved[i, j] - carved[i, j - 1]) / dz
                ch_acc[k]   = float(acc[i, j])
                ch_slope[k] = np.sqrt(gx * gx + gy * gy)
                k += 1

    max_acc = ch_acc.max() + 1.0

    # Median slope via sort (numba supports np.sort on 1-D arrays)
    sorted_slope  = np.sort(ch_slope)
    median_slope  = sorted_slope[n_ch // 2] + 1e-8

    # Pass 2: assign widths (parallel over rows — each row is independent)
    for i in prange(h):
        k_start = 0
        for ii in range(i):
            for jj in range(w):
                if channel_mask[ii, jj]:
                    k_start += 1
        kk = k_start
        for j in range(w):
            if channel_mask[i, j]:
                log_ratio    = np.log1p(ch_acc[kk]) / np.log1p(max_acc)
                slope_factor = (1.0 + ch_slope[kk] / median_slope) ** 0.25
                wv = max_width * (log_ratio ** 0.5) / slope_factor
                if wv < 0.5:
                    wv = 0.5
                if wv > max_width:
                    wv = max_width
                width_map[i, j] = wv
                kk += 1

    return width_map


def _filter_connected_lakes(lake_mask, channel_mask, lake_depth_map,
                            min_cells, max_cells, min_depth):
    """Keep only lake basins that are hydrologically meaningful.

    A basin is kept if it satisfies ALL of:
      1. ``min_cells`` ≤ area < ``max_cells``  (size bounds).
      2. Max depth within the basin ≥ ``min_depth`` (eliminates shallow
         flooded valleys — the primary cause of "entire valley = one lake").
      3. It is hydrologically connected: contains or is adjacent to at least
         one river channel cell.

    Parameters
    ----------
    lake_mask      : (H, W) bool — raw depression mask.
    channel_mask   : (H, W) bool — river channel skeleton.
    lake_depth_map : (H, W) float64 — ``filled - original`` depth per cell.
    min_cells      : int   — minimum basin area (cells) to retain.
    max_cells      : int   — maximum basin area (cells) to retain.
                             Derived from ``max_lake_area_fraction × n_cells``.
    min_depth      : float — minimum normalised basin depth to retain.
                             Derived from ``min_lake_depth_m / max_altitude``.

    Returns
    -------
    filtered : (H, W) bool — lake mask with noise-level, disconnected, too
               large, and too shallow depressions removed.
    """
    from scipy.ndimage import label as nd_label, binary_dilation

    structure = np.ones((3, 3), dtype=np.int32)   # 8-connectivity
    labeled, n_labels = nd_label(lake_mask, structure=structure)
    if n_labels == 0:
        return lake_mask

    # For each basin: count cells and check if any channel cell overlaps
    basin_ids   = labeled[lake_mask]
    channel_ids = labeled[channel_mask & lake_mask]

    sizes = np.bincount(basin_ids, minlength=n_labels + 1)

    # Max depth per basin (used for the depth threshold)
    max_depths = np.zeros(n_labels + 1, dtype=np.float64)
    np.maximum.at(max_depths, labeled[lake_mask], lake_depth_map[lake_mask])

    # Which basins touch a channel?
    connected = np.zeros(n_labels + 1, dtype=np.bool_)
    if channel_ids.size > 0:
        connected[channel_ids] = True
    channel_dilated = binary_dilation(channel_mask, structure=structure)
    adjacent_ids = labeled[channel_dilated & lake_mask]
    if adjacent_ids.size > 0:
        connected[adjacent_ids] = True

    keep = np.zeros(n_labels + 1, dtype=np.bool_)
    for lbl in range(1, n_labels + 1):
        if (sizes[lbl] >= min_cells
                and sizes[lbl] < max_cells
                and max_depths[lbl] >= min_depth
                and connected[lbl]):
            keep[lbl] = True

    return keep[labeled] & lake_mask


@njit(cache=True)
def _flood_below_level(grid, level):
    """BFS from all border cells: mark every cell reachable without crossing
    above *level* as ocean.  Uses 4-connectivity (cardinal neighbours only)
    to avoid diagonal leakage through narrow land bridges.

    Implemented as an @njit kernel with a flat array queue so the entire BFS
    runs in compiled code without any Python interpreter overhead.

    Returns
    -------
    ocean : (H, W) bool array.
    """
    h, w    = grid.shape
    ocean   = np.zeros((h, w), dtype=np.bool_)
    visited = np.zeros((h, w), dtype=np.bool_)

    # Pre-allocate a worst-case flat queue (at most h*w entries)
    q_i  = np.empty(h * w, dtype=np.int64)
    q_j  = np.empty(h * w, dtype=np.int64)
    head = np.int64(0)
    tail = np.int64(0)

    # Seed: all border cells below sea level
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

    DI = np.array([-1, 1,  0, 0], dtype=np.int64)
    DJ = np.array([ 0, 0, -1, 1], dtype=np.int64)

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


def _threshold_for_max_rivers(acc, base_thresh, max_rivers):
    """Raise *base_thresh* until at most *max_rivers* connected channel
    networks remain (8-connected components of acc >= thresh).
    Uses binary search — O(40 × label) steps.
    """
    from scipy.ndimage import label as nd_label

    structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity

    def _count(t):
        mask = acc >= int(t)
        if not np.any(mask):
            return 0
        _, n = nd_label(mask, structure=structure)
        return n

    lo, hi = base_thresh, float(acc.max())

    if _count(lo) <= max_rivers:
        return lo  # already fine

    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _count(mid) <= max_rivers:
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 1.0:
            break

    return hi


@njit(cache=True)
def _priority_flood(grid):
    """Barnes (2014) priority-flood pit filling.

    Raises every depression to the minimum pour-point elevation so that D8
    flow can reach the boundary from every interior cell.

    Algorithm
    ---------
    Seed the priority queue (min-heap) with all border cells, then process
    cells in elevation order.  Any neighbour lower than the current cell is
    raised to the current cell's elevation (plus a tiny epsilon so strict
    downhill flow is preserved) and pushed onto the heap.
    """
    h, w   = grid.shape
    out    = grid.copy()
    closed = np.zeros((h, w), dtype=np.bool_)

    # --- Manual min-heap implemented with parallel arrays ---
    # Each entry: (elevation, row, col).  We over-allocate and track size.
    MAX_HEAP = h * w + 8
    hp_e = np.empty(MAX_HEAP, dtype=np.float64)
    hp_r = np.empty(MAX_HEAP, dtype=np.int64)
    hp_c = np.empty(MAX_HEAP, dtype=np.int64)
    heap_size = 0

    def _push(e, r, c):
        nonlocal heap_size
        i = heap_size
        hp_e[i] = e; hp_r[i] = r; hp_c[i] = c
        heap_size += 1
        # sift up
        while i > 0:
            parent = (i - 1) >> 1
            if hp_e[parent] > hp_e[i]:
                hp_e[i], hp_e[parent] = hp_e[parent], hp_e[i]
                hp_r[i], hp_r[parent] = hp_r[parent], hp_r[i]
                hp_c[i], hp_c[parent] = hp_c[parent], hp_c[i]
                i = parent
            else:
                break

    def _pop():
        nonlocal heap_size
        top_e = hp_e[0]; top_r = hp_r[0]; top_c = hp_c[0]
        heap_size -= 1
        if heap_size > 0:
            hp_e[0] = hp_e[heap_size]
            hp_r[0] = hp_r[heap_size]
            hp_c[0] = hp_c[heap_size]
            # sift down
            i = 0
            while True:
                l = 2 * i + 1; r_ = 2 * i + 2; smallest = i
                if l < heap_size and hp_e[l] < hp_e[smallest]:
                    smallest = l
                if r_ < heap_size and hp_e[r_] < hp_e[smallest]:
                    smallest = r_
                if smallest == i:
                    break
                hp_e[i], hp_e[smallest] = hp_e[smallest], hp_e[i]
                hp_r[i], hp_r[smallest] = hp_r[smallest], hp_r[i]
                hp_c[i], hp_c[smallest] = hp_c[smallest], hp_c[i]
                i = smallest
        return top_e, top_r, top_c

    # Seed with border cells
    EPS = 1e-7
    for i in range(h):
        for j in range(w):
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                _push(out[i, j], i, j)
                closed[i, j] = True

    D8_I = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
    D8_J = np.array([-1,  0,  1,-1, 1,-1, 0, 1], dtype=np.int64)

    while heap_size > 0:
        e, ci, cj = _pop()
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
            _push(out[ni, nj], ni, nj)

    return out


@njit(cache=True, parallel=True)
def _flow_accumulation(grid):
    """D8 flow accumulation on a pit-filled grid.

    For each cell the steepest downhill neighbour receives all upstream
    drainage.  Cells are processed in decreasing elevation order so the
    contribution from every upslope cell arrives before its downstream
    receiver is processed.

    The steepest-descent receiver computation is embarrassingly parallel
    (prange over rows); the accumulation propagation must remain sequential
    (topological order dependency).

    Returns
    -------
    acc : int64 array, shape (h, w).
        Number of cells (including self) draining through each cell.
    """
    h, w = grid.shape

    D8_I = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
    D8_J = np.array([-1,  0,  1,-1, 1,-1, 0, 1], dtype=np.int64)
    # Distance weights for D8 (diagonal = √2)
    D8_W = np.array([1.41421356, 1.0, 1.41421356, 1.0, 1.0,
                     1.41421356, 1.0, 1.41421356], dtype=np.float64)

    # --- Compute steepest-descent receiver for every cell (parallel) ---
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

    # --- Sort cells by decreasing elevation (process uphill first) ---
    flat_elev = grid.ravel()
    order     = np.argsort(flat_elev)[::-1]   # high → low

    acc = np.ones((h, w), dtype=np.int64)

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

    For every river cell (acc ≥ threshold) the terrain is lowered by a depth
    proportional to log1p(acc/threshold).  A box-blur of half-width
    *valley_width* then slopes the surrounding cells smoothly down toward the
    channel, producing a V-shaped valley cross-section.

    The blur is applied in a second pass so channel depths are already set
    when the valley walls are shaped.

    All three embarrassingly-parallel passes (carve, H-blur, V-blur) use
    prange; only the final element-wise apply also uses prange.

    Parameters
    ----------
    grid         : pit-filled normalised height map.
    acc          : flow accumulation array (int64).
    threshold    : minimum accumulation to carve.
    carve_strength : maximum carve depth in normalised height units.
    valley_width : half-width of valley smoothing kernel (integer, cells).
    """
    h, w = grid.shape
    out  = grid.copy()

    log_denom = np.log1p(float(h * w) / threshold) + 1e-9

    # --- Pass 1: carve channel cells (parallel over rows) ---
    for i in prange(h):
        for j in range(w):
            if acc[i, j] >= threshold:
                depth = carve_strength * np.log1p(acc[i, j] / threshold) / log_denom
                out[i, j] -= depth

    # --- Pass 2: valley wall smoothing (separable box blur on river cells) ---
    # We smooth the *depression* introduced by carving so banks slope naturally.
    # delta[i,j] = how much carving lowered cell (i,j); we spread a fraction
    # of that depression to the surrounding valley_width cells.
    if valley_width < 1:
        return out

    delta = grid - out   # positive where carving occurred

    # Horizontal pass: for each cell average delta in a ±valley_width window
    # (parallel over rows — rows are independent)
    buf = np.zeros((h, w), dtype=np.float64)
    for i in prange(h):
        for j in range(w):
            j0 = max(0, j - valley_width)
            j1 = min(w - 1, j + valley_width)
            s  = 0.0
            for jj in range(j0, j1 + 1):
                s += delta[i, jj]
            buf[i, j] = s / (j1 - j0 + 1)

    # Vertical pass (parallel over columns — columns are independent)
    smoothed = np.zeros((h, w), dtype=np.float64)
    for j in prange(w):
        for i in range(h):
            i0 = max(0, i - valley_width)
            i1 = min(h - 1, i + valley_width)
            s  = 0.0
            for ii in range(i0, i1 + 1):
                s += buf[ii, j]
            smoothed[i, j] = s / (i1 - i0 + 1)

    # Apply: lower surrounding terrain toward the channel by the smoothed depression.
    # Only lower cells (never raise), so only valley walls are affected.
    # (parallel over rows)
    for i in prange(h):
        for j in range(w):
            drop = smoothed[i, j] * 0.25   # attenuate so walls aren't as deep as channel
            if drop > 0.0:
                out[i, j] -= drop

    return out




