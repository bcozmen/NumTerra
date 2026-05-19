import numpy as np
from numba import njit
from utils import MinHeap, timeit

# ---------------------------------------------------------------------------
# Neighbour tables (shared by all MFD kernels)
# ---------------------------------------------------------------------------
# 8 neighbours: offsets and their Euclidean distances (1 or sqrt(2))
_NB_DX   = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
_NB_DY   = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
_NB_DIST = np.array([1.41421356, 1.0, 1.41421356,
                     1.0,        1.0,
                     1.41421356, 1.0, 1.41421356], dtype=np.float32)

@timeit
@njit(cache=True)
def detect_sea(height_map, sea_level):
    xdim, ydim = height_map.shape
    n = xdim * ydim

    sea   = np.zeros((xdim, ydim), dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    head  = 0
    tail  = 0

    def enqueue(x, y, tail):
        sea[x, y] = True
        queue[tail] = x * ydim + y
        return tail + 1

    # ---------------------------------------------------------
    # initialize from borders
    # ---------------------------------------------------------

    # top + bottom
    for x in range(xdim):

        # top
        if height_map[x, 0] <= sea_level and sea[x, 0] == 0:
            tail = enqueue(x, 0, tail)

        # bottom
        y = ydim - 1
        if height_map[x, y] <= sea_level and sea[x, y] == 0:
            tail = enqueue(x, y, tail)

    # left + right
    for y in range(ydim):

        # left
        if height_map[0, y] <= sea_level and sea[0, y] == 0:
            tail = enqueue(0, y, tail)

        # right
        x = xdim - 1
        if height_map[x, y] <= sea_level and sea[x, y] == 0:
            tail = enqueue(x, y, tail)

    # ---------------------------------------------------------
    # BFS flood fill
    # ---------------------------------------------------------

    while head < tail:

        idx = queue[head]
        head += 1

        x = idx // ydim
        y = idx - x * ydim

        # -----------------------------------------------------
        # fully unrolled 8-neighborhood
        # -----------------------------------------------------

        nx = x - 1
        ny = y - 1
        if nx >= 0 and ny >= 0:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x - 1
        ny = y
        if nx >= 0:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x - 1
        ny = y + 1
        if nx >= 0 and ny < ydim:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x
        ny = y - 1
        if ny >= 0:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x
        ny = y + 1
        if ny < ydim:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x + 1
        ny = y - 1
        if nx < xdim and ny >= 0:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x + 1
        ny = y
        if nx < xdim:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

        nx = x + 1
        ny = y + 1
        if nx < xdim and ny < ydim:
            if sea[nx, ny] == 0 and height_map[nx, ny] <= sea_level:
                tail = enqueue(nx, ny, tail)

    return sea


# ---------------------------------------------------------------------------
# PHASE 1 — MFD (Multiple Flow Direction) hydrology
# ---------------------------------------------------------------------------
@timeit
@njit(cache=True)
def compute_mfd_weights(height_map):
    """
    For every cell compute normalised flow weights to each of its 8 neighbours.

    A neighbour only receives weight when it is strictly lower than the source
    cell.  Weights are proportional to slope (Δh / distance) and normalised so
    they sum to 1.0 per cell.  Cells that are local minima (sinks) or flat get
    all-zero weight vectors.

    Returns
    -------
    flow_weights : float32 array (xdim, ydim, 8)
        Axis-2 ordering matches _NB_DX / _NB_DY (see module constants).
    """
    xdim, ydim = height_map.shape
    flow_weights = np.zeros((xdim, ydim, 8), dtype=np.float32)

    # local copies required because numba @njit cannot capture module-level
    # numpy arrays as constants — values match _NB_DX / _NB_DY / _NB_DIST
    dx   = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy   = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    dist = np.array([1.41421356, 1.0, 1.41421356,
                     1.0,        1.0,
                     1.41421356, 1.0, 1.41421356], dtype=np.float32)

    for x in range(xdim):
        for y in range(ydim):
            h     = height_map[x, y]
            total = np.float32(0.0)

            for k in range(8):
                nx = x + dx[k]
                ny = y + dy[k]
                if 0 <= nx < xdim and 0 <= ny < ydim:
                    dh = h - height_map[nx, ny]
                    if dh > 0.0:
                        slope = dh / dist[k]
                        flow_weights[x, y, k] = slope
                        total += slope

            if total > 0.0:
                for k in range(8):
                    flow_weights[x, y, k] /= total

    return flow_weights

@timeit
@njit(cache=True)
def compute_mfd_accumulation(height_map, flow_weights):
    """
    Compute flow accumulation using MFD weights.

    Cells are visited in descending elevation order (highest first).  Each
    cell distributes its accumulated value to lower neighbours according to
    the precomputed MFD weights.

    Returns
    -------
    accumulation : float32 array (xdim, ydim)
        Each cell starts with 1.0 (its own area) and collects contributions
        from all upslope cells.
    """
    xdim, ydim = height_map.shape
    n   = xdim * ydim
    acc = np.ones(n, dtype=np.float32)

    dx = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    # descending elevation order (negate to sort ascending = descending original)
    order = np.argsort(-height_map.ravel())

    for i in range(n):
        idx = order[i]
        x   = idx // ydim
        y   = idx -  x * ydim

        for k in range(8):
            w = flow_weights[x, y, k]
            if w > 0.0:
                nx = x + dx[k]
                ny = y + dy[k]
                if 0 <= nx < xdim and 0 <= ny < ydim:
                    acc[nx * ydim + ny] += acc[idx] * w

    return acc.reshape((xdim, ydim))


# ---------------------------------------------------------------------------
# PHASE 2 — Priority-flood lake detection
# ---------------------------------------------------------------------------
@timeit
@njit(cache=True)
def compute_lake_mask(height_map, sea_mask):
    """
    Identify lake cells using the priority-flood algorithm.

    Starting from every sea cell, we expand outward in ascending fill-level
    order (like water slowly rising from the ocean inward).  The fill level
    at each cell is:

        fill_level[nb] = max(fill_level[current], height_map[nb])

    If we had to "dam" our path to reach a cell (fill_level > height_map),
    that cell sits inside a closed depression — it is a lake cell.

    This correctly captures entire basin shapes bounded by their pour point,
    avoiding both the "dot" problem (single sink pixels only) and the
    "everything is a lake" problem (MFD propagation failing on flat terrain).

    Parameters
    ----------
    height_map : float32 array (xdim, ydim)
    sea_mask   : bool array   (xdim, ydim)

    Returns
    -------
    lake_mask  : bool array   (xdim, ydim)
    """
    xdim, ydim = height_map.shape
    n = xdim * ydim

    dx = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    INF = np.float64(1e38)
    fill_level = np.full((xdim, ydim), INF, dtype=np.float64)
    processed  = np.zeros((xdim, ydim), dtype=np.bool_)

    heap = MinHeap(n)

    # Seed: all sea cells — their fill level equals their own elevation
    for x in range(xdim):
        for y in range(ydim):
            if sea_mask[x, y]:
                fl = np.float64(height_map[x, y])
                fill_level[x, y] = fl
                heap.push(x * ydim + y, fl)

    # Priority-flood expansion
    while heap.size > 0:
        idx, fl = heap.pop()
        x = idx // ydim
        y = idx -  x * ydim

        if processed[x, y]:
            continue
        processed[x, y] = True

        for k in range(8):
            nx = x + dx[k]
            ny = y + dy[k]
            if 0 <= nx < xdim and 0 <= ny < ydim and not processed[nx, ny]:
                new_fl = max(fl, np.float64(height_map[nx, ny]))
                if new_fl < fill_level[nx, ny]:
                    fill_level[nx, ny] = new_fl
                    heap.push(nx * ydim + ny, new_fl)

    # A cell is a lake where the flood had to rise above the terrain to reach it
    lake_mask = np.zeros((xdim, ydim), dtype=np.bool_)
    for x in range(xdim):
        for y in range(ydim):
            if not sea_mask[x, y] and fill_level[x, y] > np.float64(height_map[x, y]):
                lake_mask[x, y] = True

    return lake_mask, fill_level.astype(np.float32)


# ---------------------------------------------------------------------------
# PHASE 3 — D8 downstream pointer (for river path tracing)
# ---------------------------------------------------------------------------
@timeit
@njit(cache=True)
def compute_d8_downstream(flow_weights):
    """
    For each cell return the flat index of its single dominant D8 downstream
    neighbour (the direction with the highest MFD weight).  Returns -1 for
    cells that are local sinks (all weights zero).

    This collapses MFD into a single pointer per cell, which is sufficient for
    tracing river centre-line paths.

    Returns
    -------
    downstream : int32 array (xdim, ydim)
        Flat-index of the downstream neighbour, or -1 for sinks.
    """
    xdim, ydim = flow_weights.shape[0], flow_weights.shape[1]
    dx = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    downstream = np.full((xdim, ydim), -1, dtype=np.int32)

    for x in range(xdim):
        for y in range(ydim):
            best_k   = -1
            best_w   = np.float32(0.0)
            for k in range(8):
                if flow_weights[x, y, k] > best_w:
                    best_w = flow_weights[x, y, k]
                    best_k = k
            if best_k >= 0:
                nx = x + dx[best_k]
                ny = y + dy[best_k]
                if 0 <= nx < xdim and 0 <= ny < ydim:
                    downstream[x, y] = nx * ydim + ny

    return downstream


# ---------------------------------------------------------------------------
# PHASE 4 — River path tracing
# ---------------------------------------------------------------------------
@timeit
def compute_river_paths(downstream, flow_acc, sea_mask, lake_mask, river_threshold):
    """
    Trace river centre-line paths from D8 downstream pointers.

    River cells (flow_acc > threshold) are sorted by accumulation descending
    so main stems are traced before tributaries.  Each untraced river cell
    starts a new path; we follow downstream until we hit the sea, a lake,
    or an already-traced cell.

    Uses numpy to find/sort river cells and Python lists to accumulate points
    (avoids pre-allocated fixed buffer and np.int32 counter overflow).
    Converts to flat numpy arrays at the end for the numba rasterizer.

    Returns
    -------
    coords       : float32 (total_pts, 2)  — all points, normalized [0,1]²
    path_starts  : int32  (num_paths,)     — start index of each path
    path_lengths : int32  (num_paths,)     — number of points in each path
    """
    xdim, ydim = flow_acc.shape

    # --- find and sort river cells with numpy ---
    river_bool = (flow_acc > river_threshold) & (~sea_mask) & (~lake_mask)
    if not river_bool.any():
        return (np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))

    rxs, rys = np.where(river_bool)
    order    = np.argsort(-flow_acc[rxs, rys])
    rxs, rys = rxs[order], rys[order]

    inv_x = 1.0 / (xdim - 1)
    inv_y = 1.0 / (ydim - 1)

    traced       = np.zeros((xdim, ydim), dtype=np.bool_)
    buf_x        = []   # Python lists: no pre-allocation, no overflow
    buf_y        = []
    path_starts  = []
    path_lengths = []

    # --- trace paths — O(num_river) total steps ---
    for sx, sy in zip(rxs.tolist(), rys.tolist()):
        if traced[sx, sy]:
            continue

        path_start = len(buf_x)
        buf_x.append(sx * inv_x)
        buf_y.append(sy * inv_y)
        traced[sx, sy] = True

        x, y = sx, sy
        while True:
            d = int(downstream[x, y])
            if d < 0:
                break                       # local sink, no terminal point
            nx, ny = d // ydim, d % ydim
            if sea_mask[nx, ny] or lake_mask[nx, ny] or traced[nx, ny]:
                buf_x.append(nx * inv_x)   # terminal point — then stop
                buf_y.append(ny * inv_y)
                break
            traced[nx, ny] = True
            buf_x.append(nx * inv_x)
            buf_y.append(ny * inv_y)
            x, y = nx, ny

        path_len = len(buf_x) - path_start
        if path_len > 1:
            path_starts.append(path_start)
            path_lengths.append(path_len)

    if not path_starts:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))

    coords = np.column_stack([
        np.array(buf_x, dtype=np.float32),
        np.array(buf_y, dtype=np.float32),
    ])
    return (coords,
            np.array(path_starts,  dtype=np.int32),
            np.array(path_lengths, dtype=np.int32))


# ---------------------------------------------------------------------------
# PHASE 5 — River rasterization (numba Bresenham, resolution-independent)
# ---------------------------------------------------------------------------
@timeit
@njit(cache=True)
def rasterize_river_paths(coords, path_starts, path_lengths,
                           x0, x1, y0, y1, rows, cols):
    """
    Rasterize flat river-path representation onto a boolean grid.

    Uses Bresenham line drawing — sharp and correct at any resolution.
    Passes a per-path bounding-box check before drawing to avoid iterating
    over paths entirely outside the view window.

    Parameters
    ----------
    coords        : float32 (N, 2)   — all path points in [0,1]²
    path_starts   : int32  (P,)      — start index of each path in coords
    path_lengths  : int32  (P,)      — number of points in each path
    x0,x1,y0,y1  : float64           — view window in [0,1]²
    rows, cols    : int               — output grid shape

    Returns
    -------
    river_mask : bool array (rows, cols)
    """
    river_mask = np.zeros((rows, cols), dtype=np.bool_)

    if len(path_starts) == 0:
        return river_mask

    inv_x = (rows - 1) / (x1 - x0)
    inv_y = (cols - 1) / (y1 - y0)

    for pi in range(len(path_starts)):
        start  = path_starts[pi]
        length = path_lengths[pi]

        # Bounding-box quick reject
        px_min = coords[start, 0]
        px_max = coords[start, 0]
        py_min = coords[start, 1]
        py_max = coords[start, 1]
        for i in range(start + 1, start + length):
            v = coords[i, 0]
            if v < px_min: px_min = v
            if v > px_max: px_max = v
            v = coords[i, 1]
            if v < py_min: py_min = v
            if v > py_max: py_max = v

        if px_max < x0 or px_min > x1 or py_max < y0 or py_min > y1:
            continue

        for i in range(start, start + length - 1):
            x_a = coords[i,   0]
            y_a = coords[i,   1]
            x_b = coords[i+1, 0]
            y_b = coords[i+1, 1]

            r0 = int((x_a - x0) * inv_x + 0.5)
            c0 = int((y_a - y0) * inv_y + 0.5)
            r1 = int((x_b - x0) * inv_x + 0.5)
            c1 = int((y_b - y0) * inv_y + 0.5)

            dr  = r1 - r0
            dc  = c1 - c0
            if dr < 0: dr = -dr
            if dc < 0: dc = -dc
            sr  = np.int32(1) if r1 > r0 else np.int32(-1)
            sc  = np.int32(1) if c1 > c0 else np.int32(-1)
            err = dr - dc
            r, c = r0, c0

            while True:
                if 0 <= r < rows and 0 <= c < cols:
                    river_mask[r, c] = True
                if r == r1 and c == c1:
                    break
                e2 = 2 * err
                if e2 > -dc:
                    err -= dc
                    r   += sr
                if e2 < dr:
                    err += dr
                    c   += sc

    return river_mask
