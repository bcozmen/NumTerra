import numpy as np
from numba import njit, prange
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
@njit(cache=True, parallel=True)
def compute_mfd_weights(height_map, slope_exp=1.7):
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

    # local copies required because numba @njit cannot capture module-level
    # numpy arrays as constants — values match _NB_DX / _NB_DY / _NB_DIST
    dx   = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy   = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)
    dist = np.array([1.41421356, 1.0, 1.41421356,
                     1.0,        1.0,
                     1.41421356, 1.0, 1.41421356], dtype=np.float32)

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



# ---------------------------------------------------------------------------
# PHASE 2 — Priority-flood lake detection
# ---------------------------------------------------------------------------
@njit(cache=True)
def compute_lake_mask(height_map, sea_mask):
    xdim, ydim = height_map.shape
    n = xdim * ydim

    # ---- flatten inputs (CRITICAL speedup) ----
    h = height_map.ravel().astype(np.float32)
    sea = sea_mask.ravel()

    fill = np.empty(n, dtype=np.float32)
    processed = np.zeros(n, dtype=np.uint8)
    lake = np.zeros(n, dtype=np.uint8)

    INF = np.float32(1e30)
    for i in range(n):
        fill[i] = INF

    heap = MinHeap(n * 4)

    # ---- precomputed offsets (8-neighborhood) ----
    offsets = np.array([
        -ydim - 1, -ydim, -ydim + 1,
        -1,               1,
         ydim - 1,  ydim, ydim + 1
    ], dtype=np.int32)

    # ---- seed ocean ----
    for i in range(n):
        if sea[i]:
            fill[i] = h[i]
            heap.push(i, h[i])

    # ---- priority flood ----
    while heap.size > 0:
        idx, fl = heap.pop()

        if processed[idx]:
            continue
        processed[idx] = 1

        # mark lake during traversal (removes second pass)
        if fl > h[idx] and not sea[idx]:
            lake[idx] = 1

        # Row/column of current cell — needed for column-wrap guard
        cx = idx // ydim
        cy = idx - cx * ydim

        for k in range(8):
            nidx = idx + offsets[k]

            # Hard bounds: reject indices outside the flat array
            if nidx < 0 or nidx >= n:
                continue

            # Column-wrap guard: neighbours that cross the left/right edge are
            # not real neighbours (they belong to an adjacent row).
            nx = nidx // ydim
            ny = nidx - nx * ydim
            if ny - cy > 1 or cy - ny > 1:
                continue

            if processed[nidx]:
                continue

            nh = h[nidx]

            # faster than max()
            if nh > fl:
                new_fl = nh
            else:
                new_fl = fl

            if new_fl < fill[nidx]:
                fill[nidx] = new_fl
                heap.push(nidx, new_fl)

    return lake.reshape(xdim, ydim), fill.reshape(xdim, ydim)


