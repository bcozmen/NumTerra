import numpy as np
from numba import njit, prange



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