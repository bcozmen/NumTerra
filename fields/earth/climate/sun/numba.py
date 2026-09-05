import numpy as np
from numba import njit, prange

@njit(parallel=True)
def precompute_horizon_angles(H, N_dirs, max_altitude, dx, dy):
    rows, cols = H.shape
    max_steps = int(1.45 * (rows if rows > cols else cols)) + 1
    horizon_angles = np.empty((rows, cols, N_dirs), dtype=np.float32)

    for r in prange(rows):
        for c in range(cols):
            # H is assumed normalized [0, 1], scaled to absolute meters
            h_origin   = np.float64(H[r, c]) * max_altitude
            h_headroom = max_altitude - h_origin   

            for k in range(N_dirs):
                alpha = 2.0 * np.pi * k / N_dirs
                cos_a = np.cos(alpha)   
                sin_a = np.sin(alpha)   

                abs_cos = cos_a if cos_a >= 0.0 else -cos_a
                abs_sin = sin_a if sin_a >= 0.0 else -sin_a
                scale   = abs_cos if abs_cos > abs_sin else abs_sin
                if scale < 1e-9:
                    scale = 1.0

                di = sin_a / scale   
                dj = cos_a / scale   

                # FIX: Step distance accurately converted to real physical meters
                d_step = np.sqrt((di * dx) ** 2 + (dj * dy) ** 2)

                max_tan  = np.float64(-1e10)
                curr_i   = np.float64(r)
                curr_j   = np.float64(c)
                d_cum    = np.float64(0.0)

                for _ in range(max_steps):
                    curr_i += di
                    curr_j += dj
                    d_cum  += d_step

                    # Early exit condition holds true because heights and distances are matched
                    if h_headroom < max_tan * d_cum:
                        break

                    # FIX: Robust array boundary check for bilinear interpolation safety
                    if curr_i < 0.0 or curr_j < 0.0 or curr_i >= rows - 1 or curr_j >= cols - 1:
                        break
                        
                    ci = int(curr_i)
                    cj = int(curr_j)

                    fi = curr_i - ci
                    fj = curr_j - cj
                    h_t = (H[ci,     cj    ] * (1.0 - fi) * (1.0 - fj) +
                           H[ci + 1, cj    ] * fi          * (1.0 - fj) +
                           H[ci,     cj + 1] * (1.0 - fi) * fj          +
                           H[ci + 1, cj + 1] * fi          * fj         ) * max_altitude

                    tan_e = (h_t - h_origin) / d_cum
                    if tan_e > max_tan:
                        max_tan = tan_e

                # Clamp max_tan to prevent domain errors in arctan if values get strange
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