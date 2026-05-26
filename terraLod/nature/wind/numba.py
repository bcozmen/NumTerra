from numba import njit, prange
import numpy as np

@njit(cache=True, parallel=True)
def advect_numba_multi(humidity, speed_i, speed_j, max_advection):
    """
    Semi-Lagrangian advection for multiple scalar fields.

    Parameters
    ----------
    humidity : (H, W, C) array
        Multiple scalar fields (channels) advected together.
    speed_i : (H, W) array
        Vertical velocity component.
    speed_j : (H, W) array
        Horizontal velocity component.
    max_advection : float
        Maximum allowed displacement magnitude.
    """
    H, W, C = humidity.shape
    out = np.empty((H, W, C), dtype=np.float64)

    Hi, Wi = H - 1, W - 1

    for i in prange(H):
        for j in range(W):

            disp_i = speed_i[i, j]
            disp_j = speed_j[i, j]

            mag = (disp_i * disp_i + disp_j * disp_j) ** 0.5

            if mag > max_advection:
                scale = max_advection / mag
                disp_i *= scale
                disp_j *= scale

            fi = i - disp_i
            fj = j - disp_j

            # clamp sampling coords
            if fi < 0.0:
                fi = 0.0
            elif fi > Hi:
                fi = Hi

            if fj < 0.0:
                fj = 0.0
            elif fj > Wi:
                fj = Wi

            i0 = int(fi // 1)
            j0 = int(fj // 1)

            i1 = i0 + 1 if i0 + 1 < H else Hi
            j1 = j0 + 1 if j0 + 1 < W else Wi

            di = fi - i0
            dj = fj - j0

            for c in range(C):
                v00 = humidity[i0, j0, c]
                v01 = humidity[i0, j1, c]
                v10 = humidity[i1, j0, c]
                v11 = humidity[i1, j1, c]

                out[i, j, c] = (
                    v00 * (1.0 - di) * (1.0 - dj)
                    + v01 * (1.0 - di) * dj
                    + v10 * di * (1.0 - dj)
                    + v11 * di * dj
                )

    return out