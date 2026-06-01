import numpy as np
from numba import njit, prange

@njit(parallel=True)
def euler_advect(f0, f1, f2, v_x, v_y, dx, dy, dt):
    """Euler advection for three scalar fields simultaneously (shared boundary checks)."""
    rows, cols = f0.shape
    out0 = np.empty_like(f0)
    out1 = np.empty_like(f1)
    out2 = np.empty_like(f2)
    for i in prange(rows):
        for j in range(cols):
            if i == 0:
                i0 = 0; i1 = 1; sy = 1.0 / dy
            elif i == rows - 1:
                i0 = rows - 2; i1 = rows - 1; sy = 1.0 / dy
            else:
                i0 = i - 1; i1 = i + 1; sy = 0.5 / dy
            if j == 0:
                j0 = 0; j1 = 1; sx = 1.0 / dx
            elif j == cols - 1:
                j0 = cols - 2; j1 = cols - 1; sx = 1.0 / dx
            else:
                j0 = j - 1; j1 = j + 1; sx = 0.5 / dx
            vx = v_x[i, j]; vy = v_y[i, j]
            out0[i,j] = f0[i,j] - (vy*(f0[i1,j]-f0[i0,j])*sy + vx*(f0[i,j1]-f0[i,j0])*sx)*dt
            out1[i,j] = f1[i,j] - (vy*(f1[i1,j]-f1[i0,j])*sy + vx*(f1[i,j1]-f1[i,j0])*sx)*dt
            out2[i,j] = f2[i,j] - (vy*(f2[i1,j]-f2[i0,j])*sy + vx*(f2[i,j1]-f2[i,j0])*sx)*dt
    return out0, out1, out2


@njit(parallel=True)
def semi_lagrangian_advect(f0, f1, f2, v_x, v_y, dx, dy, dt):
    """Semi-Lagrangian advection for three scalar fields with a single shared backtrace."""
    rows, cols = f0.shape
    out0 = np.empty_like(f0)
    out1 = np.empty_like(f1)
    out2 = np.empty_like(f2)
    for i in prange(rows):
        for j in range(cols):
            # Shared backtrace (computed once, reused for all three fields)
            src_i = i - v_y[i, j] * dt / dy
            src_j = j - v_x[i, j] * dt / dx
            if src_i < 0.0:            src_i = 0.0
            if src_i > rows - 1.001:   src_i = rows - 1.001
            if src_j < 0.0:            src_j = 0.0
            if src_j > cols - 1.001:   src_j = cols - 1.001
            i0 = int(src_i); j0 = int(src_j)
            i1 = i0 + 1;     j1 = j0 + 1
            fy = src_i - i0; fx = src_j - j0
            w00 = (1.0 - fy) * (1.0 - fx)
            w10 = fy          * (1.0 - fx)
            w01 = (1.0 - fy) * fx
            w11 = fy          * fx
            out0[i,j] = f0[i0,j0]*w00 + f0[i1,j0]*w10 + f0[i0,j1]*w01 + f0[i1,j1]*w11
            out1[i,j] = f1[i0,j0]*w00 + f1[i1,j0]*w10 + f1[i0,j1]*w01 + f1[i1,j1]*w11
            out2[i,j] = f2[i0,j0]*w00 + f2[i1,j0]*w10 + f2[i0,j1]*w01 + f2[i1,j1]*w11
    return out0, out1, out2


@njit(parallel=True)
def wind_accelerate(P, v_x, v_y, dx, dy, rho_air, f, wind_friction, dt):
    """PGF + Coriolis + Friction applied in-place to v_x, v_y."""
    rows, cols = P.shape
    for i in prange(rows):
        for j in range(cols):
            if i == 0:          dP_dy = (P[1, j]       - P[0, j])       / dy
            elif i == rows - 1: dP_dy = (P[rows-1, j]  - P[rows-2, j])  / dy
            else:               dP_dy = (P[i+1, j]     - P[i-1, j])     / (2*dy)
            if j == 0:          dP_dx = (P[i, 1]       - P[i, 0])       / dx
            elif j == cols - 1: dP_dx = (P[i, cols-1]  - P[i, cols-2])  / dx
            else:               dP_dx = (P[i, j+1]     - P[i, j-1])     / (2*dx)
            vx = v_x[i, j]; vy = v_y[i, j]
            # Coriolis force: object moving in +x is pushed to -y (in Northern Hemisphere, sin(lat)>0 => f>0)
            v_x[i, j] += (-dP_dx / rho_air + f * vy - wind_friction * vx) * dt
            v_y[i, j] += (-dP_dy / rho_air - f * vx - wind_friction * vy) * dt


@njit(parallel=True)
def pressure_project(v_x, v_y, dx, dy, iterations):
    """Divergence → Jacobi pressure solve → velocity projection, all in-place."""
    rows, cols = v_x.shape
    dx2 = dx * dx; dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    div = np.zeros_like(v_x)
    for i in prange(rows):
        for j in range(cols):
            dv_dy = (v_y[i+1,j] - v_y[i-1,j]) / (2*dy) if 0 < i < rows-1 else 0.0
            du_dx = (v_x[i,j+1] - v_x[i,j-1]) / (2*dx) if 0 < j < cols-1 else 0.0
            div[i, j] = du_dx + dv_dy

    p = np.zeros_like(v_x)
    p_new = np.zeros_like(v_x)
    for _ in range(iterations):
        for i in prange(rows):
            for j in range(cols):
                p_d = p[i-1, j] if i > 0      else p[0,      j]
                p_u = p[i+1, j] if i < rows-1 else p[rows-1, j]
                p_l = p[i, j-1] if j > 0      else p[i,      0]
                p_r = p[i, j+1] if j < cols-1 else p[i, cols-1]
                p_new[i, j] = (dy2*(p_l+p_r) + dx2*(p_u+p_d) - dx2*dy2*div[i,j]) / denom
        p, p_new = p_new, p  # swap buffers — no data copy

    for i in prange(rows):
        for j in range(cols):
            if i == 0:          dp_dy = (p[1, j]       - p[0, j])       / dy
            elif i == rows - 1: dp_dy = (p[rows-1, j]  - p[rows-2, j])  / dy
            else:               dp_dy = (p[i+1, j]     - p[i-1, j])     / (2*dy)
            if j == 0:          dp_dx = (p[i, 1]       - p[i, 0])       / dx
            elif j == cols - 1: dp_dx = (p[i, cols-1]  - p[i, cols-2])  / dx
            else:               dp_dx = (p[i, j+1]     - p[i, j-1])     / (2*dx)
            v_x[i, j] -= dp_dx
            v_y[i, j] -= dp_dy


@njit(parallel=True)
def orographic_cooling(H, sea_level, v_x, v_y, dx, dy, lapse_rate):
    """Temperature tendency from orographic lifting (gradient of terrain clamped to land)."""
    rows, cols = H.shape
    dTa = np.empty_like(v_x)
    for i in prange(rows):
        for j in range(cols):
            if i == 0:
                h_im = max(H[0,      j] - sea_level, 0.0)
                h_ip = max(H[1,      j] - sea_level, 0.0); sy = 1.0 / dy
            elif i == rows - 1:
                h_im = max(H[rows-2, j] - sea_level, 0.0)
                h_ip = max(H[rows-1, j] - sea_level, 0.0); sy = 1.0 / dy
            else:
                h_im = max(H[i-1, j] - sea_level, 0.0)
                h_ip = max(H[i+1, j] - sea_level, 0.0); sy = 0.5 / dy
            if j == 0:
                h_jm = max(H[i, 0]      - sea_level, 0.0)
                h_jp = max(H[i, 1]      - sea_level, 0.0); sx = 1.0 / dx
            elif j == cols - 1:
                h_jm = max(H[i, cols-2] - sea_level, 0.0)
                h_jp = max(H[i, cols-1] - sea_level, 0.0); sx = 1.0 / dx
            else:
                h_jm = max(H[i, j-1] - sea_level, 0.0)
                h_jp = max(H[i, j+1] - sea_level, 0.0); sx = 0.5 / dx
            w = v_y[i, j] * (h_ip - h_im) * sy + v_x[i, j] * (h_jp - h_jm) * sx
            dTa[i, j] = -w * lapse_rate
    return dTa