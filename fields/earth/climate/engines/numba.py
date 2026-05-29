import numpy as np
from numba import njit, prange

@njit(parallel=True)
def fast_euler_advection(scalar_field, v_x, v_y, dx, dy, dt):
    """
    Fast Euler integration for advecting a scalar field.
    """
    rows, cols = scalar_field.shape
    out = np.empty_like(scalar_field)
    
    for i in prange(rows):
        for j in range(cols):
            # Compute gradients with central difference where possible
            if i == 0:
                ds_dx = (scalar_field[1, j] - scalar_field[0, j]) / dx
            elif i == rows - 1:
                ds_dx = (scalar_field[rows - 1, j] - scalar_field[rows - 2, j]) / dx
            else:
                ds_dx = (scalar_field[i + 1, j] - scalar_field[i - 1, j]) / (2 * dx)
                
            if j == 0:
                ds_dy = (scalar_field[i, 1] - scalar_field[i, 0]) / dy
            elif j == cols - 1:
                ds_dy = (scalar_field[i, cols - 1] - scalar_field[i, cols - 2]) / dy
            else:
                ds_dy = (scalar_field[i, j + 1] - scalar_field[i, j - 1]) / (2 * dy)
                
            adv = - (v_x[i, j] * ds_dx + v_y[i, j] * ds_dy)
            out[i, j] = scalar_field[i, j] + adv * dt
            
    return out

@njit(parallel=True)
def semi_lagrangian_advection(scalar_field, v_x, v_y, dx, dy, dt):
    """
    Semi-Lagrangian advection (unconditionally stable for large dt).
    Backtraces the velocity field to find the source of the scalar parcel.
    """
    rows, cols = scalar_field.shape
    out = np.empty_like(scalar_field)
    
    for i in prange(rows):
        for j in range(cols):
            # Physical position of current cell
            x = i * dx
            y = j * dy
            
            # Trace back
            src_x = x - v_x[i, j] * dt
            src_y = y - v_y[i, j] * dt
            
            # Map back to grid indices
            src_i = src_x / dx
            src_j = src_y / dy
            
            # Clamp to grid boundaries
            if src_i < 0: src_i = 0
            if src_i >= rows - 1: src_i = rows - 1.001
            if src_j < 0: src_j = 0
            if src_j >= cols - 1: src_j = cols - 1.001
            
            # Bilinear interpolation
            i0 = int(src_i)
            j0 = int(src_j)
            fx = src_i - i0
            fy = src_j - j0
            
            c00 = scalar_field[i0, j0]
            c10 = scalar_field[i0 + 1, j0]
            c01 = scalar_field[i0, j0 + 1]
            c11 = scalar_field[i0 + 1, j0 + 1]
            
            val = (c00 * (1 - fx) * (1 - fy) +
                   c10 * fx * (1 - fy) +
                   c01 * (1 - fx) * fy +
                   c11 * fx * fy)
                   
            out[i, j] = val
            
    return out

@njit(parallel=True)
def compute_wind_acceleration(P, v_x, v_y, dx, dy, rho_air, f, wind_friction):
    """
    Calculate wind acceleration using PGF, Coriolis, and Friction.
    """
    rows, cols = P.shape
    dV_x = np.empty_like(v_x)
    dV_y = np.empty_like(v_y)
    
    for i in prange(rows):
        for j in range(cols):
            # Gradients
            if i == 0:
                dP_dx = (P[1, j] - P[0, j]) / dx
            elif i == rows - 1:
                dP_dx = (P[rows - 1, j] - P[rows - 2, j]) / dx
            else:
                dP_dx = (P[i + 1, j] - P[i - 1, j]) / (2 * dx)
                
            if j == 0:
                dP_dy = (P[i, 1] - P[i, 0]) / dy
            elif j == cols - 1:
                dP_dy = (P[i, cols - 1] - P[i, cols - 2]) / dy
            else:
                dP_dy = (P[i, j + 1] - P[i, j - 1]) / (2 * dy)
                
            # Forces
            pgf_x = -dP_dx / rho_air
            pgf_y = -dP_dy / rho_air
            
            coriolis_x = f * v_y[i, j]
            coriolis_y = -f * v_x[i, j]
            
            fric_x = -wind_friction * v_x[i, j]
            fric_y = -wind_friction * v_y[i, j]
            
            dV_x[i, j] = pgf_x + coriolis_x + fric_x
            dV_y[i, j] = pgf_y + coriolis_y + fric_y
            
    return dV_x, dV_y

@njit(parallel=True)
def compute_divergence(v_x, v_y, dx, dy):
    """Computes the divergence of the 2D velocity field."""
    rows, cols = v_x.shape
    div = np.zeros_like(v_x)
    for i in prange(rows):
        for j in range(cols):
            du_dx = (v_x[i+1, j] - v_x[i-1, j]) / (2*dx) if 0 < i < rows-1 else 0.0
            dv_dy = (v_y[i, j+1] - v_y[i, j-1]) / (2*dy) if 0 < j < cols-1 else 0.0
            div[i, j] = du_dx + dv_dy
    return div

@njit(parallel=True)
def solve_poisson_jacobi(div, dx, dy, iterations=10):
    """Uses Jacobi iterations to solve the Poisson equation for pressure."""
    rows, cols = div.shape
    p = np.zeros_like(div)
    p_new = np.zeros_like(div)
    
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)
    
    for _ in range(iterations):
        for i in prange(rows):
            for j in range(cols):
                p_left = p[i-1, j] if i > 0 else p[0, j]
                p_right = p[i+1, j] if i < rows-1 else p[rows-1, j]
                p_up = p[i, j-1] if j > 0 else p[i, 0]
                p_down = p[i, j+1] if j < cols-1 else p[i, cols-1]
                
                p_new[i, j] = (dy2 * (p_left + p_right) + dx2 * (p_up + p_down) - dx2 * dy2 * div[i, j]) / denom
        
        # We must copy to state iteratively
        for i in prange(rows):
            for j in range(cols):
                p[i, j] = p_new[i, j]
                
    return p

@njit(parallel=True)
def project_velocity(v_x, v_y, p, dx, dy):
    """Subtracts the pressure gradient from the velocity field to make it divergence-free."""
    rows, cols = v_x.shape
    for i in prange(rows):
        for j in range(cols):
            dp_dx = (p[i+1, j] - p[i-1, j]) / (2*dx) if 0 < i < rows-1 else (p[1, j]-p[0, j])/dx if i==0 else (p[rows-1, j]-p[rows-2, j])/dx
            dp_dy = (p[i, j+1] - p[i, j-1]) / (2*dy) if 0 < j < cols-1 else (p[i, 1]-p[i, 0])/dy if j==0 else (p[i, cols-1]-p[i, cols-2])/dy
            v_x[i, j] -= dp_dx
            v_y[i, j] -= dp_dy

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
def raymarch_shadows(H, sx, sy, sz, max_altitude, map_size_x, map_size_y, dx, dy, step_size):
    """
    Raymarches across the terrain heightmap to figure out shadows from the solar vector.
    Returns a shadow map (1.0 = lit, 0.0 = completely shadowed).
    H should be normalized (0.0 to 1.0).
    """
    rows, cols = H.shape
    shadow_map = np.ones((rows, cols), dtype=np.float32)
    
    if sz <= 0.0:
        # Sun is below horizon, everything is in shadow
        return np.zeros((rows, cols), dtype=np.float32)
        
    norm_xy = np.sqrt(sx**2 + sy**2)
    if norm_xy < 1e-5:
        # Sun is directly overhead, no shadows
        return shadow_map
        
    
    # Grid steps
    dir_i = sx / norm_xy / dx
    dir_j = sy / norm_xy / dy
    
    # Rise in altitude (meters) per 1 unit of step_size traversed in XY
    dz_ds = sz / norm_xy
    
    # Raymarch limit: max width of map roughly
    max_steps = int(max(rows, cols))
    
    for r in prange(rows):
        for c in range(cols):
            start_h = H[r, c] * max_altitude
            
            # March along the ray
            curr_i = float(r)
            curr_j = float(c)
            curr_z = start_h
            
            for step in range(1, max_steps):
                curr_i -= dir_i * step_size
                curr_j -= dir_j * step_size
                
                # Check grid bounds
                if curr_i < 0 or curr_i >= rows - 1 or curr_j < 0 or curr_j >= cols - 1:
                    break
                    
                # Physical z of the ray
                curr_z += dz_ds * np.sqrt((dir_i * dx)**2 + (dir_j * dy)**2) * step_size
                
                # if ray goes completely above max_altitude, it won't hit anything else
                if curr_z > max_altitude:
                    break
                    
                # Bilinear interp height
                i0 = int(curr_i)
                j0 = int(curr_j)
                fi = curr_i - i0
                fj = curr_j - j0
                
                h_terrain = (H[i0, j0] * (1 - fi) * (1 - fj) +
                             H[i0 + 1, j0] * fi * (1 - fj) +
                             H[i0, j0 + 1] * (1 - fi) * fj +
                             H[i0 + 1, j0 + 1] * fi * fj) * max_altitude
                
                if h_terrain > curr_z:
                    shadow_map[r, c] = 0.0
                    break
                    
    return shadow_map
