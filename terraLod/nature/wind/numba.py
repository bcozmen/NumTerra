from numba import njit, prange
import numpy as np


@njit(cache=True, parallel=True)
def get_terrain_deflection_numba(base_i, base_j, height_gradient_i, height_gradient_j):
    H, W = base_i.shape
    terrain_i = np.empty((H, W), dtype=np.float64)
    terrain_j = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        for j in range(W):
            h_grad_i = height_gradient_i[i, j]
            h_grad_j = height_gradient_j[i, j]
            
            grad_norm = (h_grad_i * h_grad_i + h_grad_j * h_grad_j)**0.5 + 1e-6
            grad_i = h_grad_i / grad_norm
            grad_j = h_grad_j / grad_norm
            
            b_i = base_i[i, j]
            b_j = base_j[i, j]
            
            wind_slope_into = b_i * grad_i + b_j * grad_j
            blocking = wind_slope_into if wind_slope_into > 0 else 0.0
            
            tangent_i = -grad_j
            tangent_j = grad_i
            
            wind_dot_tangent = b_i * tangent_i + b_j * tangent_j
            side_sign = 1.0 if wind_dot_tangent >= 0 else -1.0
            
            tangent_i *= side_sign
            tangent_j *= side_sign
            
            terrain_i[i, j] = -blocking * grad_i + blocking * 0.6 * tangent_i
            terrain_j[i, j] = -blocking * grad_j + blocking * 0.6 * tangent_j

    return terrain_i, terrain_j

@njit(cache=True, parallel=True)
def soft_cap_numba(speed, soft, max_speed, alpha=3.0):
    H, W = speed.shape
    result = np.empty((H, W), dtype=np.float64)
    for i in prange(H):
        for j in range(W):
            s = speed[i, j]
            if s > soft:
                denom = max_speed - soft
                if denom < 1e-6:
                    denom = 1e-6
                
                t = (s - soft) / denom
                if t > 1.0: t = 1.0
                elif t < 0.0: t = 0.0
                
                shaped = 1.0 - np.exp(-alpha * t)
                result[i, j] = soft + (max_speed - soft) * shaped
            else:
                if s < 0.0:
                    result[i, j] = 0.0
                else:
                    result[i, j] = s
    return result

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

@njit(cache=True, parallel=True)
def apply_warp_numba(u, v, cx, cy, radius, max_angle_rad, direction):
    H, W = u.shape
    out_u = np.empty((H, W), dtype=np.float64)
    out_v = np.empty((H, W), dtype=np.float64)

    R2 = radius * radius + 1e-12

    for i in prange(H):
        for j in range(W):
            x = j / (W - 1.0)
            y = i / (H - 1.0)

            dx = x - cx
            dy = y - cy

            r2 = dx * dx + dy * dy
            w = np.exp(-r2 / R2)
            theta = direction * max_angle_rad * w

            c_theta = np.cos(theta)
            s_theta = np.sin(theta)

            xw = cx + dx * c_theta - dy * s_theta
            yw = cy + dx * s_theta + dy * c_theta

            xw_idx = xw * (W - 1.0)
            yw_idx = yw * (H - 1.0)

            if xw_idx < 0:
                xw_idx = -xw_idx
            elif xw_idx > W - 1.0:
                xw_idx = 2.0 * (W - 1.0) - xw_idx

            if yw_idx < 0:
                yw_idx = -yw_idx
            elif yw_idx > H - 1.0:
                yw_idx = 2.0 * (H - 1.0) - yw_idx

            if xw_idx < 0: xw_idx = 0.0
            elif xw_idx > W - 1.0: xw_idx = W - 1.0
            
            if yw_idx < 0: yw_idx = 0.0
            elif yw_idx > H - 1.0: yw_idx = H - 1.0

            x0 = int(xw_idx)
            y0 = int(yw_idx)
            x1 = x0 + 1 if x0 < W - 1 else x0
            y1 = y0 + 1 if y0 < H - 1 else y0

            dx_idx = xw_idx - x0
            dy_idx = yw_idx - y0

            u_interp = (u[y0, x0] * (1.0 - dx_idx) * (1.0 - dy_idx) +
                        u[y0, x1] * dx_idx * (1.0 - dy_idx) +
                        u[y1, x0] * (1.0 - dx_idx) * dy_idx +
                        u[y1, x1] * dx_idx * dy_idx)

            v_interp = (v[y0, x0] * (1.0 - dx_idx) * (1.0 - dy_idx) +
                        v[y0, x1] * dx_idx * (1.0 - dy_idx) +
                        v[y1, x0] * (1.0 - dx_idx) * dy_idx +
                        v[y1, x1] * dx_idx * dy_idx)

            out_u[i, j] = u_interp * c_theta - v_interp * s_theta
            out_v[i, j] = u_interp * s_theta + v_interp * c_theta

    return out_u, out_v


@njit(cache=True, parallel=True)
def scale_and_cap_numba(u, v, mean_speed, soft, max_speed, alpha=3.0):
    H, W = u.shape
    out_u = np.empty((H, W), dtype=np.float64)
    out_v = np.empty((H, W), dtype=np.float64)
    
    denom_soft = max_speed - soft
    if denom_soft < 1e-6: denom_soft = 1e-6
    
    for i in prange(H):
        for j in range(W):
            ui = u[i, j] / mean_speed
            vi = v[i, j] / mean_speed
            s = (ui * ui + vi * vi) ** 0.5
            
            if s > soft:
                t = (s - soft) / denom_soft
                if t > 1.0: t = 1.0
                elif t < 0.0: t = 0.0
                
                shaped = 1.0 - np.exp(-alpha * t)
                c_speed = soft + denom_soft * shaped
            else:
                if s < 0.0: c_speed = 0.0
                else: c_speed = s
            
            scale = c_speed / (s + 1e-5)
            out_u[i, j] = ui * scale
            out_v[i, j] = vi * scale
            
    return out_u, out_v
