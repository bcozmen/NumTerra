"""
Numba-accelerated kernels for the climate humidity simulation.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def advect_numba(humidity, speed_i, speed_j, max_advection):
    """
    Parallel Semi-Lagrangian back-tracing advection.
    speed_i and speed_j must be passed in units of: cells per iteration.
    """
    H, W = humidity.shape
    out = np.empty((H, W), dtype=np.float64)

    Hi, Wi = H - 1, W - 1
    period_i, period_j = 2.0 * Hi, 2.0 * Wi

    for i in prange(H):
        for j in range(W):
            disp_i = speed_i[i, j]
            disp_j = speed_j[i, j]
            mag = (disp_i ** 2 + disp_j ** 2) ** 0.5
            
            # Capping maximum movement to prevent tracking out-of-bounds array artifacts
            if mag > max_advection:
                scale = max_advection / mag
                disp_i *= scale
                disp_j *= scale

            fi = i - disp_i
            fj = j - disp_j

            # Reflective boundary conditions
            if period_i > 0.0:
                fi = fi % period_i
                if fi < 0.0: fi += period_i
                if fi > Hi:  fi = period_i - fi

            if period_j > 0.0:
                fj = fj % period_j
                if fj < 0.0: fj += period_j
                if fj > Wi:  fj = period_j - fj

            i0 = int(np.floor(fi))
            j0 = int(np.floor(fj))
            i1 = min(i0 + 1, Hi)
            j1 = min(j0 + 1, Wi)

            di = fi - i0
            dj = fj - j0

            out[i, j] = (
                humidity[i0, j0] * (1.0 - di) * (1.0 - dj)
                + humidity[i0, j1] * (1.0 - di) * dj
                + humidity[i1, j0] * di         * (1.0 - dj)
                + humidity[i1, j1] * di         * dj
            )
    return out


@njit(cache=True, parallel=True)
def humidity_capacity_numba(temperature):
    H, W = temperature.shape
    out = np.empty((H, W), dtype=np.float64)
    for i in prange(H):
        for j in range(W):
            t = temperature[i, j]
            out[i, j] = 6.112 * np.exp(17.67 * t / (t + 243.5))
    return out


@njit(cache=True, parallel=True)
def compute_evaporation_numba(
    temperature, sun, wind_i, wind_j,
    sea_mask, lake_mask, river_mask, soil_moisture,
    evaporation_rate, land_evaporation, sea_evaporation,
    lake_evaporation, river_evaporation, soil_capacity,
):
    H, W = temperature.shape
    out = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        for j in range(W):
            if sea_mask[i, j]:
                wf = sea_evaporation
            elif lake_mask[i, j]:
                wf = lake_evaporation
            elif river_mask[i, j]:
                wf = river_evaporation
            else:
                wf = land_evaporation
                sf = soil_moisture[i, j] / (soil_capacity + 1e-8)
                wf *= (1.0 + 1.5 * min(sf, 1.0))

            wind_speed = (wind_i[i, j] ** 2 + wind_j[i, j] ** 2) ** 0.5
            temp_factor = np.exp(0.04 * (temperature[i, j] - 15.0))
            sun_factor = 0.5 + 0.5 * sun[i, j]
            wind_factor = 1.0 + 0.08 * wind_speed

            out[i, j] = evaporation_rate * wf * temp_factor * sun_factor * wind_factor
    return out


@njit(cache=True, parallel=True)
def compute_rain_and_update_numba(
    humidity, humidity_cap, evap_frac,
    wind_i, wind_j, grad_i, grad_j,
    sea_mask, lake_mask, river_mask,
    soil_moisture, rain_accum, runoff_accum,
    condensation_rate, orographic_factor,
    soil_capacity, soil_evap_rate, itcz_factor,
    rain_shadow_fraction, wilting_point, hpa_to_mm,
    inv_iterations
):
    H, W = humidity.shape
    new_humidity = np.empty((H, W), dtype=np.float64)
    new_soil     = np.empty((H, W), dtype=np.float64)
    new_rain     = np.empty((H, W), dtype=np.float64)
    new_runoff   = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        for j in range(W):
            is_sea = sea_mask[i, j]
            is_water = is_sea or lake_mask[i, j] or river_mask[i, j]
            
            cap = humidity_cap[i, j] + 1e-8
            itcz = itcz_factor[i, j]

            # 1. Atmospheric Evaporation
            evap_hpa = evap_frac[i, j] * cap
            h = humidity[i, j] + evap_hpa
            
            if is_sea:
                h = min(h, 0.88 * cap) # Cap marine boundary layer saturation to stop endless loop downpours over open sea

            # Ground moisture loss
            s = soil_moisture[i, j]
            evap_mm = evap_hpa * hpa_to_mm

            # 2. General Rain (Convective/Frontal)
            rh = h / cap
            rain_sat = 0.0
            if rh > 0.75:
                intensity = ((rh - 0.75) / 0.25) ** 2
                rain_sat = intensity * condensation_rate * h * itcz

            # 3. Orographic Rain
            ws = (wind_i[i, j] ** 2 + wind_j[i, j] ** 2) ** 0.5 + 1e-8
            uplift = (wind_i[i, j] / ws) * grad_i[i, j] + (wind_j[i, j] / ws) * grad_j[i, j]
            
            x = max(min(3.0 * uplift, 10.0), -10.0)
            t = (np.exp(2.0 * x) - 1.0) / (np.exp(2.0 * x) + 1.0)

            rain_oro = 0.0
            if t > 0.0:
                rain_oro = t * h * orographic_factor * itcz
            else:
                # Rain shadow effect
                shadow_cap = (1.0 - (-t) * (1.0 - rain_shadow_fraction)) * cap
                h = min(h, shadow_cap)

            # 4. Vapor Pressure Allocation
            precip_hpa = min(rain_sat + rain_oro, h)
            h = max(h - precip_hpa, 0.0)

            # 5. Ground and Soil Hydrology
            precip_mm = precip_hpa * hpa_to_mm
            runoff = 0.0

            if is_water:
                s = soil_capacity
                runoff = precip_mm
            else:
                # Scale depletion by inverse iteration length so it doesn't dry out 10x too fast
                s -= (evap_mm * 0.2) * inv_iterations
                s = max(s, 0.0)
                
                sat_frac = min(s / soil_capacity, 1.0)
                immediate_runoff = precip_mm * (sat_frac ** 2)
                infiltration = precip_mm - immediate_runoff
                
                s += infiltration
                s *= (1.0 - (soil_evap_rate * 0.25) * inv_iterations)
                
                if s > soil_capacity:
                    runoff = immediate_runoff + (s - soil_capacity)
                    s = soil_capacity
                else:
                    runoff = immediate_runoff
                s = max(s, 0.0)

            new_humidity[i, j] = h
            new_soil[i, j] = s
            new_rain[i, j] = rain_accum[i, j] + precip_mm
            new_runoff[i, j] = runoff_accum[i, j] + runoff

    return new_humidity, new_soil, new_rain, new_runoff