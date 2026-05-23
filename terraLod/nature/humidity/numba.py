"""
Numba-accelerated kernels for the climate humidity simulation.
"""

import numpy as np
from numba import njit, prange
import math


@njit(cache=True, parallel=True)
def compute_evaporation_numba(temperature, sun, wind_i, wind_j,
                                    sea_mask, lake_mask, river_mask,
                                    soil_moisture,
                                    evaporation_rate,
                                    land_evaporation, sea_evaporation,
                                    lake_evaporation, river_evaporation,
                                    soil_capacity, inverse_iterations):

    H, W = temperature.shape
    out = np.empty((H, W), dtype=np.float64)

    inv_cap = 1.0 / (soil_capacity + 1e-8)

    for i in prange(H):
        for j in range(W):

            # --- wind (cheap inline math) ---
            wi = wind_i[i, j]
            wj = wind_j[i, j]
            wind_speed = (wi * wi + wj * wj) ** 0.5
            wind_factor = 1.0 + 0.08 * wind_speed

            # --- temperature factor ---
            temp_factor = np.exp(0.04 * (temperature[i, j] - 15.0))

            # --- sunlight factor ---
            sun_factor = 0.9 + 0.2 * sun[i, j]

            # --- surface type (branch once) ---
            if sea_mask[i, j]:
                wf = sea_evaporation
            elif lake_mask[i, j]:
                wf = lake_evaporation
            elif river_mask[i, j]:
                wf = river_evaporation
            else:
                sf = soil_moisture[i, j] * inv_cap
                if sf > 1.0:
                    sf = 1.0
                wf = land_evaporation * (1.0 + 1.5 * sf)

            out[i, j] = evaporation_rate * wf * temp_factor * sun_factor * wind_factor * inverse_iterations

    return out


@njit(cache=True, parallel=True)
def advect_numba(humidity, speed_i, speed_j, max_advection):
    """
    Semi-Lagrangian advection with standard clamp-to-edge boundaries.
    """
    H, W = humidity.shape
    out = np.empty((H, W), dtype=np.float64)

    Hi, Wi = H - 1, W - 1

    for i in prange(H):
        for j in range(W):

            disp_i = speed_i[i, j]
            disp_j = speed_j[i, j]
            mag = (disp_i ** 2 + disp_j ** 2) ** 0.5

            if mag > max_advection:
                scale = max_advection / mag
                disp_i *= scale
                disp_j *= scale

            fi = i - disp_i
            fj = j - disp_j

            # --- CLAMP FOR SAMPLING SAFETY ---
            # This safely handles edges without destroying humidity
            if fi < 0.0:
                fi = 0.0
            elif fi > Hi:
                fi = Hi

            if fj < 0.0:
                fj = 0.0
            elif fj > Wi:
                fj = Wi

            i0 = int(np.floor(fi))
            j0 = int(np.floor(fj))
            
            # i1 and j1 bounded by max index
            i1 = min(i0 + 1, Hi)
            j1 = min(j0 + 1, Wi)

            di = fi - i0
            dj = fj - j0

            out[i, j] = (
                humidity[i0, j0] * (1.0 - di) * (1.0 - dj)
                + humidity[i0, j1] * (1.0 - di) * dj
                + humidity[i1, j0] * di * (1.0 - dj)
                + humidity[i1, j1] * di * dj
            )

    return out

@njit(cache=True, parallel=True)
def compute_rain_and_update_numba(humidity, humidity_cap, evap_frac,
                                        wind_i, wind_j, grad_i, grad_j,
                                        sea_mask, lake_mask, river_mask,
                                        soil_moisture, rain_accum, runoff_accum,
                                        condensation_rate, orographic_factor,
                                        soil_capacity, itcz_factor,
                                        rain_humidity_threshold,  hpa_to_mm,
                                        uplift_scale):
    H, W = humidity.shape
    new_humidity = np.empty((H, W), dtype=np.float64)
    new_soil     = np.empty((H, W), dtype=np.float64)
    new_rain     = np.empty((H, W), dtype=np.float64)
    new_runoff   = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        for j in range(W):
            # 0. Prep cell data
            is_sea = sea_mask[i, j]
            is_water = is_sea or lake_mask[i, j] or river_mask[i, j]
            cap = humidity_cap[i, j] + 1e-8
            itcz = itcz_factor[i, j]

            # 1. Atmospheric Evaporation
            h_local, evap_hpa = _evaporate_cell(
                humidity[i, j], cap, evap_frac[i, j], is_sea
            )

            # 2. Convective Rain
            rain_sat = _convective_rain_cell(
                h_local, cap, condensation_rate, itcz, rain_humidity_threshold
            )

            # 3. Orographic Rain & Shadow
            rain_oro, h_local = _orographic_effects_cell(
                h_local, cap, wind_i[i, j], wind_j[i, j], 
                grad_i[i, j], grad_j[i, j], 
                orographic_factor, itcz, uplift_scale
            )

            # 4. Vapor Pressure Allocation
            precip_hpa = min(rain_sat + rain_oro, h_local)
            h_local = max(h_local - precip_hpa, 0.0)
            
            precip_mm = precip_hpa * hpa_to_mm
            evap_mm = evap_hpa * hpa_to_mm

            # 5. Hydrology
            s_local, runoff = _hydrology_cell(
                soil_moisture[i, j], soil_capacity, precip_mm, evap_mm, is_water)

            # 6. Store cell results
            new_humidity[i, j] = h_local
            new_soil[i, j] = s_local
            new_rain[i, j] = rain_accum[i, j] + precip_mm
            new_runoff[i, j] = runoff_accum[i, j] + runoff

    return new_humidity, new_soil, new_rain, new_runoff

@njit(cache=True)
def _evaporate_cell(humidity, cap, evap_frac, is_sea):
    """Calculates atmospheric evaporation for a single cell."""
    evap_hpa = evap_frac * cap
    local_humidity = humidity + evap_hpa
    
    #if is_sea:
    #    local_humidity = min(local_humidity, 0.88 * cap)
        
    return local_humidity, evap_hpa

@njit(cache=True)
def _convective_rain_cell(humidity, cap, condensation_rate, itcz, threshold=0.75):
    """Calculates convective rainfall for a single cell."""
    rh = humidity / cap
    if rh > threshold:
        intensity = ((rh - threshold) / (1.0 - threshold)) ** 2
        return intensity * condensation_rate * itcz * humidity
    return 0.0

@njit(cache=True)
def _orographic_effects_cell(humidity, cap, w_i, w_j, g_i, g_j, orographic_factor, itcz, uplift_scale):
    """Calculates terrain-driven rain and shadow drying for a single cell."""
    ws = math.hypot(w_i, w_j) + 1e-8
    uplift = (w_i / ws) * g_i + (w_j / ws) * g_j
    
    # Numba supports standard math functions
    x = uplift_scale * uplift
    modifier = math.tanh(x) 
    
    rain_oro = 0.0
    adjusted_humidity = humidity
    
    if modifier > 0.0:
        rain_oro = modifier * humidity * orographic_factor * itcz

        
    return rain_oro, adjusted_humidity

@njit(cache=True)
def _hydrology_cell(soil_moisture, soil_cap, precip_mm, evap_mm, is_water):
    """Calculates soil infiltration and runoff for a single cell."""
    if is_water:
        return soil_cap, precip_mm
        
    # Land calculation
    s = max(soil_moisture - (evap_mm ), 0.0)
    
    sat_frac = min(s / soil_cap, 1.0)
    immediate_runoff = precip_mm * (sat_frac ** 2)
    infiltration = precip_mm - immediate_runoff
    
    s += infiltration
    
    if s > soil_cap:
        runoff = immediate_runoff + (s - soil_cap)
        s = soil_cap
    else:
        runoff = immediate_runoff
        
    return max(s, 0.0), runoff
