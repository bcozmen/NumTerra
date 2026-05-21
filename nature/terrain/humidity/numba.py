"""
Numba-accelerated kernels for the humidity simulation.

Key kernels
-----------
advect_numba              – parallel semi-Lagrangian back-tracing advection
                            (replaces scipy.ndimage.map_coordinates; ~4-8× faster)
compute_evaporation_numba – parallel evaporation fraction per cell
humidity_capacity_numba   – parallel Magnus saturation pressure
compute_rain_and_update_numba – fused saturation rain + orographic rain + humidity
                                budget + soil moisture update in one parallel pass
"""

import numpy as np
from numba import njit, prange


# ------------------------------------------------------------------ advection

@njit(cache=True, parallel=True)
def advect_numba(humidity, speed_i, speed_j, max_advection_cells):
    """
    Semi-Lagrangian back-tracing advection (replaces map_coordinates).

    Parameters
    ----------
    humidity            : (H, W) float64
    speed_i / speed_j   : (H, W) float64  – wind speed in grid-cells / second
    max_advection_cells : float            – max displacement per step (per cell)

    Returns
    -------
    (H, W) float64 – advected humidity field

    Notes
    -----
    Displacement is clamped **per cell** rather than via a global effective dt.
    A global dt would be throttled by the single windiest cell on the map,
    making moisture transport stagnant everywhere else.  Per-cell clamping
    lets fast jets advect at full speed while calm regions still transport
    moisture at their natural rate.

    ``np.floor`` is used instead of ``int()`` for the bilinear base index so
    that small negative coordinates (possible before the reflect clamp fully
    settles) round toward −∞ rather than toward zero, keeping di/dj ∈ [0, 1).
    """
    H, W = humidity.shape
    out = np.empty((H, W), dtype=np.float64)

    Hi = H - 1
    Wi = W - 1
    period_i = 2.0 * Hi
    period_j = 2.0 * Wi

    for i in prange(H):
        for j in range(W):
            # Per-cell clamping: scale the displacement vector so its magnitude
            # never exceeds max_advection_cells, without touching other cells.
            disp_i = speed_i[i, j]
            disp_j = speed_j[i, j]
            mag = (disp_i ** 2 + disp_j ** 2) ** 0.5
            if mag > max_advection_cells:
                scale  = max_advection_cells / mag
                disp_i *= scale
                disp_j *= scale

            fi = i - disp_i
            fj = j - disp_j

            # --- reflect boundary (scipy 'reflect' / half-sample symmetric) ---
            if period_i > 0.0:
                fi = fi % period_i
                if fi < 0.0:
                    fi += period_i
                if fi > Hi:
                    fi = period_i - fi

            if period_j > 0.0:
                fj = fj % period_j
                if fj < 0.0:
                    fj += period_j
                if fj > Wi:
                    fj = period_j - fj

            # bilinear weights — floor is correct for negative coords
            i0 = int(np.floor(fi))
            j0 = int(np.floor(fj))
            i1 = i0 + 1
            j1 = j0 + 1
            if i1 > Hi:
                i1 = Hi
            if j1 > Wi:
                j1 = Wi

            di = fi - i0
            dj = fj - j0

            out[i, j] = (
                humidity[i0, j0] * (1.0 - di) * (1.0 - dj)
                + humidity[i0, j1] * (1.0 - di) * dj
                + humidity[i1, j0] * di         * (1.0 - dj)
                + humidity[i1, j1] * di         * dj
            )

    return out


# ------------------------------------------------------- evaporation + capacity

@njit(cache=True, parallel=True)
def humidity_capacity_numba(temperature):
    """
    Magnus / Tetens saturation vapour pressure [hPa].
    Returns (H, W) float64.
    """
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
    """
    Evaporation fraction [dimensionless / step].
    Caller does:  humidity += result * humidity_capacity(T)

    Returns (H, W) float64.
    """
    H, W = temperature.shape
    out = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        for j in range(W):
            is_sea   = sea_mask[i, j]
            is_lake  = lake_mask[i, j]
            is_river = river_mask[i, j] and not is_lake
            is_land  = not is_sea and not is_lake and not river_mask[i, j]

            if is_sea:
                wf = sea_evaporation
            elif is_lake:
                wf = lake_evaporation
            elif is_river:
                wf = river_evaporation
            else:
                wf = land_evaporation

            if is_land:
                sf = soil_moisture[i, j] / (soil_capacity + 1e-8)
                if sf > 1.0:
                    sf = 1.0
                wf *= 1.0 + 2.0 * sf

            wind_speed  = (wind_i[i, j] ** 2 + wind_j[i, j] ** 2) ** 0.5
            temp_factor = np.exp(0.05 * (temperature[i, j] - 15.0))
            sun_factor  = 0.5 + 0.5 * sun[i, j]
            wind_factor = 1.0 + 0.05 * wind_speed

            out[i, j] = evaporation_rate * wf * temp_factor * sun_factor * wind_factor

    return out


# ----------------------------------------- fused rain + budget + soil update

@njit(cache=True, parallel=True)
def compute_rain_and_update_numba(
    humidity, humidity_cap,
    evap_frac,
    wind_i, wind_j, grad_i, grad_j,
    sea_mask, lake_mask, river_mask,
    soil_moisture, rain_accum,
    condensation_rate, orographic_factor,
    soil_capacity, soil_evap_rate,
    itcz_factor,
    rain_shadow_fraction,
    wilting_point,
):
    """
    Fused kernel that applies evaporation, condensation, orographic rain,
    ITCZ multiplier, föhn rain-shadow, soil moisture, and wilting point
    in a single parallel pass.

    Steps merged
    ------------
    1. humidity += evap_frac * cap                  (atmospheric evaporation)
    2. Soil evaporation returned to air (before rain so it can rain out)
    3. Compute saturation rain and orographic rain, scaled by itcz_factor
    4. humidity -= rain_delta
    5. Föhn rain-shadow: excess humidity converted to precipitation (mass-conserving)
    6. Soil moisture update (absorb precip, track runoff overflow)
    7. Accumulate precipitation

    Parameters
    ----------
    itcz_factor          : (H, W) float64  – ≥1 at equator, <1 at ±30° horse lats
    rain_shadow_fraction : float           – max RH fraction allowed on leeward side
    wilting_point        : float           – soil fraction below which ET ramps to 0

    Returns
    -------
    new_humidity  (H, W)
    new_soil      (H, W)
    new_rain      (H, W)  – cumulative rain_accum + this step's precip
    new_runoff    (H, W)  – per-step soil overflow (mm) for hydrology systems
    """
    H, W = humidity.shape
    new_humidity = np.empty((H, W), dtype=np.float64)
    new_soil     = np.empty((H, W), dtype=np.float64)
    new_rain     = np.empty((H, W), dtype=np.float64)
    new_runoff   = np.empty((H, W), dtype=np.float64)

    for i in prange(H):
        for j in range(W):
            is_land = (
                not sea_mask[i, j]
                and not lake_mask[i, j]
                and not river_mask[i, j]
            )

            cap  = humidity_cap[i, j]
            itcz = itcz_factor[i, j]

            # --- step 1: atmospheric evaporation ---
            h = humidity[i, j] + evap_frac[i, j] * cap

            # --- step 2: soil evaporation returns to air BEFORE rain ---
            # Moving this before the rain calculation means soil moisture can
            # contribute to saturation and rain out in the same step, avoiding
            # a one-iteration lag in hot/wet climates that causes "pulsing."
            s = soil_moisture[i, j]
            soil_evap = 0.0
            if is_land:
                soil_frac = s / (soil_capacity + 1e-8)
                # Linear ramp from 0 at wilting_point to full rate above
                # 2×wilting_point.  Avoids the discontinuous step artifact
                # that causes artificial dry patches at the wilting threshold.
                if soil_frac <= wilting_point:
                    soil_evap = 0.0
                elif soil_frac < 2.0 * wilting_point:
                    ramp      = (soil_frac - wilting_point) / (wilting_point + 1e-8)
                    soil_evap = s * soil_evap_rate * ramp
                else:
                    soil_evap = s * soil_evap_rate
                h += soil_evap

            # --- step 3a: saturation rain (ITCZ-weighted condensation) ---
            excess   = h - cap
            if excess < 0.0:
                excess = 0.0
            rain_sat = excess * condensation_rate * itcz

            # --- step 3b: orographic rain (tanh uplift, ITCZ-weighted) ---
            ws = (wind_i[i, j] ** 2 + wind_j[i, j] ** 2) ** 0.5 + 1e-8
            wx = wind_i[i, j] / ws
            wy = wind_j[i, j] / ws
            uplift = wx * grad_i[i, j] + wy * grad_j[i, j]

            x = 5.0 * uplift
            if x >  20.0: x =  20.0
            if x < -20.0: x = -20.0
            ex = np.exp(2.0 * x)
            t  = (ex - 1.0) / (ex + 1.0)   # tanh(5*uplift)

            if t >= 0.0:
                rain_oro = t * h * orographic_factor * itcz   # windward
            else:
                rain_oro = t * 0.5 * h * orographic_factor    # leeward (no ITCZ on drying)

            # --- step 4: humidity budget ---
            precip         = rain_sat + (rain_oro if rain_oro > 0.0 else 0.0)
            humidity_delta = rain_sat + rain_oro
            h -= humidity_delta
            if h < 0.0:
                h = 0.0

            # --- step 5: föhn rain-shadow (mass-conserving) ---
            # On leeward side (t < 0) descending air warms adiabatically.
            # Instead of silently deleting the excess humidity, convert it to
            # precipitation at the ridge so water mass is conserved.
            if t < 0.0:
                shadow_cap = rain_shadow_fraction * cap
                if h > shadow_cap:
                    shadow_rain = h - shadow_cap
                    precip += shadow_rain   # counts as orographic ridge rain
                    h = shadow_cap

            # --- step 6: soil moisture + runoff ---
            runoff = 0.0
            if is_land:
                s_new = s + precip - soil_evap
                if s_new < 0.0:
                    s_new = 0.0
                if s_new > soil_capacity:
                    runoff = s_new - soil_capacity   # overflow → river/hydrology
                    s_new  = soil_capacity
                s = s_new

            new_humidity[i, j] = h
            new_soil[i, j]     = s
            new_rain[i, j]     = rain_accum[i, j] + precip
            new_runoff[i, j]   = runoff

    return new_humidity, new_soil, new_rain, new_runoff
