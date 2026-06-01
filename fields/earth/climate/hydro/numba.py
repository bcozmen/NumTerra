import numpy as np
import math
from numba import njit, prange

@njit(parallel=True)
def compute_hydro_step(P, Ta, Wa, Wc, M_sea, Ws, Vspeed, 
                       moisture_capacity_constant, layer_pressure_drop, pressure_lapse_rate, g,
                       lake_evap_threshold, Ce_water, Ce_land, dt,
                       rH_condensation_threshold, condensation_timescale, precip_conversion_rate, cloud_delay_factor,
                       atmospheric_layer_count):
    rows, cols = P.shape
    Wa_max = np.zeros_like(P)
    Evap = np.zeros_like(P)
    Condensation = np.zeros_like(P)
    Precip = np.zeros_like(P)

    alpha = 1.0 - math.exp(-dt / condensation_timescale)
    removal_rate = precip_conversion_rate * (1.0 - cloud_delay_factor)
    removed_fraction = 1.0 - math.exp(-removal_rate * dt)
    layer_mass = layer_pressure_drop / g

    for i in prange(rows):
        for j in range(cols):
            # 1. Wa_max calculation
            p_curr = P[i, j]
            ta_curr = Ta[i, j]
            wa_max_val = 0.0
            
            for _ in range(atmospheric_layer_count):
                p_safe = max(p_curr, 1e-5)
                # Magnus-Tetens
                es = 6.112 * math.exp((17.67 * ta_curr) / (ta_curr + 243.5)) * 100.0
                if es > 0.99 * p_safe:
                    es = 0.99 * p_safe
                
                denom = p_safe - (1.0 - moisture_capacity_constant) * es
                denom = max(denom, 1e-6)
                qs = moisture_capacity_constant * es / denom
                
                if p_curr > 0:
                    wa_max_val += qs * layer_mass
                
                p_curr -= layer_pressure_drop
                ta_curr -= pressure_lapse_rate * layer_pressure_drop
                
            Wa_max[i, j] = wa_max_val
            
            # 2. Evap calculation
            v_sp = Vspeed[i, j] + 0.1
            m_sec = M_sea[i, j]
            ws_val = Ws[i, j]
            wa_val = Wa[i, j]
            
            if ws_val > lake_evap_threshold:
                m_lake = 1.0 - m_sec
            else:
                m_lake = 0.0
                
            ce_land_eff = m_lake * Ce_water + (1.0 - m_lake) * Ce_land
            
            wa_diff = max(0.0, wa_max_val - wa_val)
            evap_pot_water = Ce_water * v_sp * wa_diff
            evap_pot_land = ce_land_eff * v_sp * wa_diff
            
            land_evap = evap_pot_land
            if land_evap > ws_val / dt:
                land_evap = ws_val / dt
                
            Evap[i, j] = m_sec * evap_pot_water + (1.0 - m_sec) * land_evap
            
            # 3. Condensation and Precip
            cond = alpha * max(0.0, wa_val - rH_condensation_threshold * wa_max_val) / dt
            Condensation[i, j] = cond
            
            Precip[i, j] = (Wc[i, j] * removed_fraction) / dt

    return Wa_max, Evap, Condensation, Precip

@njit(parallel=True)
def apply_mass_balance_numba(Ta, Wa, Wc, Ws, Wa_max, Evap, Condensation, Precip, dt, Lv, c_air):
    rows, cols = Ta.shape
    for i in prange(rows):
        for j in range(cols):
            evap = Evap[i, j]
            cond = Condensation[i, j]
            prec = Precip[i, j]
            
            wa = Wa[i, j] + (evap - cond) * dt
            wc = Wc[i, j] + (cond - prec) * dt
            ws = Ws[i, j] + (prec - evap) * dt
            
            if wa < 0.0: wa = 0.0
            if wc < 0.0: wc = 0.0
            if wc < 1e-10: wc = 0.0
            if ws < 0.0: ws = 0.0
            
            wa_m = Wa_max[i, j]
            excess = wa - wa_m
            if excess > 0.0:
                wa -= excess
                wc += excess
                
                excess_cond = excess / dt
                heat_released = (excess_cond / 3600.0) * Lv
                dta_excess = heat_released / c_air
                Ta[i, j] += dta_excess * (dt * 3600.0)
                Condensation[i, j] = cond + excess_cond
                
            Wa[i, j] = wa
            Wc[i, j] = wc
            Ws[i, j] = ws
            
@njit(parallel=True)
def d8_water_routing(surface, M_sea, Ws, max_altitude, dx, dy, dt, slope_exponent, flow_rate):
    """
    Multi-directional D8-style surface water routing (pull approach, parallel-safe).

    Water drains from each cell to all 8 downhill neighbours, weighted by
    slope^slope_exponent.  Each destination cell independently re-derives how much
    water flows into it from each uphill neighbour, so no atomic writes are needed.

    Parameters
    ----------
    surface        : float32 (rows, cols)  — effective surface height [m]
    M_sea          : float32 (rows, cols)  — sea mask (1.0 = ocean, 0.0 = land)
    Ws             : float32 (rows, cols)  — surface water [mm]
    max_altitude   : float                 — physical height of H=1 [m]
    dx, dy         : float                 — cell size [m]
    dt             : float                 — timestep [hr]
    slope_exponent : float                 — steeper slopes receive exponentially more flow
                                             (1 = linear, 2 = quadratic, …)
    flow_rate      : float                 — fraction of water drained per hour at full slope

    Returns
    -------
    Ws_out : float32 (rows, cols)
    """
    rows, cols = surface.shape
    Ws_out = np.empty_like(Ws)

    # 8-connected neighbour offsets (row-delta, col-delta)
    NDI = np.array((-1, -1, -1,  0,  0,  1,  1,  1), dtype=np.int64)
    NDJ = np.array((-1,  0,  1, -1,  1, -1,  0,  1), dtype=np.int64)
    # reverse mapping: index in neighbour that points back to the centre
    REV = np.array((7, 6, 5, 4, 3, 2, 1, 0), dtype=np.int64)

    # Precompute geometric distances for each neighbour index (avoids sqrt in inner loops)
    dx2 = dx * dx
    dy2 = dy * dy
    diag = np.sqrt(dx2 + dy2)
    DIST = np.empty(8, dtype=np.float64)
    # Matches the (NDI,NDJ) ordering above
    DIST[0] = diag
    DIST[1] = dx
    DIST[2] = diag
    DIST[3] = dy
    DIST[4] = dy
    DIST[5] = diag
    DIST[6] = dx
    DIST[7] = diag

    flow_fraction = min(1.0, flow_rate * dt)   # fraction leaving per timestep (stability cap)


    # First pass: compute routing weights to each of the 8 neighbours and totals.
    weights = np.zeros((rows, cols, 8), dtype=np.float32)
    totals = np.zeros((rows, cols), dtype=np.float32)
    outflows = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            if M_sea[i, j] > 0.5:
                totals[i, j] = 0.0
                outflows[i, j] = 0.0
                continue

            h_c = surface[i, j]
            total_w = 0.0

            for k in range(8):
                ni = i + NDI[k]
                nj = j + NDJ[k]
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue

                h_n = surface[ni, nj]
                slope = (h_c - h_n) / DIST[k]
                if slope > 0.0:
                    # Special-case common exponents to avoid slow generic pow
                    if slope_exponent == 2.0:
                        w = slope * slope
                    elif slope_exponent == 1.0:
                        w = slope
                    else:
                        w = slope ** slope_exponent

                    weights[i, j, k] = np.float32(w)
                    total_w += w

            totals[i, j] = np.float32(total_w)
            
            # Compute outgoing water limit to prevent sloshing overshoots.
            # Max loss to a neighbour is limited so the source doesn't drop below the destination.
            outflow_val = Ws[i, j] * flow_fraction
            if total_w > 0.0:
                limit = 1e9
                for k in range(8):
                    w = weights[i, j, k]
                    if w > 0.0:
                        ni = i + NDI[k]
                        nj = j + NDJ[k]
                        h_diff = h_c - surface[ni, nj]
                        # Safe transfer limit (in mm) formula ensures source's new surface >= destination's
                        k_limit = (h_diff * 1000.0) / (1.0 + w / total_w)
                        if k_limit < limit:
                            limit = k_limit
                
                if limit < outflow_val:
                    outflow_val = limit
            else:
                outflow_val = 0.0
                
            outflows[i, j] = np.float32(outflow_val)

    # Second pass: pull inflow from neighbours using precomputed neighbour weights.
    for i in prange(rows):
        for j in range(cols):
            net_inflow = 0.0

            for k in range(8):
                ni = i + NDI[k]
                nj = j + NDJ[k]
                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue

                total_n = totals[ni, nj]
                if total_n <= 0.0:
                    continue

                # weight of neighbour (ni,nj) towards this cell
                rr = REV[k]
                w_to_me = weights[ni, nj, rr]
                if w_to_me <= 0.0:
                    continue

                own_outflow_n = outflows[ni, nj]
                net_inflow += own_outflow_n * (w_to_me / total_n)

            if M_sea[i, j] > 0.5:
                Ws_out[i, j] = np.float32(Ws[i, j] + net_inflow)
            else:
                own_outflow = outflows[i, j]
                Ws_out[i, j] = max(np.float32(0.0), np.float32(Ws[i, j] - own_outflow + net_inflow))

    return Ws_out
