import numpy as np

def scale_erosion_params(world_params):
        """Return scaled *copies* of the erosion param dicts.

        The originals in world_params are never mutated, so calling erode()
        multiple times or changing resolution always starts from the same
        calibrated base values.

        Rate scaling rationale
        ----------------------
        Slopes in the simulation are Δh_normalised per 1 cell, which equals
        tan(θ) × (dx_m / dz_m).  Erosion per metre of travel is:

            erode/metre = rate × slope / dx_m
                        = rate × tan(θ) × (dx_m / dz_m) / dx_m
                        = rate × tan(θ) / dz_m   ← dx_m cancels ✓

        So the base rates are used directly without resolution scaling.
        Scaling them by dx_m would reintroduce a dx_m factor and break
        resolution invariance.

        Thermal reach scaling
        ---------------------
        Each thermal iteration spreads material at most 1 cell.
        Physical reach = iterations × dx_m  →  iterations = reach_m / dx_m.
        """
        wp      = world_params
        dx_m    = wp['max_size'] / (wp['shape'][0] - 1)
        dz_m    = wp['max_altitude']

        def slope_from_angle(deg):
            return np.tan(np.radians(deg)) * (dx_m / dz_m)

        # --- Hydraulic ---
        h_src    = wp['erosion_params']['hydraulic']
        h_scaled = dict(h_src)   # shallow copy — originals untouched
        # Do NOT scale rates by dx_m: slope already scales as (dx_m/dz_m), so
        # erode/metre = rate × slope / dx_m = rate × tan(θ)/dz_m — resolution-invariant.
        # Scaling rates by dx_m would re-introduce a dx_m factor and break invariance.
        h_scaled['erosion_rate']    = h_src['erosion_rate_base']
        h_scaled['deposition_rate'] = h_src['deposition_rate_base']
        h_scaled['min_slope']       = slope_from_angle(h_src['min_slope_angle_deg'])
        h_scaled['max_steps']       = max(200, int(h_src['max_travel_m'] / dx_m))

        # --- Thermal ---
        t_src    = wp['erosion_params']['thermal']
        t_scaled = dict(t_src)   # shallow copy
        t_scaled['talus']      = slope_from_angle(t_src['talus_angle_deg'])
        t_scaled['iterations'] = max(1, int(t_src['thermal_reach_m'] / dx_m))

        # --- Aeolian (air) ---
        a_src    = wp['erosion_params']['air']
        a_scaled = dict(a_src)   # shallow copy
        # saltation hop: physical metres → grid cells (minimum 1)
        a_scaled['saltation_cells'] = max(1, int(a_src['saltation_base_m'] / dx_m))
        # slip-face talus: degrees → normalised slope units (same as thermal)
        a_scaled['avalanche_talus'] = slope_from_angle(a_src['avalanche_talus_deg'])
        a_scaled['iterations']      = a_src['air_iterations']


        return h_scaled, t_scaled, a_scaled


def scale_hydro_params(world_params):
    """Return a *copy* of the hydro param dict with any necessary scaling.

    The original in world_params is never mutated, so calling init_hydro()
    multiple times or changing resolution always starts from the same
    calibrated base values.
    """
    wp = world_params
    h_src = wp['hydrology_params']
    h_scaled = dict(h_src)   # shallow copy — original untouched

    # No resolution scaling needed for hydro params, as they are based on flow
    # accumulation and height percentiles, which are inherently resolution-invariant.

    dx_m = wp['max_size'] / (wp['shape'][0] - 1)
    dy_m = wp['max_size'] / (wp['shape'][1] - 1)
    #km2 to cells: physical area → grid cells (minimum 1 cell)
    h_scaled['init_lake_area_threshold'] = max(1, int(h_src['init_lake_area_threshold'] * 1e6 / (dx_m * dy_m)))
    return h_scaled


def normalize(arr, axis = None,vmin = None, vmax = None, range = (0, 1)):
    if vmin is None:
        vmin = np.min(arr, axis=axis, keepdims=True)
    if vmax is None:
        vmax = np.max(arr, axis=axis, keepdims=True)
    return range[0] + (arr - vmin) * (range[1] - range[0]) / (vmax - vmin + 1e-8)

def get_grid(lim = (0, 1, 0, 1), shape = (2048, 2048)):
    x = np.linspace(lim[0], lim[1], shape[0])
    y = np.linspace(lim[2], lim[3], shape[1])
    return np.meshgrid(x, y, indexing='ij')