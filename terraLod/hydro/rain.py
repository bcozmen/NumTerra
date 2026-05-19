import numpy as np
from numba import njit
from utils import timeit

from scipy.ndimage import (label as _label,
                            minimum  as _ndi_min,
                            maximum  as _ndi_max,
                            sum      as _ndi_sum,
                            mean     as _ndi_mean)

from .helper import compute_mfd_weights, compute_lake_mask

@timeit
def compute_precipitation_accumulation(
        height_map, precipitation_map, flow_weights, temperature, sea_mask,
        basin_fill,
        infiltration_capacity=0.30,
        land_evap_fraction=0.30,
        lake_open_evap_mm=900.0,
        spill_erosion_depth=0.002,
        max_overflow_iterations=5,
        slope_exp=1.7):
    """
    Full precipitation-driven water budget with lake equilibrium.

    Pipeline
    --------
    1. Land losses      — ``_compute_runoff``
    2. Routing pass 1   — ``_route_runoff`` on original terrain
    3. Basin equilibrium loop (up to ``max_overflow_iterations``):
         a. ``_process_all_basins`` — overflow / partial-fill / dry decision
              per basin, highest spill first.  Overflowing basins erode their
              spill rim and deposit surplus water at the outlet.
         b. ``_route_surplus``     — MFD routing of surplus on the (now eroded)
              terrain.  Returns new ``sink_water`` for the next iteration.
         Iteration stops early when no basin overflows.

    Parameters
    ----------
    height_map            : (R,C) float32 — normalised terrain [0, 1]
    precipitation_map     : (R,C) float32 — mm / year
    flow_weights          : (R,C,8) float32
    temperature           : (R,C) float32 — °C
    sea_mask              : (R,C) bool
    basin_fill            : (R,C) float32 — priority-flood fill heights
    infiltration_capacity : fraction [0,1] of P absorbed by soil
    land_evap_fraction    : max PET fraction [0,1] at 30 °C on land cells
    lake_open_evap_mm     : open-water potential evaporation (mm / year)
    spill_erosion_depth   : normalised height eroded at each overflowing spill point
    max_overflow_iterations : int — cascade depth cap (default 5)

    Returns
    -------
    throughput     : (R,C) float32 — water through each cell (river signal)
    lake_mask      : (R,C) bool
    lake_level     : (R,C) float32 — water-surface height (0 outside lakes)
    height_map_out : (R,C) float32 — terrain with eroded spill points
    """
    height_map     = height_map.astype(np.float32)
    height_map_out = height_map.copy()

    runoff                 = _compute_runoff(precipitation_map.astype(np.float32),
                                             temperature.astype(np.float32),
                                             float(infiltration_capacity),
                                             float(land_evap_fraction))
    throughput, sink_water = _route_runoff(height_map, runoff, flow_weights, sea_mask)

    lake_mask  = np.zeros(height_map.shape, dtype=bool)
    lake_level = np.zeros(height_map.shape, dtype=np.float32)

    # Recompute basin_fill from the (potentially updated) terrain each iteration
    current_basin_fill = basin_fill.copy()

    for _ in range(max_overflow_iterations):
        lake_mask, lake_level, surplus_runoff, any_overflow = _process_all_basins(
            height_map, height_map_out, sink_water, temperature, sea_mask,
            current_basin_fill, lake_open_evap_mm, spill_erosion_depth,
            lake_mask, lake_level,
        )

        # Stop if nothing overflowed AND no surplus to route.
        # Note: route surplus even when any_overflow is False — partial lakes
        # can still generate a small spill that would otherwise be silently lost.
        if not surplus_runoff.any():
            break

        # Route surplus on eroded terrain; new sink_water feeds the next iteration.
        # IMPORTANT: raise already-flooded lake cells to their water-surface level
        # before computing routing weights.  Without this, the MFD weights from
        # an eroded rim cell strongly favour the deep basin interior (steep slope
        # back to the lake floor) over the shallow outflow slope toward the next
        # basin — roughly 95 % of surplus flows back into the full lake and only
        # ~5 % reaches the next basin, killing the river cascade after 1–2 hops.
        # By raising flooded cells to lake_level + ε the basin interior appears
        # *higher* than the eroded rim, so all routing weight points downstream.
        height_for_routing = height_map_out.copy()
        if lake_mask.any():
            height_for_routing[lake_mask] = np.clip(
                lake_level[lake_mask] + 1e-3, 0.0, 1.0
            )
        flow_weights_eroded    = compute_mfd_weights(height_for_routing, slope_exp)
        throughput2, sink_water = _route_surplus(
            height_map_out, surplus_runoff, flow_weights_eroded, sea_mask,
        )
        throughput = throughput + throughput2

        # Recompute fill on eroded terrain so the next pass sees lowered saddles
        _, current_basin_fill = compute_lake_mask(height_map_out, sea_mask)

    return throughput, lake_mask.astype(bool), lake_level.astype(np.float32), height_map_out


# ---------------------------------------------------------------------------
# Basin equilibrium helpers
# ---------------------------------------------------------------------------

def _lake_evaporation_capacity(basin_areas, T_means, lake_open_evap_mm):
    """
    Vectorised: total open-water evaporation capacity for *every* basin.

    Parameters
    ----------
    basin_areas       : (B,) int   — number of cells in each basin
    T_means           : (B,) float — mean temperature of each basin
    lake_open_evap_mm : float

    Returns
    -------
    evap_full : (B,) float64
    """
    t_norm    = np.clip(T_means / 30.0, 0.0, 1.0)
    evap_cell = lake_open_evap_mm * (0.2 + 0.8 * t_norm)
    return evap_cell * basin_areas


@njit(cache=True)
def _erode_all_spill_rims(height_map_out, labeled, overflow_ids,
                           spill_heights, basin_floors,
                           sea_mask_flat, spill_erosion_depth,
                           xdim, ydim):
    """
    Single-pass Numba kernel: find and erode the spill rim for every overflow
    basin in one sweep of the grid, then deposit surplus at rim cells.

    Parameters
    ----------
    labeled        : (N,) int32   flat labeled array (0 = background)
    overflow_ids   : (K,) int32   basin IDs that are overflowing (1-indexed)
    spill_heights  : (K,) float32 spill saddle height per overflow basin
    basin_floors   : (K,) float32 basin floor height per overflow basin

    Returns
    -------
    rim_count : (K,) int32  — number of rim cells eroded per basin
    """
    n       = xdim * ydim
    K       = len(overflow_ids)

    # Build a lookup: basin_id (1-indexed) → index in overflow_ids (or -1)
    max_id  = 0
    for i in range(K):
        if overflow_ids[i] > max_id:
            max_id = overflow_ids[i]

    id_to_k = np.full(max_id + 1, -1, dtype=np.int32)
    for i in range(K):
        id_to_k[overflow_ids[i]] = i

    rim_count = np.zeros(K, dtype=np.int32)

    dx = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    hout_flat = height_map_out.ravel()

    for idx in range(n):
        bid = labeled[idx]
        if bid <= 0 or bid > max_id:
            continue
        k = id_to_k[bid]
        if k < 0:
            continue

        # This cell is inside an overflow basin — check its neighbours for rim
        cx = idx // ydim
        cy = idx - cx * ydim

        for d in range(8):
            nx = cx + dx[d]
            ny = cy + dy[d]
            if nx < 0 or nx >= xdim or ny < 0 or ny >= ydim:
                continue
            nidx = nx * ydim + ny
            # rim cell: outside basin, not sea, at or below spill saddle
            if labeled[nidx] == 0 and not sea_mask_flat[nidx]:
                if hout_flat[nidx] <= spill_heights[k] + 1e-4:
                    floor_h = basin_floors[k]
                    new_h   = hout_flat[nidx] - spill_erosion_depth
                    if new_h < floor_h:
                        new_h = floor_h
                    hout_flat[nidx] = new_h
                    rim_count[k]   += 1

    return rim_count


def _process_all_basins(height_map, height_map_out, sink_water, temperature,
                        sea_mask, basin_fill, lake_open_evap_mm, spill_erosion_depth,
                        lake_mask=None, lake_level=None):
    """
    Run basin equilibrium for every depression, highest spill first.

    All per-basin stats are computed with vectorised ``scipy.ndimage``
    aggregation (one pass per statistic).  Partial-fill assignment is fully
    vectorised.  Rim erosion for overflowing basins is handled by a single
    Numba sweep (``_erode_all_spill_rims``) instead of per-basin
    ``binary_dilation`` calls.

    Returns
    -------
    lake_mask      : (R,C) bool
    lake_level     : (R,C) float32
    surplus_runoff : (R,C) float32
    any_overflow   : bool
    """
    basin_depression = (basin_fill > height_map) & (~sea_mask)

    if lake_mask is None:
        lake_mask  = np.zeros(height_map.shape, dtype=bool)
    if lake_level is None:
        lake_level = np.zeros(height_map.shape, dtype=np.float32)

    surplus_runoff = np.zeros(height_map.shape, dtype=np.float32)
    any_overflow   = False

    if not basin_depression.any():
        return lake_mask, lake_level, surplus_runoff, any_overflow

    labeled, n_basins = _label(basin_depression)
    if n_basins == 0:
        return lake_mask, lake_level, surplus_runoff, any_overflow

    bids = np.arange(1, n_basins + 1, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Bulk per-basin stats — each is a single C-level ndimage pass        #
    # ------------------------------------------------------------------ #
    basin_inflows  = np.asarray(_ndi_sum(sink_water,   labeled, bids), dtype=np.float64)
    basin_floors   = np.asarray(_ndi_min(height_map,   labeled, bids), dtype=np.float64)
    spill_heights  = np.asarray(_ndi_max(basin_fill,   labeled, bids), dtype=np.float64)
    T_means        = np.asarray(_ndi_mean(temperature, labeled, bids), dtype=np.float64)
    # basin_areas: number of cells per basin
    basin_areas    = np.bincount(labeled.ravel())[1:].astype(np.float64)

    # ------------------------------------------------------------------ #
    # "Already settled" test — vectorised via bincount                    #
    # ------------------------------------------------------------------ #
    settled_count  = np.bincount(labeled.ravel(),
                                  weights=lake_mask.ravel().astype(np.float64))[1:]
    settled        = settled_count >= basin_areas   # all cells already in lake

    # ------------------------------------------------------------------ #
    # Evaporation capacity (vectorised)                                   #
    # ------------------------------------------------------------------ #
    evap_full = _lake_evaporation_capacity(basin_areas, T_means, lake_open_evap_mm)

    valid    = (~settled) & (spill_heights > basin_floors) & (basin_inflows > 0.0)
    overflow = valid & (basin_inflows >= evap_full)
    partial  = valid & (basin_inflows <  evap_full)

    # ------------------------------------------------------------------ #
    # Partial fill — fully vectorised, no per-basin Python loop           #
    # ------------------------------------------------------------------ #
    if partial.any():
        pids       = bids[partial]            # 1-indexed
        fill_frac  = basin_inflows[partial] / evap_full[partial]
        lvls       = (basin_floors[partial]
                      + fill_frac * (spill_heights[partial] - basin_floors[partial]))
        lvls_f32   = lvls.astype(np.float32)

        # Build a per-cell target-level map in one vectorised pass
        # labeled[i] == bid  →  target_level[i] = lvls[bid_index]
        target_level_1d = np.zeros(n_basins + 1, dtype=np.float32)
        target_level_1d[pids] = lvls_f32
        level_map = target_level_1d[labeled]          # (R,C), 0 outside partial basins

        partial_union = np.isin(labeled, pids)
        active        = partial_union & (height_map <= level_map)
        lake_mask[active]  = True
        lake_level[active] = level_map[active]

    # ------------------------------------------------------------------ #
    # Overflow — one Numba sweep erodes all rims simultaneously           #
    # ------------------------------------------------------------------ #
    if overflow.any():
        any_overflow  = True
        oids          = bids[overflow].astype(np.int32)
        surpluses     = basin_inflows[overflow] - evap_full[overflow]
        o_spill_h     = spill_heights[overflow].astype(np.float32)
        o_floor_h     = basin_floors[overflow].astype(np.float32)

        xdim, ydim = height_map.shape
        rim_count = _erode_all_spill_rims(
            height_map_out,
            labeled.ravel().astype(np.int32),
            oids, o_spill_h, o_floor_h,
            sea_mask.ravel(),
            float(spill_erosion_depth),
            xdim, ydim,
        )

        # Mark overflow basins as fully flooded
        overflow_union = np.isin(labeled, oids)
        lake_mask[overflow_union] = True

        spill_level_1d        = np.zeros(n_basins + 1, dtype=np.float32)
        spill_level_1d[oids]  = o_spill_h
        spill_level_map       = spill_level_1d[labeled]
        lake_level[overflow_union] = spill_level_map[overflow_union]

        # Surplus at outlet: concentrate at the single lowest rim cell so that
        # the outflow forms one strong river rather than many weak trickles.
        for i, (bid, surplus, rc) in enumerate(zip(oids, surpluses, rim_count)):
            if surplus <= 0.0:
                continue
            if rc > 0:
                # Get the eroded rim mask for this basin
                outlet = _get_basin_rim_mask(height_map_out, labeled, bid,
                                              o_spill_h[i], sea_mask, xdim, ydim)
                if outlet.any():
                    # Deposit ALL surplus at the single lowest rim cell — this
                    # is the true spill point and seeds one coherent outlet river.
                    outlet_heights = np.where(outlet, height_map_out, np.inf)
                    lowest = np.unravel_index(np.argmin(outlet_heights), height_map_out.shape)
                    surplus_runoff[lowest] += float(surplus)
                    continue
            # Fallback (no identifiable rim): spread over the whole basin
            outlet = labeled == bid
            n_outlet = max(int(outlet.sum()), 1)
            surplus_runoff[outlet] += float(surplus) / n_outlet

    return lake_mask, lake_level, surplus_runoff, any_overflow


def _get_basin_rim_mask(height_map_out, labeled, basin_id,
                         spill_h, sea_mask, xdim, ydim):
    """
    Return a bool mask of the eroded rim cells for *one* basin.
    Uses numpy roll-based dilation — no scipy, no per-cell Python loop.
    """
    basin = labeled == basin_id
    # 8-connected dilation via roll
    dilated = basin.copy()
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        dilated |= np.roll(np.roll(basin, dx, axis=0), dy, axis=1)
    rim = dilated & ~basin & ~sea_mask & (height_map_out <= spill_h + 1e-4)
    return rim


def _route_surplus(height_map_out, surplus_runoff, flow_weights_eroded, sea_mask):
    """
    Route surplus overflow water on the current (eroded) terrain.

    Returns the throughput contribution from this pass and the new sink_water
    that feeds the next basin-equilibrium iteration.

    Returns
    -------
    throughput2 : (R,C) float32
    sink_water2 : (R,C) float32
    """
    return _route_runoff(height_map_out, surplus_runoff, flow_weights_eroded, sea_mask)


# ---------------------------------------------------------------------------
# Low-level Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _compute_runoff(precipitation_map, temperature, infiltration_capacity, land_evap_fraction):
    """
    Per-cell land losses: infiltration + temperature-scaled evapotranspiration.
    """
    xdim, ydim = precipitation_map.shape
    runoff = np.empty((xdim, ydim), dtype=np.float32)

    for x in range(xdim):
        for y in range(ydim):
            p = precipitation_map[x, y]

            p -= p * infiltration_capacity

            t_norm = temperature[x, y] / 30.0
            if t_norm < 0.0:
                t_norm = 0.0
            elif t_norm > 1.0:
                t_norm = 1.0
            p -= p * land_evap_fraction * t_norm

            runoff[x, y] = p if p > 0.0 else 0.0

    return runoff


@njit(cache=True)
def _route_runoff(height_map, runoff, flow_weights, sea_mask):
    """
    MFD routing: distribute runoff downslope, recording throughput per cell.

    Cells are processed highest-first.  Each cell records its total incoming
    water as ``throughput`` (river signal) *before* distributing it.
    Local minima with no downhill neighbours accumulate water in ``sink_water``
    (lake inflow signal).  Sea cells absorb all incoming water.

    Returns
    -------
    throughput : (R,C) float32 — cumulative water through each cell
    sink_water : (R,C) float32 — water pooled at local minima
    """
    xdim, ydim = height_map.shape
    n = xdim * ydim

    water      = runoff.ravel().astype(np.float32).copy()
    throughput = np.zeros(n, dtype=np.float32)
    sink_water = np.zeros(n, dtype=np.float32)

    dx = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.int32)
    dy = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.int32)

    order = np.argsort(-height_map.ravel())

    for i in range(n):
        idx = order[i]
        cx  = idx // ydim
        cy  = idx - cx * ydim
        w   = water[idx]

        throughput[idx] += w        # record before routing

        if sea_mask[cx, cy] or w <= 0.0:
            continue

        has_outflow = False
        for k in range(8):
            if flow_weights[cx, cy, k] > 0.0:
                has_outflow = True
                break

        if not has_outflow:
            sink_water[idx] = w
            continue

        for k in range(8):
            fw = flow_weights[cx, cy, k]
            if fw > 0.0:
                nx = cx + dx[k]
                ny = cy + dy[k]
                if 0 <= nx < xdim and 0 <= ny < ydim:
                    water[nx * ydim + ny] += w * fw

        water[idx] = 0.0

    return throughput.reshape(xdim, ydim), sink_water.reshape(xdim, ydim)



