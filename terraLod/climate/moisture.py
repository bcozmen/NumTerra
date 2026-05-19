"""
Moisture transport — physically-based moisture advection and source building.

Algorithm: iterative upwind sweep (Numba)
=========================================
Each sweep visits cells in upwind-first order.  Transport is **conservative**:
orographic precipitation is extracted from the moisture budget as air rises,
producing genuine rain shadows.  Multi-sample advection (1- and 2-step
upstream) adds transport inertia.  Loops exit early once max change < tol.

Physical units
--------------
* Exponential decay anchored to a physical half-life in km.
* Orographic blocking uses altitude in metres.
* Wind-efficiency uses dimensionless slope [m/m].
All are resolution-independent.
"""

import numpy as np
import numba
from scipy.ndimage import label as _scipy_label, gaussian_filter
from utils import timeit

# ---------------------------------------------------------------------------
# Moisture source map
# ---------------------------------------------------------------------------

def build_moisture_sources(sea_mask: np.ndarray,
                            lake_mask: np.ndarray,
                            river_map: np.ndarray | None = None,
                            
                            river_strength: float = 0.12,
                            ) -> np.ndarray:
    """
    Return a float32 source-strength map.

    * Sea   cells → 1.0
    * Lake  cells → area-scaled per connected component, clamped to [0.1, 1.0]
                    (area_scale=50 → a lake covering 2 % of the map ≈ 1.0).
    * River cells → ``river_strength`` × normalised accumulation (optional).
    """
    total_cells = sea_mask.size
    area_scale  = 10.0

    out = np.zeros(sea_mask.shape, dtype=np.float32)
    out[sea_mask] = 1.0

    labeled, n_lakes = _scipy_label(lake_mask)
    if n_lakes:
        counts    = np.bincount(labeled.ravel(), minlength=n_lakes + 1)
        fracs     = counts[1:].astype(np.float64) / total_cells
        strengths = np.clip(np.sqrt(fracs * area_scale), 0.1, 1.0).astype(np.float32)
        lut       = np.zeros(n_lakes + 1, dtype=np.float32)
        lut[1:]   = strengths
        out       = np.where(lake_mask, lut[labeled], out).astype(np.float32)

    if river_map is not None:
        river_contrib = (river_map * river_strength).astype(np.float32)
        out = np.where(sea_mask | lake_mask, out,
                       np.maximum(out, river_contrib)).astype(np.float32)

    return out


@timeit
def advect_moisture(sources:       np.ndarray,
                    source_mask:   np.ndarray,
                    lake_mask:     np.ndarray,
                    height_m:      np.ndarray,
                    wy_field:      np.ndarray,
                    wx_field:      np.ndarray,
                    pixel_size_m:  float,
                    wetness:       float,
                    orog_k_per_km: float = 0.05,
                    slope_beta:    float = 0.025,
                    n_iters:       int   = 20,
                    tol:           float = 1e-4,
                    moisture_diffusion_sigma_km: float = 25.0,
                    slope_mag:     np.ndarray | None = None,
                    order:         np.ndarray | None = None,
                    lake_floor:    np.ndarray | None = None,
                    plains_halflife_mult: float = 3.0,
                    plains_flat_slope:   float = 0.01,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Advect moisture from evaporation sources along the terrain-deflected
    wind field and return ``(moisture, orog_precip_proxy)``.

    Orographic lifting extracts moisture conservatively at each cell,
    directly producing rain shadows.  Multi-sample advection (1-step +
    2-step upstream) adds transport inertia.  Early exit when the maximum
    per-cell change drops below ``tol``.

    Parameters
    ----------
    sources       : (R,C) float32 — evaporation strength [0..1]
    source_mask   : (R,C) bool    — cells whose value is fixed (sea)
    lake_mask     : (R,C) bool    — lake cells
    height_m      : (R,C) float32 — altitude above sea level [m]
    wy_field      : (R,C) float32 — local wind y-component (unit vectors)
    wx_field      : (R,C) float32 — local wind x-component (unit vectors)
    pixel_size_m  : float — physical pixel size [m]
    wetness       : float [0,1] — global moisture scalar
    orog_k_per_km : orographic condensation rate [km⁻¹]
    slope_beta    : slope-based wind-efficiency decay coefficient
    n_iters       : maximum sweep repetitions
    tol           : early-exit tolerance on max per-cell change
    slope_mag     : pre-computed terrain slope magnitude [m/m] (avoids recompute)
    order         : pre-computed upwind-first cell ordering (avoids re-argsort)
    lake_floor    : soft-floor moisture for lake cells; advection can exceed it
    plains_halflife_mult : halflife multiplier on flat terrain (> 1 = more inland reach)
    plains_flat_slope    : slope threshold [m/m] defining "flat" (~1 % grade)

    Returns
    -------
    moisture    : (R,C) float32 — atmospheric moisture [0..1]
    orog_precip : (R,C) float32 — orographic rain-out proxy [unnormalised]
    """
    rows, cols = sources.shape

    if slope_mag is None:
        grad_y, grad_x = np.gradient(height_m, pixel_size_m)
        slope_mag = np.sqrt(grad_y**2 + grad_x**2).astype(np.float32)

    moisture    = sources.copy()
    orog_precip = np.zeros((rows, cols), dtype=np.float32)

    # Per-pixel decay: half-life ranges from 50 km (arid) to 500 km (wet)
    halflife_pix = (50.0 + wetness * 450.0) * 1_000.0 / pixel_size_m
    step_decay   = float(0.5 ** (1.0 / halflife_pix))

    # Plains boost: flat cells get a longer effective half-life
    plains_factor = (
        (1.0 + (plains_halflife_mult - 1.0) * np.exp(-slope_mag / plains_flat_slope))
        .astype(np.float32)
        if plains_halflife_mult > 1.0
        else np.ones(sources.shape, dtype=np.float32)
    )

    if order is None:
        ii    = np.arange(rows, dtype=np.float32)[:, None]
        jj    = np.arange(cols, dtype=np.float32)[None, :]
        order = np.argsort(
            (ii * float(wy_field.mean()) + jj * float(wx_field.mean())).ravel()
        ).astype(np.int64)

    _advect_kernel(
        moisture, orog_precip,
        height_m, slope_mag,
        source_mask, lake_mask, sources,
        wy_field, wx_field,
        step_decay, float(orog_k_per_km), float(slope_beta),
        order, n_iters, float(tol),
        lake_mask if lake_floor is not None else None,
        lake_floor,
        plains_factor,
    )

    sigma_px    = min(moisture_diffusion_sigma_km * 1_000.0 / pixel_size_m,
                      max(rows, cols) * 0.04)
    moisture    = gaussian_filter(moisture,    sigma=sigma_px).astype(np.float32)
    orog_precip = gaussian_filter(orog_precip, sigma=sigma_px * 0.5).astype(np.float32)
    moisture[source_mask] = sources[source_mask]

    return moisture, orog_precip


# ---------------------------------------------------------------------------
# Numba kernel helpers
# ---------------------------------------------------------------------------

@numba.njit(inline='always', cache=True)
def _bilinear(arr: np.ndarray, fi: float, fj: float,
              rows: int, cols: int) -> float:
    """Bilinear sample of *arr* at floating-point index (fi, fj), clamped."""
    # Use floor (not truncation) so negative coords work correctly
    i0 = min(max(int(np.floor(fi)),     0), rows - 1)
    i1 = min(max(int(np.floor(fi)) + 1, 0), rows - 1)
    j0 = min(max(int(np.floor(fj)),     0), cols - 1)
    j1 = min(max(int(np.floor(fj)) + 1, 0), cols - 1)
    # fractional parts are always in [0, 1) because we used floor
    di = min(max(fi - np.floor(fi), 0.0), 1.0)
    dj = min(max(fj - np.floor(fj), 0.0), 1.0)
    return (arr[i0, j0] * (1.0 - di) * (1.0 - dj)
          + arr[i1, j0] *        di  * (1.0 - dj)
          + arr[i0, j1] * (1.0 - di) *        dj
          + arr[i1, j1] *        di  *        dj)


@numba.njit(cache=True)
def _advect_kernel(moisture:        np.ndarray,   # (R,C) float32, modified in-place
                   orog_precip:     np.ndarray,   # (R,C) float32, modified in-place
                   height_m:        np.ndarray,   # (R,C) float32
                   slope_mag:       np.ndarray,   # (R,C) float32  [m/m, dimensionless]
                   source_mask:     np.ndarray,   # (R,C) bool
                   lake_mask:       np.ndarray,   # (R,C) bool
                   sources:         np.ndarray,   # (R,C) float32
                   wy_field:        np.ndarray,   # (R,C) float32
                   wx_field:        np.ndarray,   # (R,C) float32
                   step_decay:      float,
                   orog_k_per_km:   float,
                   slope_beta:      float,
                   order:           np.ndarray,   # (R*C,) int64 upwind-first
                   n_iters:         int,
                   tol:             float = 1e-4,
                   lake_floor_mask: np.ndarray | None = None,  # (R,C) bool  — soft-floor cells
                   lake_floor_vals: np.ndarray | None = None,  # (R,C) float32 — floor strengths
                   plains_factor:   np.ndarray | None = None,  # (R,C) float32 — local halflife multiplier
                   ) -> None:
    """
    Run up to `n_iters` upwind-sorted sweeps with conservative moisture transport.
    Exits early when the maximum per-cell change across a full sweep drops below
    `tol` (default 1e-4), typically saving 50–70 % of iterations on most terrains.

    For each non-source cell:
      1. Sample moisture at 1-step and 2-step upwind (multi-sample advection).
      2. Compute total orographic lifting over those two upstream steps.
      3. Remove condensed moisture conservatively (actual budget reduction).
      4. Apply wind-speed efficiency based on local terrain slope.
      5. Apply per-pixel exponential decay (halflife anchored in km).
      6. Max-accumulate so cells fed by multiple paths keep the best value.
         For ``lake_floor_mask`` cells the moisture is additionally clamped to
         at least ``lake_floor_vals[i,j]``, making them a *soft floor* —
         advection can still exceed the floor, but the basin always contributes
         a minimum moisture bias.

    This directly generates:
      - Wet windward slopes (moisture peaks before the crest)
      - Dry rain shadows    (moisture has been rained out by the time air descends)
      - Gradual inland drying via the decay term
      - Sharper moisture fronts via multi-sample inertia
    """
    rows = moisture.shape[0]
    cols = moisture.shape[1]
    n    = len(order)

    for _ in range(n_iters):
        max_change = 0.0

        for idx in range(n):
            flat = order[idx]
            i    = flat // cols
            j    = flat  % cols

            # Sources keep their fixed evaporation strength
            if source_mask[i, j]:
                moisture[i, j] = sources[i, j]
                continue

            wy = wy_field[i, j]
            wx = wx_field[i, j]

            # --- Multi-sample advection (1-step + 2-step upstream) -----------
            fi1 = float(i) - wy
            fj1 = float(j) - wx
            fi2 = float(i) - 2.0 * wy
            fj2 = float(j) - 2.0 * wx

            m1 = _bilinear(moisture, fi1, fj1, rows, cols)
            m2 = _bilinear(moisture, fi2, fj2, rows, cols)
            h1 = _bilinear(height_m, fi1, fj1, rows, cols)
            h2 = _bilinear(height_m, fi2, fj2, rows, cols)

            # Blend: 65 % near-upstream, 35 % far-upstream — transport inertia
            m_up = 0.65 * m1 + 0.35 * m2

            # --- Integrated orographic uplift over last two upstream steps ---
            # dh > 0 means air is rising (windward); only rising air precipitates
            dh1_km = max(0.0, height_m[i, j] - h1) / 1_000.0
            dh2_km = max(0.0, h1 - h2)             / 1_000.0
            # Recent lift weighted fully; penultimate step at half-weight
            total_dh_km = dh1_km + 0.5 * dh2_km

            # --- Conservative precipitation extraction -----------------------
            # Fraction of carried moisture that condenses and falls as rain
            orog_loss_frac = 1.0 - np.exp(-orog_k_per_km * total_dh_km)
            orog_extracted = m_up * orog_loss_frac
            m_after_rain   = m_up - orog_extracted   # conserved budget

            # Accumulate orographic precip deposit at this cell
            if orog_extracted > orog_precip[i, j]:
                orog_precip[i, j] = orog_extracted

            # --- Wind-speed efficiency (steep slopes retard transport) -------
            eff = np.exp(-slope_beta * slope_mag[i, j])

            # Plains halflife boost: step_decay^(1/plains_factor) is a slower
            # (less aggressive) decay on flat terrain.
            if plains_factor is not None:
                pf = plains_factor[i, j]
                local_decay = step_decay ** (1.0 / pf) if pf > 1.0 else step_decay
            else:
                local_decay = step_decay

            candidate = local_decay * eff * m_after_rain

            # Soft floor for lake-basin cells: advection can exceed the floor,
            # but basins always retain at least their seed moisture value.
            if lake_floor_mask is not None and lake_floor_mask[i, j]:
                floor_val = lake_floor_vals[i, j]   # type: ignore[index]
                if candidate < floor_val:
                    candidate = floor_val

            # Max-accumulate: moisture can only grow across iterations
            delta = candidate - moisture[i, j]
            if delta > 0.0:
                moisture[i, j] = candidate
                if delta > max_change:
                    max_change = delta

        # Early exit: converged when no cell changed by more than tol
        if max_change < tol:
            break

