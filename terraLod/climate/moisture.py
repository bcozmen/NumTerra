"""
Moisture transport — physically-based moisture advection and source building.

Algorithm: iterative upwind sweep (inside Numba)
=================================================
Each iteration visits every non-source cell once, in upwind-first order.
This pass is **conservative**: orographic precipitation is extracted from
the moisture budget at each step, so moisture is actually *removed* as air
rises over terrain, creating genuine rain shadows.

Key design choices
------------------
* **Conservative transport** — when air rises, a fraction of its moisture
  condenses and is removed as orographic precipitation.  The remaining
  moisture continues downwind.  This directly produces wet windward slopes
  and dry leeward rain shadows without any artificial blocking factor.

* **Multi-sample advection** — each cell blends two upstream samples
  (1-step and 2-step back), introducing transport inertia and sharper
  moisture fronts rather than pure diffusion.

* **Integrated orographic uplift** — the altitude gain over the *last two*
  upstream steps is accumulated, giving a more realistic estimate of total
  lifting and suppressing noise from single-cell spikes.

* **Wind-speed modulation** — a terrain-slope-based transport efficiency
  factor is applied per cell: ``eff = exp(-beta * |slope|)``.  Steep slopes
  slow moisture transport; gentle terrain allows free flow.

* **Upwind-first order** — cells are sorted by their projection onto the
  mean wind axis, achieving full-domain propagation in one pass for uniform
  winds and converging in a handful of passes for rugged terrain.

* **Loop inside Numba** — n_iters iterations run entirely inside a single
  @njit kernel; there is zero Python overhead per pass.

Physical units
--------------
* Exponential decay anchored to a physical half-life in km → independent of
  grid resolution and map size.
* Orographic blocking uses altitude in metres → independent of max_altitude.
* Wind-speed efficiency uses dimensionless slope [m/m] → resolution-independent.
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
    * Lake  cells → area-scaled per connected component, clamped to [0.1, 1.0].
                    (area_scale=50 → a lake covering 2 % of the map ≈ 1.0)
    * River cells → ``river_strength`` × normalised river accumulation (optional).
                    Rivers add a weaker evaporation bias without overriding lakes/sea.
    """
    shape       = sea_mask.shape
    total_cells = shape[0] * shape[1]
    area_scale  = 50.0

    out = np.zeros(shape, dtype=np.float32)
    out[sea_mask] = 1.0

    # Area-scale each lake component so large lakes emit more moisture than tiny ones.
    labeled, n_lakes = _scipy_label(lake_mask)
    for k in range(1, n_lakes + 1):
        mask_k   = labeled == k
        fraction = float(mask_k.sum()) / total_cells
        # sqrt gives smooth monotonic scaling; area_scale=50 → 2% map area → 1.0
        strength = float(np.sqrt(fraction * area_scale))
        out[mask_k] = np.float32(np.clip(strength, 0.1, 1.0))

    # River evaporation: rivers add a soft moisture floor proportional to
    # their normalised flow accumulation, capped to river_strength.
    # Only applied where not already a stronger source (sea / lake).
    if river_map is not None:
        river_contrib = (river_map * river_strength).astype(np.float32)
        water_source  = sea_mask | lake_mask
        out = np.where(water_source, out, np.maximum(out, river_contrib)).astype(np.float32)

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
    wind field and return (moisture, orog_precip_proxy).

    Parameters
    ----------
    sources       : (R,C) float32 — evaporation strength [0..1]
    source_mask   : (R,C) bool    — cells whose value is fixed (sea / lake)
    lake_mask     : (R,C) bool    — lake cells (passed through to the kernel)
    height_m      : (R,C) float32 — altitude above sea level in metres
    wy_field      : (R,C) float32 — local wind y-component (unit vectors)
    wx_field      : (R,C) float32 — local wind x-component (unit vectors)
    pixel_size_m  : float — physical size of one grid pixel in metres
    wetness       : float [0,1] — global moisture scalar
    orog_k_per_km : float — orographic condensation rate [km⁻¹].
                    Higher → more rain per km of uplift, sharper rain shadows.
    slope_beta    : float — slope-based wind efficiency decay coefficient.
                    0 = no slowdown; 0.5 = slope of 2 m/m → ~37 % efficiency.
    n_iters       : int   — maximum sweep repetitions.
    tol           : float — early-exit tolerance; sweeps stop when the maximum
                    per-cell moisture change falls below this value.
    slope_mag     : (R,C) float32 (optional) — pre-computed terrain slope
                    magnitude [m/m].  Pass this to avoid an extra np.gradient
                    call when the caller already has it (e.g. Climate._slope_mag).
    order         : (R*C,) int64 (optional) — pre-computed upwind-first cell
                    ordering.  Pass this to skip the argsort when the wind
                    field has not changed since the last call.
    lake_floor    : (R,C) float32 (optional) — if supplied, lake-basin cells act
                    as *soft floors*: advection can still exceed the floor value,
                    but each lake cell retains at least this moisture regardless
                    of what the upwind path delivers.  Useful during init_run to
                    keep basin locations as a weak moisture bias without hard-
                    pinning them like sea cells.
    plains_halflife_mult : float — multiplier applied to the moisture halflife on
                    completely flat terrain (slope ≈ 0).  Values > 1 let moisture
                    travel further inland over plains.  E.g. 3.0 → 3× longer
                    halflife on flat land, smoothly decaying back to 1× on steep
                    slopes.  Default 3.0.
    plains_flat_slope    : float — slope threshold [m/m] that defines "flat":
                    cells with slope < this value receive the full
                    ``plains_halflife_mult`` boost; cells steeper than ~2×
                    this value are unaffected.  Default 0.01 (1 % grade).

    Returns
    -------
    moisture    : (R,C) float32 — atmospheric moisture after transport [0..1]
    orog_precip : (R,C) float32 — orographic rain-out proxy [0..1, unnormalised]
    """
    rows, cols = sources.shape

    # Terrain slope magnitude [m/m] — only computed when not supplied by caller.
    if slope_mag is None:
        grad_y, grad_x = np.gradient(height_m, pixel_size_m)
        slope_mag = np.sqrt(grad_y ** 2 + grad_x ** 2).astype(np.float32)

    moisture    = sources.copy()
    orog_precip = np.zeros((rows, cols), dtype=np.float32)

    # Per-pixel decay anchored to a physical half-life: 50 km (arid) … 500 km (wet)
    halflife_km  = 50.0 + wetness * 450.0
    halflife_pix = halflife_km * 1_000.0 / pixel_size_m
    step_decay   = float(0.5 ** (1.0 / halflife_pix))

    # Plains halflife boost: on flat land the effective halflife is multiplied
    # by `plains_halflife_mult`, smoothly falling back to 1.0 on steep terrain.
    # plains_factor[i,j] ∈ [1, plains_halflife_mult]
    # per-cell effective decay = step_decay ** (1 / plains_factor[i,j])
    # which is slower (less decay) where plains_factor > 1.
    if plains_halflife_mult > 1.0:
        plains_factor = 1.0 + (plains_halflife_mult - 1.0) * np.exp(
            -slope_mag / plains_flat_slope
        )
        plains_factor = plains_factor.astype(np.float32)
    else:
        plains_factor = np.ones(sources.shape, dtype=np.float32)

    # Upwind-first cell ordering based on mean wind direction.
    if order is None:
        wy_mean = float(wy_field.mean())
        wx_mean = float(wx_field.mean())
        ii      = np.arange(rows, dtype=np.float32)[:, None]
        jj      = np.arange(cols, dtype=np.float32)[None, :]
        proj    = ii * wy_mean + jj * wx_mean
        order   = np.argsort(proj.ravel()).astype(np.int64)

    _advect_kernel(
        moisture, orog_precip,
        height_m, slope_mag,
        source_mask.astype(np.bool_), lake_mask.astype(np.bool_), sources,
        wy_field, wx_field,
        step_decay, float(orog_k_per_km), float(slope_beta),
        order, n_iters, float(tol),
        lake_mask.astype(np.bool_) if lake_floor is not None else None,
        lake_floor,
        plains_factor,
    )

    # Lateral diffusion: spread moisture perpendicular to the wind axis so that
    # advection streaks (the "45-degree line" artifact from a pure directional
    # sweep) are filled in.  Physical basis: turbulent mixing and sub-grid
    # convective transport diffuse moisture across ~20-40 km.
    # We diffuse after the kernel so sources stay sharp; sigma = 25 km.
    sigma_px = moisture_diffusion_sigma_km * 1_000.0 / pixel_size_m
    # Cap sigma so it doesn't become huge on tiny maps
    sigma_px = min(sigma_px, max(rows, cols) * 0.04)
    moisture    = gaussian_filter(moisture,    sigma=sigma_px).astype(np.float32)
    orog_precip = gaussian_filter(orog_precip, sigma=sigma_px * 0.5).astype(np.float32)

    # Re-pin source cells to their fixed values after diffusion
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

