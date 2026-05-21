import numpy as np
from numba import njit, prange

def air_erosion(grid, iterations, saltation_cells, wind_strength, threshold,
                erosion_rate, deposition_rate, avalanche_talus, wind_x, wind_y,
                **kwargs):
    return air_erosion_numba(grid, int(iterations), wind_x, wind_y,
                             wind_strength, threshold, int(saltation_cells),
                             erosion_rate, deposition_rate, avalanche_talus)


@njit(cache=True, parallel=True)
def air_erosion_numba(
    grid,
    iterations=8,
    wind_x=1.0,
    wind_y=0.3,
    wind_strength=0.06,
    threshold=0.008,
    saltation_cells=4,
    erosion_rate=0.012,
    deposition_rate=1.0,
    avalanche_talus=0.05,
):
    """Aeolian erosion via saltation + slip-face avalanching.

    Physics
    -------
    Shear stress on a cell is proportional to its *windward slope* — how much
    it rises above its upwind neighbour.  Windward faces (positive slope) are
    eroded; lee faces have zero windward slope so they are naturally protected
    without a separate shadow parameter.

    Eroded grains hop exactly *saltation_cells* downwind and are deposited
    there (``deposition_rate = 1.0`` → mass exactly conserved).

    A slip-face avalanche pass after each saltation sweep enforces the maximum
    stable slope *avalanche_talus*, which causes material to pile up into
    dune-like asymmetric ridges.

    Parallelism
    -----------
    Saltation pass — sweep the *wind axis* sequentially (upwind → downwind)
    so that each row's deposits land in a row that has not yet been processed,
    eliminating all RAW hazards.  The *perpendicular* axis is independent
    within a row and runs in parallel via ``prange``.

    * ``di != 0``: outer loop over rows (sequential), inner over columns
      (``prange``).  Deposits from row *i* land in row *i + di* which is
      processed later → race-free.
    * ``di == 0`` (pure crosswind): roles swap — outer loop over columns
      (sequential), inner over rows (``prange``).

    Avalanche pass — 9-colour (3×3 tile) decomposition identical to the one
    used in ``thermal_erosion_numba``: same-colour cells are ≥ 3 apart in
    every dimension so their 8-neighbourhoods never overlap → zero races.

    Parameters
    ----------
    grid             : normalised [0, 1] height map.
    iterations       : number of full wind passes (each = saltation + avalanche).
    wind_x / wind_y  : wind direction vector (need not be unit length).
    wind_strength    : shear coefficient — multiplied by windward slope.
    threshold        : minimum shear required to mobilise grains.
    saltation_cells  : hop length in grid cells (derive from physical metres
                       via ``saltation_base_m / dx_m`` before calling).
    erosion_rate     : fraction of available excess shear converted to volume.
    deposition_rate  : fraction of eroded volume deposited at landing
                       (1.0 → exact mass conservation).
    avalanche_talus  : maximum stable slope in normalised height units;
                       derive from angle via ``tan(θ) × dx_m / dz_m``.
    """
    h, w = grid.shape
    out = grid.copy()
    SQRT2 = 1.41421356

    # Normalise wind direction
    wl = (wind_x * wind_x + wind_y * wind_y) ** 0.5 + 1e-12
    wx = wind_x / wl
    wy = wind_y / wl

    # Integer saltation hop (downwind)
    di = int(round(wx * saltation_cells))
    dj = int(round(wy * saltation_cells))
    if di == 0 and dj == 0:
        di = 1  # guarantee at least 1-cell transport

    # Upwind look-back offset (one hop upstream, for windward slope)
    ui = -di
    uj = -dj

    for _ in range(iterations):

        # ---------------------------------------------------------------
        # Saltation pass
        # Sequential axis: wind direction  |  Parallel axis: perpendicular
        # ---------------------------------------------------------------
        if di != 0:
            # Sweep rows upwind → downwind; columns in parallel.
            # Deposits from (i, j) land at (i+di, j+dj) — a different row
            # that has not been processed yet → no RAW hazard across threads.
            for ii in range(h - 2):
                i = (1 + ii) if wx >= 0.0 else (h - 2 - ii)
                for j in prange(1, w - 1):
                    z = out[i, j]
                    pui = i + ui
                    puj = j + uj
                    if 0 <= pui < h and 0 <= puj < w:
                        upwind_z = out[pui, puj]
                    else:
                        upwind_z = z  # boundary → treat as flat
                    windward_slope = z - upwind_z
                    if windward_slope <= 0.0:
                        continue
                    shear = wind_strength * windward_slope
                    if shear < threshold:
                        continue
                    erode = erosion_rate * (shear - threshold)
                    if erode > out[i, j]:
                        erode = out[i, j]
                    ni = i + di
                    nj = j + dj
                    if 1 <= ni < h - 1 and 1 <= nj < w - 1:
                        out[i, j]   -= erode
                        out[ni, nj] += erode * deposition_rate
        else:
            # di == 0 (pure crosswind): sweep columns upwind → downwind;
            # rows in parallel.  Deposits land at (i, j+dj) — a different
            # column → no RAW hazard across threads.
            for jj in range(w - 2):
                j = (1 + jj) if wy >= 0.0 else (w - 2 - jj)
                for i in prange(1, h - 1):
                    z = out[i, j]
                    pui = i + ui
                    puj = j + uj
                    if 0 <= pui < h and 0 <= puj < w:
                        upwind_z = out[pui, puj]
                    else:
                        upwind_z = z  # boundary → treat as flat
                    windward_slope = z - upwind_z
                    if windward_slope <= 0.0:
                        continue
                    shear = wind_strength * windward_slope
                    if shear < threshold:
                        continue
                    erode = erosion_rate * (shear - threshold)
                    if erode > out[i, j]:
                        erode = out[i, j]
                    ni = i + di
                    nj = j + dj
                    if 1 <= ni < h - 1 and 1 <= nj < w - 1:
                        out[i, j]   -= erode
                        out[ni, nj] += erode * deposition_rate

        # ---------------------------------------------------------------
        # Slip-face avalanche pass — 9-colour parallel decomposition
        # Same-colour cells are ≥ 3 apart in every dimension so their
        # 8-neighbourhoods never overlap → zero race conditions.
        # ---------------------------------------------------------------
        for ci in range(3):
            for cj in range(3):
                ni_count = (h - 2 - ci + 2) // 3
                nj_count = (w - 2 - cj + 2) // 3
                for pi in prange(ni_count):
                    i = 1 + ci + pi * 3
                    if i >= h - 1:
                        continue
                    for pj in range(nj_count):
                        j = 1 + cj + pj * 3
                        if j >= w - 1:
                            continue
                        for ddi in range(-1, 2):
                            for ddj in range(-1, 2):
                                if ddi == 0 and ddj == 0:
                                    continue
                                t = avalanche_talus * SQRT2 if (ddi != 0 and ddj != 0) else avalanche_talus
                                diff = out[i, j] - out[i + ddi, j + ddj]
                                if diff > t:
                                    move = (diff - t) * 0.5
                                    out[i, j]              -= move
                                    out[i + ddi, j + ddj]  += move

    return out