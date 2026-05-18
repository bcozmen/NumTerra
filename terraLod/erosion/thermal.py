import numpy as np
from numba import njit, prange

def thermal_erosion(grid, iterations, talus, **kwargs):
    return thermal_erosion_numba(grid, iterations, talus)

@njit(cache=True, parallel=True)
def thermal_erosion_numba(grid, iterations: int = 0, talus: float = 0.025):
    """Talus-angle thermal erosion — parallel 9-colour implementation.

    Redistributes material from steep slopes to their lower neighbours
    when the height difference exceeds *talus*.

    Parallelism strategy — 9-colour (3×3 tile) decomposition
    ---------------------------------------------------------
    Each iteration is split into 9 sequential sub-passes, one per colour
    ``(ci, cj)`` where ``ci = i % 3`` and ``cj = j % 3``.  Within a
    sub-pass all cells of the same colour are processed in parallel via
    ``prange``.

    Correctness proof: two cells of the same colour are at least 3 cells
    apart in every dimension.  Their 8-neighbourhoods extend only 1 cell
    in each direction, so neighbourhoods of same-colour cells never overlap.
    Each cell and its neighbours are therefore touched by exactly one thread
    per sub-pass → **zero race conditions**, no locking required.

    Other improvements
    ------------------
    * Diagonal neighbours use ``talus × √2`` (distance-corrected).
    * Material is distributed proportionally to each neighbour's excess,
      producing smooth alluvial fans.
    * Mass is exactly conserved.

    Parameters
    ----------
    grid       : 2-D float32 height map.
    iterations : number of full passes (each pass = 9 parallel sub-passes).
    talus      : maximum stable height difference for axial neighbours.

    Returns
    -------
    Modified copy of *grid*.
    """
    h, w = grid.shape
    out = grid.copy()
    SQRT2 = 1.41421356

    for _ in range(iterations):
        # 9 sub-passes — one per (ci, cj) colour in {0,1,2}²
        for ci in range(3):
            for cj in range(3):
                # Number of cells of this colour (ceiling division)
                ni = (h - 2 - ci + 2) // 3   # rows in [1, h-2] with row%3 == ci
                nj = (w - 2 - cj + 2) // 3

                for pi in prange(ni):
                    i = 1 + ci + pi * 3
                    if i >= h - 1:
                        continue
                    for pj in range(nj):
                        j = 1 + cj + pj * 3
                        if j >= w - 1:
                            continue

                        total_excess = 0.0

                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                if di == 0 and dj == 0:
                                    continue
                                t = talus * SQRT2 if (di != 0 and dj != 0) else talus
                                diff = out[i, j] - out[i + di, j + dj]
                                if diff > t:
                                    total_excess += (diff - t)

                        if total_excess > 0.0:
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    if di == 0 and dj == 0:
                                        continue
                                    t = talus * SQRT2 if (di != 0 and dj != 0) else talus
                                    diff = out[i, j] - out[i + di, j + dj]
                                    if diff > t:
                                        excess = diff - t
                                        # Distribute proportionally: each neighbour receives
                                        # (excess / total_excess) of the total material to move
                                        # (total to move = total_excess * 0.5 for stability).
                                        # Simplifies to: move = excess * 0.5
                                        move = 0.5 * excess
                                        out[i + di, j + dj] += move
                                        out[i, j]           -= move

    return out