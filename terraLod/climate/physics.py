"""
Pure physics utility functions for the climate module.

All functions here are stateless and dependency-free (only NumPy).
They can be used and tested independently of the Climate class.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

def prevailing_wind(sea_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-cell (wy, wx) unit-vector arrays representing a sea-to-inland
    prevailing wind.

    For every cell the wind direction is the unit vector pointing **from the
    nearest sea cell toward the current cell** — i.e. the wind blows off the
    ocean and penetrates into the continent.  Sea cells themselves carry a
    zero vector (no net advection source at the origin).

    Parameters
    ----------
    sea_mask : 2-D bool array
        True where the cell is ocean / open sea.

    Returns
    -------
    wy, wx : 2-D float32 arrays of the same shape as *sea_mask*
        Per-cell unit wind vectors (row-axis component, column-axis component).
    """
    rows, cols = sea_mask.shape

    # Distance transform: for each cell find the nearest sea cell.
    # distance_transform_edt treats zero-pixels as "background" (sea),
    # so we pass (~sea_mask) to measure distance from sea.
    _, indices = distance_transform_edt(~sea_mask, return_indices=True)
    # indices[0] → row of nearest sea cell; indices[1] → col of nearest sea cell

    ii = np.broadcast_to(
        np.arange(rows, dtype=np.float32)[:, None], (rows, cols)
    ).copy()
    jj = np.broadcast_to(
        np.arange(cols, dtype=np.float32)[None, :], (rows, cols)
    ).copy()

    # Vector from nearest-sea-cell → current cell  (= sea-to-inland direction)
    wy = (ii - indices[0]).astype(np.float32)
    wx = (jj - indices[1]).astype(np.float32)

    norm = np.sqrt(wy ** 2 + wx ** 2) + 1e-9
    wy /= norm
    wx /= norm

    # Sea cells have no meaningful "from-sea" direction — zero them out
    wy[sea_mask] = 0.0
    wx[sea_mask] = 0.0

    return wy, wx


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def lat_base_temp(lat_deg: float | np.ndarray) -> np.ndarray:
    """
    Approximate annual-mean surface temperature (°C) as a function of latitude.

    Uses a simple empirical cosine fit calibrated to real-world means:
      equator  ~27 °C, 30° ~22 °C, 45° ~10 °C, 60° ~0 °C, 90° ~-20 °C
    """
    return 27.0 * np.cos(np.radians(lat_deg)) ** 1.5 - 2.0


def saturation_capacity(T_celsius: np.ndarray) -> np.ndarray:
    """
    Approximate saturation vapour pressure (relative scale) as a function
    of temperature, based on the Magnus formula (Clausius–Clapeyron approx).

    Returns values in [0, 1] range via normalisation against the equatorial
    peak (~27 °C), so the result can be used directly as a capacity multiplier
    against the moisture field.

    Higher temperature → higher capacity → air can hold more moisture before
    condensing → deserts in hot subsidence zones unless moisture is also high.

    q_sat ∝ exp(17.27 * T / (T + 237.3))
    """
    T = np.asarray(T_celsius, dtype=np.float64)
    q = np.exp(17.27 * T / (T + 237.3 + 1e-6))
    q_ref = np.exp(17.27 * 27.0 / (27.0 + 237.3))   # normalise at 27 °C equatorial baseline
    return (q / q_ref).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalize_land(arr: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """
    Normalise *arr* to [0, 1] using only the land (non-water) cells.
    Water cells are left untouched and must be overwritten by the caller.

    Note: only used for visualisation helpers.  Physical pipeline maps
    (humidity, precipitation) are intentionally NOT normalised so that
    absolute values and spatial gradients are preserved.
    """
    land_vals = arr[land_mask]
    lo, hi    = float(land_vals.min()), float(land_vals.max())
    if hi - lo < 1e-9:
        arr[land_mask] = 0.0
    else:
        arr[land_mask] = (land_vals - lo) / (hi - lo)
    return arr
