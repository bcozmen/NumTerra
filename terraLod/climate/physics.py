"""
Pure physics utility functions for the climate module.

All functions here are stateless and dependency-free (only NumPy).
They can be used and tested independently of the Climate class.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

def prevailing_wind(lat_deg: float) -> tuple[float, float]:
    """
    Return a unit (wy, wx) prevailing-wind vector for a given central latitude.

    Follows the three classic circulation cells:
      - Tropics    (|lat| < 30°): north-east trades  → wind blows toward SW
      - Temperate  (30–60°):      south-westerlies    → wind blows toward NE
      - Polar      (> 60°):       north-east polars   → wind blows toward SW

    The *y* axis of the height-map is interpreted as south→north (row 0 = south).
    """
    abs_lat = abs(lat_deg)
    if abs_lat < 30.0:
        wy, wx = -0.5, -0.87   # trade winds: SW
    elif abs_lat < 60.0:
        wy, wx = 0.5, 0.87     # westerlies: NE
    else:
        wy, wx = -0.5, -0.87   # polar easterlies: SW
    if lat_deg < 0:
        wy = -wy
    norm = np.sqrt(wx ** 2 + wy ** 2)
    return wy / norm, wx / norm


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
