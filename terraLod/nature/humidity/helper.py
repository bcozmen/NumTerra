import numpy as np

def get_itcz(lat_grid):
    lat_abs = np.abs(lat_grid)
    return np.clip(1.0 + 0.6 * np.exp(-(lat_abs / 12.0)**2) - 0.4 * np.exp(-((lat_abs - 30.0) / 8.0)**2), 0.3, 1.8)