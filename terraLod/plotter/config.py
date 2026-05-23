from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class LayerSpec:
    """Everything Plotter needs to visualise one named map layer."""
    unit:           str
    cmap:           str                         = "viridis"
    vrange:         Tuple[float, float]         = (0.0, 1.0)
    auto_range:     bool                        = False
    norm_type:      str                         = "linear"
    renderer:       Optional[Callable]          = field(default=None, repr=False)

    def resolve_range(self, data: np.ndarray) -> Tuple[float, float]:
        #auto_range (min_percentile, max_percentile) 
        if not self.auto_range:
            r1 = min(self.vrange[0], np.nanmin(data))
            r2 = max(self.vrange[1], np.nanmax(data))
            return (r1, r2)
        #convert range percentiles to 0-100 scale and compute percentiles
        sea_mask = np.zeros(data.shape, dtype=bool)

        lo, hi = np.percentile(data[~sea_mask], [self.auto_range[0] * 100, self.auto_range[1] * 100])
        
        return lo, hi
        


def _default_layer_specs() -> dict[str, LayerSpec]:
    return {
        "height":        LayerSpec("norm",    cmap="terrain", vrange=(-0.2, 1.0)),
        "temperature":   LayerSpec("°C",       cmap="coolwarm", vrange=(-10, 35)),
        "sun":           LayerSpec("norm",     cmap="gray"),
        "humidity":      LayerSpec("hPa",      cmap="Blues",   vrange=(0, 60), auto_range = (0,1)),
        "rain":          LayerSpec("mm/yr",    cmap="Blues",   vrange=(1.0, 2000), auto_range=(0,1)),
        "soil_moisture": LayerSpec("mm",       cmap="Greens",  vrange=(0, 200),  auto_range=(0,1)),
        "wind":          LayerSpec("m/s",      cmap="plasma",    vrange=(0, 25), auto_range=(0,1)),
        "runoff":        LayerSpec("mm/step",  cmap="YlGnBu",  vrange=(0.1, 50), auto_range=(0,1)),
        "water_depth":   LayerSpec("m",        cmap="Blues",   vrange=(0, 10), auto_range=(0,1)),
        "discharge":     LayerSpec("m/step",   cmap="YlGnBu",  vrange=(1e-3, 0.5), auto_range=(0,1)),
    }

WATER_COLORS: dict[str, np.ndarray] = {
    "sea":   np.array([0.00, 0.20, 0.50]),
    "lake":  np.array([0.00, 0.25, 0.85]),
    "river": np.array([0.25, 0.60, 1.00]),
}

@dataclass
class PlotterConfig:
    """Global settings for the Plotter."""
    wind_sample_points: int = 100  # Target number of points to plot in wind streamplot (controls subsampling density)