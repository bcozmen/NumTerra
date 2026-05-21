"""
Plotter — visualisation module for the map-maker pipeline.

Design
------
* Each renderable layer is described by a ``LayerSpec`` dataclass that carries
  its colormap, value range, unit label, and an optional custom renderer.
* ``Plotter.plot(key)`` looks up the spec, draws the base image, then composites
  sun-shading, contour lines, wind streamlines, and water masks as needed.
* New layers can be registered at runtime via ``Plotter.register(key, spec)``.

Wind note: axis-0 (i) is the row axis running top → bottom.
  wind[..., 0] > 0  ⟹  air moving toward higher row indices (southward).
  wind[..., 1] > 0  ⟹  air moving toward higher column indices (eastward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Layer specification
# ---------------------------------------------------------------------------

@dataclass
class LayerSpec:
    """Everything Plotter needs to visualise one named map layer."""
    unit:       str
    cmap:       str                         = "viridis"
    vrange:     Tuple[float, float]         = (0.0, 1.0)
    # When True the range is recomputed per-render from the data (robust percentile).
    auto_range: bool                        = False
    # Wind streamline colormap that contrasts well against this layer's background.
    wind_cmap:  str                         = "cool"
    # Optional hook: fn(ax, world) for fully custom rendering.
    renderer:   Optional[Callable]          = field(default=None, repr=False)

    def resolve_range(self, data: np.ndarray) -> Tuple[float, float]:
        if self.auto_range:
            return tuple(np.nanpercentile(data, [0.5, 99.5]))
        return self.vrange


def _default_layer_specs() -> dict[str, LayerSpec]:
    return {
        "height":        LayerSpec(unit="norm",    cmap="terrain", vrange=(-0.2, 1.0), wind_cmap="Reds"),
        "temperature":   LayerSpec(unit="°C",       cmap="inferno", vrange=(-10, 30),   wind_cmap="Blues",   auto_range=True),
        "sun":           LayerSpec(unit="norm",     cmap="gray",    vrange=(0, 1),       wind_cmap="autumn"),
        "humidity":      LayerSpec(unit="hPa",      cmap="Blues",   vrange=(0, 60),      wind_cmap="autumn",  auto_range=True),
        "rain":          LayerSpec(unit="mm/yr",    cmap="Blues",   vrange=(0, 2000),    wind_cmap="autumn",  auto_range=True),
        "soil_moisture": LayerSpec(unit="mm",       cmap="Greens",  vrange=(0, 200),     wind_cmap="YlOrRd",  auto_range=True),
        "runoff":        LayerSpec(unit="mm/step",  cmap="YlGnBu",  vrange=(0, 50),      wind_cmap="autumn",  auto_range=True),
        "wind":          LayerSpec(unit="m/s",      cmap="cool",    vrange=(-25, 25),    wind_cmap="Reds"),
    }


WATER_COLORS: dict[str, np.ndarray] = {
    "sea":   np.array([0.00, 0.20, 0.15]),   # deep ocean teal
    "lake":  np.array([0.00, 0.25, 0.25]),   # lake blue
    "river": np.array([0.25, 0.60, 0.45]),   # stream cyan
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Plotter:
    """Visualisation model.  Attach to a World with ``Plotter(world)``."""

    def __init__(self, world):
        self.world  = world
        self.specs: dict[str, LayerSpec] = _default_layer_specs()
        self._sync_dynamic_ranges()
        world["model_plotter"] = self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, key: str, spec: LayerSpec) -> None:
        """Add or overwrite a layer specification."""
        self.specs[key] = spec

    def plot(
        self,
        key: str,
        *,
        show_wind:    bool = False,
        show_contour: bool = True,
        show_water:   bool = True,
        show_sun:     bool = True,
        figsize: Tuple[int, int] = (8, 8),
    ) -> Tuple[Figure, Axes]:
        """
        Render *key* as a 2-D colour map.

        Returns ``(fig, ax)`` so callers can further customise the plot.
        """
        if key not in self.specs:
            raise KeyError(
                f"No LayerSpec registered for '{key}'. "
                f"Available: {sorted(self.specs)}"
            )

        spec = self.specs[key]
        fig, ax = plt.subplots(figsize=figsize)

        # --- custom renderer short-circuit ---
        if spec.renderer is not None:
            spec.renderer(ax, self.world)
            ax.set_title(key.capitalize(), fontsize=14)
            self._set_axis_ticks(ax)
            plt.tight_layout()
            plt.show()
            return fig, ax

        # --- base image ---
        data = self._fetch(key)
        if key != "height":  # height map should still influence colour scaling even under water
            data = self._make_water_cells_nan(data)
        vmin, vmax = spec.resolve_range(data)
        print(f"Plotting '{key}' with vmin={vmin:.2f}, vmax={vmax:.2f}")
        im = ax.imshow(data, cmap=spec.cmap, vmin=vmin, vmax=vmax, origin="upper")

        # --- overlays (order matters: sun → contours → wind → water) ---
        height = self._height_above_sea()

        if show_sun and "sun" in self.world.maps:
            sun_alpha = 0.45 if key == "height" else 0.20
            ax.imshow(self._fetch("sun"), cmap="gray", alpha=sun_alpha, origin="upper")

        if show_contour and key != "height":
            self._draw_contours(ax, height)

        if show_wind and "wind" in self.world.maps:
            self._draw_wind(ax, key)

        if show_water:
            self._draw_water(ax)

        # --- decorations ---
        plt.colorbar(im, ax=ax, label=f"{key} ({spec.unit})", shrink=0.5)
        ax.set_title(f"{key.capitalize()} Map", fontsize=14)
        self._set_axis_ticks(ax)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_all(
        self,
        keys: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Plot every registered layer that exists in world.maps (skips 'wind' and 'sun')."""
        keys = keys or [k for k in self.specs if k in self.world.maps and (k != "wind" and k != "sun")]
        for key in keys:
            self.plot(key, **kwargs)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _fetch(self, key: str) -> np.ndarray:
        return self.world[key]()

    def _height_above_sea(self) -> np.ndarray:
        h = self._fetch("height").copy()
        sea_level = float(getattr(self.world, "sea_level", 0.0))
        h -= sea_level
        return h

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------
    def _make_water_cells_nan(self, map: np.ndarray) -> np.ndarray:
        """Set water cells to NaN so they don't affect colour scaling."""
        map = map.copy()   # never mutate the stored orig_arr in the FastInterpolator
        for name in WATER_COLORS:
            mask_key = f"{name}_mask"
            if mask_key in self.world.maps:
                mask = self._fetch(mask_key).astype(bool)
                map[mask] = np.nan
        return map
    def _draw_contours(self, ax: Axes, height: np.ndarray, levels: int = 10) -> None:
        contour = ax.contour(
            height, levels=levels,
            colors="black", linewidths=0.5, alpha=0.6,
        )
        ax.clabel(contour, inline=True, fontsize=7, fmt="%.2f")

    def _draw_wind(self, ax: Axes, key: str) -> None:
        wind  = self._fetch("wind")
        u     = wind[..., 1]            # eastward  (col-axis)
        v     = -wind[..., 0]           # northward (negate row-axis for plot convention)
        speed_norm = np.hypot(u, v) / (np.nanmax(np.hypot(u, v)) + 1e-8)

        wind_cmap = self.specs.get(key, LayerSpec(unit="")).wind_cmap
        sp = ax.streamplot(
            np.arange(wind.shape[1]), np.arange(wind.shape[0]),
            u, v,
            color=speed_norm, cmap=wind_cmap,
            linewidth=1.2, density=1.5, arrowsize=0.8,
        )
        cbar = plt.colorbar(sp.lines, ax=ax, shrink=0.4)
        cbar.set_label("wind speed (norm.)", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    def _draw_water(self, ax: Axes, alpha: float = 0.75) -> None:
        for name, color in WATER_COLORS.items():
            mask_key = f"{name}_mask"
            if mask_key not in self.world.maps:
                continue
            mask = self._fetch(mask_key).astype(float)
            rgba = np.zeros((*mask.shape, 4))
            rgba[..., :3] = color
            rgba[..., 3]  = mask * alpha
            ax.imshow(rgba, origin="upper")

    # ------------------------------------------------------------------
    # Axis decoration
    # ------------------------------------------------------------------

    def _set_axis_ticks(self, ax: Axes) -> None:
        rows, cols = self.world.size
        cy, cx     = self.world.cell_size      # (row_meters, col_meters)
        x_step = max(1, cols // 5)
        y_step = max(1, rows // 5)

        x_ticks = np.arange(0, cols, x_step)
        y_ticks = np.arange(0, rows, y_step)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{int(x * cx / 1_000)} km" for x in x_ticks])
        ax.set_yticklabels([f"{int(y * cy / 1_000)} km" for y in y_ticks])

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _sync_dynamic_ranges(self) -> None:
        """Pull runtime limits from already-initialised models."""
        models = self.world.models

        if "model_wind" in models:
            v = models["model_wind"].config.max_wind_speed
            self.specs["wind"].vrange = (-v, v)

        if "model_humidity" in models:
            cfg = models["model_humidity"].config
            self.specs["rain"].vrange          = (0, cfg.max_rain)
            self.specs["soil_moisture"].vrange = (0, cfg.soil_capacity) 

