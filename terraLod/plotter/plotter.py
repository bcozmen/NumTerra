from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm, PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import LogFormatterMathtext, LogLocator


from terraLod.utils import timeit


@dataclass
class LayerSpec:
    """Everything Plotter needs to visualise one named map layer."""
    unit:           str
    cmap:           str                         = "viridis"
    vrange:         Tuple[float, float]         = (0.0, 1.0)
    auto_range:     bool                        = False
    norm_type:      str                         = "linear"
    wind_cmap:      str                         = "plasma" # Special colormap for wind speed in streamplot, since it uses color for magnitude not value
    renderer:       Optional[Callable]          = field(default=None, repr=False)

    def resolve_range(self, data: np.ndarray) -> Tuple[float, float]:
        #auto_range (min_percentile, max_percentile) 
        if not self.auto_range:
            return self.vrange
        #convert range percentiles to 0-100 scale and compute percentiles
        sea_mask = np.zeros(data.shape, dtype=bool)

        lo, hi = np.percentile(data[~sea_mask], [self.auto_range[0] * 100, self.auto_range[1] * 100])
        
        return lo, hi
        


def _default_layer_specs() -> dict[str, LayerSpec]:
    return {
        "height":        LayerSpec("norm",    cmap="terrain", vrange=(-0.2, 1.0)),
        "temperature":   LayerSpec("°C",       cmap="inferno", vrange=(-10, 30),    auto_range=(0,1)),
        "sun":           LayerSpec("norm",     cmap="gray"),
        "humidity":      LayerSpec("hPa",      cmap="Blues",   vrange=(0, 60), auto_range = (0,1)),
        "rain":          LayerSpec("mm/yr",    cmap="Blues",   vrange=(1.0, 2000), auto_range=(0,1)),
        "soil_moisture": LayerSpec("mm",       cmap="Greens",  vrange=(0, 200),  auto_range=(0,1)),
        "wind":          LayerSpec("m/s",      cmap="cool",    vrange=(0, 25), auto_range=(0,1)),
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
    wind_subsample: int = 4  # Subsampling stride for wind vector field. Higher = faster but less detailed. Only applies to wind layer.

class Plotter:
    """Visualisation module. Attach to a World with ``Plotter(world)``."""

    def __init__(self, world):
        self.world = world
        self.config = PlotterConfig()
        self.specs = _default_layer_specs()
        self.origin = "lower"
        world["model_plotter"] = self

    def plot_all(self, keys: Optional[List[str]] = None, **kwargs) -> Tuple[Figure, List[Axes]]:
        """Plots all matching layers cleanly inside a unified Nx2 subplot grid layout."""
        keys = keys or [k for k in self.specs if k in self.world.maps and k not in ("wind", "sun")]
        if not keys:
            return plt.figure(), []

        n_plots = len(keys)
        n_rows = (n_plots + 1) // 2
        
        # Pop standard grid figsize out so it isn't passed down downstream to single plot logic
        figsize = kwargs.pop("figsize", (12, 5 * n_rows))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes_flat = axes.flatten()

        for i, key in enumerate(keys):
            # Setup the specific layer overrides used during grid builds
            opts = {"show_wind": (key == "height")}
            opts.update(kwargs)
            
            # Delegate entirely to the refactored plot method!
            self.plot(key, ax=axes_flat[i], **opts)

        # Clear out any leftover empty subplots if we have an odd count
        for j in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        title = f"Season : {self.world.season.capitalize()} | Hour: {self.world.worldConfig.hour}:00 | Latitude: {self.world.worldConfig.latitude}°"

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        return fig, list(axes_flat[:n_plots])
    
    def plot(self, key: str, ax: Optional[Axes] = None, **kwargs) -> Tuple[Optional[Figure], Axes]:
        """Render a single map layer. If an ax is provided, draws directly onto it."""
        if key not in self.specs:
            raise KeyError(f"No LayerSpec for '{key}'. Available: {sorted(self.specs)}")

        # Track if we are responsible for showing the final layout window
        standalone = ax is None
        fig = None

        if standalone:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 8)))

        im, max_wind = self._render_map(key, ax, kwargs)
        self._finalize_axis(ax, key, im)

        if standalone:
            plt.tight_layout()
            plt.show()
            
        return fig, ax

    def _render_map(self, key: str, ax: Axes, opts: dict) -> Optional[plt.cm.ScalarMappable]:
        """Internal heavy-lifter to draw a single layer onto a given axis."""
        spec = self.specs[key]
        
        if spec.renderer is not None:
            spec.renderer(ax, self.world)
            return None

        im = self._render_layer(key, ax, opts, spec)
        self._render_sun(opts, ax, key)
        max_wind = self._render_wind(opts, ax, key, spec)
        self._render_sea(opts, ax, key)
        self._render_contour(opts, ax, key)
        self._render_filter(opts, ax, key)
        return im, max_wind

    def _render_filter(self, opts, ax, key):
        c = np.ones(self.world.size)
        ax.imshow(c, cmap="gray", alpha=0.05, origin=self.origin, vmin=0, vmax=1.0)
    def _finalize_axis(self, ax: Axes, key, im=None) -> None:
        """Standardized labels, ticks, and colorbars."""
        title = f"{key.capitalize()} Map"
        label = f"{key} ({self.specs[key].unit})"


        ax.set_title(title, fontsize=14)
        rows, cols = self.world.size
        cy, cx = self.world.cell_size
        
        ax.set_xticks(np.linspace(0, cols - 1, 5, dtype=int))
        ax.set_yticks(np.linspace(0, rows - 1, 5, dtype=int))
        ax.set_xticklabels([f"{int(x * cx / 1000)} km" for x in ax.get_xticks()])
        ax.set_yticklabels([f"{int(y * cy / 1000)} km" for y in ax.get_yticks()])

        ax.set_xlim(0, cols - 1)
        ax.set_ylim(0, rows - 1)
        if im  and key != "height":
            if isinstance(im.norm, LogNorm):
                formatter = LogFormatterMathtext(base=10, labelOnlyBase=True)
                locator = LogLocator(base=10, subs=(1.0,))
                plt.colorbar(im, ax=ax, label=label, shrink=0.5, format=formatter, ticks=locator)
            else:
                plt.colorbar(im, ax=ax, label=label, shrink=0.5)

    
    def _get_range_cmap(self, key, spec):
        vmin, vmax = spec.resolve_range(self.world[key]())
        cmap = spec.cmap
        if key == "height":
            # Plot height normally first (like sea_level 0 should start as green and up to vmax 1).
            vmin = getattr(self.world, "sea_level", 0.0)
            vmax = 1.0
            cmap = LinearSegmentedColormap.from_list(
                "terrain_land", plt.cm.terrain(np.linspace(0.20, 1.0, 256))
            )
        return vmin, vmax, cmap

    def _get_norm(self, spec, vmin, vmax):
        if spec.norm_type == "log":
            floor = spec.vrange[0] if spec.vrange[0] > 0 else 0.1
            vmin = max(vmin, floor) 
            vmax = max(vmax, vmin + 1.0)
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        return norm
    # ------ Rendering Helpers ------
    @timeit(label="Map Rendering")
    def _render_layer(self, key, ax, opts, spec):
        data = self.world[key]().copy()

        vmin, vmax, cmap = self._get_range_cmap(key, spec)
        norm = self._get_norm(spec, vmin, vmax)

        cmap_obj = plt.get_cmap(cmap)
        normed_data = norm(data)
        rgba = cmap_obj(normed_data)
        ax.imshow(rgba, origin=self.origin)
        im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return im

    def _render_sun(self,opts, ax, key):
        if opts.get("show_sun", True) and "sun" in self.world.maps:
            sun = self.world["sun"]()
            alpha = 0.25 if key == "height" else 0.05
            ax.imshow(1 - sun, cmap="gray", alpha=alpha, origin=self.origin)
    
    @timeit(label="Contour Rendering")
    def _render_contour(self, opts, ax, key):
        if opts.get("show_contour", True):
            h = self.world["height"]() - float(getattr(self.world, "sea_level", 0.0))
            cnt = ax.contour(h, levels=10, colors="black", linewidths=0.5, alpha=0.75)
            ax.clabel(cnt, inline=True, fontsize=7, fmt="%.2f")
            self._render_sea_contour(opts, ax, key)

    def _render_sea_contour(self,opts, ax, key):
        sea_mask = self.world["sea_mask"]().astype(bool)
        if np.any(sea_mask):
            ax.contour(sea_mask, levels=[0.5], colors="darkblue", linewidths=0.65, alpha=0.75)  

    def _subsample_wind(self, u, v):
        stride = self.config.wind_subsample
        return u[::stride, ::stride], v[::stride, ::stride], np.arange(0, u.shape[1], stride), np.arange(0, u.shape[0], stride)
    
    @timeit(label="Wind Rendering")
    def _render_wind(self, opts, ax, key, spec):
        max_wind = 0
        if opts.get("show_wind", False) and "wind" in self.world.maps:
            w = self.world["wind"]()
            u, v = w[..., 1], w[..., 0]
            u, v, x, y = self._subsample_wind(u, v)
            sq = np.hypot(u, v)
            
            vmin, vmax, cmap = self._get_range_cmap(key, spec)
            norm = self._get_norm(spec, vmin, vmax)

            ax.streamplot(x, y, u, v, color=sq, cmap=spec.wind_cmap, norm=norm, linewidth=0.7, arrowsize=0.5)
            
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=spec.wind_cmap)
            plt.colorbar(mappable, ax=ax, label="Wind Speed (m/s) [Power Scale]", shrink=0.5)

        return max_wind
    
    def _render_sea(self, opts, ax, key):
        if key != "height":
            return
        sea_mask = self.world["sea_mask"]().astype(bool)

        height_mask = self.world["height"]()
        height_mask[~sea_mask] = np.nan  # Mask out non-sea areas for accurate coloring
        height_mask /= np.nanmax(height_mask)  # Normalize to [0, 1] based on sea depth
        height_mask = 1 - height_mask  # Invert so deeper water is darker
        height_mask = 0.35 + 0.5 * height_mask  # Scale to [0.25, 1] to avoid pure white for shallow water
        ax.imshow(height_mask, cmap="Blues", origin=self.origin, vmin=0, vmax=1.0)
        #self._specular_highlight(opts, ax, key)

    def _specular_highlight(self, opts, ax, key):
        h, w = self.world.size
        water_shade = np.zeros((h, w), dtype=np.float32)
        masks = {
            "sea": self.world["sea_mask"]().astype(bool),
            #"river": self.world["river_mask"]().astype(bool),
            #"lake": self.world["lake_mask"]().astype(bool),
        }

        sx, sy, sz = self.world.worldConfig.solar_vectors
        L = np.array([sx, sy, sz])
        V = np.array([0.0, 0.0, 1.0])
        H = L + V
        H_norm = H / (np.linalg.norm(H) + 1e-8)
        spec_val = float(max(0.0, H_norm[2]) ** 20) 

        water_any = np.zeros((h, w), dtype=bool)
        for m_data in masks.values():
            water_any |= m_data
        
        arr_to_show = np.where(water_any, spec_val, np.nan)
        ax.imshow(arr_to_show, cmap="Greys", origin=self.origin, vmin=0, vmax=1.0, alpha=0.2)