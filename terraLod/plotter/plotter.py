from typing import Callable, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm, PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import matplotlib.colors as mcolors

from terraLod.utils import timeit

from .config import LayerSpec, PlotterConfig, _default_layer_specs


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
        if keys is None:
            keys = ["height", "wind", "temperature", "humidity", "rain", "soil_moisture", "runoff","sun"]
        
        n_plots = len(keys) + 1
        n_rows = (n_plots + 1) // 2
        
        # Pop standard grid figsize out so it isn't passed down downstream to single plot logic
        figsize = kwargs.pop("figsize", (12, 5 * n_rows))
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes_flat = axes.flatten()

        for i, key in enumerate(keys):
            # Setup the specific layer overrides used during grid builds
            opts = {}
            if key == "wind":
                opts["show_wind"] = True  
                key = "height"  # Wind layer spec is mostly the same as height, but with the show_wind flag enabled to trigger streamplot rendering
            
            opts.update(kwargs)
            
            # Delegate entirely to the refactored plot method!
            self.plot(key, ax=axes_flat[i], **opts)

        

        # Clear out any leftover empty subplots if we have an odd count
        for j in range(n_plots, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        title = f"Season : {self.world.time} | Latitude: {self.world.worldConfig.latitude}°"

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
        self._finalize_axis(ax, key, im, kwargs)

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
    def _finalize_axis(self, ax: Axes, key, im=None, opts=None) -> None:
        """Standardized labels, ticks, and colorbars."""
        title = f"{key.capitalize()} Map"
        label = f"{key} ({self.specs[key].unit})"

        map = self.world[key]()
        sea_mask = self.world["sea_mask"]().astype(bool)
        mean = np.nanmean(map)
        mean_without_sea = np.nanmean(map[~sea_mask]) 

        title += f"\nMean: {mean:.2f}" 
        title += f" | Land Mean: {mean_without_sea:.2f}"
        ax.set_title(title, fontsize=14)
        rows, cols = self.world.size
        cy, cx = self.world.cell_size
        
        ax.set_xticks(np.linspace(0, cols - 1, 5, dtype=int))
        ax.set_yticks(np.linspace(0, rows - 1, 5, dtype=int))
        ax.set_xticklabels([f"{int(x * cx / 1000)} km" for x in ax.get_xticks()])
        ax.set_yticklabels([f"{int(y * cy / 1000)} km" for y in ax.get_yticks()])

        ax.set_xlim(0, cols - 1)
        ax.set_ylim(0, rows - 1)
        if im:
            if key == "height" and opts.get("show_wind", False):
                ax.set_title(f"Wind", fontsize=14)
                return
            if isinstance(im.norm, LogNorm):
                formatter = LogFormatterMathtext(base=10, labelOnlyBase=True)
                locator = LogLocator(base=10, subs=(1.0,))
                plt.colorbar(im, ax=ax, label=label, shrink=0.5, format=formatter, ticks=locator)
            else:
                plt.colorbar(im, ax=ax, label=label, shrink=0.5)

    
    def _get_range_cmap(self, key, data, spec):
        vmin, vmax = spec.resolve_range(data)
        cmap = spec.cmap
        if key == "height":
            # Plot height normally first (like sea_level 0 should start as green and up to vmax 1).
            vmin = 0
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

        vmin, vmax, cmap = self._get_range_cmap(key, data, spec)
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
            alpha = 0.5 if key == "height" else 0.1
            ax.imshow(1 - sun, cmap="gray", alpha=alpha, origin=self.origin)
    
    @timeit(label="Contour Rendering")
    def _render_contour(self, opts, ax, key):
        if opts.get("show_contour", True):
            h = self.world["height"]() - self.world["sea_level"]()
            cnt = ax.contour(h, levels=10, colors="black", linewidths=0.5, alpha=0.75)
            ax.clabel(cnt, inline=True, fontsize=7, fmt="%.2f")
            self._render_sea_contour(opts, ax, key)

    def _render_sea_contour(self,opts, ax, key):
        sea_mask = self.world["sea_mask"]().astype(bool)
        if np.any(sea_mask):
            ax.contour(sea_mask, levels=[0.5], colors="darkblue", linewidths=0.65, alpha=0.75)  

    def _subsample_wind(self, u, v):
        h, w  = u.shape
        total = h * w
        stride = int(np.sqrt(total / self.config.wind_sample_points))

        u_new = u[::stride, ::stride]
        v_new = v[::stride, ::stride]
        x_new = np.linspace(0, w - 1, u_new.shape[1])
        y_new = np.linspace(0, h - 1, u_new.shape[0])
         
        return u_new, v_new, x_new, y_new
    
    def _darken_cmap(self,cmap, factor):
        cmap = plt.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))
        colors[:, :3] *= factor  # Darken RGB channels
        return mcolors.ListedColormap(colors)
    @timeit(label="Wind Rendering")
    def _render_wind(self, opts, ax, key, spec):
        max_wind = 0
        if opts.get("show_wind", False) and "wind" in self.world.maps:
            w = self.world["wind"]()
            u, v = w[..., 1], w[..., 0]
            u, v, x, y = self._subsample_wind(u, v)
            sq = np.hypot(u, v)
            
            vmin, vmax, cmap = self._get_range_cmap("wind", sq, self.specs["wind"])
            
            norm = self._get_norm(spec, vmin, vmax)

            #linewidth -> faster is smaller, 
            linewidth = 1 + 1.0 * (sq / vmax)  # Scale line width by wind speed             
            
            sp = ax.streamplot(x, y, u, v, color=sq, cmap=self._darken_cmap(cmap, 0.9), norm=norm, 
                        linewidth=linewidth, arrowsize=1.5, minlength=0.25, 
                        broken_streamlines=False,  density=0.8)
            
            
            plt.colorbar(sp.lines, ax=ax, label="Wind Speed (m/s)" , shrink=0.5)

        return max_wind
        
    
    def _render_sea(self, opts, ax, key):
        if key != "height":
            return
        sea_mask = self.world["sea_mask"]().astype(bool)

        height_mask = self.world["height"]().copy()
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

        sx, sy, sz = self.world.worldConfig.time.solar_vectors
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