import copy

from fields import BaseModel

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.colors import (
    Normalize, LogNorm, PowerNorm, SymLogNorm,
    LinearSegmentedColormap,
)

# Auto-generated derivative maps are always skipped regardless of render config.
_SKIP_SUFFIXES = ("_grad_magnitude", "_grad_i", "_grad_j", "_magnitude")

_PLOT_ORDER = [
    "H", "Sun", "P",
    "Ta", "Ts", "Tw",
    "Wa", "Wc", "Ws",
    "Evap", "Condensation", "Precip",
]


class WorldRenderer(BaseModel):
    """Lightweight real-time renderer attached to a ``fields.World`` instance.

    Render settings are read from each map's ``map_info[key]["render"]`` dict:
      - ``cmap``      - matplotlib colormap name (default ``"viridis"``)
      - ``vrange``    - ``(vmin, vmax)`` tuple to pin the colour scale
      - ``scale``     - ``"linear"`` (default), ``"log"``, ``"power"``, or ``"symlog"``
      - ``gamma``     - exponent for ``"power"`` scale (default ``0.4``)
      - ``linthresh`` - linear threshold for ``"symlog"`` scale
      - ``mask_sea``  - if True, sea tiles are fully excluded from stats and rendering

    All panels get terrain-contour and sea-boundary overlays automatically.
    The height panel uses a composite land/sea rendering with shadow shading
    and displays a wind-speed colour bar.
    """

    info = {
        "name": "renderer",
        "map_info": {}
    }

    def __init__(
        self,
        world,
        keys: list[str] | None = None,
        cmaps: dict[str, str] | None = None,
        cols: int = 3,
        figsize: tuple[float, float] | None = None,
    ):
        super().__init__(world)
        self._extra_cmaps = cmaps or {}

        self.keys = keys if keys is not None else self._available_maps()
        if not self.keys:
            raise ValueError("No plottable maps found in world.area. "
                             "Run at least one model step first.")

        n = len(self.keys)
        self._cols = min(cols, n)
        self._rows = (n + self._cols - 1) // self._cols

        if figsize is None:
            figsize = (5.2 * self._cols, 4.2 * self._rows)

        # squeeze=False ensures we always get a 2-D ndarray of axes,
        # avoiding a crash when there is only a single panel.
        self.fig, _axes = plt.subplots(
            self._rows, self._cols,
            squeeze=False,
            figsize=figsize,
            layout="constrained",
        )
        self._axes: list = list(_axes.flatten())

        self._ims:      dict = {}   # key -> AxesImage
        self._cbars:    dict = {}   # key -> Colorbar
        self._titles:   dict = {}   # key -> Text
        self._ax_map:   dict = {}   # key -> Axes
        self._contours: dict = {}   # key -> list[QuadContourSet | tuple]

        self._build()

        # w_pad / h_pad control panel spacing (inches)
        self.fig.get_layout_engine().set(
            w_pad=0.01, h_pad=0.01, rect=[0, 0, 1, 0.95],
            hspace=0.01, wspace=0.01,
        )

    # -- public API ------------------------------------------------------------

    def step(self, keys = None, save_path = None) -> None:
        """Redraw all panels with the current world state (no simulation step)."""
        if keys is None:
            keys = self._ims
        for key in keys:
            ax = self._ax_map[key]
            self._remove_contours(key)

            if key == "H":
                self._ims[key].set_data(self._height_rgba())
                self._titles[key].set_text(self._height_title())
            else:
                data, vmin, vmax = self._prepare_data(key)
                self._ims[key].set_data(data)
                self._ims[key].set_norm(self._make_norm(key, vmin, vmax))
                self._titles[key].set_text(self._panel_title(key, data))

            self._draw_contours(ax, key)

        time_model = self.world.models.get("time")
        if time_model is not None:
            self.fig.suptitle(str(time_model), fontsize=11)

        if save_path is not None:
            date = str(time_model).replace("/", "-").replace(" ", "_").replace(":", "-")
            self.fig.savefig(save_path + '_' + date + '.jpg', dpi=150)
        self._flush()

    # -- render-config helpers -------------------------------------------------

    def _render_info(self, key: str) -> dict:
        """Merged render dict: map_info defaults + caller cmap override."""
        info = dict(self.world.map_info.get(key, {}).get("render", {}))
        if key in self._extra_cmaps:
            info["cmap"] = self._extra_cmaps[key]
        return info

    def _build_cmap(self, key: str):
        """Return a (copied) colormap for *key*.

        When ``mask_sea`` is set the bad-value colour is made fully transparent
        so that NaN sea pixels are invisible in the rendered image.
        """
        ri   = self._render_info(key)
        cmap = copy.copy(plt.get_cmap(ri.get("cmap", "viridis")))
        if ri.get("mask_sea", False):
            cmap.set_bad(alpha=0)
        return cmap

    def _make_norm(self, key: str, vmin: float, vmax: float):
        """Build the appropriate Normalize subclass from the map's render config."""
        ri    = self._render_info(key)
        scale = ri.get("scale", "linear")


        if vmin == vmax == 0.0 and key == "Sun":
            vmax = 1.0 
            vmin = 0.0

        if scale == "log":
            safe_min = max(vmin, 1e-6)
            safe_max = max(vmax, safe_min * 10)
            return LogNorm(vmin=safe_min, vmax=safe_max)

        if scale == "power":
            return PowerNorm(gamma=ri.get("gamma", 0.4), vmin=vmin, vmax=vmax)

        if scale == "symlog":
            linthresh = ri.get("linthresh", max(abs(vmax) * 0.01, 1e-4))
            return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

        return Normalize(vmin=vmin, vmax=vmax)  # "linear" (default)

    def _prepare_data(self, key: str) -> tuple[np.ndarray, float, float]:
        """Return ``(data, vmin, vmax)`` with sea masking and range pinning applied.

        Sea tiles are set to NaN so they are excluded from both statistics and
        rendering (when paired with a cmap whose bad colour is transparent).
        """
        data = self.world.area[key]().copy().astype(np.float64)
        ri   = self._render_info(key)

        if ri.get("mask_sea", False):
            sea_mask = self.world.area["M_sea"]().astype(bool)
            data[sea_mask] = np.nan

        vrange = ri.get("vrange")
        if vrange is not None:
            return data, vrange[0], vrange[1]

        return data, float(np.nanmin(data)), float(np.nanmax(data))

    def _panel_title(self, key: str, data: np.ndarray) -> str:
        """Format: ``'KEY (unit)  [min ... max]'``."""
        info     = self.world.map_info.get(key, {})
        unit     = info.get("unit", "")
        unit_str = f" ({unit})" if unit else ""
        return (f"{key}{unit_str}")#f"[{float(np.nanmin(data)):.2f} … {float(np.nanmax(data)):.2f}]")

    # -- internal build --------------------------------------------------------

    def _available_maps(self) -> list[str]:
        result = [
            k for k in self.world.area.maps
            if not any(k.endswith(s) for s in _SKIP_SUFFIXES)
            and "render" in self.world.map_info.get(k, {})
            and self.world.area[k]().ndim == 2
        ]
        result.sort(key=lambda k: _PLOT_ORDER.index(k) if k in _PLOT_ORDER else 1e6)
        return result

    def _build(self) -> None:
        for i, key in enumerate(self.keys):
            ax = self._axes[i]
            self._ax_map[key] = ax

            if key == "H":
                im    = ax.imshow(self._height_rgba(), origin="lower",
                                  aspect="equal", interpolation="nearest")
                title = self._height_title()
                cbar  = self._build_wind_colorbar(ax)
            else:
                data, vmin, vmax = self._prepare_data(key)
                im    = ax.imshow(data, cmap=self._build_cmap(key),
                                  norm=self._make_norm(key, vmin, vmax),
                                  origin="lower", aspect="equal",
                                  interpolation="nearest")
                title = self._panel_title(key, data)
                cbar  = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01, shrink=0.7)

            self._set_distance_ticks(ax)
            self._ims[key]    = im
            self._cbars[key]  = cbar
            self._titles[key] = ax.set_title(title, fontsize=9)
            self._draw_contours(ax, key)

        for ax in self._axes[len(self.keys):]:
            ax.set_visible(False)

    def _set_distance_ticks(self, ax) -> None:
        """Display image-index ticks as physical distances in kilometres."""
        # imshow uses columns for x and rows for y.  Area.cell_size is stored
        # in array order (row, column), so the values are intentionally
        # reversed here.
        cell_y, cell_x = self.world.area.cell_size
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{int(value * cell_x / 1000):g}")
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{int(value * cell_y / 1000):g}")
        )
        ax.set_xlabel("x (km)", fontsize=8)
        ax.set_ylabel("y (km)", fontsize=8)

    # -- height panel ----------------------------------------------------------

    def _build_wind_colorbar(self, ax):
        """Wind-speed colour bar shown on the H panel."""
        sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno,
                                   norm=Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        self._wind_sm = sm
        cbar = self.fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.01, shrink=0.7)
        cbar.set_label("wind speed (m/s)", fontsize=7)
        return cbar

    def _height_rgba(self) -> np.ndarray:
        """Composite RGBA: terrain cmap for land, Blues depth for sea, shadow shading."""
        H         = self.world.area["H"]().copy()
        sea_level = float(self.world.area["sea_level"]())
        M_sea     = self.world.area["M_sea"]().astype(bool)
        max_alt   = float(self.world.max_altitude)

        H_m  = (H - sea_level) * max_alt   # metres relative to sea level
        rgba = np.zeros((*H.shape, 4), dtype=np.float32)
        rgba[..., 3] = 1.0

        # Land -- fixed scale [0, max_alt] m
        land_cmap = LinearSegmentedColormap.from_list(
            "terrain_land", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
        )
        land_rgba = land_cmap(Normalize(vmin=0.0, vmax=max_alt)(np.clip(H_m, 0.0, None)))
        rgba[~M_sea] = land_rgba[~M_sea]

        # Sea -- fixed scale [-max_alt, 0], deeper -> darker blue
        sea_normed = 0.35 + 0.50 * (
            1.0 - Normalize(vmin=-max_alt, vmax=0.0)(np.clip(H_m, -max_alt, 0.0))
        )
        rgba[M_sea] = plt.cm.Blues(sea_normed)[M_sea]

        # Shadow / hillshade
        ambient = 0.5
        shadow  = self.world.area["Shadow"]().copy()
        shade   = ambient + (1.0 - ambient) * np.clip(shadow, 0.0, 1.0)
        rgba[..., :3] = np.clip(rgba[..., :3] * shade[..., np.newaxis], 0.0, 1.0)

        # Gamma lift for better shadow/highlight contrast
        rgba[..., :3] = np.power(rgba[..., :3], 1.0 / 1.25)
        return rgba

    def _height_title(self) -> str:
        sea_level = float(self.world.area["sea_level"]())
        max_alt   = self.world.max_altitude
        land_max  = (1.0 - sea_level) * max_alt
        sea_min   = -sea_level * max_alt
        return "Wind Map"
        return (f"H  (m rel. sea level)"
                f"  │  land: 0 … {land_max:.0f} m"
                f"  │  sea: {sea_min:.0f} … 0 m")

    # -- overlay rendering -----------------------------------------------------

    def _draw_contours(self, ax, key: str) -> None:
        """Terrain contours + coastline on every panel; wind streamlines on H."""
        H         = self.world.area["H"]().copy()
        sea_level = float(self.world.area["sea_level"]())
        M_sea     = self.world.area["M_sea"]().astype(bool)

        overlays = [
            ax.contour(H - sea_level, levels=10,
                       colors="black", linewidths=0.4, alpha=0.45),
        ]
        if np.any(M_sea):
            overlays.append(
                ax.contour(M_sea.astype(float), levels=[0.5],
                           colors="darkblue", linewidths=0.8, alpha=0.7)
            )

        if key == "H":
            wind = self._draw_wind_overlay(ax)
            if wind is not None:
                overlays.append(wind)

        self._contours[key] = overlays

    def _draw_wind_overlay(self, ax) -> tuple | None:
        """Draw wind streamlines on *ax*; returns ``(StreamplotSet, patches)`` or None."""
        if "V" not in self.world.area.maps:
            return None

        V     = self.world.area["V"]().copy()
        V_mag = self.world.area["V_magnitude"]().copy()
        if V.ndim != 3 or V.shape[-1] != 2:
            return None

        rows, cols = V.shape[:2]
        X   = np.arange(cols)
        Y   = np.arange(rows)
        U   = V[..., 0].astype(np.float64)
        V_y = V[..., 1].astype(np.float64)

        vmax = float(np.nanmax(V_mag))
        vmax = vmax if vmax > 0 else 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)

        # Keep the H-panel colour bar in sync with the current wind-speed range
        if hasattr(self, "_wind_sm"):
            self._wind_sm.set_norm(norm)
            self._wind_sm.set_array(np.linspace(0.0, vmax, 256))
            if "H" in self._cbars:
                self._cbars["H"].update_normal(self._wind_sm)

        lw = 0.5 + 1.5 * (V_mag / vmax)
        patches_before = set(id(p) for p in ax.patches)
        sp = ax.streamplot(
            X, Y, U, V_y,
            color=V_mag, cmap=plt.cm.inferno, norm=norm,
            linewidth=lw, broken_streamlines=False,
            arrowsize=0.6, density=0.75,
        )
        new_patches = [p for p in ax.patches if id(p) not in patches_before]
        return (sp, new_patches)

    def _remove_contours(self, key: str) -> None:
        for cs in self._contours.get(key, []):
            try:
                if isinstance(cs, tuple):       # (StreamplotSet, [patches])
                    sp, patches = cs
                    sp.lines.remove()
                    for p in patches:
                        try:
                            p.remove()
                        except ValueError:
                            pass
                else:                           # QuadContourSet
                    cs.remove()
            except (ValueError, AttributeError):
                pass
        self._contours[key] = []

    # -- display flush ---------------------------------------------------------

    def _flush(self) -> None:
        """Push the updated figure to Jupyter cell output or the interactive canvas."""
        backend = matplotlib.get_backend().lower()
        if "inline" in backend:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
