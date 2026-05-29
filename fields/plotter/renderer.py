from fields import BaseModel

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


# Fallback colormaps for maps that don't define render.cmap in their map_info.
_DEFAULT_CMAPS: dict[str, str] = {
    "H":            "terrain",
    "Sun":          "hot",
    "P":            "RdBu_r",
    "Ta":           "RdYlBu_r",
    "Ts":           "RdYlBu_r",
    "Tw":           "Blues_r",
    "Wa":           "YlGnBu",
    "Wa_max":       "YlGnBu",
    "Wc":           "Blues",
    "Ws":           "YlGn",
    "Evap":         "PuBu",
    "Condensation": "PuBuGn",
    "Precip":       "Blues",
}

# Auto-generated derivative maps always skipped regardless of render.plot.
_SKIP_SUFFIXES = ("_grad_magnitude", "_grad_i", "_grad_j", "_magnitude")


class WorldRenderer(BaseModel):
    info = {
        'name': 'renderer',
        'map_info': {}
    }
    """Lightweight real-time renderer attached to a ``fields.World`` instance.

    Render settings (cmap, plot flag, vrange) are read from each map's
    ``map_info[key]['render']`` dict.  Colormaps fall back to ``_DEFAULT_CMAPS``
    and finally ``'viridis'`` when not specified.

    All panels get terrain-contour and sea-boundary overlays automatically.
    The height panel uses a composite land/sea rendering instead of a plain cmap.
    """

    def __init__(
        self,
        world,
        keys: list[str] | None = None,
        cmaps: dict[str, str] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
        cols: int = 3,
        figsize: tuple[float, float] | None = None,
        rescale: bool = True,
    ):
        super().__init__(world)
        self._extra_cmaps = cmaps or {}   # caller overrides, highest priority
        self.ranges       = dict(ranges or {})
        self.rescale      = rescale

        self.keys = keys if keys is not None else self._available_maps()
        if not self.keys:
            raise ValueError("No plottable maps found in world.area. "
                             "Run at least one model step first.")

        n = len(self.keys)
        self._cols = min(cols, n)
        self._rows = (n + self._cols - 1) // self._cols

        if figsize is None:
            figsize = (5.2 * self._cols, 4.2 * self._rows)

        self.fig, _axes = plt.subplots(
            self._rows, self._cols,
            squeeze=False,
            figsize=figsize,
        )
        self._axes: list = list(_axes.flatten())

        self._ims:      dict = {}   # key -> AxesImage
        self._cbars:    dict = {}   # key -> Colorbar
        self._titles:   dict = {}   # key -> Text
        self._ax_map:   dict = {}   # key -> Axes
        self._contours: dict = {}   # key -> [matplotlib collections]

        self._build()
        # Reserve top margin so the suptitle (time) is never hidden by the subplots.
        self.fig.tight_layout(rect=[0, 0, 1, 0.93])

    # ── public API ────────────────────────────────────────────────────────────



    def step(self, rescale: bool = True) -> None:
        """Redraw all panels with the current world state (no simulation step)."""
        do_rescale = self.rescale if rescale is None else rescale

        for key, im in self._ims.items():
            ax = self._ax_map[key]

            if key == 'H':
                im.set_data(self._height_rgba())
                self._titles[key].set_text(self._height_title())
            else:
                data = self.world.area[key]().copy()
                if data.ndim != 2:
                    continue
                im.set_data(data)
                if do_rescale and key not in self.ranges:
                    vmin, vmax = self._clim(key, data)
                    im.set_clim(vmin, vmax)
                self._titles[key].set_text(
                    f"{key}  [{float(np.nanmin(data)):.2f} \u2026 {float(np.nanmax(data)):.2f}]"
                )

            # Refresh contour overlays
            self._remove_contours(key)
            self._draw_contours(ax, key)

        time_model = self.world.models.get("time")
        if time_model is not None:
            self.fig.suptitle(str(time_model), fontsize=11)

        self._flush()

    # ── internal setup ────────────────────────────────────────────────────────

    def _available_maps(self) -> list[str]:
        result = []
        for k in self.world.area.maps:
            if any(k.endswith(s) for s in _SKIP_SUFFIXES):
                continue
            if 'render' not in self.world.map_info.get(k, {}):  # no render dict = don't plot
                continue
            if self.world.area[k]().ndim != 2:
                continue
            result.append(k)
        return result

    def _get_cmap(self, key: str) -> str:
        """Cmap priority: caller override > map_info render.cmap > _DEFAULT_CMAPS > viridis."""
        if key in self._extra_cmaps:
            return self._extra_cmaps[key]
        render = self.world.map_info.get(key, {}).get('render', {})
        return render.get('cmap', _DEFAULT_CMAPS.get(key, 'viridis'))

    def _build(self) -> None:
        for i, key in enumerate(self.keys):
            ax = self._axes[i]
            self._ax_map[key] = ax
            data = self.world.area[key]().copy()

            if data.ndim != 2:
                ax.set_title(f"{key}\n(vector field \u2013 skipped)", fontsize=9)
                ax.axis("off")
                continue

            if key == 'H':
                im = ax.imshow(self._height_rgba(), origin="lower",
                               aspect="equal", interpolation="nearest")
                title_text = self._height_title()
            else:
                cmap       = self._get_cmap(key)
                vmin, vmax = self._clim(key, data)
                norm       = Normalize(vmin=vmin, vmax=vmax)
                im = ax.imshow(data, cmap=cmap, norm=norm, origin="lower",
                               aspect="equal", interpolation="nearest")
                title_text = (f"{key}  [{float(np.nanmin(data)):.2f} \u2026"
                              f" {float(np.nanmax(data)):.2f}]")

            # For the composite H panel, the AxesImage has no norm so its
            # colorbar would default to [0, 1].  Instead build a proxy
            # ScalarMappable using the fixed [0, max_altitude] land scale.
            if key == 'H':
                _lcmap = LinearSegmentedColormap.from_list(
                    "_tcb", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
                )
                _sm = plt.cm.ScalarMappable(
                    cmap=_lcmap, norm=Normalize(vmin=0.0, vmax=float(self.world.max_altitude))
                )
                _sm.set_array([])
                cbar = self.fig.colorbar(_sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("land alt (m)", fontsize=7)
            else:
                cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._ims[key]    = im
            self._cbars[key]  = cbar
            self._titles[key] = ax.set_title(title_text, fontsize=9)

            self._draw_contours(ax, key)

        for ax in self._axes[len(self.keys):]:
            ax.set_visible(False)

    # ── height composite rendering ────────────────────────────────────────────

    def _height_rgba(self) -> np.ndarray:
        """Composite RGBA using metres relative to sea level (0 = coast).

        Land: terrain cmap on fixed scale [0, max_altitude] metres.
        Sea:  Blues cmap on fixed scale [-max_altitude, 0] metres.
        Sun/shadow shading applied on top when the Shadow map is available.
        """
        H         = self.world.area['H']().copy()
        sea_level = float(self.world.area['sea_level']())
        M_sea     = self.world.area['M_sea']().astype(bool)
        max_alt   = float(self.world.max_altitude)

        H_m = (H - sea_level) * max_alt   # metres rel. sea level

        rows, cols = H.shape
        rgba = np.zeros((rows, cols, 4), dtype=np.float32)
        rgba[..., 3] = 1.0

        # Land — fixed scale [0, max_alt]
        land_cmap = LinearSegmentedColormap.from_list(
            "terrain_land", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
        )
        land_norm = Normalize(vmin=0.0, vmax=max_alt)
        land_rgba = land_cmap(land_norm(np.clip(H_m, 0.0, None)))
        rgba[~M_sea] = land_rgba[~M_sea]

        # Sea — fixed scale [-max_alt, 0], deeper → darker
        sea_norm   = Normalize(vmin=-max_alt, vmax=0.0)
        sea_normed = 1.0 - sea_norm(np.clip(H_m, -max_alt, 0.0))
        sea_normed = 0.35 + 0.50 * sea_normed
        sea_rgba   = plt.cm.Blues(sea_normed)
        rgba[M_sea] = sea_rgba[M_sea]

        # Sun/shadow shading: ambient + directional from Shadow map

        ambient = 0.5
        shadow = self.world.area['Shadow']().copy() 
        
        shade = ambient + (1.0 - ambient) * np.clip(shadow, 0.0, 1.0)
        rgba[..., :3] = np.clip(rgba[..., :3] * shade[..., np.newaxis], 0.0, 1.0)

        #apply gamma correction for better contrast in shadows and highlights
        contrast = 1.25
        gamma = 1.0 / contrast
        rgba[..., :3] = np.power(rgba[..., :3], gamma)

        return rgba

    def _height_title(self) -> str:
        """Title using fixed [0, max_altitude] scale (no data scan needed)."""
        sea_level = float(self.world.area['sea_level']())
        max_alt   = self.world.max_altitude
        land_max  = (1.0 - sea_level) * max_alt
        sea_min   = -sea_level * max_alt
        return (f"H  (m rel. sea level) │ "
                f"land: 0 … {land_max:.0f} m  │  sea: {sea_min:.0f} … 0 m")

    # ── contour overlays ──────────────────────────────────────────────────────

    def _draw_contours(self, ax, key: str) -> None:
        """Overlay terrain contours and sea boundary on any panel."""
        H         = self.world.area['H']().copy()
        sea_level = float(self.world.area['sea_level']())
        M_sea     = self.world.area['M_sea']().astype(bool)

        # Store QuadContourSet objects directly (.collections was removed in mpl 3.8)
        contour_sets = []

        # Terrain contours relative to sea level (thin black)
        cs = ax.contour(H - sea_level, levels=10, colors='black', linewidths=0.4, alpha=0.45)
        contour_sets.append(cs)

        # Sea boundary (thicker dark blue)
        if np.any(M_sea):
            sb = ax.contour(M_sea.astype(float), levels=[0.5],
                            colors='darkblue', linewidths=0.8, alpha=0.7)
            contour_sets.append(sb)

        self._contours[key] = contour_sets

    def _remove_contours(self, key: str) -> None:
        for cs in self._contours.get(key, []):
            try:
                cs.remove()
            except (ValueError, AttributeError):
                pass
        self._contours[key] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _clim(self, key: str, data: np.ndarray) -> tuple[float, float]:
        """Return (vmin, vmax): caller ranges > map_info render.vrange > auto.

        To fix a map's range via map_info, add ``'vrange': (vmin, vmax)`` to its
        ``render`` dict, e.g. ``'render': {'cmap': 'RdYlBu_r', 'vrange': (-30, 50)}``.
        """
        if key in self.ranges:
            return self.ranges[key]
        vrange = self.world.map_info.get(key, {}).get('render', {}).get('vrange', None)
        if vrange is not None:
            return vrange
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if vmin == vmax:
            vmax = vmin + 1.0
        return vmin, vmax

    def _flush(self) -> None:
        """Push the updated figure to the Jupyter cell output."""
        backend = matplotlib.get_backend().lower()
        if "inline" in backend:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
