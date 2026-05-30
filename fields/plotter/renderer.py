from fields import BaseModel

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import (
    Normalize, LogNorm, PowerNorm, SymLogNorm,
    LinearSegmentedColormap,
)

# Auto-generated derivative maps always skipped regardless of render.plot.
_SKIP_SUFFIXES = ("_grad_magnitude", "_grad_i", "_grad_j", "_magnitude")


class WorldRenderer(BaseModel):
    """Lightweight real-time renderer attached to a ``fields.World`` instance.

    Render settings are read from each map's ``map_info[key]['render']`` dict:
      - ``cmap``      – matplotlib colormap name (default ``'viridis'``)
      - ``vrange``    – ``(vmin, vmax)`` tuple to pin the color scale
      - ``scale``     – ``'linear'`` (default), ``'log'``, ``'power'``, or ``'symlog'``
      - ``gamma``     – exponent for ``'power'`` scale (default ``0.4``)
      - ``linthresh`` – linear threshold for ``'symlog'`` scale

    All panels get terrain-contour and sea-boundary overlays automatically.
    The height panel uses a composite land/sea rendering with shadow shading.
    """

    info = {
        'name': 'renderer',
        'map_info': {}
    }

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
            squeeze=True,
            figsize=figsize,
            layout='constrained',
        )
        self._axes: list = list(_axes.flatten())

        self._ims:      dict = {}   # key -> AxesImage
        self._cbars:    dict = {}   # key -> Colorbar
        self._titles:   dict = {}   # key -> Text
        self._ax_map:   dict = {}   # key -> Axes
        self._contours: dict = {}   # key -> list[QuadContourSet]

        self._build()
        
        # w_pad and h_pad control spacing between panels (in inches)
        self.fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, rect=[0, 0, 1, 0.95], hspace=0.01, wspace=0.01)


    # ── public API ────────────────────────────────────────────────────────────

    def step(self, rescale: bool = True) -> None:
        """Redraw all panels with the current world state (no simulation step)."""
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
                if rescale and key not in self.ranges:
                    vmin, vmax = self._clim(key, data)
                    im.set_norm(self._make_norm(key, vmin, vmax))
                self._titles[key].set_text(self._panel_title(key, data))

            self._remove_contours(key)
            self._draw_contours(ax, key)

        time_model = self.world.models.get("time")
        if time_model is not None:
            self.fig.suptitle(str(time_model), fontsize=11)

        self._flush()

    # ── render-config helpers ─────────────────────────────────────────────────

    def _render_info(self, key: str) -> dict:
        """Merged render dict: map_info defaults + caller cmap override."""
        info = dict(self.world.map_info.get(key, {}).get('render', {}))
        if key in self._extra_cmaps:
            info['cmap'] = self._extra_cmaps[key]
        return info

    def _get_cmap(self, key: str) -> str:
        return self._render_info(key).get('cmap', 'viridis')

    def _clim(self, key: str, data: np.ndarray) -> tuple[float, float]:
        """Return (vmin, vmax): caller ranges > render.vrange > auto from data."""
        if key in self.ranges:
            return self.ranges[key]
        vrange = self._render_info(key).get('vrange')
        if vrange is not None:
            return tuple(vrange)
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        return (vmin, vmax + 1.0) if vmin == vmax else (vmin, vmax)

    def _make_norm(self, key: str, vmin: float, vmax: float):
        """Build the appropriate matplotlib Normalize subclass for the map's scale."""
        ri    = self._render_info(key)
        scale = ri.get('scale', 'linear')

        if scale == 'log':
            safe_min = max(vmin, 1e-6)
            safe_max = max(vmax, safe_min * 10)
            return LogNorm(vmin=safe_min, vmax=safe_max)

        if scale == 'power':
            gamma = ri.get('gamma', 0.4)
            return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

        if scale == 'symlog':
            linthresh = ri.get('linthresh', max(abs(vmax) * 0.01, 1e-4))
            return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

        return Normalize(vmin=vmin, vmax=vmax)  # 'linear' (default)

    def _panel_title(self, key: str, data: np.ndarray) -> str:
        """Format: 'KEY (unit)  [min … max]'."""
        info     = self.world.map_info.get(key, {})
        unit     = info.get('unit', '')
        unit_str = f" ({unit})" if unit else ""
        return (f"{key}{unit_str}  "
                f"[{float(np.nanmin(data)):.2f} \u2026 {float(np.nanmax(data)):.2f}]")

    # ── internal setup ────────────────────────────────────────────────────────

    def _available_maps(self) -> list[str]:
        result = []
        for k in self.world.area.maps:
            if any(k.endswith(s) for s in _SKIP_SUFFIXES):
                continue
            if 'render' not in self.world.map_info.get(k, {}):
                continue
            if self.world.area[k]().ndim != 2:
                continue
            result.append(k)
        return result

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
                im         = ax.imshow(self._height_rgba(), origin="lower",
                                       aspect="equal", interpolation="nearest")
                title_text = self._height_title()
                cbar       = self._build_height_colorbar(ax)
            else:
                vmin, vmax = self._clim(key, data)
                norm       = self._make_norm(key, vmin, vmax)
                im = ax.imshow(data, cmap=self._get_cmap(key), norm=norm,
                               origin="lower", aspect="equal", interpolation="nearest")
                title_text = self._panel_title(key, data)
                cbar       = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01, shrink=0.7)

            self._ims[key]    = im
            self._cbars[key]  = cbar
            self._titles[key] = ax.set_title(title_text, fontsize=9)
            self._draw_contours(ax, key)

        for ax in self._axes[len(self.keys):]:
            ax.set_visible(False)

    def _build_height_colorbar_old(self, ax):
        """Proxy colorbar for the composite H panel (land scale only)."""
        _lcmap = LinearSegmentedColormap.from_list(
            "_tcb", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
        )
        sm = plt.cm.ScalarMappable(
            cmap=_lcmap,
            norm=Normalize(vmin=0.0, vmax=float(self.world.max_altitude)),
        )
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("land alt (m)", fontsize=7)
        return cbar

    def _build_height_colorbar(self, ax):
        cmap = plt.cm.inferno
        # Use a placeholder norm; will be updated on first real draw via _draw_contours
        self._wind_sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=1.0))
        self._wind_sm.set_array([])
        cbar = self.fig.colorbar(self._wind_sm, ax=ax, fraction=0.046, pad=0.01, shrink=0.7)
        cbar.set_label("wind speed (m/s)", fontsize=7)
        return cbar

    # ── height composite rendering ────────────────────────────────────────────

    def _height_rgba(self) -> np.ndarray:
        """Composite RGBA: terrain cmap for land, Blues depth for sea, shadow shading."""
        H         = self.world.area['H']().copy()
        sea_level = float(self.world.area['sea_level']())
        M_sea     = self.world.area['M_sea']().astype(bool)
        max_alt   = float(self.world.max_altitude)

        H_m  = (H - sea_level) * max_alt   # metres relative to sea level
        rgba = np.zeros((*H.shape, 4), dtype=np.float32)
        rgba[..., 3] = 1.0

        # Land — fixed scale [0, max_alt] m
        land_cmap  = LinearSegmentedColormap.from_list(
            "terrain_land", plt.cm.terrain(np.linspace(0.25, 1.0, 256))
        )
        land_norm  = Normalize(vmin=0.0, vmax=max_alt)
        land_rgba  = land_cmap(land_norm(np.clip(H_m, 0.0, None)))
        rgba[~M_sea] = land_rgba[~M_sea]

        # Sea — fixed scale [-max_alt, 0], deeper → darker blue
        sea_normed = 0.35 + 0.50 * (
            1.0 - Normalize(vmin=-max_alt, vmax=0.0)(np.clip(H_m, -max_alt, 0.0))
        )
        rgba[M_sea] = plt.cm.Blues(sea_normed)[M_sea]

        # Shadow / hillshade
        ambient = 0.5
        shadow  = self.world.area['Shadow']().copy()
        shade   = ambient + (1.0 - ambient) * np.clip(shadow, 0.0, 1.0)
        rgba[..., :3] = np.clip(rgba[..., :3] * shade[..., np.newaxis], 0.0, 1.0)

        # Gamma for better shadow/highlight contrast
        rgba[..., :3] = np.power(rgba[..., :3], 1.0 / 1.25)
        return rgba

    def _height_title(self) -> str:
        sea_level = float(self.world.area['sea_level']())
        max_alt   = self.world.max_altitude
        land_max  = (1.0 - sea_level) * max_alt
        sea_min   = -sea_level * max_alt
        return (f"H  (m rel. sea level)"
                f"  \u2502  land: 0 \u2026 {land_max:.0f} m"
                f"  \u2502  sea: {sea_min:.0f} \u2026 0 m")

    # ── contour overlays ──────────────────────────────────────────────────────

    def _draw_contours(self, ax, key: str) -> None:
        """Terrain contours + coastline on every panel."""
        H         = self.world.area['H']().copy()
        sea_level = float(self.world.area['sea_level']())
        M_sea     = self.world.area['M_sea']().astype(bool)

        contour_sets = [
            ax.contour(H - sea_level, levels=10,
                       colors='black', linewidths=0.4, alpha=0.45),
        ]
        if np.any(M_sea):
            contour_sets.append(
                ax.contour(M_sea.astype(float), levels=[0.5],
                           colors='darkblue', linewidths=0.8, alpha=0.7)
            )

        if key == 'H' and 'V' in self.world.area.maps:
            V = self.world.area['V']().copy()
            V_mag = self.world.area['V_magnitude']().copy()
            if V.ndim == 3 and V.shape[-1] == 2:
                rows, cols = V.shape[:2]
                Y = np.arange(cols)
                X = np.arange(rows)
                V_y = V[..., 1].astype(np.float64)   # j-component → screen x (horizontal)
                U = V[..., 0].astype(np.float64) # i-component → screen y (vertical)
                cmap = plt.cm.inferno
                vmax = float(np.nanmax(V_mag))
                vmax = vmax if vmax > 0 else 1.0
                norm = Normalize(vmin=0.0, vmax=vmax)
                # Keep colorbar in sync with the current wind-speed range
                if hasattr(self, '_wind_sm'):
                    self._wind_sm.set_norm(norm)
                    self._wind_sm.set_array(np.linspace(0.0, vmax, 256))
                    if 'H' in self._cbars:
                        self._cbars['H'].update_normal(self._wind_sm)
                lw = 0.5 + 1.5 * (V_mag / vmax)
                patches_before = set(id(p) for p in ax.patches)
                sp = ax.streamplot(X, Y, U, V_y, color=V_mag, cmap=cmap, norm=norm, linewidth=lw,
                                   broken_streamlines=False, arrowsize=0.6, density=0.75)
                new_patches = [p for p in ax.patches if id(p) not in patches_before]
                contour_sets.append((sp, new_patches))

        self._contours[key] = contour_sets

    def _remove_contours(self, key: str) -> None:
        ax = self._ax_map.get(key)
        for cs in self._contours.get(key, []):
            try:
                if isinstance(cs, tuple):  # (StreamplotSet, [patches])
                    sp, patches = cs
                    sp.lines.remove()
                    for p in patches:
                        try:
                            p.remove()
                        except ValueError:
                            pass
                elif hasattr(cs, 'lines') and hasattr(cs, 'arrows'):
                    cs.lines.remove()
                    try:
                        cs.arrows.remove()
                    except (NotImplementedError, ValueError):
                        pass
                else:
                    cs.remove()
            except (ValueError, AttributeError):
                pass
        self._contours[key] = []

    # ── display flush ─────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Push updated figure to Jupyter cell output or interactive canvas."""
        backend = matplotlib.get_backend().lower()
        if "inline" in backend:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
