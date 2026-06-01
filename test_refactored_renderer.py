import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import (
    Normalize, LogNorm, PowerNorm, SymLogNorm,
    LinearSegmentedColormap,
)
from fields import BaseModel

_SKIP_SUFFIXES = ("_grad_magnitude", "_grad_i", "_grad_j", "_magnitude")
_PLOT_ORDER = [
    "H", "Sun", "P",
    "Ta", "Ts", "Tw",
    "Wa", "Wc", "Ws",
    "Evap", "Condensation", "Precip",
]

class WorldRenderer(BaseModel):
    info = {
        "name": "renderer",
        "map_info": {}
    }

    def __init__(self, world, keys=None, cmaps=None, cols=3, figsize=None):
        super().__init__(world)
        self._extra_cmaps = cmaps or {}
        self.keys = keys if keys is not None else self._available_maps()
        if not self.keys:
            raise ValueError("No plottable maps found in world.area.")

        n = len(self.keys)
        self._cols = min(cols, n)
        self._rows = (n + self._cols - 1) // self._cols

        if figsize is None:
            figsize = (5.2 * self._cols, 4.2 * self._rows)

        self.fig, axes = plt.subplots(
            self._rows, self._cols,
            squeeze=False, figsize=figsize, layout="constrained"
        )
        self._axes = list(axes.flatten())

        self._state = {}  # key -> {"im": AxesImage, "cbar": Colorbar, "title": Text, "ax": Axes, "contours": list}
        
        self.fig.get_layout_engine().set(
            w_pad=0.01, h_pad=0.01, rect=[0, 0, 1, 0.95],
            hspace=0.01, wspace=0.01
        )
        
        self._build()

    def step(self):
        for key in self._state:
            self._update_panel(key)

        time_model = self.world.models.get("time")
        if time_model is not None:
            self.fig.suptitle(str(time_model), fontsize=11)
        self._flush()

    def _render_info(self, key):
        info = dict(self.world.map_info.get(key, {}).get("render", {}))
        if key in self._extra_cmaps:
            info["cmap"] = self._extra_cmaps[key]
        return info

    def _build_cmap(self, ri):
        cmap = plt.get_cmap(ri.get("cmap", "viridis")).copy()
        if ri.get("mask_sea", False):
            cmap.set_bad(alpha=0)
        return cmap

    def _make_norm(self, ri, vmin, vmax):
        scale = ri.get("scale", "linear")
        if scale == "log":
            return LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, max(vmin, 1e-6)*10))
        if scale == "power":
            return PowerNorm(gamma=ri.get("gamma", 0.4), vmin=vmin, vmax=vmax)
        if scale == "symlog":
            return SymLogNorm(linthresh=ri.get("linthresh", max(abs(vmax)*0.01, 1e-4)), vmin=vmin, vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    def _prepare_data(self, key):
        data = self.world.area[key]().copy().astype(np.float64)
        ri = self._render_info(key)

        if ri.get("mask_sea", False):
            data[self.world.area["M_sea"]().astype(bool)] = np.nan

        vrange = ri.get("vrange")
        if vrange is not None:
            return data, vrange[0], vrange[1]

        valid_data = data[~np.isnan(data)]
        if valid_data.size == 0:
            return data, 0.0, 1.0

        return data, float(valid_data.min()), float(valid_data.max())

    def _panel_title(self, key, vmin, vmax):
        unit = self.world.map_info.get(key, {}).get("unit", "")
        unit_str = f" ({unit})" if unit else ""
        return f"{key}{unit_str}  [{vmin:.2f} … {vmax:.2f}]"

    def _available_maps(self):
        result = [
            k for k in self.world.area.maps
            if not any(k.endswith(s) for s in _SKIP_SUFFIXES)
            and "render" in self.world.map_info.get(k, {})
            and self.world.area[k]().ndim == 2
        ]
        result.sort(key=lambda k: _PLOT_ORDER.index(k) if k in _PLOT_ORDER else 1e6)
        return result

    def _build(self):
        for i, key in enumerate(self.keys):
            ax = self._axes[i]
            self._state[key] = {"ax": ax, "contours": []}

            if key == "H":
                im = ax.imshow(self._height_rgba(), origin="lower", aspect="equal", interpolation="nearest")
                cbar = self._build_wind_colorbar(ax)
                title = ax.set_title(self._height_title(), fontsize=9)
            else:
                ri = self._render_info(key)
                data, vmin, vmax = self._prepare_data(key)
                im = ax.imshow(
                    data, cmap=self._build_cmap(ri),
                    norm=self._make_norm(ri, vmin, vmax), origin="lower", aspect="equal", interpolation="nearest"
                )
                cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01, shrink=0.7)
                title = ax.set_title(self._panel_title(key, vmin, vmax), fontsize=9)

            self._state[key].update({"im": im, "cbar": cbar, "title": title})
            self._draw_contours(key)

        for ax in self._axes[len(self.keys):]:
            ax.set_visible(False)

    def _update_panel(self, key):
        state = self._state[key]
        self._remove_contours(key)

        if key == "H":
            state["im"].set_data(self._height_rgba())
            state["title"].set_text(self._height_title())
        else:
            ri = self._render_info(key)
            data, vmin, vmax = self._prepare_data(key)
            state["im"].set_data(data)
            state["im"].set_norm(self._make_norm(ri, vmin, vmax))
            state["title"].set_text(self._panel_title(key, vmin, vmax))
        
        self._draw_contours(key)

    def _build_wind_colorbar(self, ax):
        self._wind_sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=Normalize(vmin=0.0, vmax=1.0))
        self._wind_sm.set_array([])
        cbar = self.fig.colorbar(self._wind_sm, ax=ax, fraction=0.046, pad=0.01, shrink=0.7)
        cbar.set_label("wind speed (m/s)", fontsize=7)
        return cbar

    def _height_rgba(self):
        H, sea_level, M_sea = self.world.area["H"]().copy(), float(self.world.area["sea_level"]()), self.world.area["M_sea"]().astype(bool)
        max_alt = float(self.world.max_altitude)
        H_m = (H - sea_level) * max_alt

        rgba = np.zeros((*H.shape, 4), dtype=np.float32)
        rgba[..., 3] = 1.0

        land_cmap = LinearSegmentedColormap.from_list("terrain_land", plt.cm.terrain(np.linspace(0.25, 1.0, 256)))
        rgba[~M_sea] = land_cmap(Normalize(vmin=0.0, vmax=max_alt)(np.clip(H_m, 0.0, None)))[~M_sea]
        
        sea_normed = 0.35 + 0.50 * (1.0 - Normalize(vmin=-max_alt, vmax=0.0)(np.clip(H_m, -max_alt, 0.0)))
        rgba[M_sea] = plt.cm.Blues(sea_normed)[M_sea]

        shadow = self.world.area["Shadow"]().copy()
        shade = 0.5 + 0.5 * np.clip(shadow, 0.0, 1.0)
        rgba[..., :3] = np.power(np.clip(rgba[..., :3] * shade[..., np.newaxis], 0.0, 1.0), 1.0 / 1.25)
        
        return rgba

    def _height_title(self):
        sea_level, max_alt = float(self.world.area["sea_level"]()), self.world.max_altitude
        return f"H  (m rel. sea level)  │  land: 0 … {(1.0 - sea_level) * max_alt:.0f} m  │  sea: {-sea_level * max_alt:.0f} … 0 m"

    def _draw_contours(self, key):
        ax = self._state[key]["ax"]
        H, sea_level, M_sea = self.world.area["H"]().copy(), float(self.world.area["sea_level"]()), self.world.area["M_sea"]().astype(bool)
        ri = self._render_info(key)
        
        # Determine base mesh for contouring depending on mask_sea
        H_plot = H - sea_level
        if ri.get("mask_sea", False):
            H_plot[M_sea] = np.nan

        overlays = []
        if key == "H" or not ri.get("mask_sea", False):
            overlays.append(ax.contour(H_plot, levels=10, colors="black", linewidths=0.4, alpha=0.45))
            if np.any(M_sea):
                overlays.append(ax.contour(M_sea.astype(float), levels=[0.5], colors="darkblue", linewidths=0.8, alpha=0.7))
        else:
            # When masking sea, perhaps we just want land terrain contours, which H_plot gives since sea is NaN.
            overlays.append(ax.contour(H_plot, levels=10, colors="black", linewidths=0.4, alpha=0.45))

        if key == "H":
            wind = self._draw_wind_overlay(ax)
            if wind: overlays.append(wind)

        self._state[key]["contours"].extend(overlays)

    def _draw_wind_overlay(self, ax):
        if "V" not in self.world.area.maps or "V_magnitude" not in self.world.area.maps:
            return None

        V, V_mag = self.world.area["V"]().copy(), self.world.area["V_magnitude"]().copy()
        if V.ndim != 3 or V.shape[-1] != 2: return None

        vmax = max(float(np.nanmax(V_mag)), 1e-6)
        norm = Normalize(vmin=0.0, vmax=vmax)

        if hasattr(self, "_wind_sm"):
            self._wind_sm.set_norm(norm)
            self._wind_sm.set_array(np.linspace(0.0, vmax, 256))
            if "H" in self._state:
                self._state["H"]["cbar"].update_normal(self._wind_sm)

        patches_before = {id(p) for p in ax.patches}
        sp = ax.streamplot(
            np.arange(V.shape[0]), np.arange(V.shape[1]), V[..., 0].astype(np.float64), V[..., 1].astype(np.float64),
            color=V_mag, cmap=plt.cm.inferno, norm=norm,
            linewidth=0.5 + 1.5 * (V_mag / vmax), broken_streamlines=False, arrowsize=0.6, density=0.75
        )
        return sp, [p for p in ax.patches if id(p) not in patches_before]

    def _remove_contours(self, key):
        for cs in self._state[key].get("contours", []):
            try:
                if isinstance(cs, tuple):
                    cs[0].lines.remove()
                    for p in cs[1]:
                        try: p.remove()
                        except ValueError: pass
                else: cs.remove()
            except (ValueError, AttributeError): pass
        self._state[key]["contours"] = []

    def _flush(self):
        backend = matplotlib.get_backend().lower()
        if "inline" in backend:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
