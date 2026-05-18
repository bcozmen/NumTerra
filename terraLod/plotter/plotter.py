import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

from ..helper import normalize, get_grid
from .shade import get_3D_shade, hillshade
from .helper import get_cell_size, set_labels, find_z_limits




class Plotter():
    def __init__(self, plotter_params):
        for key in plotter_params:
            setattr(self, key, plotter_params.get(key, plotter_params[key]))
    def plot(self, height_map, lim=(0.0, 1.0, 0.0, 1.0), 
                masks = None,save_path=None, shade = True, plot_slope_histogram = True):
        
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.05)

        z = height_map * self.max_altitude
        cell_size, max_range = get_cell_size(lim, self.max_size, height_map.shape)
        dzdy, dzdx = np.gradient(z, cell_size[1], cell_size[0])
        gradients = (dzdy, dzdx)

        ax1 = fig.add_subplot(gs[0, 0])
        self.plot2D(height_map, gradients, ax=ax1, lim=lim, shade=shade, masks=masks)

        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        self.plot3D(height_map, gradients, ax=ax2, lim=lim, shade=shade, masks=masks)

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        if plot_slope_histogram:
            self.plot_slope_histogram(height_map, gradients, lim=lim)
    def plot3D(self, height_map, gradients, ax = None, lim = (0.0, 1.0, 0.0,1.0), shade = True, masks=None):
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        x, y = get_grid(lim=lim, shape=height_map.shape)
        cell_size, max_range = get_cell_size(lim, self.max_size, height_map.shape)

        base_color = plt.cm.terrain(height_map)[..., :3]  # RGB only
        terrain_shade = None
        if shade:
            terrain_shade = get_3D_shade(self, height_map, gradients, lim, self.max_size, self.max_altitude, self.ambient)
            base_color = base_color * terrain_shade

        # Paint water onto base_color, passing terrain_shade so valley
        # shadows are preserved under water.
        if masks is not None:
            base_color = self._paint_water_colors(base_color, masks, terrain_shade)

        ax.plot_surface(x, y, height_map, facecolors=base_color, linewidth=0, antialiased=True)

        z_lim = find_z_limits(height_map, lim, height_map.shape, self.max_size)
        set_labels(ax, zlim=z_lim, z_label='Height (normalized)', title='3D Terrain')
        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        return ax


    def plot2D(self, height_map, gradients, ax= None, lim = (0.0, 1.0, 0.0,1.0), shade = True, masks=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(height_map, cmap='terrain', extent=lim, origin='lower', vmin=-0.25, vmax=1)

        # Draw water BEFORE hillshade so that terrain shadows and valley
        # darkness naturally fall on top of water — this gives rivers/lakes
        # in valleys the correct shadow for free.
        if masks is not None:
            self._overlay_water_masks(ax, masks, lim)

        if shade:
            hillshade_map = hillshade(height_map, gradients, lim, self.max_altitude, self.max_size, self.shade_azim, self.shade_elev)
            ax.imshow(hillshade_map, cmap='gray', extent=lim, origin='lower', alpha=0.45)

        set_labels(ax, z_label=None, title='Height Map with Hillshade')
        return ax

    # ------------------------------------------------------------------
    # Water-mask overlay helpers
    # ------------------------------------------------------------------

    # Colour palette — RGB in [0,1]
    _WATER_COLOURS = {
        'sea':   np.array([0.00, 0.2, 0.5]),   # deep ocean blue
        'lake':  np.array([0.00, 0.25, 0.85]),   # lighter lake blue
        'river': np.array([0.25, 0.60, 1.00]),   # bright stream cyan
    }

    def _paint_water_colors(self, base_color, masks, terrain_shade=None):
        """Return a copy of *base_color* (H×W×3) with water cells painted.

        The terrain shade multiplier is preserved under water so valley
        shadows remain visible.  A specular glint is added on top.
        """
        out = base_color.copy()
        c   = self._WATER_COLOURS
        blend = 0.80

        water_shade = self._water_shade_map(masks, terrain_shade)

        def _apply(mask_float, rgb):
            alpha  = np.clip(mask_float, 0.0, 1.0)[..., None] * blend
            colour = np.clip(rgb[None, None, :] * water_shade[..., None], 0.0, 1.0)
            out[:] = out * (1.0 - alpha) + colour * alpha

        if 'sea_mask'   in masks: _apply(masks['sea_mask'].astype(np.float32),   c['sea'])
        if 'lake_mask'  in masks: _apply(masks['lake_mask'].astype(np.float32),  c['lake'])
        if 'river_mask' in masks: _apply(masks['river_mask'],                    c['river'])
        return out

    def _water_shade_map(self, masks, terrain_shade=None):
        """(H×W) float32 shading multiplier for water surfaces.

        Three effects stacked:
        1. **Terrain shadow inheritance** — reuse the terrain hillshade value
           at each water cell so rivers in valleys are dark, rivers on lit
           slopes are bright.
        2. **Depth darkening** — water far from shore is darker (absorption).
        3. **Specular glint** — Phong highlight using light dir (shade_azim/elev)
           and view dir (azim/elev from plotter params).
        """
        from scipy.ndimage import distance_transform_edt

        ref = next(iter(masks.values()))
        h, w = ref.shape

        # 1. Start from terrain shade (if provided) or neutral 1.0
        if terrain_shade is not None:
            # get_3D_shade may return (H, W, 1) — squeeze to (H, W) to avoid
            # accidental outer-product broadcasts when boolean-indexing.
            ts = np.asarray(terrain_shade, dtype=np.float32)
            if ts.ndim == 3:
                ts = ts[..., 0]
            shade = ts.copy()
        else:
            shade = np.ones((h, w), dtype=np.float32)

        # 2. Depth darkening for ocean / lakes
        for key, dark_strength in (('sea_mask', 0.45), ('lake_mask', 0.30)):
            if key not in masks:
                continue
            body = np.asarray(masks[key]).squeeze()   # ensure exactly (H, W)
            body = body.astype(np.bool_)
            if not np.any(body):
                continue
            dist_shore = distance_transform_edt(body)                # (H, W) float64
            max_d = float(dist_shore.max()) + 1e-6
            vals  = shade[body] * (1.0 - dark_strength * (dist_shore[body] / max_d))
            shade[body] = vals.astype(np.float32)

        # 3. Specular glint — Blinn-Phong on flat water (N = (0,0,1))
        #    Light direction L uses shade_azim / shade_elev (sun position).
        #    View  direction V uses azim     / elev      (camera position).
        #    Half-vector H = normalise(L + V).
        #    Specular = max(0, N·H)^shininess.
        light_az = np.radians(getattr(self, 'shade_azim', 45))
        light_el = np.radians(getattr(self, 'shade_elev', 30))
        view_az  = np.radians(getattr(self, 'azim',  -210))
        view_el  = np.radians(getattr(self, 'elev',   30))

        def _dir(az, el):
            return np.array([
                np.cos(el) * np.sin(az),
                np.cos(el) * np.cos(az),
                np.sin(el),
            ])

        L = _dir(light_az, light_el)
        V = _dir(view_az,  view_el)
        H = L + V
        H_norm = H / (np.linalg.norm(H) + 1e-8)
        # N = (0, 0, 1) so N·H = H_norm[2]
        spec_val = float(max(0.0, H_norm[2]) ** 20) * 0.45

        water_any = np.zeros((h, w), dtype=np.bool_)
        for key, val in masks.items():
            water_any |= np.asarray(val).squeeze() > 0.05
        shade[water_any] = np.clip(shade[water_any] + spec_val, 0.0, 1.5)

        return shade

    def _overlay_water_masks(self, ax, masks, lim):
        """Composite sea → lake → river layers onto *ax* using RGBA images.

        Water shading (depth darkening + specular) is baked into each pixel's
        colour before compositing, so rivers in shadowed valleys look dark
        automatically (because hillshade is drawn on top of this layer).
        """
        h = next(iter(masks.values())).shape[0]
        w = next(iter(masks.values())).shape[1]

        # No terrain_shade available here — hillshade is drawn on top instead,
        # which achieves the same valley-shadow effect.
        water_shade = self._water_shade_map(masks, terrain_shade=None)

        def _draw(mask_float, rgb, base_alpha=1.0):
            alpha  = np.asarray(mask_float, dtype=np.float32) * base_alpha
            colour = np.clip(rgb[None, None, :] * water_shade[..., None], 0.0, 1.0)
            img    = np.zeros((h, w, 4), dtype=np.float32)
            img[..., :3] = colour
            img[..., 3]  = alpha
            ax.imshow(img, extent=lim, origin='lower', interpolation='nearest')

        c = self._WATER_COLOURS
        if 'sea_mask'   in masks: _draw(masks['sea_mask'],   c['sea'],   base_alpha=1.0)
        if 'lake_mask'  in masks: _draw(masks['lake_mask'],  c['lake'],  base_alpha=1.0)
        if 'river_mask' in masks: _draw(masks['river_mask'], c['river'], base_alpha=1.0)

        from matplotlib.patches import Patch
        legend_items = []
        if 'sea_mask'   in masks: legend_items.append(Patch(color=(*c['sea'],   1.0), label='Ocean'))
        if 'lake_mask'  in masks: legend_items.append(Patch(color=(*c['lake'],  1.0), label='Lake'))
        if 'river_mask' in masks: legend_items.append(Patch(color=(*c['river'], 1.0), label='River'))
        if legend_items:
            ax.legend(handles=legend_items, loc='lower right', framealpha=0.8, fontsize=8)
    def plot_slope_histogram(self, height_map, gradients, lim=(0.0, 1.0, 0.0, 1.0), ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        cell_size, max_range = get_cell_size(lim, self.max_size, height_map.shape)
        
        z = height_map * self.max_altitude
        dzdy, dzdx = gradients
        slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope = np.degrees(slope) #convert slope to degrees for better interpretability

        ax.hist(slope.flatten(), bins=100)
        set_labels(ax, x_label='Slope (degrees)', y_label='Frequency', title='Slope Histogram', z_label=None)
        return ax

