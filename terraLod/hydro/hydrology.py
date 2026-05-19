import numpy as np
from scipy.ndimage import label

from .helper import (
    detect_sea,
    compute_mfd_weights,
    compute_mfd_accumulation,
    compute_lake_mask,
    compute_d8_downstream,
    compute_river_paths,
    rasterize_river_paths,
)

class Hydrology:
    """
    MFD (Multiple Flow Direction) hydrology.

    Pipeline
    --------
    1. sea_mask               — border flood-fill BFS
    2. flow_weights           — MFD slopes, normalised per cell
    3. flow_acc               — accumulation in descending-elevation order
    4. lake_mask              — priority-flood; depressions where fill > height.
                                Small lakes with no river input are removed.
    5. river_{coords,         — D8 centre-line paths as a flat numba array:
          path_starts,          all points concatenated + per-path start/length.
          path_lengths}         Passed directly to the numba rasterizer.
    6. river_mask             — base-resolution rasterization of above

    Infinite zoom
    -------------
    Call `set_interpolators(height_interp, fill_interp)` once, then call
    `get_masks_at(lim, shape)` at any zoom.  Lakes use interpolated fill_level
    vs height; rivers use the numba Bresenham rasterizer on stored paths —
    both are sharp at any resolution with no re-simulation.
    """

    def __init__(self, height_map, sea_level_percentile, river_threshold,
                 min_lake_river_acc=None):
        """
        Parameters
        ----------
        height_map            : float32 array (xdim, ydim)
        sea_level_percentile  : float in (0, 1)
        river_threshold       : int  — min flow_acc to define a river cell
        min_lake_river_acc    : int or None
            Lakes whose peak inflow is below this are evaporated.
            Defaults to `river_threshold`.
        """
        if min_lake_river_acc is None:
            min_lake_river_acc = river_threshold

        self.height_map      = height_map
        self.xdim, self.ydim = height_map.shape
        self.river_threshold = river_threshold

        self.sea_level = np.percentile(height_map, sea_level_percentile * 100)
        self.sea_mask  = detect_sea(height_map, self.sea_level)

        self.flow_weights  = compute_mfd_weights(height_map)
        self.flow_acc      = compute_mfd_accumulation(height_map, self.flow_weights)
        self.d8_downstream = compute_d8_downstream(self.flow_weights)

        raw_lake_mask, self.fill_level = compute_lake_mask(height_map, self.sea_mask)

        self.lake_mask = self._filter_small_lakes(
            raw_lake_mask, self.flow_acc, min_lake_river_acc
        )

        # Flat numba-friendly path representation
        self.river_coords, self.river_path_starts, self.river_path_lengths = \
            compute_river_paths(
                self.d8_downstream, self.flow_acc,
                self.sea_mask, self.lake_mask, river_threshold
            )

        self.river_mask = rasterize_river_paths(
            self.river_coords, self.river_path_starts, self.river_path_lengths,
            0.0, 1.0, 0.0, 1.0, self.xdim, self.ydim
        )

        self._height_interp = None
        self._fill_interp   = None

        print(f"sea   cells : {np.sum(self.sea_mask):>8}")
        print(f"lake  cells : {np.sum(self.lake_mask):>8}")
        print(f"river paths : {len(self.river_path_starts):>8}")
        print(f"river cells : {np.sum(self.river_mask):>8}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_small_lakes(lake_mask, flow_acc, min_acc):
        """
        Remove connected lake components whose maximum inflow (flow_acc)
        is below `min_acc`.  Vectorized with np.maximum.at — O(n) single pass,
        no Python loop over label components.
        """
        labeled, n_labels = label(lake_mask)
        if n_labels == 0:
            return lake_mask

        label_max = np.zeros(n_labels + 1, dtype=flow_acc.dtype)
        np.maximum.at(label_max, labeled.ravel(), flow_acc.ravel())
        keep = label_max >= min_acc   # shape (n_labels+1,), index 0 = background
        keep[0] = False               # index 0 is background — never a lake
        return keep[labeled]          # broadcast to grid shape

    # ------------------------------------------------------------------
    # Interpolator injection (called by Terrain after building interps)
    # ------------------------------------------------------------------

    def set_interpolators(self, height_interp, fill_interp):
        """
        Inject RegularGridInterpolators so get_masks_at() works at any zoom.

        Parameters
        ----------
        height_interp : RegularGridInterpolator over height_map  in [0,1]²
        fill_interp   : RegularGridInterpolator over fill_level  in [0,1]²
        """
        self._height_interp = height_interp
        self._fill_interp   = fill_interp

    # ------------------------------------------------------------------
    # Zoom-level mask generation
    # ------------------------------------------------------------------

    def get_masks_at(self, lim, shape):
        """
        Generate sea / lake / river masks at an arbitrary zoom level.

        Lakes  — interpolate fill_level and height_map; lake where fill > h.
        Rivers — Bresenham rasterization of the stored vector polylines.

        Parameters
        ----------
        lim   : (x0, x1, y0, y1) — view window in [0, 1]²
        shape : (rows, cols)

        Returns
        -------
        dict with keys 'sea_mask', 'lake_mask', 'river_mask'
        """
        if self._height_interp is None or self._fill_interp is None:
            raise RuntimeError("Call set_interpolators() before get_masks_at().")

        x0, x1, y0, y1 = lim
        rows, cols = shape

        xs = np.linspace(x0, x1, rows)
        ys = np.linspace(y0, y1, cols)
        X, Y   = np.meshgrid(xs, ys, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)

        h  = self._height_interp(points).reshape(shape).astype(np.float32)
        fl = self._fill_interp(points).reshape(shape).astype(np.float32)

        sea_mask   = h <= self.sea_level
        lake_mask  = (fl > h) & ~sea_mask
        river_mask = rasterize_river_paths(
            self.river_coords, self.river_path_starts, self.river_path_lengths,
            float(x0), float(x1), float(y0), float(y1), rows, cols
        ) & ~sea_mask

        return {
            'sea_mask':   sea_mask,
            'lake_mask':  lake_mask,
            'river_mask': river_mask,
        }

    def get_masks(self):
        """Return base-resolution masks (no interpolation needed)."""
        return {
            'sea_mask':   self.sea_mask,
            'lake_mask':  self.lake_mask,
            'river_mask': self.river_mask,
        }

