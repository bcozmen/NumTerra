import numpy as np
from scipy.ndimage import label, binary_dilation
from scipy.interpolate import RegularGridInterpolator
from .helper import (
    compute_mfd_weights,
    compute_lake_mask,
    detect_sea,
)

from .rain import compute_precipitation_accumulation




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

    def __init__(self, height_map, sea_level_percentile, init_lake_area_threshold, river_threshold,
                 min_lake_river_acc=None, infiltration_capacity=0.30, land_evap_fraction=0.30,
                 lake_open_evap_mm=900.0, spill_erosion_depth=0.002, max_overflow_iterations=5,
                 slope_exp=1.7):
        """
        Parameters
        ----------
        height_map            : float32 array (xdim, ydim)
        sea_level_percentile  : float in (0, 1)
        river_threshold       : int  — min flow_acc to define a river cell
        min_lake_river_acc    : int or None
            Lakes whose peak inflow is below this are evaporated.
            Defaults to `river_threshold`.
        infiltration_capacity    : fraction [0,1] of precipitation absorbed by soil
        land_evap_fraction       : max PET fraction [0,1] on land at 30 °C
        lake_open_evap_mm        : open-water potential evaporation (mm / year)
        spill_erosion_depth      : depth by which the spill-point rim is lowered
        max_overflow_iterations  : cascade depth cap (1 = single pass)
        """
        if min_lake_river_acc is None:
            min_lake_river_acc = river_threshold

        self.height_map      = height_map
        self.xdim, self.ydim = height_map.shape

        self.river_threshold     = river_threshold
        self.min_lake_river_acc  = min_lake_river_acc   # was never saved — lakes with no river input survived
        self.sea_level_percentile = sea_level_percentile
        self.init_lake_area_threshold = init_lake_area_threshold

        # water-budget parameters used by run()
        self.infiltration_capacity   = infiltration_capacity
        self.land_evap_fraction      = land_evap_fraction
        self.lake_open_evap_mm       = lake_open_evap_mm
        self.spill_erosion_depth     = spill_erosion_depth
        self.max_overflow_iterations = max_overflow_iterations
        self.slope_exp               = slope_exp

        self.mfd_weights = compute_mfd_weights(height_map, slope_exp)

        self.base_sea_mask, self.sea_level = self.init_sea()
        self.sea_interp = self.get_interpolators(self.base_sea_mask.astype(float))

        self.base_lake_mask, self.base_lake_fill = self.init_lake()
        self._init_lake_mask = self.base_lake_mask.copy()   # snapshot before budget modifies it

    def run(self, climate):
        """
        Run the precipitation-driven water budget using climate outputs.

        Computes lake equilibrium, lake masks and levels, and a modified
        height map with eroded spill points.  Results are stored as instance
        attributes.

        Returns
        -------
        height_map_out : (R,C) float32 — terrain with eroded spill points
        """
        water_acc, sink_water_base, lake_mask, lake_level, height_map_out = compute_precipitation_accumulation(
            height_map            = self.height_map,
            precipitation_map     = climate.precipitation_map,
            flow_weights          = self.mfd_weights,
            temperature           = climate.temperature_map,
            sea_mask              = self.base_sea_mask,
            basin_fill            = self.base_lake_fill,
            infiltration_capacity = self.infiltration_capacity,
            land_evap_fraction    = self.land_evap_fraction,
            lake_open_evap_mm     = self.lake_open_evap_mm,
            spill_erosion_depth   = self.spill_erosion_depth,
            max_overflow_iterations = self.max_overflow_iterations,
            slope_exp             = self.slope_exp,
        )

        self.water_acc      = water_acc
        self.base_lake_mask = lake_mask
        self.lake_level     = lake_level
        self.height_map_out = height_map_out

        # Remove lakes that have no meaningful river input — their water came
        # only from direct precipitation on a few cells with no upstream catchment.
        # Compare against sink_water_base (water that *pools* at the basin floor
        # from precipitation routing), NOT water_acc (throughput), which is the
        # cumulative flow through every cell and is enormous even for tiny lakes
        # because it includes the entire upstream catchment passing through the
        # lake's slope cells.  A 10-cell isolated puddle with 260 mm/yr runoff
        # per cell has sink_water_base.sum() ≈ 2 600 — well below 50 000 — while
        # its water_acc.max() ≈ 260 000+, which the old code never filtered out.
        if self.min_lake_river_acc > 0 and lake_mask.any():
            labeled_lakes, n_lakes = label(lake_mask)
            for i in range(1, n_lakes + 1):
                lake_cells   = labeled_lakes == i
                basin_inflow = float(sink_water_base[lake_cells].sum())
                if basin_inflow < self.min_lake_river_acc:
                    lake_mask[lake_cells]  = False
                    lake_level[lake_cells] = 0.0
            self.base_lake_mask = lake_mask
            self.lake_level     = lake_level

        # Continuous river field — stored in log space for smooth interpolation.
        # log1p compresses the huge dynamic range of flow accumulation so that
        # bilinear/cubic interpolation at arbitrary zoom levels stays well-behaved.
        # At query time: interpolate log_acc → expm1 → threshold.
        log_acc = np.log1p(water_acc).astype(np.float32)
        self.log_acc_max   = float(log_acc.max()) + 1e-9
        log_acc_norm       = log_acc / self.log_acc_max   # [0, 1]
        self.river_interp  = self.get_interpolators(log_acc_norm)

        # Lake: interpolate the fill-level surface so get_lake_mask can compare
        # against the zoomed height map — same physics as sea_mask but per-lake.
        self.lake_level_interp = self.get_interpolators(lake_level)

        return self.height_map_out


    def init_sea(self):
        sea_level = np.percentile(self.height_map, self.sea_level_percentile * 100)
        sea_mask = detect_sea(self.height_map, sea_level)
        return sea_mask, sea_level

    def init_lake(self):
        lake, fill = compute_lake_mask(self.height_map, self.base_sea_mask)

        labeled_lakes, num_lakes = label(lake)
        for i in range(1, num_lakes + 1):
            if np.sum(labeled_lakes == i) < self.init_lake_area_threshold:
                mask = labeled_lakes == i
                lake[mask] = False
                fill[mask] = 0.0
        fill[~lake] = 0.0
        return lake.astype(bool), fill

    def get_interpolators(self, map_base):
        x = np.linspace(0, 1, self.xdim)
        y = np.linspace(0, 1, self.ydim)
        return RegularGridInterpolator((x, y), map_base)

    def get_sea_mask(self, height_map, pts):
        sea_mask = self.sea_interp(pts).reshape(height_map.shape)
        sea_prior = sea_mask > 0
        valid_region = sea_prior | binary_dilation(sea_prior, iterations=10)

        sea_true = height_map < self.sea_level
        final_sea_mask = sea_true & valid_region
        return final_sea_mask.astype(bool)

    def get_lake_mask(self, height_map, pts):
        """
        Zoom-aware lake mask.

        Interpolates the base-resolution fill-level surface to the query
        resolution, then marks a cell as lake when:
          1. It falls within the dilated footprint of a base-resolution lake, AND
          2. The zoomed height_map is below the interpolated fill level.

        This gives the same sharp lake boundaries as get_sea_mask — correct
        at any zoom without re-running the simulation.
        """
        fill_level = self.lake_level_interp(pts).reshape(height_map.shape)
        lake_prior = fill_level > 0
        valid_region = lake_prior | binary_dilation(lake_prior, iterations=10)

        lake_true = (height_map < fill_level) & (fill_level > 0)
        return (lake_true & valid_region).astype(bool)

    def get_river_field(self, height_map, pts, sea_mask, lake_mask, river_threshold_norm=None):
        """
        Continuous river accumulation field at arbitrary zoom / resolution.

        Returns
        -------
        river_acc : (R,C) float32
            Physical accumulation (expm1 of the normalised log field).
            Use this for rendering river width / colour ramps.
        river_mask : (R,C) bool
            True where river_acc exceeds ``river_threshold_norm`` (a fraction
            of the max log-accumulation, default 0.15).  Rivers are excluded
            from sea and lake cells.

        How width scales with zoom
        --------------------------
        ``river_acc`` is a smooth continuous field.  At low zoom a river is
        1-2 pixels wide; at 4× zoom the same physical river spans 4-8 pixels
        because the field is sampled at higher density.  The threshold stays
        fixed in normalised log space so the *physical* river width is
        resolution-independent.
        """
        if river_threshold_norm is None:
            river_threshold_norm = 0.67   # tune: lower = more/thinner rivers

        log_acc_norm = self.river_interp(pts).reshape(height_map.shape)
        river_acc    = np.expm1(log_acc_norm * self.log_acc_max).astype(np.float32)

        river_mask = (log_acc_norm > river_threshold_norm) & ~sea_mask & ~lake_mask

        return river_acc, river_mask.astype(bool)

    def get_masks(self, height_map, grid):
        X, Y = grid
        pts = np.stack([X.flatten(), Y.flatten()], axis=-1)

        sea_mask  = self.get_sea_mask(height_map, pts)
        lake_mask = self.get_lake_mask(height_map, pts) 
        river_acc, river_mask = self.get_river_field(height_map, pts, sea_mask, lake_mask) 

        return {
            "sea_mask":   sea_mask,
            "lake_mask":  lake_mask,
            "river_acc":  river_acc,
            "river_mask": river_mask,
            "sea_level":  self.sea_level,
        }



        

        
