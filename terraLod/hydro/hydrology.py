import numpy as np
from scipy.ndimage import label
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation
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
                 lake_open_evap_mm=900.0, spill_erosion_depth=0.002, max_overflow_iterations=5):
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

        self.river_threshold = river_threshold
        self.sea_level_percentile = sea_level_percentile
        self.init_lake_area_threshold = init_lake_area_threshold

        # water-budget parameters used by run()
        self.infiltration_capacity   = infiltration_capacity
        self.land_evap_fraction      = land_evap_fraction
        self.lake_open_evap_mm       = lake_open_evap_mm
        self.spill_erosion_depth     = spill_erosion_depth
        self.max_overflow_iterations = max_overflow_iterations

        self.mfd_weights = compute_mfd_weights(height_map)

        self.base_sea_mask, self.sea_level = self.init_sea()
        self.sea_interp = self.get_interpolators(self.base_sea_mask.astype(float))

        self.base_lake_mask, self.base_lake_fill = self.init_lake()
        #self.lake_interp = self.get_interpolators(self.base_lake_fill)

    def run(self, climate):
        """
        Run the precipitation-driven water budget using climate outputs.

        Computes lake equilibrium, lake masks and levels, and a modified
        height map with eroded spill points.  Results are stored as instance
        attributes and returned as a dict.

        Returns
        -------
        dict with keys:
            water_acc      — water throughput per cell (river strength signal)
            lake_mask      — bool mask of standing water after budget
            lake_level     — water-surface height per cell
            height_map_out — terrain with eroded spill points
        """
        water_acc, lake_mask, lake_level, height_map_out = compute_precipitation_accumulation(
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
        )

        self.water_acc      = water_acc
        self.lake_mask      = lake_mask
        self.lake_level     = lake_level
        self.height_map_out = height_map_out

        return self.height_map_out


    def init_sea(self):
        sea_level = np.percentile(self.height_map, self.sea_level_percentile * 100)
        sea_mask = detect_sea(self.height_map, sea_level)
        return sea_mask, sea_level

    def init_lake(self):
        lake, fill = compute_lake_mask(self.height_map, self.base_sea_mask)

        labeled_lakes, num_lakes = label(lake)
        for i in range(1, num_lakes + 1):
            lake_size = np.sum(labeled_lakes == i)
            if lake_size < self.init_lake_area_threshold:
                lake[labeled_lakes == i] = False
                fill[labeled_lakes == i] = 0
        fill[lake == False] = 0
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
        lake_mask = self.lake_interp(pts).reshape(height_map.shape)
        lake_mask = lake_mask > 0
        valid_region = lake_mask | binary_dilation(lake_mask, iterations=10)

        #for each lake, use the fill level to determine actual boundary of the lake
        
        return lake_mask.astype(bool)
    def get_masks(self, height_map, grid):
        X, Y = grid
        pts = np.stack([X.flatten(), Y.flatten()], axis=-1)
        
        sea_mask = self.get_sea_mask(height_map, pts)

        return {
            "sea_mask": sea_mask,
            "sea_level": self.sea_level,
        }



        

        
