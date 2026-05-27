"""
Hydro — physically-based surface hydrology model.

Design overview
---------------
State (persisted between world() calls):
  water_depth    (H,W) [m]    : surface water on each land cell
  height_erosion (H,W) [norm] : cumulative net erosion (+) / deposition (-)
  _base_height   (H,W) [norm] : pristine terrain snapshot from Terrain

Per world() call the model:
1.  Converts annual rainfall (mm/yr) to surface runoff [m per sub-step].
2.  Runs ``config.iterations`` water-routing sub-steps:
      a. Recomputes MFD from *water surface elevation* (terrain + water_depth)
         so that basin filling and rim overflow are captured automatically.
      b. Routes ``drain_frac`` of each cell's water depth downstream.
      c. Sea cells drain completely; pit cells accumulate → lakes form.
      d. Applies per-step evaporation from open-water surfaces.
3.  Runs one erosion/deposition pass on the average discharge:
      stream power Ω = discharge × slope → erosion / sediment transport.
4.  Updates the worldConfig height (triggers slope / MFD recompute for
    Wind, Thermal, etc.) and publishes lake_mask, river_mask, water_depth.

Fill-spill-incise mechanism
----------------------------
* Rain accumulates in closed depressions.  The water surface rises.
* When the water surface equals the basin rim, compute_water_surface_mfd
  assigns positive weight toward the lowest gap — overflow begins.
* The overflow concentrates discharge at the rim → high stream power →
  the rim erodes faster than the surrounding terrain → the notch deepens →
  a permanent river channel is cut.  No artificial basin pre-filling.
"""

from dataclasses import dataclass
import numpy as np

from terraLod.utils import Interpolator, timeit

from .numba import (
    compute_mfd_weights,
    compute_water_surface_mfd,
    topo_sort,
    route_water_step,
    erode_step,
)


@dataclass
class HydroConfig:
    # ---- Inner routing loop -----------------------------------------------
    # Number of water-routing sub-steps per world() call.
    # Each sub-step represents a fraction of one "year" of transport.
    # More iterations → longer rivers, better-filled basins, smoother flow.
    iterations: int = 10

    # How often (in sub-steps) to recompute MFD weights and topo-sort.
    # The flow graph changes slowly, so recomputing every 3–5 steps gives
    # a 3–5× speedup on that pair of calls with negligible accuracy loss.
    # Set to 1 to recompute every step (original behaviour).
    # Must be < iterations, otherwise MFD is never updated mid-loop.
    mfd_recompute_interval: int = 3

    # Fraction of each cell's water depth routed downstream per sub-step.
    # Must be < 1 for numerical stability.  0.3 works well for most grids.
    drain_frac: float = 0.30

    # Fraction of standing water evaporated per sub-step (open-water loss).
    # 0.01 × 10 steps ≈ 10 % / year from lake/river surfaces — realistic.
    evap_rate: float = 0.01

    # ---- Rainfall coupling ------------------------------------------------
    # Fraction of annual rainfall (mm/yr) converted to surface runoff [m].
    # The remainder is handled by Humidity's soil-moisture model.
    rain_to_surface: float = 0.55   # increased from 0.35 — more runoff to sustain rivers

    # ---- Erosion / deposition ---------------------------------------------
    # Bedrock erodibility K [normalised height / (m²/step)].
    # erosion/step = K × discharge × slope,  capped at max_erosion_norm.
    # Increase K for faster landscape evolution (deeper canyons, wider valleys).
    erodibility: float = 3e-5

    # Fraction of excess-capacity sediment deposited per step.
    # Lower → sediment travels further before settling (braided rivers).
    deposition_rate: float = 0.08

    # Sediment transport capacity multiplier: C = capacity_k × Ω.
    # Higher → rivers carry more before depositing (more energetic erosion).
    capacity_k: float = 0.6

    # Hard cap on erosion per sub-step [normalised height].
    # 5e-5 normalised ≈ 0.15 m at max_altitude=3000 m — prevents blow-up.
    max_erosion_norm: float = 1e-5

    # ---- Mask thresholds --------------------------------------------------
    # Percentile of land-cell discharge above which a cell is a river.
    # 97 → top 3 % of land cells carry rivers.  Tune between 90 (many streams)
    # and 99 (only major rivers).
    river_discharge_percentile: float = 97.0   # top 3 % of land cells = rivers

    # Absolute minimum discharge [m/step] a cell must carry to be a river.
    # With max_rain=600mm/yr, rain_to_surface=0.55, iterations=10:
    # per-cell rain/step ≈ 600*0.001*0.55/10 = 0.033 m/step
    # A 10-cell catchment ≈ 0.033 * 10 * drain_efficiency ≈ 0.06 m/step.
    # Threshold must be well below that to see any rivers at all.
    river_discharge_threshold: float = 0.002   # was 0.1 — was larger than actual discharge values

    # Standing water depth [m] needed to classify a cell as a lake.
    lake_depth_threshold: float = 0.005   # was 0.05 — lowered to match actual water depth magnitudes

    # ---- Stability guard --------------------------------------------------
    # Maximum change in normalised height per world() call.
    # Prevents catastrophic terrain collapse in early, far-from-equilibrium
    # iterations when discharge is unrealistically large.
    max_height_change_per_call: float = 0.05

    slope_exponent: float = 2.5  # exponent applied to slope in erosion formula (Ω = Q × slope^slope_exponent)


class Hydro:
    """
    Surface hydrology: river routing, lake formation, and stream-power erosion.

    Usage
    -----
    Instantiate *after* Terrain, Thermal, Wind, Humidity::

        world = World()
        Terrain(world)
        Thermal(world)
        Wind(world)
        Humidity(world)
        Hydro(world)          # ← add here
        Plotter(world)

        for i in range(iterations):
            world()           # Hydro.run() is called automatically
    """

    @timeit(label="Hydro Initialization")
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.config      = HydroConfig()
        worldConfig["model_hydro"] = self

        H, W = worldConfig.size
        # Surface water depth [m], persisted between world() calls.
        self.water_depth    = np.zeros((H, W), dtype=np.float64)
        # Cumulative height change from erosion/deposition [normalised].
        self.height_erosion = np.zeros((H, W), dtype=np.float64)
        # Pristine terrain snapshot — erosion is always applied as a *delta*
        # on top of this so we never lose the original landscape signal.
        self._base_height   = worldConfig["height"]().copy().astype(np.float64)

        self.run()

    # ------------------------------------------------------------------
    def __call__(self, area=None):
        if area is None:
            self.run()
        else:
            self.generate(area)

    # ------------------------------------------------------------------
    @timeit(label="Hydro Simulation")
    def run(self):
        cfg     = self.config
        wc      = self.worldConfig
        max_alt = wc.max_altitude
        cx, cy  = wc.cell_size

        sea_mask = wc["sea_mask"]().astype(np.bool_)

        # Reset surface water each call — water_depth is NOT a persistent
        # state across world() calls.  Each call re-derives the steady-state
        # routing from scratch given the current (evolved) terrain.
        # Accumulating water across calls causes unbounded growth because
        # only a fraction drains per inner step.
        self.water_depth = np.zeros(wc.size, dtype=np.float64)

        # Rain [mm/yr] → metres of surface runoff per sub-step.
        if "rain" in wc.maps:
            rain_mm_yr = wc["rain"]().astype(np.float64)
        else:
            rain_mm_yr = np.zeros(wc.size, dtype=np.float64)
        rain_m_per_step = rain_mm_yr * 1e-3 * cfg.rain_to_surface / cfg.iterations

        # Current terrain = pristine + cumulative erosion, clamped to [0, 1].
        height = np.clip(self._base_height + self.height_erosion, 0.0, 1.0)

        # ------------------------------------------------------------------
        # Inner water-routing loop
        # ------------------------------------------------------------------
        cumulative_discharge = np.zeros(wc.size, dtype=np.float64)

        mfd, order, n_valid = None, None, None
        for step in range(cfg.iterations):
            # Recompute MFD only every mfd_recompute_interval steps.
            # The water surface evolves slowly so the flow graph barely changes
            # between sub-steps; skipping recomputes gives a large speedup.
            if step % cfg.mfd_recompute_interval == 0:
                mfd = compute_water_surface_mfd(
                    height, self.water_depth, max_alt, cx, cy, slope_exp = cfg.slope_exponent
                )
                order, n_valid = topo_sort(mfd)

            self.water_depth, discharge = route_water_step(
                self.water_depth, rain_m_per_step, mfd, order, n_valid,
                sea_mask, cfg.drain_frac, cfg.evap_rate,
            )
            cumulative_discharge += discharge

        # ------------------------------------------------------------------
        # Erosion / deposition  (one pass, using time-averaged discharge)
        # Reuse the MFD from the last routing sub-step — avoids a redundant
        # compute_water_surface_mfd + topo_sort call.
        # ------------------------------------------------------------------
        avg_discharge = cumulative_discharge / cfg.iterations

        height_delta = erode_step(
            height, self.water_depth, avg_discharge,
            mfd, order, n_valid, sea_mask,
            cx, cy, max_alt,
            cfg.erodibility, cfg.deposition_rate,
            cfg.capacity_k, cfg.max_erosion_norm,
        )

        # Guard against early-iteration blow-up.
        height_delta       = np.clip(
            height_delta,
            -cfg.max_height_change_per_call,
             cfg.max_height_change_per_call,
        )
        self.height_erosion += height_delta

        # ------------------------------------------------------------------
        # Update worldConfig height — triggers slope / MFD recompute so that
        # Wind, Thermal, and Humidity see the geomorphically-evolved terrain.
        # ------------------------------------------------------------------
        new_height = np.clip(self._base_height + self.height_erosion, 0.0, 1.0)
        wc["height"] = Interpolator(new_height.astype(np.float32), order=1, can_call=True)

        # ------------------------------------------------------------------
        # Derive masks and publish all hydro maps.
        # Use order=0 (nearest-neighbour) for boolean masks so that
        # Humidity's astype(np.bool_) retrieval stays clean.
        # ------------------------------------------------------------------
        lake_mask  = (self.water_depth > cfg.lake_depth_threshold) & ~sea_mask

        # River mask: percentile threshold relative to the TOP of the discharge
        # distribution, combined with a hard minimum so cells with near-zero
        # flow are never labelled as rivers even if they rank in the top N%.
        # After many outer iterations erosion creates many channels; without
        # the minimum floor the percentile alone marks far too many cells.
        land_discharge = avg_discharge[~sea_mask & ~lake_mask]
        if land_discharge.size > 0 and land_discharge.max() > 1e-10:
            river_thresh = max(
                np.percentile(land_discharge, cfg.river_discharge_percentile),
                cfg.river_discharge_threshold,
            )
        else:
            river_thresh = cfg.river_discharge_threshold   # fallback
        river_mask = (
            (avg_discharge > river_thresh) & ~sea_mask & ~lake_mask
        )

        wc["lake_mask"]   = Interpolator(lake_mask.astype(bool),
                                             order=0, can_call=True)
        wc["river_mask"]  = Interpolator(river_mask.astype(bool),
                                             order=0, can_call=True)
        wc["water_depth"] = Interpolator(self.water_depth.astype(np.float32),
                                             order=1, can_call=True)
        wc["discharge"]   = Interpolator(avg_discharge.astype(np.float32),
                                             order=1, can_call=True)

        n_lake  = int(lake_mask.sum())
        n_river = int(river_mask.sum())
        if wc.debug:
            erosion_m = (height_delta * max_alt)
            print(
                f"  [Hydro]  lakes={n_lake}  rivers={n_river}  "
                f"max_water={self.water_depth.max():.2f}m  "
                f"erosion max={erosion_m.min() * -1:.4f}m  "
                f"deposition max={erosion_m.max():.4f}m"
            )

    # ------------------------------------------------------------------
    @timeit(label="Hydro Generation")
    def generate(self, area):
        """Interpolate hydro maps onto a zoomed-in area grid."""
        points, size = area.points, area.size
        for key in ("lake_mask", "river_mask", "water_depth", "discharge"):
            if key in self.worldConfig.maps:
                area[key] = self.worldConfig[key](points).reshape(size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
