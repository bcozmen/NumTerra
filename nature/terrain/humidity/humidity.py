from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import numpy as np

from utils.functions import FastInterpolator
from utils import timeit
from ..wind.helper import get_lat_grid
from .numba import (
    advect_numba,
    humidity_capacity_numba,
    compute_evaporation_numba,
    compute_rain_and_update_numba,
)


@dataclass
class HumidityConfig:
    # 10 inner steps = one "year" of humidity convergence.
    # The outer Thermal→Wind→Humidity→River loop repeats ~10 times (= ~10 years to equilibrium).
    iterations: int = 10

    # --- Evaporation [fraction of saturation capacity added per step] ---
    # compute_evaporation() returns a dimensionless fraction; the caller multiplies by
    # humidity_capacity(T) to get hPa.  This avoids any dt-scaling issues.
    # Sea at 20°C, full sun: evap_frac ≈ 0.3 → adds 0.3 × 23.4 ≈ 7 hPa / step.
    evaporation_rate: float = 0.3   # fraction of sat. capacity / step  (sea baseline)

    diffusion_sigma: float = 0.6

    orographic_factor: float = 1.0    # extra rain per unit uplift per unit humidity / step
    condensation_rate: float = 0.5    # fraction of excess above saturation removed / step

    # Surface type multipliers on evaporation_rate
    sea_evaporation:    float = 1.0
    lake_evaporation:   float = 0.7
    river_evaporation:  float = 0.3
    land_evaporation:   float = 0.05

    # Soil moisture [mm]
    soil_capacity:   float = 200.0   # mm water equivalent (field capacity)
    soil_evap_rate:  float = 0.05    # fraction of soil moisture returned to air / step

    # Advection uses a short effective dt so the semi-Lagrangian back-trace stays
    # within max_advection_cells grid cells per step.  Large back-traces with
    # mode='reflect' would wrap thousands of times and destroy spatial structure.
    max_advection_cells: float = 5.0   # max grid-cell displacement per advection step

    # Output rain scale — only the spatial distribution comes from physics.
    # Tune this to match your desired climate:
    #   Sahara      ~  25 mm/year
    #   UK          ~ 600 mm/year
    #   Amazon      ~2000 mm/year
    #   Wet tropics ~3000 mm/year
    max_rain: float = 1500.0   # mm / year

    # Föhn rain-shadow: on leeward side of a mountain, cap humidity to this
    # fraction of saturation to simulate descending warm dry air (adiabatic
    # warming raises capacity, dropping relative humidity sharply).
    # 0.6 → RH never exceeds 60 % immediately downwind of a ridge.
    rain_shadow_fraction: float = 0.60

    # Soil wilting point: below this fraction of soil_capacity evapotranspiration
    # essentially stops.  Prevents arid soils from pumping artificial humidity.
    wilting_point: float = 0.10


class Humidity:
    @timeit(label="Humidity Initialization")
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.worldConfig["model_humidity"] = self
        self.config = HumidityConfig()
        self.run()

    def __call__(self, area=None):
        if area is None:
            self.run()
        else:
            self.generate(area)
    @timeit(label="Humidity Simulation")
    def run(self):
        humidity_map, rain_map, soil_map, runoff_map = self.init_humidity_map()
        self.worldConfig["humidity"]      = FastInterpolator(humidity_map, order=1) 
        self.worldConfig["rain"]          = FastInterpolator(rain_map,     order=1)
        self.worldConfig["soil_moisture"] = FastInterpolator(soil_map,     order=1)
        self.worldConfig["runoff"]        = FastInterpolator(runoff_map,   order=1)
    @timeit(label="Humidity Generation")
    def generate(self, area):
        grid, points, size = area.grid, area.points, area.size
        area["humidity"]      = self.worldConfig["humidity"](points).reshape(size)
        area["rain"]          = self.worldConfig["rain"](points).reshape(size)
        area["soil_moisture"] = self.worldConfig["soil_moisture"](points).reshape(size)
        area["runoff"]        = self.worldConfig["runoff"](points).reshape(size)

    # ------------------------------------------------------------------ helpers

    def _get_masks(self):
        sea_mask   = self.worldConfig["sea_mask"]()
        zero_bool  = np.zeros_like(sea_mask, dtype=bool)
        lake_mask  = (self.worldConfig["lake_mask"]()
                      if "lake_mask"  in self.worldConfig.maps else zero_bool.copy())
        river_mask = (self.worldConfig["river_mask"]()
                      if "river_mask" in self.worldConfig.maps else zero_bool.copy())
        return sea_mask, lake_mask, river_mask

    def _init_humidity(self, sea_mask, lake_mask, river_mask):
        """
        Initialise atmospheric humidity (hPa) from Magnus saturation capacity.

        Relative humidity by surface type:
          sea    -> 95 %
          lake   -> 85 %
          river  -> 70 %
          land   -> 20-85 %, S-curve in temperature (warm tropics start humid)
        """
        temperature = self.worldConfig["temperature"]()
        cap = humidity_capacity(temperature=temperature)   # hPa, Magnus formula

        humidity = np.zeros_like(temperature)
        humidity[sea_mask]                = cap[sea_mask]  * 0.95
        humidity[lake_mask]               = cap[lake_mask] * 0.85
        humidity[river_mask & ~lake_mask] = cap[river_mask & ~lake_mask] * 0.70

        land_mask = ~sea_mask & ~lake_mask & ~river_mask
        if land_mask.any():
            t_land  = temperature[land_mask]
            # tanh S-curve: 50 % RH at 15 C, 85 % at ~35 C, 20 % at ~-15 C
            rh_base = np.clip(0.50 + 0.35 * np.tanh((t_land - 15.0) / 15.0), 0.20, 0.85)
            humidity[land_mask] = cap[land_mask] * rh_base

        return humidity

    def _init_soil_moisture(self, sea_mask, lake_mask, river_mask):
        """Soil moisture (mm).  Zero over open water; land starts at 30 % capacity."""
        soil = np.zeros_like(self.worldConfig["temperature"]())
        land_mask = ~sea_mask & ~lake_mask & ~river_mask
        soil[land_mask] = self.config.soil_capacity * 0.30
        return soil

    def get_maps(self):
        sea_mask, lake_mask, river_mask = self._get_masks()
        zero_float = np.zeros_like(self.worldConfig["temperature"]())

        # Carry converged state forward between outer iterations so each world()
        # call continues from where the previous one left off instead of
        # re-initialising from a temperature-only guess every time.
        if "humidity" in self.worldConfig.maps:
            humidity_init = self.worldConfig["humidity"]()
        else:
            humidity_init = self._init_humidity(sea_mask, lake_mask, river_mask)

        if "soil_moisture" in self.worldConfig.maps:
            soil_init = self.worldConfig["soil_moisture"]()
        else:
            soil_init = self._init_soil_moisture(sea_mask, lake_mask, river_mask)

        return {
            "config":        self.config,
            # cell_size (m, m) needed by advect() for correct back-tracing displacement
            "cell_size":     self.worldConfig.cell_size,
            "temperature":   self.worldConfig["temperature"](),
            "sun":           self.worldConfig["sun"](),
            "wind":          self.worldConfig["wind"](),
            "sea_mask":      sea_mask,
            "lake_mask":     lake_mask,
            "river_mask":    river_mask,
            "grad_i":        self.worldConfig["grad_i"](),
            "grad_j":        self.worldConfig["grad_j"](),
            "humidity":      humidity_init,
            "soil_moisture": soil_init,
            # Start rain accumulator at zero each run; the stored worldConfig rain
            # (mm/year) is the OUTPUT of the previous outer iteration and is NOT fed
            # back into the physics loop (rain is a cumulative output, not a state var).
            "rain":          zero_float.copy(),
            # Runoff accumulates per-step soil overflow for the hydrology system.
            "runoff":        zero_float.copy(),
        }

    # ----------------------------------------------------------------- main loop

    def init_humidity_map(self):
        maps = self.get_maps()
        cfg  = self.config

        wind      = maps["wind"]
        wind_i    = np.ascontiguousarray(wind[..., 0], dtype=np.float64)
        wind_j    = np.ascontiguousarray(wind[..., 1], dtype=np.float64)
        cell_size = maps["cell_size"]

        # Pre-compute grid-cell wind speeds for advection (grid cells / s)
        speed_i = wind_i / cell_size[0]
        speed_j = wind_j / cell_size[1]

        # ------------------------------------------------------------------
        # ITCZ / Hadley-cell precipitation multiplier (constant per run).
        #   > 1.0  at the equator   (ITCZ convective uplift)
        #   < 1.0  at ±30°          (horse-latitude subsidence / trade-wind deserts)
        #   ≈ 1.0  at ±60°          (Ferrel / polar-front zone)
        # ------------------------------------------------------------------
        lat_rows = get_lat_grid(self.worldConfig)          # degrees, shape (rows,)
        lat_abs  = np.abs(lat_rows)
        itcz_boost     = 0.5  * np.exp(-(lat_abs / 10.0) ** 2)
        horse_suppress = 0.40 * np.exp(-((lat_abs - 30.0) / 8.0) ** 2)
        itcz_col    = np.clip(1.0 + itcz_boost - horse_suppress, 0.4, 1.8)
        itcz_factor = np.ascontiguousarray(
            np.broadcast_to(itcz_col[:, None], maps["temperature"].shape).copy(),
            dtype=np.float64,
        )

        # ------------------------------------------------------------------
        # Pre-convert static (loop-invariant) arrays once so the repeated
        # np.ascontiguousarray calls inside the loop don't copy on every iter.
        # ------------------------------------------------------------------
        temperature_c  = np.ascontiguousarray(maps["temperature"],  dtype=np.float64)
        sun_c          = np.ascontiguousarray(maps["sun"],          dtype=np.float64)
        sea_mask_c     = np.ascontiguousarray(maps["sea_mask"],     dtype=np.bool_)
        lake_mask_c    = np.ascontiguousarray(maps["lake_mask"],    dtype=np.bool_)
        river_mask_c   = np.ascontiguousarray(maps["river_mask"],   dtype=np.bool_)
        grad_i_c       = np.ascontiguousarray(maps["grad_i"],       dtype=np.float64)
        grad_j_c       = np.ascontiguousarray(maps["grad_j"],       dtype=np.float64)

        for _ in range(cfg.iterations):
            # 1. Evaporation
            evap_frac = compute_evaporation_numba(
                temperature_c, sun_c,
                wind_i, wind_j,
                sea_mask_c, lake_mask_c, river_mask_c,
                np.ascontiguousarray(maps["soil_moisture"], dtype=np.float64),
                cfg.evaporation_rate,
                cfg.land_evaporation, cfg.sea_evaporation,
                cfg.lake_evaporation, cfg.river_evaporation,
                cfg.soil_capacity,
            )

            # 2. Semi-Lagrangian advection (parallel numba — no scipy)
            maps["humidity"] = advect_numba(
                np.ascontiguousarray(maps["humidity"], dtype=np.float64),
                speed_i, speed_j,
                cfg.max_advection_cells,
            )

            # 3. Saturation rain + orographic rain + budget + soil — single fused pass.
            #    Diffusion is intentionally applied AFTER this step so the orographic
            #    moisture gradient (windward wet / leeward dry) is not pre-blurred.
            humidity_cap = humidity_capacity_numba(temperature_c)
            maps["humidity"], maps["soil_moisture"], maps["rain"], maps["runoff"] = (
                compute_rain_and_update_numba(
                    np.ascontiguousarray(maps["humidity"],      dtype=np.float64),
                    humidity_cap,
                    evap_frac,
                    wind_i, wind_j,
                    grad_i_c, grad_j_c,
                    sea_mask_c, lake_mask_c, river_mask_c,
                    np.ascontiguousarray(maps["soil_moisture"], dtype=np.float64),
                    np.ascontiguousarray(maps["rain"],          dtype=np.float64),
                    cfg.condensation_rate,
                    cfg.orographic_factor,
                    cfg.soil_capacity,
                    cfg.soil_evap_rate,
                    itcz_factor,
                    cfg.rain_shadow_fraction,
                    cfg.wilting_point,
                )
            )

            # 4. Turbulent diffusion — applied AFTER rain so orographic/ITCZ spatial
            #    structure is sharpened before being smoothed by mesoscale mixing.
            maps["humidity"] = gaussian_filter(maps["humidity"], sigma=cfg.diffusion_sigma)

        # Normalise to mm/year
        raw_max = maps["rain"].max()
        if raw_max > 1e-8:
            maps["rain"] = maps["rain"] / raw_max * cfg.max_rain

        return maps["humidity"], maps["rain"], maps["soil_moisture"], maps["runoff"]


# ============================================================ physics functions
# NOTE: The functions below are kept for reference / testing only.
# The hot-path in Humidity.init_humidity_map() now uses the numba kernels in
# humidity/numba.py which are 4-8× faster on large grids.


def compute_evaporation(config, temperature, sun, wind, sea_mask, lake_mask,
                        river_mask, soil_moisture, **kwargs):
    """
    Evaporation rate [dimensionless fraction of saturation capacity / step].

    Caller does:  humidity += compute_evaporation(...) * humidity_capacity(T)

    Units:
      water_factor   [0.05 – 1.0]   surface type
      temp_factor    [dimensionless] doubles every ~14 °C (Clausius-Clapeyron slope)
      sun_factor     [0.5 – 1.0]
      wind_factor    [1.0 – ~2.0]
      evaporation_rate [fraction / step]  (sea baseline, default 0.3)
    → result [fraction / step], typically 0.01 – 0.5
    """
    wind_speed = np.linalg.norm(wind, axis=-1)

    water_factor = np.full_like(temperature, config.land_evaporation)
    water_factor[sea_mask]                = config.sea_evaporation
    water_factor[lake_mask]               = config.lake_evaporation
    water_factor[river_mask & ~lake_mask] = config.river_evaporation

    # Wet soil evaporates up to 3× more than dry soil
    land_mask     = ~sea_mask & ~lake_mask & ~river_mask
    soil_fraction = np.clip(soil_moisture / (config.soil_capacity + 1e-8), 0.0, 1.0)
    water_factor[land_mask] *= (1.0 + 2.0 * soil_fraction[land_mask])

    temp_factor = np.exp(0.05 * (temperature - 15.0))   # ≈ Clausius-Clapeyron
    sun_factor  = 0.5 + 0.5 * sun
    wind_factor = 1.0 + 0.05 * wind_speed

    return config.evaporation_rate * water_factor * temp_factor * sun_factor * wind_factor


def humidity_capacity(temperature, **kwargs):
    """
    Saturation vapour pressure (Magnus / Tetens formula) [hPa].

    Reference values:
      -20 C ->  1.3 hPa
        0 C ->  6.1 hPa
       15 C -> 17.1 hPa
       20 C -> 23.4 hPa
       35 C -> 56.2 hPa
    """
    return 6.112 * np.exp(17.67 * temperature / (temperature + 243.5))


def saturation_rain(config, humidity, humidity_max, **kwargs):
    """
    Condensation rate when humidity exceeds saturation [hPa / step].

    Removes a fraction of the excess above saturation each step.
    """
    excess = np.maximum(humidity - humidity_max, 0.0)
    return excess * config.condensation_rate   # [hPa / step]


def orographic_rain(config, humidity, wind, grad_i, grad_j, **kwargs):
    """
    Orographic precipitation and leeward drying (föhn effect) [hPa / step].

    grad_i, grad_j are dimensionless [m/m] slopes from get_slope(scale=max_altitude).
    Uplift = (unit wind vector) · (slope vector) is dimensionless.

    Returns:
      positive → windward rain [hPa / step]
      negative → leeward drying [hPa / step]  (not counted as precipitation by caller)
    """
    wind_speed = np.linalg.norm(wind, axis=-1) + 1e-8
    wx = wind[..., 0] / wind_speed
    wy = wind[..., 1] / wind_speed

    uplift = wx * grad_i + wy * grad_j     # dimensionless, [-1, 1] after tanh
    uplift = np.tanh(5.0 * uplift)

    return np.where(
        uplift >= 0,
        uplift       * humidity * config.orographic_factor,   # windward
        uplift * 0.5 * humidity * config.orographic_factor,   # leeward (50 % weaker)
    )   # [hPa / step]


def advect(config, humidity, wind, cell_size, **kwargs):
    """
    Semi-Lagrangian back-tracing advection.

    The displacement is clamped to config.max_advection_cells grid cells per step.
    Without this, a 15 m/s wind over a 195 m grid with a yearly dt would back-trace
    ~240,000 cells, wrapping thousands of times and destroying all spatial structure.

    The clamp sets an effective transport speed per inner step:
      max transport = max_advection_cells × cell_size / step
    Over 10 steps/year × 10 outer loops = 100 steps total, moisture can travel:
      5 cells × 195 m × 100 = 97.5 km — appropriate for a 200 km domain.

    Units:
      wind        [m / s]
      cell_size   [m, m]
      displacement [grid cells, dimensionless]  — clamped to max_advection_cells
    """
    h, w = humidity.shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Raw displacement in grid cells for a 1-second step, scaled to max_advection_cells
    # by finding the effective dt that keeps max displacement = max_advection_cells.
    speed_i = wind[..., 0] / cell_size[0]   # grid cells / second
    speed_j = wind[..., 1] / cell_size[1]

    max_speed = np.sqrt(speed_i**2 + speed_j**2).max() + 1e-8
    # Clamp: effective dt = time needed for fastest cell to travel max_advection_cells
    eff_dt = config.max_advection_cells / max_speed   # seconds

    prev_i = ii - speed_i * eff_dt
    prev_j = jj - speed_j * eff_dt

    coords   = np.array([prev_i.ravel(), prev_j.ravel()])
    advected = map_coordinates(humidity, coords, order=1, mode='reflect')
    return advected.reshape(h, w)
