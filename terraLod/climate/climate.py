"""
Climate orchestrator — ties together wind, temperature, moisture and precipitation.

The ``Climate`` class is the only public surface of this module.  All
physics computations are delegated to sub-modules:

  physics.py  — pure stateless functions (lat → temperature, saturation, wind)
  moisture.py — moisture-source building and Numba advection kernel
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

from utils import timeit
from .physics  import prevailing_wind, lat_base_temp, saturation_capacity
from .moisture import build_moisture_sources, advect_moisture


class Climate:
    """
    Full-resolution climate simulator.

    Works directly on the base-resolution height map — no downsampling.
    Since climate is generated once and used as a static lookup (zooming
    uses nearest/bilinear interpolation of these maps), full resolution
    gives maximum quality without significant extra cost.

    Physics pipeline
    ----------------
    1. **Evaporation** — sea and lake cells emit moisture proportional to
       their surface area.  Source strength is scaled by ``wetness``.
    2. **Conservative advection** — moisture is transported along a
       terrain-deflected wind field.  At each cell, orographic lifting
       extracts a fraction of carried moisture as precipitation, reducing
       the available moisture budget for downwind cells (rain shadows).
       Multi-sample blending (1-step + 2-step upstream) introduces transport
       inertia for sharper fronts.  Terrain slope reduces transport efficiency.
    3. **Saturation (Clausius–Clapeyron)** — the advected moisture is divided
       by a temperature-dependent saturation capacity to yield relative
       humidity (RH).  Warm tropical air has a higher capacity; cold polar air
       saturates easily.
    4. **Precipitation** — two physically distinct regimes:
       - *Convective*: wherever RH > 1 (supersaturation), the excess falls
         immediately as convective rain.
       - *Orographic*: the orographic rain-out from the advection step,
         gated by the soft-saturation RH so dry air produces no rain even
         when lifted.

    Parameters
    ----------
    height_map          : 2-D float array  (values normalised 0–1)
    hydro               : Hydrology object — must expose base_sea_mask, base_lake_mask
    world_params        : dict — must contain 'max_altitude' and 'max_size'
    latitude            : float — central latitude in degrees (−90 to 90)
    wetness             : float in [0, 1] — global moisture scalar (0=arid, 1=very wet)
    precipitation_range : (min_mm, max_mm) annual precipitation bounds
    """

    def __init__(self, height_map, hydro, world_params,
                 latitude=25.0, precipitation_range=(200, 2000), wetness=0.5):
        self.world_params = world_params
        self.shape        = height_map.shape

        self.hydro      = hydro
        self.sea_level  = hydro.sea_level
        self.sea_mask   = hydro.base_sea_mask
        self.lake_mask  = hydro.base_lake_mask

        self.max_altitude         = world_params['max_altitude']
        self.max_size             = world_params['max_size']
        self.latitude             = latitude
        self.wetness              = wetness
        self.precipitation_range  = precipitation_range

        rows, cols = height_map.shape
        self.pixel_size_m = self.max_size / max(rows, cols)

        # Height map variants
        land_scale = max(1.0 - self.sea_level, 1e-6)
        h_raw = height_map.astype(np.float32)
        self.height_map      = h_raw
        self.height_map_land = np.clip(
            (h_raw - self.sea_level) / land_scale, 0.0, 1.0
        ).astype(np.float32)

        # Global prevailing wind direction (unit vector)
        self._wy0, self._wx0 = prevailing_wind(latitude)
        self._wy = self._wx = None

        # Cached intermediates
        self._T_cache    = None
        self._slope_mag  = None
        self._wind_order = None

        # Output maps
        self.temperature_map   = None
        self.humidity_map      = None
        self.precipitation_map = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, init_run = True):
        self._T_cache = None
        self.temperature_map = None
        self.humidity_map = None
        self.precipitation_map = None
        self.lake_mask = self.hydro.base_lake_mask
        """Compute all climate maps and store them as instance attributes."""
        self._build_wind_field()

        T          = self._compute_temperature(init_run)
        RH, P_orog = self._compute_humidity_and_orog_precip(init_run)
        P          = self._compute_precipitation(RH, P_orog, init_run)

        #water_mask               = self.sea_mask | self.lake_mask
        humidity_out             = np.clip(RH, 0.0, 1.0)
        #humidity_out[water_mask] = 0.0
        #P[water_mask]            = 0.0

        self.temperature_map   = T
        self.humidity_map      = humidity_out
        self.precipitation_map = P
        return self
        

    def get_maps(self) -> dict:
        """Return dict of full-resolution climate maps, running the simulation if needed."""
        if self.temperature_map is None:
            self.run()
        return {
            'temperature':   self.temperature_map,
            'humidity':      self.humidity_map,
            'precipitation': self.precipitation_map,
        }

    # ------------------------------------------------------------------
    # Wind field
    # ------------------------------------------------------------------

    @timeit
    def _build_wind_field(self, alpha: float = 0.5) -> None:
        """
        Local wind = global_wind − α · ∇h_land, normalised per cell.

        Gradient is in physical slope units [m/m], making wind deflection
        independent of grid resolution and map size.  The slope magnitude
        (``self._slope_mag``) is stored for use by the advection kernel.
        Pre-computes upwind-first cell ordering (O(N log N), done once).
        """
        grad_y, grad_x = np.gradient(
            self.height_map_land * self.max_altitude,
            self.pixel_size_m,
        )
        wy   = self._wy0 - alpha * grad_y
        wx   = self._wx0 - alpha * grad_x
        norm = np.sqrt(wy ** 2 + wx ** 2) + 1e-9
        self._wy        = (wy / norm).astype(np.float32)
        self._wx        = (wx / norm).astype(np.float32)
        self._slope_mag = np.sqrt(grad_y ** 2 + grad_x ** 2).astype(np.float32)

        rows, cols = self.shape
        ii = np.arange(rows, dtype=np.float32)[:, None]
        jj = np.arange(cols, dtype=np.float32)[None, :]
        proj = ii * float(self._wy.mean()) + jj * float(self._wx.mean())
        self._wind_order = np.argsort(proj.ravel()).astype(np.int64)

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------
    def get_water_mask(self, init_run):
        if init_run:
            return self.sea_mask
        else:
            return self.sea_mask | self.lake_mask
    @timeit
    def _compute_temperature(self, init_run) -> np.ndarray:
        """
        T = T_base(lat) − lapse_rate × altitude + continentality − sea_cooling.

        Components
        ----------
        T_base         : cosine fit (equator ~27 °C, poles ~−20 °C)
        lapse_rate     : 6.5 °C / 1 000 m
        continentality : up to +8 °C inland; 150 km exponential decay from water
        sea_cooling    : −3 °C on water cells

        Result is cached so subsequent calls are free.
        """
        if self._T_cache is not None:
            return self._T_cache

        rows = self.shape[0]
        lat_span = self.max_size / 111_000.0
        lat_grid = np.linspace(
            self.latitude - lat_span / 2.0,
            self.latitude + lat_span / 2.0,
            rows,
        )[:, None]

        T_base     = lat_base_temp(lat_grid).astype(np.float32)
        altitude_m = self.height_map_land * float(self.max_altitude)
        lapse      = (6.5e-3 * altitude_m).astype(np.float32)

        water_mask = self.get_water_mask(init_run)
        dist_to_water = distance_transform_edt(~water_mask).astype(np.float32)
        decay_len     = 150_000.0 / self.pixel_size_m
        maritime      = np.exp(-dist_to_water / decay_len)
        continentality = 8.0 * (1.0 - maritime)
        sea_cooling    = np.where(water_mask, 3.0, 0.0).astype(np.float32)

        T = T_base - lapse + continentality - sea_cooling
        T = gaussian_filter(T, sigma=2.0).astype(np.float32)

        self._T_cache = T
        return T

    # ------------------------------------------------------------------
    # Humidity (coupled with orographic precipitation)
    # ------------------------------------------------------------------

    @timeit
    def _compute_humidity_and_orog_precip(self, init_run) -> tuple[np.ndarray, np.ndarray]:
        """
        Advect moisture from evaporation sources along the terrain-deflected
        wind field, simultaneously computing orographic precipitation via
        the conservative Numba kernel in moisture.py.

        Clausius–Clapeyron soft-saturation converts raw moisture to RH:
            RH = 1 − exp(−moisture / q_sat)

        Returns
        -------
        humidity    : (R,C) float32 — relative humidity [0, 1]
        orog_precip : (R,C) float32 — orographic rain-out proxy (unnormalised)
        """
        h_m = self.height_map_land * float(self.max_altitude)

        # Build normalised river accumulation map for moisture sourcing (2nd pass only).
        river_map = None
        if not init_run and hasattr(self.hydro, 'water_acc'):
            raw = self.hydro.water_acc.astype(np.float32)
            log_raw = np.log1p(raw)
            peak = float(log_raw.max()) + 1e-9
            river_map = (log_raw / peak).astype(np.float32)   # [0, 1], log-compressed

        sources = build_moisture_sources(self.sea_mask, self.lake_mask, river_map=river_map)
        sources *= np.float32(float(self.wetness) * 0.6 + 0.4)

        if init_run:
            # Lakes are not real yet — treat them as soft-floor bias (unpinned).
            # Sea cells are hard sources; lake cells provide a moisture floor
            # that advection can exceed but never drop below.
            source_mask = self.sea_mask
            lake_floor  = sources * np.float32(0.25)   # floor = 25 % of source strength
            lake_floor[~self.lake_mask] = 0.0
        else:
            # Only sea cells are hard-pinned.  Lakes and rivers both act as
            # soft floors so that advection from nearby sea can freely exceed
            # the lake's own evaporation — a small lake near the coast won't
            # become a humidity hole just because its area-scaled value is low.
            source_mask = self.sea_mask
            lake_floor  = sources.copy()
            lake_floor[self.sea_mask] = 0.0   # sea is already hard-pinned above

        moisture, orog_precip = advect_moisture(
            sources, source_mask, self.lake_mask, h_m,
            self._wy, self._wx,
            self.pixel_size_m, float(self.wetness),
            slope_mag=self._slope_mag,
            order=self._wind_order,
            lake_floor=lake_floor,
        )

        T        = self._compute_temperature(init_run)
        q_sat    = saturation_capacity(T)
        humidity = (1.0 - np.exp(-moisture / (q_sat + 1e-6))).astype(np.float32)

        return humidity, orog_precip.astype(np.float32)

    # ------------------------------------------------------------------
    # Precipitation
    # ------------------------------------------------------------------

    @timeit
    def _compute_precipitation(self, RH: np.ndarray,
                               orog_precip_raw: np.ndarray, init_run: bool) -> np.ndarray:
        """
        Precipitation = convective component + orographic component.

        Convective  : ``RH²``  — quadratic amplification in saturated regions.
        Orographic  : normalised orographic rain-out from the advection kernel
                      (no extra RH re-weighting to avoid double-counting).

        99th-percentile land normalisation prevents one extreme peak from
        flattening the rest of the map.  Final values are scaled to physical
        units (mm / year) via ``precipitation_range``.
        """
        conv_precip = (RH ** 2).astype(np.float32)
        op_max      = float(orog_precip_raw.max())
        orog_precip = (orog_precip_raw / (op_max + 1e-9)).astype(np.float32)

        precip = conv_precip + 0.5 * orog_precip
        precip = gaussian_filter(precip, sigma=1.5).astype(np.float32)

        water_mask = self.get_water_mask(init_run)
        land_mask = ~water_mask

        land_vals = precip[land_mask]
        if land_vals.size > 0:
            scale = float(np.percentile(land_vals, 99))
            if scale > 1e-9:
                precip[land_mask] = np.clip(land_vals / scale, 0.0, 1.0)

        precip[water_mask] = np.clip(RH[water_mask] * 0.8, 0.0, 1.0)
        precip = np.clip(precip, 0.0, 1.0).astype(np.float32)

        lo, hi = self.precipitation_range
        return precip * (hi - lo) + lo
