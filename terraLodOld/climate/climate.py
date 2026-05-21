import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, distance_transform_edt

from utils import timeit
from .helper import get_temperature_grid

#from .physics  import prevailing_wind, lat_base_temp, saturation_capacity
#from .moisture import build_moisture_sources, advect_moisture



@dataclass
class ClimateParams:
    latitude            : float = 25.0
    wetness             : float = 0.5
    precipitation_range : tuple = (200, 2000)  # mm/year
    alpha               : float = 0.5
    orog_k_per_km       : float = 0.05
    slope_beta          : float = 0.025
    moisture_diffusion_sigma_km : float = 25.0
    plains_halflife_mult        : float = 3.0
    plains_flat_slope           : float = 0.01
    thermal_wind_factor         : float = 0.4

class WorldParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@timeit
def _compute_temperature(self, init_run) -> np.ndarray:
    """
    T = T_base(lat) − lapse_rate × altitude + continentality − sea_cooling.

    Components: T_base (cosine fit), lapse 6.5 °C/km, continentality
    (+8 °C inland with 150 km exponential decay), sea_cooling (−3 °C on water).
    Cached after first call.
    """
    if self._T_cache is not None:
        return self._T_cache

    rows     = self.shape[0]
    lat_span = self.max_size / 111_000.0
    lat_grid = np.linspace(
        self.latitude - lat_span / 2.0,
        self.latitude + lat_span / 2.0,
        rows,
    )[:, None]

    T_base     = lat_base_temp(lat_grid).astype(np.float32)
    altitude_m = self.height_map_land * float(self.max_altitude)
    lapse      = (6.5e-3 * altitude_m).astype(np.float32)

    water_mask     = self._water_mask(init_run)
    dist_px        = distance_transform_edt(~water_mask).astype(np.float32)
    maritime       = np.exp(-dist_px / (150_000.0 / self.pixel_size_m))
    continentality = (8.0 * (1.0 - maritime)).astype(np.float32)
    sea_cooling    = np.where(water_mask, 3.0, 0.0).astype(np.float32)

    T = gaussian_filter(T_base - lapse + continentality - sea_cooling,
                        sigma=2.0).astype(np.float32)
    self._T_cache = T
    return T

class Climate:
    def __init__(self, height_map, hydro, world_params, climate_params: ClimateParams = ClimateParams()):
        self.height_map = height_map
        self.height_map_land = np.clip(height_map - hydro.maps['sea_level'], 0.0, 1.0)
        self.hydro = hydro

        self.maps = {}

        
        self.world_params = WorldParams(**world_params) if isinstance(world_params, dict) else world_params
        self.climate_params = climate_params

        self.init_temperature()

    def init_temperature(self):
        base_temp = get_temperature_grid(
            rows=self.height_map.shape[0],
            cols = self.height_map.shape[1],
            max_size=self.world_params.max_size,
            latitude=self.climate_params.latitude,
        )    
        

        altitude_lapse = self.height_map_land * float(self.world_params.max_altitude) * 6.5e-3
        
        lake_mask = self.hydro.maps.get('lake_mask', np.zeros_like(self.height_map, dtype=bool))
        sea_mask = self.hydro.maps.get('sea_mask', np.zeros_like(self.height_map, dtype=bool))
        river_mask = self.hydro.maps.get('river_mask', np.zeros_like(self.height_map, dtype=bool))

        inland_dist_px = distance_transform_edt(~sea_mask)
        pixel_size_m = self.world_params.max_size / max(self.height_map.shape)
        maritime = np.exp(-inland_dist_px / (150_000.0 / pixel_size_m))
        continentality = 8.0 * (1.0 - maritime)

        sea_cooling = np.where(sea_mask, 3.0, 0.0)
        lake_ccoling = np.where(lake_mask, 1.0, 0.0)
        river_cooling = np.where(river_mask, 0.5, 0.0)

        T = gaussian_filter(base_temp - altitude_lapse + continentality - sea_cooling - lake_ccooling - river_cooling, sigma=2.0)
        self.maps['temperature'] = T


class ClimateOLd:
    """
    Full-resolution climate simulator.

    Works directly on the base-resolution height map — no downsampling.

    Parameters
    ----------
    height_map          : 2-D float array  (values normalised 0–1)
    hydro               : Hydrology object exposing ``base_sea_mask``, ``base_lake_mask``
    world_params        : dict with ``'max_altitude'`` and ``'max_size'``
    latitude            : central latitude in degrees (−90 to 90)
    wetness             : global moisture scalar [0=arid … 1=very wet]
    precipitation_range : (min_mm, max_mm) annual precipitation bounds
    """

    def __init__(self, height_map, hydro, world_params,
                 latitude=25.0, precipitation_range=(200, 2000), wetness=0.5,
                 alpha=0.5, orog_k_per_km=0.05, slope_beta=0.025,
                 moisture_diffusion_sigma_km=25.0,
                 plains_halflife_mult=3.0, plains_flat_slope=0.01,
                 thermal_wind_factor=0.4):
        self.world_params = world_params
        self.shape        = height_map.shape
        self.hydro        = hydro
        self.sea_level    = hydro.sea_level
        self.sea_mask     = hydro.base_sea_mask
        self.lake_mask    = hydro.base_lake_mask

        self.max_altitude  = world_params['max_altitude']
        self.max_size      = world_params['max_size']
        self.latitude      = latitude
        self.wetness       = wetness
        self.precipitation_range = precipitation_range

        # Physics knobs
        self.alpha                       = alpha
        self.orog_k_per_km               = orog_k_per_km
        self.slope_beta                  = slope_beta
        self.moisture_diffusion_sigma_km = moisture_diffusion_sigma_km
        self.plains_halflife_mult        = plains_halflife_mult
        self.plains_flat_slope           = plains_flat_slope
        self.thermal_wind_factor         = thermal_wind_factor

        rows, cols = height_map.shape
        self.pixel_size_m = self.max_size / max(rows, cols)

        # Height map variants
        land_scale = max(1.0 - self.sea_level, 1e-6)
        h_raw = height_map.astype(np.float32)
        self.height_map      = h_raw
        self.height_map_land = np.clip(
            (h_raw - self.sea_level) / land_scale, 0.0, 1.0
        ).astype(np.float32)

        # Global prevailing wind direction (sea-to-inland, terrain-independent)
        self._wy0, self._wx0 = prevailing_wind(self.sea_mask)
        self._wy = self._wx = None

        # Cached intermediates
        self._T_cache    = None
        self._slope_mag  = None
        self._wind_order = None

        # Output maps
        self.temperature_map   = None
        self.humidity_map      = None
        self.precipitation_map = None
        self.wind_map          = None  # (R, C, 2) float32 — [wy, wx] per cell

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, init_run = True):
        """Compute all climate maps and store them as instance attributes."""
        self._T_cache = None
        self.temperature_map = None
        self.humidity_map = None
        self.precipitation_map = None
        self.wind_map = None
        self.lake_mask = self.hydro.base_lake_mask

        # Temperature first — wind field uses it for the thermal component
        T = self._compute_temperature(init_run)
        self._build_wind_field(T)

        RH, P_orog = self._compute_humidity_and_orog_precip(init_run)
        P          = self._compute_precipitation(RH, P_orog, init_run)

        self.temperature_map   = T
        self.humidity_map      = np.clip(RH, 0.0, 1.0)
        self.precipitation_map = P
        # Store wind field as a single (R, C, 2) array: channel 0 = wy, 1 = wx
        self.wind_map = np.stack([self._wy, self._wx], axis=-1)
        return self
        

    def get_maps(self) -> dict:
        """Return dict of climate maps, running the simulation if needed."""
        if self.temperature_map is None:
            self.run()
        return {
            'temperature':   self.temperature_map,
            'humidity':      self.humidity_map,
            'precipitation': self.precipitation_map,
            'wind':          self.wind_map,   # (R, C, 2) — [wy, wx] unit vectors
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _water_mask(self, init_run: bool) -> np.ndarray:
        return self.sea_mask if init_run else (self.sea_mask | self.lake_mask)

    # ------------------------------------------------------------------
    # Wind field
    # ------------------------------------------------------------------

    @timeit
    def _build_wind_field(self, T: np.ndarray) -> None:
        """
        Local wind = global_wind − α·∇h_land + thermal_wind_factor·∇T̂, normalised.

        Three components are blended per cell:
          1. **Prevailing (sea-to-inland)** — from `prevailing_wind`, resolution-
             independent sea-to-land unit vector.
          2. **Orographic deflection** − α·∇h  — air deflects around mountains;
             gradient in physical m/m units makes it resolution-independent.
          3. **Thermal (pressure gradient)** + β·∇T̂  — surface wind blows from
             cold high-pressure toward warm low-pressure regions (∝ +∇T).
             ∇T is smoothed and normalised before blending so it can't overwhelm
             the prevailing direction.

        The slope magnitude is stored for the advection kernel; upwind-first
        ordering is pre-computed once (O(N log N)).
        """
        grad_y, grad_x = np.gradient(
            self.height_map_land * self.max_altitude,
            self.pixel_size_m,
        )

        # --- thermal component: wind blows toward warm (low-pressure) areas ---
        # Smooth T first so small-scale noise doesn't create spurious wind cells
        T_smooth = gaussian_filter(T.astype(np.float64), sigma=4.0).astype(np.float32)
        tgrad_y, tgrad_x = np.gradient(T_smooth, self.pixel_size_m)
        t_norm   = np.hypot(tgrad_y, tgrad_x) + 1e-9
        tgrad_y /= t_norm
        tgrad_x /= t_norm

        wy   = self._wy0 - self.alpha * grad_y + self.thermal_wind_factor * tgrad_y
        wx   = self._wx0 - self.alpha * grad_x + self.thermal_wind_factor * tgrad_x
        norm = np.hypot(wy, wx) + 1e-9
        self._wy        = (wy / norm).astype(np.float32)
        self._wx        = (wx / norm).astype(np.float32)
        self._slope_mag = np.sqrt(grad_y**2 + grad_x**2).astype(np.float32)

        rows, cols = self.shape
        ii = np.arange(rows, dtype=np.float32)[:, None]
        jj = np.arange(cols, dtype=np.float32)[None, :]
        proj = ii * float(self._wy.mean()) + jj * float(self._wx.mean())
        self._wind_order = np.argsort(proj.ravel()).astype(np.int64)

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------
    @timeit
    def _compute_temperature(self, init_run) -> np.ndarray:
        """
        T = T_base(lat) − lapse_rate × altitude + continentality − sea_cooling.

        Components: T_base (cosine fit), lapse 6.5 °C/km, continentality
        (+8 °C inland with 150 km exponential decay), sea_cooling (−3 °C on water).
        Cached after first call.
        """
        if self._T_cache is not None:
            return self._T_cache

        rows     = self.shape[0]
        lat_span = self.max_size / 111_000.0
        lat_grid = np.linspace(
            self.latitude - lat_span / 2.0,
            self.latitude + lat_span / 2.0,
            rows,
        )[:, None]

        T_base     = lat_base_temp(lat_grid).astype(np.float32)
        altitude_m = self.height_map_land * float(self.max_altitude)
        lapse      = (6.5e-3 * altitude_m).astype(np.float32)

        water_mask     = self._water_mask(init_run)
        dist_px        = distance_transform_edt(~water_mask).astype(np.float32)
        maritime       = np.exp(-dist_px / (150_000.0 / self.pixel_size_m))
        continentality = (8.0 * (1.0 - maritime)).astype(np.float32)
        sea_cooling    = np.where(water_mask, 3.0, 0.0).astype(np.float32)

        T = gaussian_filter(T_base - lapse + continentality - sea_cooling,
                            sigma=2.0).astype(np.float32)
        self._T_cache = T
        return T

    # ------------------------------------------------------------------
    # Humidity (coupled with orographic precipitation)
    # ------------------------------------------------------------------

    @timeit
    def _compute_humidity_and_orog_precip(self, init_run) -> tuple[np.ndarray, np.ndarray]:
        """
        Advect moisture along the wind field (conservative orographic rain-out).

        Clausius–Clapeyron converts raw moisture to RH:
            RH = 1 − exp(−moisture / q_sat(T))

        Returns (humidity [0,1], orog_precip_proxy [unnormalised]).
        """
        h_m = self.height_map_land * float(self.max_altitude)

        river_map = None
        if not init_run and hasattr(self.hydro, 'water_acc'):
            raw = np.log1p(self.hydro.water_acc.astype(np.float32))
            river_map = (raw / (raw.max() + 1e-9)).astype(np.float32)

        sources  = build_moisture_sources(self.sea_mask, self.lake_mask, river_map=river_map)
        sources *= np.float32(self.wetness * 0.6 + 0.4)

        # Sea cells are always hard-pinned.
        # On init_run, lakes are a weak floor (25 %) — not yet finalised.
        # On full run, lakes/rivers are full soft floors that advection can exceed.
        lake_floor              = sources.copy()
        lake_floor[self.sea_mask] = 0.0
        if init_run:
            lake_floor[self.lake_mask] *= np.float32(0.25)

        moisture, orog_precip = advect_moisture(
            sources, self.sea_mask, self.lake_mask, h_m,
            self._wy, self._wx,
            self.pixel_size_m, float(self.wetness),
            orog_k_per_km=self.orog_k_per_km,
            slope_beta=self.slope_beta,
            slope_mag=self._slope_mag,
            order=self._wind_order,
            lake_floor=lake_floor,
            plains_halflife_mult=self.plains_halflife_mult,
            plains_flat_slope=self.plains_flat_slope,
        )

        q_sat    = saturation_capacity(self._compute_temperature(init_run))
        humidity = (1.0 - np.exp(-moisture / (q_sat + 1e-6))).astype(np.float32)
        return humidity, orog_precip.astype(np.float32)

    # ------------------------------------------------------------------
    # Precipitation
    # ------------------------------------------------------------------

    @timeit
    def _compute_precipitation(self, RH: np.ndarray,
                               orog_precip_raw: np.ndarray, init_run: bool) -> np.ndarray:
        """
        Precipitation = RH² (convective) + 0.5 × normalised orographic.

        99th-percentile land normalisation prevents a single extreme peak from
        flattening the rest of the map.  Output is scaled to mm/year.
        """
        orog   = (orog_precip_raw / (orog_precip_raw.max() + 1e-9)).astype(np.float32)
        precip = gaussian_filter(RH**2 + 0.5 * orog, sigma=1.5).astype(np.float32)

        water_mask = self._water_mask(init_run)
        land_vals  = precip[~water_mask]
        if land_vals.size > 0:
            scale = float(np.percentile(land_vals, 99))
            if scale > 1e-9:
                precip[~water_mask] = np.clip(land_vals / scale, 0.0, 1.0)

        precip[water_mask] = np.clip(RH[water_mask] * 0.8, 0.0, 1.0)
        precip = np.clip(precip, 0.0, 1.0).astype(np.float32)

        lo, hi = self.precipitation_range
        return precip * (hi - lo) + lo
