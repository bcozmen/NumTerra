from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import numpy as np



from terraLod.utils import FastInterpolator, timeit
from ..wind.helper import get_lat_grid
from .numba import (
    advect_numba,
    humidity_capacity_numba,
    compute_evaporation_numba,
    compute_rain_and_update_numba,
)

@dataclass
class HumidityConfig:
    world_max_size : float 
    cell_size : tuple
    iterations: int = 12      
    max_advection_percent: float = 0.02     # Max fraction of world size a parcel with 1m/s speed can move in total      
    evaporation_rate: float = 0.08        # Dropped slightly to stop aggressive sea-rain loops
    diffusion_sigma: float = 2.0          # Slightly sharper transitions
    orographic_factor: float = 0.12       # Increased so mountains squeeze out water effectively
    condensation_rate: float = 0.14       
    
    # Surface Parameters
    sea_evaporation: float = 1.0          
    lake_evaporation: float = 0.7         
    river_evaporation: float = 0.4        
    land_evaporation: float = 0.05        
    
    # Soil Mechanics
    soil_capacity: float = 200.0          
    soil_evap_rate: float = 0.02          # Lowered so soils don't instantly vaporize
    wilting_point: float = 0.10           
    
    max_rain: float = 2000.0              
    rain_shadow_fraction: float = 0.50    # Deserts behind mountains are now drier
    hpa_to_mm_factor: float = 25.0        # Normalized scaling factor

    @property
    def cells_per_ms_per_iter(self):
        L_cells = self.world_max_size / self.cell_size[0]
        return (self.max_advection_percent * L_cells) / self.iterations

    @property
    def max_advection_cells(self):
        return 2 * self.cells_per_ms_per_iter  # Max movement in cells per iteration, doubled for safety margin


class Humidity:
    @timeit(label="Humidity Initialization")
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.worldConfig["model_humidity"] = self
        self.config = HumidityConfig(world_max_size=self.worldConfig.max_size, cell_size=self.worldConfig.cell_size)
        self.run()

    def __call__(self, area=None):
        if area is None: self.run()
        else: self.generate(area)

    @timeit(label="Humidity Simulation")
    def run(self):
        humidity_map, rain_map, soil_map, runoff_map = self.simulate_climate()
        self.worldConfig["humidity"]      = FastInterpolator(humidity_map, order=1) 
        self.worldConfig["rain"]          = FastInterpolator(rain_map,     order=1)
        self.worldConfig["soil_moisture"] = FastInterpolator(soil_map,     order=1)
        self.worldConfig["runoff"]        = FastInterpolator(runoff_map,   order=1)

    @timeit(label="Humidity Generation")
    def generate(self, area):
        pts, size = area.points, area.size
        area["humidity"]      = self.worldConfig["humidity"](pts).reshape(size)
        area["rain"]          = self.worldConfig["rain"](pts).reshape(size)
        area["soil_moisture"] = self.worldConfig["soil_moisture"](pts).reshape(size)
        area["runoff"]        = self.worldConfig["runoff"](pts).reshape(size)

    def _get_masks(self):
        sea = self.worldConfig["sea_mask"]()
        zero = np.zeros_like(sea, dtype=bool)
        lake = self.worldConfig["lake_mask"]() if "lake_mask" in self.worldConfig.maps else zero
        river = self.worldConfig["river_mask"]() if "river_mask" in self.worldConfig.maps else zero
        return sea, lake, river

    def simulate_climate(self):
        sea_m, lake_m, river_m = self._get_masks()
        temp_field = np.ascontiguousarray(self.worldConfig["temperature"](), dtype=np.float64)
        
        h_init = self.worldConfig["humidity"]() if "humidity" in self.worldConfig.maps else (humidity_capacity_numba(temp_field) * 0.5)
        s_init = self.worldConfig["soil_moisture"]() if "soil_moisture" in self.worldConfig.maps else (np.ones_like(temp_field) * self.config.soil_capacity * 0.4)
        
        rain_accum = np.zeros_like(temp_field, dtype=np.float64)
        runoff_accum = np.zeros_like(temp_field, dtype=np.float64)

        wind = self.worldConfig["wind"]()
        w_i = wind[..., 0] # m/s eastward
        w_j = wind[..., 1] # m/s northward (positive j is south, so this is actually southward speed)
        
        speed_i = w_i * self.config.cells_per_ms_per_iter
        speed_j = w_j * self.config.cells_per_ms_per_iter

        lat_rows = get_lat_grid(self.worldConfig)
        lat_abs = np.abs(lat_rows)
        itcz = np.clip(1.0 + 0.6 * np.exp(-(lat_abs / 12.0)**2) - 0.4 * np.exp(-((lat_abs - 30.0) / 8.0)**2), 0.3, 1.8)
        itcz_factor = np.ascontiguousarray(np.broadcast_to(itcz[:, None], temp_field.shape).copy(), dtype=np.float64)

        grad_i = np.ascontiguousarray(self.worldConfig["grad_i"](), dtype=np.float64)
        grad_j = np.ascontiguousarray(self.worldConfig["grad_j"](), dtype=np.float64)
        sun_map = np.ascontiguousarray(self.worldConfig["sun"](), dtype=np.float64)

        inv_iter = 1.0 / self.config.iterations

        for _ in range(self.config.iterations):
            evap_frac = compute_evaporation_numba(
                temp_field, sun_map, w_i, w_j, sea_m, lake_m, river_m, s_init,
                self.config.evaporation_rate, self.config.land_evaporation,
                self.config.sea_evaporation, self.config.lake_evaporation,
                self.config.river_evaporation, self.config.soil_capacity
            )

            h_init = advect_numba(h_init, speed_i, speed_j, self.config.max_advection_cells)
            cap = humidity_capacity_numba(temp_field)

            h_init, s_init, rain_accum, runoff_accum = compute_rain_and_update_numba(
                h_init, cap, evap_frac, w_i, w_j, grad_i, grad_j, sea_m, lake_m, river_m,
                s_init, rain_accum, runoff_accum, self.config.condensation_rate,
                self.config.orographic_factor, self.config.soil_capacity,
                self.config.soil_evap_rate, itcz_factor, self.config.rain_shadow_fraction,
                self.config.wilting_point, self.config.hpa_to_mm_factor, inv_iter
            )

            h_init = gaussian_filter(h_init, sigma=self.config.diffusion_sigma)

        # Scale down annual accumulations to reasonable bounds
        rain_accum = (rain_accum / self.config.iterations) * 12.0
        runoff_accum = (runoff_accum / self.config.iterations) * 12.0

        return h_init, rain_accum, s_init, runoff_accum