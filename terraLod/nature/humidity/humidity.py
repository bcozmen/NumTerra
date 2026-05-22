from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import numpy as np

from terraLod.utils import FastInterpolator, timeit, get_lat_grid, get_water_masks


from .helper import get_itcz

from .numba import advect_numba, humidity_capacity, compute_evaporation_numba, compute_rain_and_update_numba

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

    #### ========== Simulation & Generation ==========

    @timeit(label="Humidity Simulation")
    def run(self):
        humidity_map, rain_map, soil_map, runoff_map = self._simulate_climate()
        self.set_maps(humidity_map, rain_map, soil_map, runoff_map)

    @timeit(label="Humidity Generation")
    def generate(self, area):
        pts, size = area.points, area.size
        area["humidity"]      = self.worldConfig["humidity"](pts).reshape(size)
        area["rain"]          = self.worldConfig["rain"](pts).reshape(size)
        area["soil_moisture"] = self.worldConfig["soil_moisture"](pts).reshape(size)
        area["runoff"]        = self.worldConfig["runoff"](pts).reshape(size)

    ### ========== Map Management ==========

    def set_maps(self, humidity_map, rain_map, soil_map, runoff_map):
        self.worldConfig["humidity"]      = FastInterpolator(humidity_map, order=1) 
        self.worldConfig["rain"]          = FastInterpolator(rain_map,     order=1)
        self.worldConfig["soil_moisture"] = FastInterpolator(soil_map,     order=1)
        self.worldConfig["runoff"]        = FastInterpolator(runoff_map,   order=1)
    
    def get_maps(self):
        sea, lake, river = get_water_masks(self.worldConfig)
        temperature = self.worldConfig["temperature"]()
        wind = self.worldConfig["wind"]() 
        sun = self.worldConfig["sun"]() 

        grad_i, grad_j = self.worldConfig["grad_i"](), self.worldConfig["grad_j"]()
        
        humidity = self.worldConfig["humidity"]() if "humidity" in self.worldConfig.maps else (humidity_capacity(temperature) * 0.5)
        soil_moisture = self.worldConfig["soil_moisture"]() if "soil_moisture" in self.worldConfig.maps else (np.ones_like(temperature) * self.config.soil_capacity * 0.4)
        
        return sea, lake, river, temperature, humidity, soil_moisture, wind, sun, grad_i, grad_j
    
    ## ========= Climate Simulation Core ==========

    def _simulate_climate(self):
        sea_m, lake_m, river_m, temperature, humidity, soil_moisture, wind, sun, grad_i, grad_j = self.get_maps()
        w_i, w_j = wind[..., 0], wind[..., 1]
        
        rain = np.zeros_like(temperature, dtype=np.float32)
        runoff = np.zeros_like(temperature, dtype=np.float32)

        speed_i, speed_j = w_i * self.config.cells_per_ms_per_iter, w_j * self.config.cells_per_ms_per_iter

        itcz = get_itcz(get_lat_grid(self.worldConfig.latitude, temperature.shape, self.worldConfig.max_size))

        inv_iter = 1.0 / self.config.iterations

        for _ in range(self.config.iterations):
            evap_frac = compute_evaporation_numba(
                temperature, sun, w_i, w_j, sea_m, lake_m, river_m, soil_moisture,
                self.config.evaporation_rate, self.config.land_evaporation,
                self.config.sea_evaporation, self.config.lake_evaporation,
                self.config.river_evaporation, self.config.soil_capacity
            )

            humidity = advect_numba(humidity, speed_i, speed_j, self.config.max_advection_cells)
            cap = humidity_capacity(temperature)

            humidity, soil_moisture, rain, runoff = compute_rain_and_update_numba(
                humidity, cap, evap_frac, w_i, w_j, grad_i, grad_j, sea_m, lake_m, river_m,
                soil_moisture, rain, runoff, self.config.condensation_rate,
                self.config.orographic_factor, self.config.soil_capacity,
                self.config.soil_evap_rate, itcz, self.config.rain_shadow_fraction,
                self.config.wilting_point, self.config.hpa_to_mm_factor, inv_iter
            )

            humidity = gaussian_filter(humidity, sigma=self.config.diffusion_sigma)

        rain = (rain / self.config.iterations) * 12.0
        runoff = (runoff / self.config.iterations) * 12.0

        return humidity, rain, soil_moisture, runoff
