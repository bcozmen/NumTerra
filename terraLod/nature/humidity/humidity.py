from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
import numpy as np

from terraLod.utils import get_lat_grid, get_water_masks, timeit


from .helper import get_itcz, humidity_capacity
from .numba import advect_numba, compute_evaporation_numba, compute_rain_and_update_numba

@dataclass
class HumidityConfig:
    world_max_size: float              # [meters] Maximum dimension of the simulation world
    cell_size: tuple                   # [meters] Dimensions of a single grid cell (dx, dy)
    iterations: int = 10               # [count] Number of simulation ticks per generation step

    advection_iterations: int = 1             # [count] Number of sub-steps for advection within each main iteration (higher = more accurate but slower)
    max_advection_percent: float = 0.1 # [fraction, 0-1] Max fraction of world size a 1m/s wind parcel can move over one iterations
    diffusion_sigma: float = 1.0       # [cells] Gaussian blur radius for humidity smoothing (lower = sharper transitions)
    
    evaporation_rate: float = 3.0     # [multiplier] Base atmospheric evaporation scale. Doubled vs the old cap-based formula because evaporation now scales with the vapour-pressure deficit (cap - humidity), so the effective rate is halved at 50 % RH; 2.0 restores equivalent global moisture flux while still self-limiting near saturation.
    sea_evaporation: float = 1.0       # [multiplier] Evaporation efficiency factor over oceans
    lake_evaporation: float = 0.8      # [multiplier] Evaporation efficiency factor over lakes
    river_evaporation: float = 0.6     # [multiplier] Evaporation efficiency factor over rivers
    land_evaporation: float = 0.1      # [multiplier] Evaporation efficiency factor over bare land
    
    rain_humidity_threshold: float = 0.6 # [fraction, 0-1] Relative humidity required before rain triggers (higher = less frequent but more intense)
    condensation_rate: float = 0.7        # [fraction, 0-1] Proportion of excess humidity condensing into rain per iteration
    vapor_column_height: float = 6000.0   # [meters] Effective water-vapor column scale for hPa to mm conversion
    
    uplift_scale: float = 1.0          # [multiplier] How strongly terrain slope/uplift forces air upward (higher = dramatic mountain rain)
    orographic_factor: float = 0.01     # [multiplier] Efficiency of mountains squeezing water out of humid air masses
    

    soil_capacity: float = 200.0       # [mm] Maximum amount of water the soil layer can retain

    @property
    def cells_per_ms_per_iter(self):
        L_cells = self.world_max_size / self.cell_size[0]
        return (self.max_advection_percent * L_cells) / self.iterations / self.advection_iterations

    @property
    def max_advection_cells(self):
        return 5 * self.cells_per_ms_per_iter  # Max movement in cells per iteration, doubled for safety margin


class Humidity:
    def __init__(self, worldConfig):
        self.worldConfig = worldConfig
        self.worldConfig["model_humidity"] = self
        self.config = HumidityConfig(world_max_size=self.worldConfig.max_size, cell_size=self.worldConfig.cell_size)
        self.run()

    def __call__(self, area=None):
        if area is None: self.run()
        else: self.generate(area)

    #### ========== Simulation & Generation ==========
    @timeit(name="Humidity Simulation")
    def run(self):
        humidity_map, rain_map, soil_map, runoff_map = self._simulate_climate()
        self.set_maps(humidity_map, rain_map, soil_map, runoff_map)

    def generate(self, area):
        pts, size = area.points, area.size
        area["humidity"]      = self.worldConfig["humidity"](pts).reshape(size)
        area["rain"]          = self.worldConfig["rain"](pts).reshape(size)
        area["soil_moisture"] = self.worldConfig["soil_moisture"](pts).reshape(size)
        area["runoff"]        = self.worldConfig["runoff"](pts).reshape(size)

    ### ========== Map Management ==========

    def set_maps(self, humidity_map, rain_map, soil_map, runoff_map):
        self.worldConfig["humidity"]      = humidity_map
        self.worldConfig["rain"]          = rain_map
        self.worldConfig["soil_moisture"] = soil_map
        self.worldConfig["runoff"]        = runoff_map
    
    def get_maps(self):
        sea, lake, river = get_water_masks(self.worldConfig)
        temperature = self.worldConfig["temperature"]()
        wind = self.worldConfig["wind"]() 
        sun = self.worldConfig["sun"]() 

        grad_i, grad_j = self.worldConfig["height_grad_i"](), self.worldConfig["height_grad_j"]()
        
        humidity = self.worldConfig["humidity"]() if "humidity" in self.worldConfig.maps else (humidity_capacity(temperature) * 0.5)
        soil_moisture = self.worldConfig["soil_moisture"]() if "soil_moisture" in self.worldConfig.maps else (np.ones_like(temperature) * self.config.soil_capacity * 0.4)
        
        return sea, lake, river, temperature, humidity, soil_moisture, wind, sun, grad_i, grad_j
    
    

    def _simulate_climate(self):
        sea_m, lake_m, river_m, temperature, humidity, soil_moisture, wind, sun, grad_i, grad_j = self.get_maps()
        w_i, w_j = wind[..., 0], wind[..., 1]
        
        rain = np.zeros_like(temperature, dtype=np.float32)
        runoff = np.zeros_like(temperature, dtype=np.float32)

        speed_i, speed_j = w_i * self.config.cells_per_ms_per_iter, w_j * self.config.cells_per_ms_per_iter
        #print("Max wind advection per iteration (cells):", np.max(np.sqrt(speed_i**2 + speed_j**2)))
        #print("Mean wind advection per iteration (cells):", np.mean(np.sqrt(speed_i**2 + speed_j**2)))
        #print(f"Max advection allowed per iteration (cells): {self.config.max_advection_cells:.2f}")

        itcz = get_itcz(get_lat_grid(self.worldConfig.latitude, temperature.shape, self.worldConfig.max_size))

        inv_iter = 1.0 / self.config.iterations

        cap = humidity_capacity(temperature)

        # Compute hPa->mm conversion per cell using local temperature
        hpa_to_mm = 100.0 * self.config.vapor_column_height / (461.0 * (temperature + 273.15))

        for iter_ix in range(self.config.iterations):
            evap_frac = compute_evaporation_numba(
                temperature, sun, w_i, w_j, sea_m, lake_m, river_m, soil_moisture,
                self.config.evaporation_rate, self.config.land_evaporation,
                self.config.sea_evaporation, self.config.lake_evaporation,
                self.config.river_evaporation, self.config.soil_capacity, inv_iter
            )
            
            pre_adv_mass = np.sum(humidity)
            for _ in range(self.config.advection_iterations):
                humidity = advect_numba(humidity, speed_i, speed_j, self.config.max_advection_cells)
            post_adv_mass = np.sum(humidity)
            if post_adv_mass > 0:
                humidity *= (pre_adv_mass / post_adv_mass)
            
            humidity, soil_moisture, rain, runoff = compute_rain_and_update_numba(
                humidity, cap, evap_frac, w_i, w_j, grad_i, grad_j, sea_m, lake_m, river_m,
                soil_moisture, rain, runoff, self.config.condensation_rate,
                self.config.orographic_factor, self.config.soil_capacity,
                itcz, self.config.rain_humidity_threshold,
                hpa_to_mm,  self.config.uplift_scale,
                )


            if self.config.diffusion_sigma > 0:
                pre_diff_mass = np.sum(humidity)
                humidity = gaussian_filter(humidity, sigma=self.config.diffusion_sigma)
                post_diff_mass = np.sum(humidity)
                if post_diff_mass > 0:
                    humidity *= (pre_diff_mass / post_diff_mass)
            

        return humidity, rain, soil_moisture, runoff
