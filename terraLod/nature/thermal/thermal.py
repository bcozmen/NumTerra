from dataclasses import dataclass, field
import numpy as np
from .helper import get_temperature_grid, get_sea_cooling, get_sun_heating, season_phase

from terraLod.utils import  get_water_masks, timeit
from scipy.ndimage import gaussian_filter


#TODO -
# Rain, wind, humidity, soil moisture is calculated now
# add feedback loops
#1. The Albedo Feedback (Snow & Ice) -> less heat absorbed
#2. Evaporative Cooling (Soil Moisture & Vegetation)

@dataclass
class ThermalConfig:    
    lapse_rate: float = 6.5  # °C drop per 1000m elevation. Natural physics default is 6.5. Increase to make mountains even colder.

    cooling_effects: dict = field(default_factory=lambda: {
        'sea': (5.0, 50_000.0),            # (Max effect in °C, Scale in meters) How far inland sea breezes cool the land.
        'lake': (2.0, 5_000.0),            # Localized cooling around lakes. 
        'river': (0.5, 1_000.0),            # Minor cooling near rivers.
        'continentality': (0.75, 200_000.0), # % of seasonal swing added back per 200km from coast. Makes deep interiors have more extreme seasons.
        'sun': (4.0,)                       # Max solar heating amplitude in °C. South-facing slopes get this much hotter.
    })
    marine_drift_amplitude: float = 4.0 # Max seasonal shift in mean temperature at the coast due to ocean thermal inertia (°C).
    latent_heat_factor: float = 1.0         # How much °C the air warms when heavy rain falls (simulating latent heat release).
    latent_heat_ref_rain: float = 300.0 # Scaling factor for latent heat reference rain.
    humidity_greenhouse_factor: float = 1.5 # Extra heat trapped near surface in wet/humid tropical areas (°C).
    humidity_greenhouse_ref: float = 15.0   # hPa normalization baseline for greenhouse warming calculation.
    blur_sigma: float = 10.0                   # Smoothing applied to temperature map to prevent extreme spikes and create more natural transitions.

class Thermal:
    def __init__(self, worldConfig):
        self.config = ThermalConfig()
        self.worldConfig = worldConfig
        self.worldConfig["model_thermal"] = self

        self.water_buffer_effect, self.continentality = None, None
        self.run()
        
    def __call__(self, area=None):
        if area is None: 
            self.run()
        else: 
            self.generate(area)

    #### ========== Simulation & Generation ==========
    @timeit(name="Thermal Simulation")
    def run(self):
        temperature, sun = self._init_temperature_map()
        self.set_maps(temperature, sun)

    def generate(self, area):
        area["temperature"] = self.worldConfig["temperature"](area.points).reshape(area.size)
        area["sun"] = self.worldConfig["sun"](area.points).reshape(area.size)

    ### ======== Maps Management ==========
    def set_maps(self, temperature_map, sun_map):
        self.worldConfig["temperature"] = temperature_map
        self.worldConfig["sun"] = sun_map

    def get_maps(self):
        sea_mask, lake_mask, river_mask = get_water_masks(self.worldConfig)
        sea_level = self.worldConfig["sea_level"]()

        height = self.worldConfig["height"]()
        di, dj = self.worldConfig["height_grad_i"](), self.worldConfig["height_grad_j"]()
        sun = self.worldConfig["sun"]()

        return sea_mask, lake_mask, river_mask, sea_level, height, sun, di, dj

    ### ========= Temperature Map Initialization Core ==========
    def _init_temperature_map(self):
        sea_mask, lake_mask, river_mask, sea_level, height, sun, di, dj = self.get_maps()
        #phase = season_phase(self.worldConfig.season)
        phase = self.worldConfig.time.season_phase
        # 1. Latitude-Based Temperature Gradient
        temp_mean, temp_delta = get_temperature_grid(self.worldConfig.size, self.worldConfig.max_size, self.worldConfig.latitude, phase)

        # 2. Topographic Altitude Effect (Lapse Rate)
        altitude = np.maximum(height - sea_level, 0.0)  # Treat anything below sea level as sea level for temperature purposes
        altitude_effect = ((altitude * self.worldConfig.max_altitude) / 1000.0) * self.config.lapse_rate

        # 3. Microclimate Alterations (Water Buffers & Continentality grids)
        if self.water_buffer_effect is None or self.continentality is None:
             self.water_buffer_effect, self.continentality = get_sea_cooling(self.worldConfig, self.config, sea_mask)
        

        # 4. Aspect/Hillshade Solar Radiative Heating
        sun = get_sun_heating(di, dj, self.worldConfig.latitude, self.worldConfig.time.declination, self.worldConfig.time.solar_vectors)
        sun_effect = self.config.cooling_effects["sun"][0] * sun
        sun_effect[sea_mask] = 0.0  # No solar heating on water cells

        # 5. Seasonal Swing Modulation by Continentality & Water Buffers
        thermal_damping = np.clip(1.0 - (0.8 * self.water_buffer_effect) / (self.config.cooling_effects['sea'][0] + 1e-5), 0.2, 1.0)  # Damping factor between 0.5 and 1.0
        effective_seasonal_swing = (temp_delta * (1.0 + self.continentality)) * thermal_damping

        # 6. Marine Thermal Inertia Drift
        hemisphere_sign = np.where(self.worldConfig.latitude >= 0, 1, -1)
        marine_drift = (phase * hemisphere_sign * self.config.marine_drift_amplitude) * (self.water_buffer_effect / self.config.cooling_effects['sea'][0])  # Max shift of 6°C at the coast, tapering off with distance from water
        temp_mean_shifted = temp_mean + marine_drift

        # 7. Planetary Feedback Ticks (e.g. latent heat from rain, greenhouse effect from humidity) could be added here as additional layers
        pass

        # Final combined temperature map
        temperature = temp_mean_shifted + effective_seasonal_swing - altitude_effect + sun_effect
        return temperature, sun

   