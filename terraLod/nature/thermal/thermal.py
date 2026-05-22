from dataclasses import dataclass, field
import numpy as np
from .helper import get_temperature_grid, get_water_cooling, get_sun_heating, season_phase

from terraLod.utils import FastInterpolator, timeit


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
    marine_drift_amplitude: float = 6.0 # Max seasonal shift in mean temperature at the coast due to ocean thermal inertia (°C).
    latent_heat_factor: float = 1.0         # How much °C the air warms when heavy rain falls (simulating latent heat release).
    latent_heat_ref_rain: float = 2000.0 # Scaling factor for latent heat reference rain.
    humidity_greenhouse_factor: float = 1.5 # Extra heat trapped near surface in wet/humid tropical areas (°C).
    humidity_greenhouse_ref: float = 50.0   # hPa normalization baseline for greenhouse warming calculation.


class Thermal:
    @timeit(label="Thermal Initialization")
    def __init__(self, worldConfig):
        self.config = ThermalConfig()
        self.worldConfig = worldConfig
        self.worldConfig["model_thermal"] = self
        self.run()
        
    def __call__(self, area=None):
        if area is None: 
            self.run()
        else: 
            self.generate(area)

    @timeit(label="Thermal Simulation")
    def run(self):
        temperature, sun = self.init_temperature_map()
        self.worldConfig["temperature"] = temperature
        self.worldConfig["sun"] = sun

    @timeit(label="Thermal Generation")
    def generate(self, area):
        """Samples sub-regional grids from the precomputed global interpolators."""
        area["temperature"] = self.worldConfig["temperature"](area.points).reshape(area.size)
        area["sun"] = self.worldConfig["sun"](area.points).reshape(area.size)

    # ---- Initialization ----
    def init_temperature_map(self):
        altitude = self.worldConfig["height"]()
        sea_mask, sea_level = self.worldConfig["sea_mask"](), self.worldConfig["sea_level"]()
        declination = self.worldConfig.declination
        season = self.worldConfig.season

        solar_vectors = self.worldConfig.solar_vectors
        di, dj = self.worldConfig["grad_i"](), self.worldConfig["grad_j"]()

        phase = season_phase(season)

        # 1. Latitude-Based Temperature Gradient
        temp_mean, temp_delta = get_temperature_grid(self.worldConfig.size, self.worldConfig.max_size, self.worldConfig.latitude, phase)
        
        # 2. Topographic Altitude Effect (Lapse Rate)
        altitude = np.maximum(altitude - sea_level, 0.0)  # Treat anything below sea level as sea level for temperature purposes
        altitude_effect = ((altitude * self.worldConfig.max_altitude) / 1000.0) * self.config.lapse_rate

        # 3. Microclimate Alterations (Water Buffers & Continentality grids)
        water_buffer_effect, continentality = get_water_cooling(self.worldConfig, self.config)

        # 4. Aspect/Hillshade Solar Radiative Heating
        sun = self.config.cooling_effects['sun'][0] * get_sun_heating(di, dj, self.worldConfig.latitude, declination, solar_vectors) 
        sun[sea_mask] = 0.0  # No solar heating on water cells

        # Linear water buffer to multipplicative thermal inertia
        thermal_damping = np.clip(1.0 - (0.8 * water_buffer_effect) / (self.config.cooling_effects['sea'][0] + 1e-5), 0.2, 1.0)  # Damping factor between 0.5 and 1.0
        effective_seasonal_swing = (temp_delta * (1.0 + continentality)) * thermal_damping
        # Combine primary thermodynamic layers
        
        hemisphere_sign = np.where(self.worldConfig.latitude >= 0, 1, -1)
        # This creates a localized map that applies only to water and immediate coastal bands
        marine_drift = (phase * hemisphere_sign * self.config.marine_drift_amplitude) * (water_buffer_effect / self.config.cooling_effects['sea'][0])  # Max shift of 6°C at the coast, tapering off with distance from water
        temp_mean_shifted = temp_mean + marine_drift 
        
        temperature = temp_mean_shifted + effective_seasonal_swing - altitude_effect + sun
        print(temperature.mean(), temperature.min(), temperature.max())
        
        # 7. Planetary Feedback Ticks
        #if "rain" in self.worldConfig.maps:
        #    rain_norm = np.clip(self.worldConfig["rain"]() / self.config.latent_heat_ref_rain, 0.0, 1.0)
        #    temperature += self.config.latent_heat_factor * rain_norm

        #if "humidity" in self.worldConfig.maps:
        #    humidity_norm = np.clip(self.worldConfig["humidity"]() / self.config.humidity_greenhouse_ref, 0.0, 1.0)
        #    temperature += self.config.humidity_greenhouse_factor * humidity_norm

        return FastInterpolator(temperature, order=1), FastInterpolator(sun, order=1)
        
