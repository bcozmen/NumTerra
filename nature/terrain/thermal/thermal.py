from dataclasses import dataclass, field
import numpy as np
from .helper import get_temperature_grid, get_water_cooling, get_sun_heating
from utils.functions import FastInterpolator
from utils import timeit


#TODO -
# Rain, wind, humidity, soil moisture is calculated now
# add feedback loops

@dataclass
class ThermalConfig:    
    lapse_rate: float = 4.5  # °C drop per 1000m elevation. Natural physics default is 6.5. Increase to make mountains even colder.
    
    cooling_effects: dict = field(default_factory=lambda: {
        'sea': (3.0, 150_000.0),            # (Max effect in °C, Scale in meters) How far inland sea breezes cool the land.
        'lake': (2.0, 15_000.0),            # Localized cooling around lakes. 
        'river': (0.3, 8_000.0),            # Minor cooling near rivers.
        'continentality': (5.0, 50_000.0), # Extreme inland warming effect. Deeper inland = much hotter summers.
        'sun': (3.5,)                       # Max solar heating amplitude in °C. South-facing slopes get this much hotter.
    })
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
        # 1. Base Planetary Temperature (Latitude Gradient)
        declination = self.worldConfig.worldConfig.declination
        solar_vectors = self.worldConfig.worldConfig.solar_vectors
        base_temp = get_temperature_grid(self.worldConfig.size, self.worldConfig.max_size, self.worldConfig.latitude, declination)
        
        # 2. Topographic Altitude Effect (Lapse Rate)
        altitude = self.worldConfig["height"]() * self.worldConfig.max_altitude
        altitude_effect = (altitude / 1000.0) * self.config.lapse_rate

        # 3. Microclimate Alterations (Water Buffers, Upwelling Currents, & Continentality)
        water_cooling_effect, continentality = get_water_cooling(self.worldConfig, self.config)

        # 4. Seasonality
        seasonal_amplitude = np.sin(declination) * 2  # Varies from -2 to +2 over the year
        # in summer makes inland hotter, in winter makes it colder (less moderating ocean influence)
        continentality *= seasonal_amplitude

        if seasonal_amplitude > 0:
            # In summer, amplify continentality effect (hotter inland)
            water_cooling_effect = water_cooling_effect * seasonal_amplitude
        else:
            water_cooling_effect = 0.5 * water_cooling_effect * seasonal_amplitude  # In winter, reduce cooling effect (less ocean moderation)

        # 4. Aspect/Hillshade Solar Radiative Heating
        sun = get_sun_heating(self.worldConfig, declination, solar_vectors) 
        sun_heating = self.config.cooling_effects['sun'][0] * sun

        # Combine primary thermodynamic layers
        temperature = base_temp - altitude_effect - water_cooling_effect + continentality + sun_heating

        # 5. Planetary Feedback Ticks (Multi-pass integrations)
        if "rain" in self.worldConfig.maps:
            rain_norm = np.clip(self.worldConfig["rain"]() / self.config.latent_heat_ref_rain, 0.0, 1.0)
            temperature += self.config.latent_heat_factor * rain_norm

        if "humidity" in self.worldConfig.maps:
            humidity_norm = np.clip(self.worldConfig["humidity"]() / self.config.humidity_greenhouse_ref, 0.0, 1.0)
            temperature += self.config.humidity_greenhouse_factor * humidity_norm

        return FastInterpolator(temperature, order=1), FastInterpolator(sun, order=1)
        
    


    




