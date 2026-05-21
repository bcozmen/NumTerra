from dataclasses import dataclass, field

import numpy as np

from .helper import get_temperature_grid, get_water_masks, get_water_cooling, get_sun_heating
from utils.functions import FastInterpolator
from utils import  timeit

@dataclass
class ThermalConfig:    
    lapse_rate : float = 6.5 #°C per 1000m

    season_to_declination : dict = field(default_factory=lambda: {
            "spring": 10,
            "summer": 23.44,
            "autumn": -10,
            "winter": -23.44
    })
    
    cooling_effects : dict = field(default_factory=lambda: {
        'sea': (3, 150_000), #°C, meters
        'lake': (2, 15_000), #°C, meters
        'river': (0.3, 8_000), #°C, meters
        'continentality': (8, 150_000), #°C, meters
        'sun' : (5, 1) #°C, unitless
    })

    # Latent-heat feedback (from previous humidity iteration).
    # Condensation releases ~2500 J/g; this simplified factor maps normalised rain
    # [0-1] to °C of surface warming.  Realistic order-of-magnitude: 1-4 °C.
    # Reference: 2000 mm/yr ~ tropical rainforest maximum.  Normalising by this
    # fixed value (rather than rain.max()) keeps the scale consistent across
    # iterations even as the precipitation distribution converges.
    latent_heat_factor: float = 3.0
    latent_heat_ref_rain: float = 2000.0   # mm/yr — normalisation reference

    # Greenhouse-like humidity warming: high atmospheric humidity traps outgoing
    # longwave radiation.  Maps normalised humidity [0-1] to °C of warming.
    # Reference: 50 hPa ≈ saturation at ~35 °C (tropical maximum).
    humidity_greenhouse_factor: float = 1.5
    humidity_greenhouse_ref: float = 50.0  # hPa — normalisation reference



class Thermal:
    @timeit(label="Thermal Initialization")
    def __init__(self, worldConfig):
        self.config = ThermalConfig()
        self.worldConfig = worldConfig
        self.worldConfig["model_thermal"] = self

        self.run()
        
    
    def __call__(self, area = None):
        if area is None: self.run()
        else: self.generate(area)
    @timeit(label="Thermal Simulation")
    def run(self):
        temperature, sun = self.init_temperature_map()
        self.worldConfig["temperature"] = temperature
        self.worldConfig["sun"] = sun
    @timeit(label="Thermal Generation")
    def generate(self, area):
        grid, points, size = area.grid, area.points, area.size
        area["temperature"] = self.worldConfig["temperature"](points).reshape(size)
        area["sun"] = self.worldConfig["sun"](points).reshape(size)

    # ---- Initialization ----
    def init_temperature_map(self):
        size, max_size, latitude = self.worldConfig.size, self.worldConfig.max_size, self.worldConfig.latitude
        base_temp = get_temperature_grid(size, max_size, latitude)
        
        altitude = self.worldConfig["height"]() * self.worldConfig.max_altitude
        altitude_effect = (altitude / 1000) * self.config.lapse_rate

        water_cooling_effect, continentality = get_water_cooling(self.worldConfig, self.config)

        sun = get_sun_heating(self.worldConfig, self.config) 
        sun_heating = self.config.cooling_effects['sun'][0] * sun

        temperature = base_temp - altitude_effect - water_cooling_effect + continentality + sun_heating

        # ---- Feedbacks from previous humidity iteration (2nd+ pass) ----
        # Latent heat: condensation/rain releases heat in the atmosphere.
        # Normalised by a fixed reference (not rain.max()) so the temperature
        # contribution is stable and comparable across iterations.
        if "rain" in self.worldConfig.maps:
            rain = self.worldConfig["rain"]()
            rain_norm = np.clip(rain / self.config.latent_heat_ref_rain, 0.0, 1.0)
            temperature += self.config.latent_heat_factor * rain_norm

        # Greenhouse-like humidity warming: moist air traps outgoing longwave radiation.
        # Normalised by a fixed saturation-pressure reference (not humidity.max()).
        if "humidity" in self.worldConfig.maps:
            humidity = self.worldConfig["humidity"]()
            humidity_norm = np.clip(humidity / self.config.humidity_greenhouse_ref, 0.0, 1.0)
            temperature += self.config.humidity_greenhouse_factor * humidity_norm

        temperature = FastInterpolator(temperature, order=1)
        sun = FastInterpolator(sun, order=1)
        return temperature, sun
    
    


    




