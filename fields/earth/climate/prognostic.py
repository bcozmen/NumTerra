import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

#T = (Ta, Ts, Tw)

map_info = {
    'Sun' : {
        'unit' : 'W/m2',
        'description' : 'Solar energy input to the surface'
    },
    'P' : {
        'unit' : 'Pa',
        'description' : 'Atmospheric pressure'
    },
    'Wa_max' : {
        'unit' : 'kg/m2',
        'description' : 'Maximum atmospheric water capacity'
    },
    'Evap' : {
        'unit' : 'mm/hr',
        'description' : 'Evaporation rate'
    },
    'Precip' : {
        'unit' : 'mm/hr',
        'description' : 'Precipitation rate'
    },
}

@dataclass
class PrognosticClimateConfig:
    solar_constant: float = 1361.0  # Solar constant (W/m2)
    moisture_capacity_constant: float = 0.622 # Ratio of molecular weights of water to dry air, used in max moisture calculation
    moisture_scale_height: float = 2500.0 # Scale height for moisture distribution in the atmosphere (m)

    P0 : float = 101325.0  # Reference pressure at sea level (Pa)
    R : float = 287.05     # Specific gas constant for dry air (J/kg/K)
    g : float = 9.80665    # Gravitational acceleration (m/s2)

    Ce : float = 0.0015 # Evaporation coefficient (tunable parameter for evaporation rate)
    precip_conversion_rate : float = 0.8 # Tunable parameter to convert from potential precipitation to actual precipitation, accounting for factors like runoff and infiltration
class PrognosticClimate(BaseModel):
    info = {
        'name':'prognostic_climate',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.config = PrognosticClimateConfig()  # Use default config values
        self.init()  # Run the simulation immediately to initialize maps

    ## ========== Simulation & Generation ==========
    def init(self):
        H, M_sea = self.world.area['H'](), self.world.area['M_sea']()  # Terrain height and sea mask
        # Initialize pressure and moisture maps based on terrain and sea mask
    
    def step(self):
        self._calculate()  # Recalculate maps based on updated terrain and temperature

    def generate(self, area):
        pass

    ## ========== Map Management ==========
    def get_maps(self):
        H = self.world.area['H']()
        H_grad_i, H_grad_j = self.world.area['H_grad_i'](), self.world.area['H_grad_j']()
        T = self.world.area['T']()
        Wa = self.world.area['Wa']()
        M_sea = self.world.area['M_sea']()
        Ws = self.world.area['Ws']()
        Vspeed = self.world.area['V_magnitude']()
        Evap = self.world.area['Evap']() 
        return H, H_grad_i, H_grad_j, T, Wa, M_sea, Ws, Vspeed

    def _calculate(self):
        H, H_grad_i, H_grad_j, T, Wa, M_sea, Ws, Vspeed = self.get_maps()

        maps = {}
        maps['Sun'] = self._calculate_sun(H_grad_i, H_grad_j, Wa, self.config.solar_constant, self.world['time'].solar_vectors)
        maps['P'] = self._calculate_pressure(H, T, Wa, self.config.P0, self.config.R, self.config.g)
        maps['Wa_max'] = self._calculate_max_moisture(T, maps['P'], self.config.moisture_capacity_constant, self.config.R, self.config.moisture_scale_height)
        maps['Evap'] = self._calculate_evaporation(M_sea, maps['Wa_max'], Wa, Vspeed, Ws, self.config.Ce)
        maps['Precip'] = self._calculate_precipitation(Wa, maps['Wa_max'], self.config.precip_conversion_rate)
        self.set_maps(maps)



    def _calculate_sun(self, H_grad_i, H_grad_j, Wa, solar_constant, solar_vectors):
        #TODO add shadows
        sx, sy, sz = solar_vectors  # (sx, sy, sz)
        sz = np.clip(sz, 0.0, 1.0)

        # Atmospheric attenuation
        transmission = 0.8 * np.exp(-0.05 * Wa)

        #
        nx, ny, nz = -H_grad_i, -H_grad_j, 1.0
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        solar_norm = np.sqrt(sx**2 + sy**2 + sz**2) + 1e-6
        sx, sy, sz = sx / solar_norm, sy / solar_norm, sz / solar_norm

        cos_incidence = np.clip(nx * sx + ny * sy + nz * sz, 0.0, 1.0)
        # Surface solar flux
        solar_flux = solar_constant * cos_incidence * transmission
        return solar_flux

    def _calculate_pressure(self, H, T, Wa, P0, R, g):
        Tk = T[0] + 273.15  # Convert to Kelvin

        column_air_mass = P0 / g
        q= Wa / (column_air_mass)
        
        Tv = Tk * (1.0 + 0.61 * q)  # Virtual temperature
        P = P0 * np.exp(-g * H / (R * Tv))
        return P

    def _calculate_max_moisture(self, T, P, moisture_capacity_constant, R, moisture_scale_height):
        """Estimate column maximum water (kg/m2) using local pressure `P`."""
        Ta = T[0]
        # Saturation vapour pressure in Pa (Magnus-Tetens approximation)
        es = 6.112 * np.exp((17.67 * Ta) / (Ta + 243.5)) * 100.0

        # Prevent es exceeding local pressure (numerical safety)
        es = np.minimum(es, 0.99 * P)

        # epsilon = ratio of molecular weights of water vapour/dry air
        epsilon = moisture_capacity_constant

        # More exact saturation specific humidity (qs, kg/kg):
        # qs = epsilon * es / (P - (1 - epsilon) * es)
        denom = P - (1.0 - epsilon) * es
        denom = np.maximum(denom, 1e-6)
        qs = epsilon * es / denom

        # Local surface air density (kg/m3)
        rho_surface = P / (R * (Ta + 273.15))

        # Convert to column-integrated kg/m2 using an effective moisture
        # scale height for the atmospheric column.
        return rho_surface * qs * moisture_scale_height

    def _calculate_evaporation(self, M_sea, Wa_max, Wa, Vspeed, Ws, Ce):
        Vspeed = Vspeed + 0.1  # Avoid zero wind speed for evaporation

        evap_potential = Ce * Vspeed * np.maximum(0.0, Wa_max - Wa)

        sea_evaporation = evap_potential
        land_evaporation = np.minimum(evap_potential, Ws)

        return M_sea * sea_evaporation + (1 - M_sea) * land_evaporation

    def _calculate_precipitation(self, Wa, Wa_max, precip_conversion_rate):
        return np.maximum(0.0, Wa - Wa_max) * precip_conversion_rate

    