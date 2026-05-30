import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

from .engines.numba import precompute_horizon_angles, lookup_shadow_from_horizon

#T = (Ta, Ts, Tw)

map_info = {
    'Sun' : {
        'unit' : 'W/m²',
        'description' : 'Solar energy input to the surface',
        'render' : {'cmap': 'hot'},
    },
    'Shadow' : {
        'unit' : 'bool/float',
        'description' : 'Shadow map (0.0 = completely shadowed, 1.0 = fully lit)',
        'render' : {'cmap': 'gray'},
    },
    'P' : {
        'unit' : 'Pa',
        'description' : 'Atmospheric pressure',
        'render' : {'cmap': 'RdBu_r', 'scale': 'linear'},
    },
    'Wa_max' : {
        'unit' : 'kg/m²',
        'description' : 'Maximum atmospheric water capacity',
        'render' : {'cmap': 'YlGnBu', 'scale': 'linear'},
    },
    'Evap' : {
        'unit' : 'mm/hr',
        'description' : 'Evaporation rate',
        'render' : {'cmap': 'PuBu', 'scale': 'linear'},
    },
    'Condensation' : {
        'unit' : 'mm/hr',
        'description' : 'Condensation rate (vapor to cloud)',
        'render' : {'cmap': 'PuBuGn', 'scale': 'linear'},
    },
    'Precip' : {
        'unit' : 'mm/hr',
        'description' : 'Precipitation rate (cloud to surface)',
        'render' : {'cmap': 'Blues', 'scale': 'linear'},
    },
}

    


@dataclass
class PrognosticClimateConfig:
    solar_constant: float = 1361.0 # Solar constant (W/m²) at Earth's distance from the Sun
    moisture_capacity_constant: float = 0.622 # Ratio of molecular weights of water to dry air
    moisture_scale_height: float = 2500.0 # Scale height for moisture distribution (m) - Left for legacy/other uses

    layer_pressure_drop: float = 15000.0 # Pressure drop per atmospheric layer for moisture estimation (Pa)
    pressure_lapse_rate: float = 0.0008  # Rough temperature drop per Pa

    P0 : float = 101325.0  # Reference pressure at sea level (Pa)
    R : float = 287.05     # Specific gas constant for dry air (J/kg/K)
    g : float = 9.80665    # Gravitational acceleration (m/s2)

    Ce_water : float = 0.0015 # Evaporation coefficient over water (tunable parameter for evaporation rate)
    Ce_land : float = 0.0008  # Evaporation coefficient over land
    precip_conversion_rate : float = 1.0 # Tunable parameter for actual precipitation conversion
    cloud_delay_factor : float = 0.5 # Proportion of precip that remains as clouds per tick

    horizon_n_dirs : int = 16  # Number of azimuth directions for the precomputed horizon shadow map.
                               # Higher = more accurate shadow edges; 16 is a good default.

    # Solar transmission parameters
    cloud_transmission_coef: float = 0.5 # Extinction coeff for clouds in shortwave
    vapor_transmission_coef: float = 0.005 # Extinction coeff for water vapor in shortwave
    base_solar_transmission: float = 0.8 # Global baseline atmospheric shortwave transmission

class PrognosticClimate(BaseModel):
    info = {
        'name':'prognostic_climate',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.config = PrognosticClimateConfig()  # Use default config values
        self._horizon_angles = None  # Lazily computed; cached after first real terrain is available
        self.init()  # Run the simulation immediately to initialize maps

    ## ========== Simulation & Generation ==========
    def init(self):
        self._calculate()  # Bootstrap maps
    
    def step(self):
        self._calculate()  # Recalculate maps based on updated terrain and temperature

    def generate(self, area):
        pass

    def invalidate_horizon_cache(self):
        """Discard the cached horizon angles. Call after terrain (H) has changed."""
        self._horizon_angles = None

    ## ========== Map Management ==========
    def get_maps(self):
        H = self.world.area['H']()
        H_grad_i, H_grad_j = self.world.area['H_grad_i'](), self.world.area['H_grad_j']()
        Ta = self.world.area['Ta']()
        Wa = self.world.area['Wa']()
        Wc = self.world.area['Wc']()
        M_sea = self.world.area['M_sea']()
        Ws = self.world.area['Ws']()
        Vspeed = self.world.area['V_magnitude']()
        Evap = self.world.area['Evap']() 
        return H, H_grad_i, H_grad_j, Ta, Wa, Wc, M_sea, Ws, Vspeed

    def _calculate(self):
        """Computes dependent maps (Sun, Pressure, Moisture Capacity, Evaporation, Precipitation) sequentially."""
        H, H_grad_i, H_grad_j, Ta, Wa, Wc, M_sea, Ws, Vspeed = self.get_maps()

        maps = {}
        maps['Sun'], maps['Shadow'] = self._calculate_sun(
            H, H_grad_i, H_grad_j, M_sea, Wa, Wc, self.config.solar_constant, self.world['time'].solar_vectors,
            self.config.cloud_transmission_coef, self.config.vapor_transmission_coef, self.config.base_solar_transmission
        )
        maps['P'] = self._calculate_pressure(H, Ta, Wa, self.config.P0, self.config.R, self.config.g)
        
        # Updated call matching the new layered calculation signature
        maps['Wa_max'] = self._calculate_max_moisture(Ta, maps['P'], self.config.moisture_capacity_constant, self.config.g)
        
        maps['Evap'] = self._calculate_evaporation(M_sea, maps['Wa_max'], Wa, Vspeed, Ws, self.config.Ce_water, self.config.Ce_land)
        maps['Condensation'], maps['Precip'] = self._calculate_precipitation(Wa, Wc, maps['Wa_max'], self.config.precip_conversion_rate, self.config.cloud_delay_factor)
        self.set_maps(maps)

    def _calculate_sun(self, H, H_grad_i, H_grad_j, M_sea, Wa, Wc, solar_constant, solar_vectors,
                       cloud_transmission_coef, vapor_transmission_coef, base_solar_transmission):
        """Calculates solar flux across the map, accounting for terrain slope incident angles."""
        sx, sy, sz = solar_vectors
        dx, dy = self.world.area.cell_size

        # --- Shadow map via precomputed horizon angles ----------------------------
        # Precompute once (lazily) when real terrain is available; O(1) per step after that.
        if self._horizon_angles is None and H.max() > 0.001:
            self._horizon_angles = precompute_horizon_angles(
                H, self.config.horizon_n_dirs, self.world.max_altitude, dx, dy
            )

        if self._horizon_angles is not None:
            shadow_map = lookup_shadow_from_horizon(self._horizon_angles, sx, sy, sz)
        else:
            # Flat / uninitialised terrain: no cast shadows, respect sun below horizon
            shadow_map = np.ones(H.shape, dtype=np.float32)
            if sz <= 0.0:
                shadow_map[:] = 0.0
        # --------------------------------------------------------------------------

        if sz <= 0.0:
            return np.zeros_like(H), np.zeros_like(H)
            
        sz = np.clip(sz, 0.0, 1.0)

        # Clouds heavily block solar transmission. Water vapor has a moderate effect.
        cloud_transmission = np.exp(-cloud_transmission_coef * Wc) # Liquid water blocks heavily
        # Water vapor is mostly transparent to visible light (shortwave), so we lower the coefficient
        vapor_transmission = np.exp(-vapor_transmission_coef * Wa)
        transmission = base_solar_transmission * vapor_transmission * cloud_transmission

        # With imshow(origin='lower'): row-axis = North, col-axis = East.
        # The outward surface normal in (East, North, Up) order is (-∂H/∂col, -∂H/∂row, 1)
        # = (-H_grad_j, -H_grad_i, 1).  Pairing with solar vector (sx=East, sy=North, sz=Up).
        # Water/ocean surfaces are flat — suppress horizontal normal components on sea cells
        # so their incidence is determined only by the solar elevation angle, not terrain tilt.
        land_mask = 1.0 - M_sea
        nx = -H_grad_j * land_mask
        ny = -H_grad_i * land_mask
        nz = np.ones_like(H)
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        solar_norm = np.sqrt(sx**2 + sy**2 + sz**2) + 1e-6
        sx, sy, sz = sx / solar_norm, sy / solar_norm, sz / solar_norm

        cos_incidence = np.clip(nx * sx + ny * sy + nz * sz, 0.0, 1.0)
        
        # solar_flux: cos_incidence handles slope incidence, shadow_map handles cast shadows
        solar_flux = solar_constant * cos_incidence * transmission * shadow_map

        # Shadow map stored for display: slope shading × cast shadow → continuous [0, 1]
        # 0 = back-facing slope or blocked by terrain; 1 = fully lit, directly facing sun
        display_shadow = shadow_map * cos_incidence
        return solar_flux, display_shadow

    def _calculate_pressure(self, H, Ta, Wa, P0, R, g):
        """Calculates global atmospheric surface pressure maps."""
        Tk = Ta + 273.15  # Convert to Kelvin
        
        H_m = H * self.world.max_altitude  # Scale normalized H to meters
        
        # Calculate local column mass by assuming initial standard pressure,
        # then iteratively adjusting to local conditions. 
        P_approx = P0 * np.exp(-g * H_m / (R * Tk))
        column_air_mass = P_approx / g
        q = Wa / np.maximum(column_air_mass, 1e-6)

        Tv = Tk * (1.0 + 0.61 * q)  # Virtual temperature
        P = P0 * np.exp(-g * H_m / (R * Tv))
        return P

    def _calculate_max_moisture(self, Ta, P, moisture_capacity_constant, g):
        """Estimate column maximum water (kg/m2) by integrating through 5 atmospheric layers."""
        Wa_max = np.zeros_like(P)
        P_current = P.copy()
        Ta_current = Ta.copy()  # Surface temperature in Celsius
        
        # Iterate through ~4-5 atmospheric layers
        for _ in range(5): 
            # Create safety masks for layers pushing past the top of the atmosphere
            valid_mask = P_current > 0
            P_safe = np.maximum(P_current, 1e-5)

            # Saturation vapor pressure in Pa (Magnus-Tetens approximation)
            es = 6.112 * np.exp((17.67 * Ta_current) / (Ta_current + 243.5)) * 100.0
            es = np.minimum(es, 0.99 * P_safe)
            
            # Saturation specific humidity (qs, kg/kg)
            denom = P_safe - (1.0 - moisture_capacity_constant) * es
            qs = moisture_capacity_constant * es / np.maximum(denom, 1e-6)
            
            # Add this layer's capacity to the total (mass of layer = delta_P / g)
            layer_mass = self.config.layer_pressure_drop / g
            Wa_max += np.where(valid_mask, qs * layer_mass, 0.0)
            
            # Move up to the next layer
            P_current -= self.config.layer_pressure_drop
            Ta_current -= self.config.pressure_lapse_rate * self.config.layer_pressure_drop
            
        return Wa_max

    def _calculate_evaporation(self, M_sea, Wa_max, Wa, Vspeed, Ws, Ce_water, Ce_land):
        """Calculates global evaporation rates, considering surface water availability and wind speed."""
        Vspeed = Vspeed + 0.1  # Avoid zero wind speed
        dt = self.world['time'].dt
        
        # Calculate distinct evaporation potentials for land and water using their unique coefficients
        evap_potential_water = Ce_water * Vspeed * np.maximum(0.0, Wa_max - Wa)
        evap_potential_land = Ce_land * Vspeed * np.maximum(0.0, Wa_max - Wa)

        sea_evaporation = evap_potential_water
        # Land evaporation is limited by the actual soil moisture available per hour
        land_evaporation = np.minimum(evap_potential_land, Ws / dt)

        return M_sea * sea_evaporation + (1 - M_sea) * land_evaporation

    def _calculate_precipitation(self, Wa, Wc, Wa_max, precip_conversion_rate, cloud_delay_factor):
        """Calculates precipitation rate and condensation rates."""
        dt = self.world['time'].dt
        # Calculate condensation first: vapor exceeding max capacity turns into liquid clouds.
        # Condense the entire excess over the current time step (dt).
        condensation = np.maximum(0.0, Wa - Wa_max) / dt
        
        # Precipitation falls from already formed clouds
        # Delay factor moderates how much liquid rapidly drops vs stays afloat
        # We use exponential decay to prevent overshooting (Wc going negative) for large dt.
        removal_rate = precip_conversion_rate * (1.0 - cloud_delay_factor)
        removed_fraction = 1.0 - np.exp(-removal_rate * dt)
        precip = (Wc * removed_fraction) / dt
        
        return condensation, precip