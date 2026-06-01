import numpy as np
from dataclasses import dataclass, field
from .numba import compute_hydro_step, apply_mass_balance_numba

@dataclass
class HydroConfig:
    moisture_capacity_constant: float = 0.622 # Ratio of molecular weights of water to dry air

    layer_pressure_drop: float = 15000.0 # Pressure drop per atmospheric layer for moisture estimation (Pa)
    pressure_lapse_rate: float = 0.0008  # Rough temperature drop per Pa

    Ce_water : float = 4.16e-7 # Evaporation coefficient over water (dim-less roughly ~1e-3, but previously absorbed 3600)
    Ce_land : float = 2.22e-7  # Evaporation coefficient over land
    lake_evap_threshold: float = 20.0  # mm; land cells with Ws above this are treated as inland lakes and use Ce_water
    precip_conversion_rate : float = 1.0 # Tunable parameter for actual precipitation conversion
    cloud_delay_factor : float = 0.5 # Proportion of precip that remains as clouds per tick
    rH_condensation_threshold: float = 0.85  # Relative humidity at which clouds start forming (0-1)
    condensation_timescale: float = 3.0       # Hours over which excess vapor relaxes to clouds; shorter = harder threshold

class Hydro:
    def __init__(self, world, atmospheric_layer_count):
        self.world = world
        self.config = HydroConfig()
        self.atmospheric_layer_count = atmospheric_layer_count

    def __call__(self, P, Ta, Ts, Tw, Wa, Wc, M_sea, Ws, Vspeed):
        dt = self.world['time'].dt
        Wa_max, Evap, Condensation, Precip = compute_hydro_step(
            P, Ta, Ts, Tw, Wa, Wc, M_sea, Ws, Vspeed,
            self.config.moisture_capacity_constant, self.config.layer_pressure_drop, self.config.pressure_lapse_rate, self.world.constants['g'],
            self.config.lake_evap_threshold, self.config.Ce_water, self.config.Ce_land, dt,
            self.config.rH_condensation_threshold, self.config.condensation_timescale, self.config.precip_conversion_rate, self.config.cloud_delay_factor,
            self.atmospheric_layer_count
        )
        return Wa_max, Evap, Condensation, Precip

    def apply_mass_balance(self, Ta, Wa, Wc, Ws, Wa_max, Evap, Condensation, Precip, dt, thermal):
        """Applies mass balance and enforces non-negativity and hard saturation cap.
        
        Returns updated (Ta, Wa, Wc, Ws, Condensation).
        """
        Lv = self.world.constants['Lv']
        c_air = thermal.config.c_air
        
        apply_mass_balance_numba(Ta, Wa, Wc, Ws, Wa_max, Evap, Condensation, Precip, dt, Lv, c_air)
        return Ta, Wa, Wc, Ws, Condensation



class HydroNoNumba:
    def __init__(self, world):
        self.world = world
        self.config = HydroConfig()

    def __call__(self, P, Ta, Ts, Tw, Wa, Wc, M_sea, Ws, Vspeed):
        Wa_max = self._calculate_max_moisture(Ta, P)
        Wa_max_surf = self._calculate_max_moisture(Ts * (1 - M_sea) + Tw * M_sea, P)
        Evap = self._calculate_evaporation(M_sea, Wa_max_surf, Wa, Vspeed, Ws)
        Condensation, Precip = self._calculate_precipitation(Wa, Wc, Wa_max)
        return Wa_max, Evap, Condensation, Precip

    def _calculate_max_moisture(self, Ta, P):
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
            denom = P_safe - (1.0 - self.config.moisture_capacity_constant) * es
            qs = self.config.moisture_capacity_constant * es / np.maximum(denom, 1e-6)
            
            # Add this layer's capacity to the total (mass of layer = delta_P / g)
            layer_mass = self.config.layer_pressure_drop / self.world.constants['g']
            Wa_max += np.where(valid_mask, qs * layer_mass, 0.0)
            
            # Move up to the next layer
            P_current -= self.config.layer_pressure_drop
            Ta_current -= self.config.pressure_lapse_rate * self.config.layer_pressure_drop
            
        return Wa_max

    def _calculate_evaporation(self, M_sea, Wa_max, Wa, Vspeed, Ws):
        Vspeed = Vspeed + 0.1  # Avoid zero wind speed
        dt = self.world['time'].dt
        
        # Inland lake mask: non-sea land cells with standing water above threshold evaporate at the water rate,
        # preventing unlimited basin accumulation. Below the threshold, use the slower land coefficient.
        M_lake = np.float32(Ws > self.config.lake_evap_threshold) * (1.0 - M_sea)
        Ce_land_eff = M_lake * self.config.Ce_water + (1.0 - M_lake) * self.config.Ce_land

        evap_potential_water = self.config.Ce_water * Vspeed * np.maximum(0.0, Wa_max - Wa) * 3600.0
        evap_potential_land  = Ce_land_eff         * Vspeed * np.maximum(0.0, Wa_max - Wa) * 3600.0

        sea_evaporation = evap_potential_water
        # Land evaporation is limited by the actual soil moisture available per hour
        land_evaporation = np.minimum(evap_potential_land, Ws / dt)

        return M_sea * sea_evaporation + (1 - M_sea) * land_evaporation

    def _calculate_precipitation(self, Wa, Wc, Wa_max):
        """Calculates precipitation rate and condensation rates."""
        dt = self.world['time'].dt

        threshold = self.config.rH_condensation_threshold
        tau       = self.config.condensation_timescale
        alpha     = 1.0 - np.exp(-dt / tau)
        condensation = alpha * np.maximum(0.0, Wa - threshold * Wa_max) / dt
        
        # Precipitation falls from already formed clouds
        # Delay factor moderates how much liquid rapidly drops vs stays afloat
        # We use exponential decay to prevent overshooting (Wc going negative) for large dt.
        removal_rate = self.config.precip_conversion_rate * (1.0 - self.config.cloud_delay_factor)
        removed_fraction = 1.0 - np.exp(-removal_rate * dt)
        precip = (Wc * removed_fraction) / dt
        
        return condensation, precip

    def apply_mass_balance(self, Ta, Wa, Wc, Ws, Wa_max, Evap, Condensation, Precip, dt, thermal):
        """Applies mass balance and enforces non-negativity and hard saturation cap.
        
        Returns updated (Ta, Wa, Wc, Ws, Condensation).
        """
        # Apply mass balance; land evaporation removes water from soil (sea Ws is reset after advection)
        Wa += (Evap - Condensation) * dt
        Wc += (Condensation - Precip) * dt
        Ws += (Precip - Evap) * dt

        # Flush numerical negatives and subnormal float32 values
        Wa = np.maximum(Wa, 0.0)
        Wc = np.maximum(Wc, 0.0)
        Wc[Wc < 1e-10] = 0.0   # subnormals are ~100x slower in float32 ops
        Ws = np.maximum(Ws, 0.0)

        # Hard cap: condense any Wa that exceeds saturation into Wc
        excess = np.maximum(Wa - Wa_max, 0.0)
        Wa -= excess
        Wc += excess

        excess_cond = excess / dt
        if np.any(excess_cond > 0):
            Condensation += excess_cond
            
            # Release latent heat
            Lv = self.world.constants['Lv']
            c_air = thermal.config.c_air
            Ta += (excess * Lv) / c_air

        return Ta, Wa, Wc, Ws, Condensation