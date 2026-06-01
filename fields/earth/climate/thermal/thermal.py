from dataclasses import dataclass
import numpy as np

from .numba import compute_thermal_step

@dataclass
class ThermalConfig:
    # Effective heat capacities for the whole vertical column (J/m2/K)
    c_air: float = 1.004e7   # 1004 J/kg/K * ~10000 kg/m2 of air column
    c_land: float = 2.0e6    # Land surface / ~1 m soil column capacity
    c_water: float = 4.184e7 # ~10m deep active mixing layer (1000kg/m3 * 10m * 4184 J/kg/K)
    
    # Radiation parameters
    sensible_heat_coef: float = 4.0 # Coefficient for sensible heat exchange
    albedo_land: float = 0.25  # Average albedo for land
    albedo_water: float = 0.06 # Average albedo for water

    # Thermodynamics parameters
    lapse_rate: float = 0.0065 # Temperature drop per meter altitude (K/m)
        
    # Longwave radiation parameters
    greenhouse_base_emissivity: float = 0.55  # Baseline emissivity from well-mixed GHGs (CO2, etc)
    greenhouse_water_vapor_emissivity_multiplier: float = 0.2 # Water vapor contribution; base + this must stay <= 1.0
    greenhouse_water_vapor_absorption_coef: float = 0.04  # Absorption coef; saturates around Wa~50 kg/m²

    # Atmospheric solar absorption: fraction of surface-reaching solar that was absorbed by the atmosphere
    # above (by ozone, water vapour, aerosols). ~30 % of TOA solar never reaches the ground; roughly half
    # of that is absorbed (rest reflected). 0.20 ≈ 15% of TOA absorbed / 70% transmitted = ~0.21.
    solar_atm_absorption: float = 0.20



class Thermal:
    def __init__(self, world):
        self.world = world
        self.config = ThermalConfig()

    def init(self, H, M_sea):
        """Returns baseline temperature map depending on latitude, day of year, and time of day."""
        time_obj = self.world['time']

        H_m = np.maximum(0.0, H - self.world.area['sea_level']()) * self.world.max_altitude
        
        # Mean temp dependent on latitude
        lat_factor = np.cos(np.radians(self.world.latitude))
        T_mean = 30.0 * lat_factor - 15.0 * (1.0 - lat_factor) # Ex: 30C at equator, -15C at pole
        
        # Seasonal variation
        season_amplitude = 15.0 * (1.0 - lat_factor) # Stronger seasons near poles
        T_season_offset = season_amplitude * time_obj.season_phase
        
        # Diurnal variation
        hour_angle = (time_obj.fractional_hour - 14.0) / 24.0 * 2 * np.pi # Peak temp around 14:00
        T_diurnal_offset = 5.0 * np.cos(hour_angle) # +/- 5C swing

        T_base = T_mean + T_season_offset + T_diurnal_offset
        Ta = (T_base - (self.config.lapse_rate/4) * H_m).astype(np.float32)
        Ts = Ta.copy()
        Tw = np.where(M_sea, np.float32(T_base), Ta).astype(np.float32)
        
        return Ta, Ts, Tw

    def __call__(self, M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc):
        """Computes thermodynamic changes (dTa, dTs, dTw) tracking sensible/latent heat, solar, and evaporative cooling."""
        return compute_thermal_step(
            M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc,
            self.config.sensible_heat_coef, self.config.c_air, self.config.c_land, self.config.c_water,
            self.world.constants['Lv'], self.world.constants['sigma'],
            self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef,
            self.config.albedo_land, self.config.albedo_water, self.config.solar_atm_absorption
        )

    def __call_without_numba__(self, M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc):
        """Computes thermodynamic changes (dTa, dTs, dTw) tracking sensible/latent heat, solar, and evaporative cooling."""
        dT_air_from_land,  dT_land_loss  = self.calculate_sensible_heat(Ts, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_land)
        dT_air_from_water, dT_water_loss = self.calculate_sensible_heat(Tw, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_water)

        dT_air_latent = self.calculate_atmosphere_latent_heat(Condensation, self.world.constants['Lv'], self.config.c_air)
        
        # Calculate Longwave / Greenhouse Radiation
        dT_air_lw_land,  dT_land_lw  = self.calculate_longwave_radiation(Ts, Ta, Wa, self.world.constants['sigma'], self.config.c_air, self.config.c_land, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef)
        dT_air_lw_water, dT_water_lw = self.calculate_longwave_radiation(Tw, Ta, Wa, self.world.constants['sigma'], self.config.c_air, self.config.c_water, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef)
        dT_air_lw = (1 - M_sea) * dT_air_lw_land + M_sea * dT_air_lw_water

        dT_land_solar_evap   = self.calculate_surface_heating(Sun, Evap, self.config.albedo_land, self.world.constants['Lv'], self.config.c_land)
        dT_water_solar_evap  = self.calculate_surface_heating(Sun, Evap, self.config.albedo_water, self.world.constants['Lv'], self.config.c_water)

        # Direct solar heating of the air column: the fraction of surface-reaching solar that was
        # absorbed by the atmosphere above (clouds, water vapour, ozone) rather than being reflected
        # to space. Without this term the energy is simply discarded and Ta has no diurnal cycle.
        dT_air_solar = Sun * self.config.solar_atm_absorption / self.config.c_air

        dT_air_sensible = (M_sea * dT_air_from_water) + ((1 - M_sea) * dT_air_from_land)
        dTa = dT_air_sensible + dT_air_latent + dT_air_lw + dT_air_solar
        dTs = (dT_land_solar_evap  + dT_land_lw  - dT_land_loss)  * (1 - M_sea)
        dTw = (dT_water_solar_evap + dT_water_lw - dT_water_loss) * M_sea
    ## ========== Vertical Thermodynamics ==========
    # Wind-driven turbulent heat exchange between surface and overlying air.
    def calculate_sensible_heat(self, T_surface, Ta, Vspeed, sensible_heat_coef, c_air, c_surface):
        heat_transfer_coef = sensible_heat_coef * Vspeed
        flux = heat_transfer_coef * (T_surface - Ta)
        return flux / c_air, flux / c_surface   # (dT_air_gain, dT_surface_loss)

    # Net surface temperature change from solar gain and evaporative cooling.
    def calculate_surface_heating(self, Sun, Evap, albedo, Lv, c_surface):
        # Evap is in mm/hr, convert to kg/m2/s by dividing by 3600
        evap_flux = (Evap / 3600.0) * Lv
        net = (Sun * (1 - albedo)) - evap_flux
        return net / c_surface
        
    def calculate_longwave_radiation(self, T_surface, Ta, Wa, stefan_boltzmann_constant, c_air, c_surface, gh_base_eps, gh_wv_mult, gh_wv_coef):
        """Calculates longwave radiation exchange between surface, atmosphere (greenhouse effect), and space."""
        # Prevent negative Kelvin temperatures
        Tk_surf = np.maximum(T_surface + 273.15, 0.0)
        Tk_air = np.maximum(Ta + 273.15, 0.0)
        
        outgoing_surface = stefan_boltzmann_constant * (Tk_surf ** 4)
        
        # Emissivity of the atmosphere depends on baseline GHGs (CO2, etc) plus water vapor (Wa)
        # Clamped to [0, 1] — physical emissivity cannot exceed 1
        eps_a = np.clip(gh_base_eps + gh_wv_mult * (1.0 - np.exp(-gh_wv_coef * np.maximum(Wa, 0.0))), 0.0, 1.0)
        
        
        # Atmosphere emits both up to space and down to surface based on its emissivity
        downwelling_atmosphere = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
        upwelling_atmosphere = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
        
        # Atmosphere absorbs a fraction of outgoing surface radiation
        absorbed_by_atmosphere = eps_a * outgoing_surface
        
        # Net fluxes
        net_surface_lw = downwelling_atmosphere - outgoing_surface
        # Atmosphere gains from absorbed surface LW; loses only what it emits upward to space.
        # The downwelling term belongs to the surface budget, not the atmospheric energy balance.
        # This gives net_air_lw = eps*sigma*(Ts^4 - Ta^4): 0 at equilibrium, stable.
        net_air_lw = absorbed_by_atmosphere - upwelling_atmosphere
        
        return net_air_lw / c_air, net_surface_lw / c_surface

    # Air warming from latent heat released when water vapour condenses into clouds.
    def calculate_atmosphere_latent_heat(self, Condensation, Lv, c_air):
        # When water vapor condenses into clouds, it releases latent heat into the air.
        # Condensation is in mm/hr, convert to kg/m2/s
        heat_released = (Condensation / 3600.0) * Lv
        dT_air_latent = heat_released / c_air
        return dT_air_latent
