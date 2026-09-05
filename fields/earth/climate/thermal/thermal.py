from dataclasses import dataclass
import numpy as np

from .numba import compute_thermal_step

@dataclass
class ThermalConfig:
    # Effective heat capacities for the whole vertical column (J/m2/K)
    c_air: float = 1.03e6    # 1004 J/kg/K * ~10000 kg/m2 of air column
    c_land: float = 5.0e6    # Land surface / ~1 m soil column capacity
    c_water: float = 5.184e7 # ~10m deep active mixing layer (1000kg/m3 * 10m * 4184 J/kg/K)
    
    # Radiation parameters
    sensible_heat_coef: float = 20.0 # Coefficient for sensible heat exchange
    albedo_land: float = 0.15  # Average albedo for land 0.25
    albedo_water: float = 0.03 # Average albedo for water 0.06

    # Thermodynamics parameters
    lapse_rate: float = 0.0065 # Temperature drop per meter altitude (K/m)
        
    # Longwave radiation parameters
    greenhouse_base_emissivity: float = 0.75  # Baseline emissivity from well-mixed GHGs (CO2, etc)

    greenhouse_water_vapor_emissivity_multiplier: float = 0.2 # Water vapor contribution
    greenhouse_water_vapor_absorption_coef: float = 0.05  # Absorption coef; saturates around Wa~50 kg/m²

    greenhouse_cloud_emissivity_multiplier: float = 0.2 # Clouds are very efficient absorbers/emitters    
    greenhouse_cloud_emissivity_coeff: float = 1.5 # Clouds are very efficient absorbers/emitters

    # Istanbul baseline parameters.  These are approximate regional values,
    # not a replacement for observed boundary conditions.
    istanbul_mean_temperature: float = 16.0       # annual mean air temperature [°C]
    istanbul_seasonal_amplitude: float = 9.0      # seasonal amplitude [K]
    air_diurnal_amplitude: float = 3.0            # air-temperature amplitude [K]
    land_diurnal_amplitude: float = 5.0           # land-surface amplitude [K]
    water_diurnal_amplitude: float = 0.75         # water-surface amplitude [K]
    diurnal_peak_hour: float = 14.0

    external_heat_flux: float = 80.0 # W/m², external heat flux into the system (e.g., from ocean currents or geothermal sources)





class Thermal:
    def __init__(self, world):
        self.world = world
        self.config = ThermalConfig()

    def init(self, H, M_sea):
        """Returns baseline temperature map depending on latitude, day of year, and time of day."""
        time_obj = self.world['time']

        H_m = np.maximum(0.0, H - self.world.area['sea_level']()) * self.world.max_altitude
        
        # The old latitude-only expression produced only about 4 K of
        # seasonal variation at Istanbul (41 N).  Use a calibrated Istanbul
        # baseline while retaining a modest latitude adjustment for nearby
        # domains.
        lat_delta = self.world.latitude - 41.0
        T_mean = self.config.istanbul_mean_temperature - 0.12 * lat_delta
        season_amplitude = self.config.istanbul_seasonal_amplitude + 0.08 * abs(lat_delta)
        T_season = season_amplitude * time_obj.season_phase

        hour_angle = (time_obj.fractional_hour - self.config.diurnal_peak_hour) / 24.0 * 2 * np.pi
        air_diurnal = self.config.air_diurnal_amplitude * np.cos(hour_angle)
        land_diurnal = self.config.land_diurnal_amplitude * np.cos(hour_angle)
        water_diurnal = self.config.water_diurnal_amplitude * np.cos(hour_angle)

        T_base = T_mean + T_season
        Ta = (T_base + air_diurnal - self.config.lapse_rate * H_m).astype(np.float32)
        Ts = (T_base + land_diurnal - self.config.lapse_rate * H_m).astype(np.float32)
        Tw = np.where(
            M_sea,
            np.float32(T_base + water_diurnal),
            Ta,
        ).astype(np.float32)
        
        return Ta, Ts, Tw

    def __call__(self, M_sea, Sun, Sun_atm, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc, dt):
        """Computes thermodynamic changes (dTa, dTs, dTw) tracking sensible/latent heat, solar, and evaporative cooling."""
        return compute_thermal_step(
            M_sea, Sun, Sun_atm, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc,
            self.config.sensible_heat_coef, self.config.c_air, self.config.c_land, self.config.c_water,
            self.world.constants['Lv'], self.world.constants['sigma'],
            self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef, self.config.greenhouse_cloud_emissivity_multiplier, self.config.greenhouse_cloud_emissivity_coeff,
            self.config.albedo_land, self.config.albedo_water, self.config.external_heat_flux, dt
        )

    def __call_without_numba__(self, M_sea, Sun, Sun_atm, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc):
        """Computes thermodynamic changes (dTa, dTs, dTw) tracking sensible/latent heat, solar, and evaporative cooling."""
        dT_air_from_land,  dT_land_loss  = self.calculate_sensible_heat(Ts, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_land)
        dT_air_from_water, dT_water_loss = self.calculate_sensible_heat(Tw, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_water)

        dT_air_latent = self.calculate_atmosphere_latent_heat(Condensation, self.world.constants['Lv'], self.config.c_air)
        
        # Calculate Longwave / Greenhouse Radiation (includes both water vapour and cloud emissivity)
        dT_air_lw_land,  dT_land_lw  = self.calculate_longwave_radiation(Ts, Ta, Wa, Wc, self.world.constants['sigma'], self.config.c_air, self.config.c_land, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef, self.config.greenhouse_cloud_emissivity_multiplier, self.config.greenhouse_cloud_emissivity_coeff)
        dT_air_lw_water, dT_water_lw = self.calculate_longwave_radiation(Tw, Ta, Wa, Wc, self.world.constants['sigma'], self.config.c_air, self.config.c_water, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef, self.config.greenhouse_cloud_emissivity_multiplier, self.config.greenhouse_cloud_emissivity_coeff)
        dT_air_lw = (1 - M_sea) * dT_air_lw_land + M_sea * dT_air_lw_water

        dT_land_solar_evap   = self.calculate_surface_heating(Sun, Evap, self.config.albedo_land, self.world.constants['Lv'], self.config.c_land)
        dT_water_solar_evap  = self.calculate_surface_heating(Sun, Evap, self.config.albedo_water, self.world.constants['Lv'], self.config.c_water)

        # True atmospheric solar heating: energy absorbed before reaching the surface (= S0_toa - Sun).
        dT_air_solar = Sun_atm / self.config.c_air

        dT_air_sensible = (M_sea * dT_air_from_water) + ((1 - M_sea) * dT_air_from_land)
        dT_air_external = self.config.external_heat_flux / self.config.c_air
        dTa = dT_air_sensible + dT_air_latent + dT_air_lw + dT_air_solar + dT_air_external
        # Scale by sea fraction so mixed cells are blended correctly (matches numba path)
        dTs = np.where(M_sea < 1.0, (dT_land_solar_evap  + dT_land_lw  - dT_land_loss)  * (1.0 - M_sea), 0.0)
        dTw = np.where(M_sea > 0.0, (dT_water_solar_evap + dT_water_lw - dT_water_loss) * M_sea,         0.0)
        return dTa, dTs, dTw

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
        
    def calculate_longwave_radiation(self, T_surface, Ta, Wa, Wc, stefan_boltzmann_constant, c_air, c_surface, gh_base_eps, gh_wv_mult, gh_wv_coef, gh_cloud_mult, gh_cloud_coeff):
        """Calculates longwave radiation exchange between surface, atmosphere (greenhouse effect), and space."""
        # Prevent negative Kelvin temperatures
        Tk_surf = np.maximum(T_surface + 273.15, 0.0)
        Tk_air = np.maximum(Ta + 273.15, 0.0)
        
        outgoing_surface = stefan_boltzmann_constant * (Tk_surf ** 4)
        
        # Emissivity: baseline GHGs + water vapour + clouds (matches numba path)
        tau_base  = 1.0 - gh_base_eps
        tau_wv    = 1.0 - (gh_wv_mult * (1.0 - np.exp(-gh_wv_coef * np.maximum(Wa, 0.0))))
        tau_cloud = 1.0 - (gh_cloud_mult * (1.0 - np.exp(-gh_cloud_coeff * np.maximum(Wc, 0.0))))
        tau_total = tau_base * tau_wv * tau_cloud
        eps_a = 1.0 - tau_total
        
        
        # Atmosphere emits both up to space and down to surface based on its emissivity.
        # Both are energy losses for the atmospheric layer.
        atmos_emission = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
        downwelling_atmosphere = atmos_emission
        upwelling_atmosphere = atmos_emission
        
        # Atmosphere absorbs a fraction of outgoing surface radiation
        absorbed_by_atmosphere = eps_a * outgoing_surface
        
        # Net fluxes
        net_surface_lw = downwelling_atmosphere - outgoing_surface
        # Atmosphere gains from absorbed surface LW and loses both upward and downward emission.
        net_air_lw = absorbed_by_atmosphere - (upwelling_atmosphere + downwelling_atmosphere)
        
        return net_air_lw / c_air, net_surface_lw / c_surface

    # Air warming from latent heat released when water vapour condenses into clouds.
    def calculate_atmosphere_latent_heat(self, Condensation, Lv, c_air):
        # When water vapor condenses into clouds, it releases latent heat into the air.
        # Condensation is in mm/hr, convert to kg/m2/s
        heat_released = (Condensation / 3600.0) * Lv
        dT_air_latent = heat_released / c_air
        return dT_air_latent
