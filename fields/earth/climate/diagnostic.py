import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

from .engines import AdvectionEngine, ErosionEngine, WaterAdvectionEngine
from . wind import prevailing_wind_degrees, ou_process, vm_process, Mr2OU
map_info = {
    'Ta' : {
        'unit' : '°C',
        'description' : 'Air temperature',
        'render' : {'cmap': 'RdYlBu_r', 'vrange': (-15, 40)},
    },
    'Ts' : {
        'unit' : '°C',
        'description' : 'Surface (land) temperature',
        'render' : {'cmap': 'RdYlBu_r', 'vrange': (-15, 40)},
    },
    'Tw' : {
        'unit' : '°C',
        'description' : 'Water (ocean/lake) temperature',
        'render' : {'cmap': 'Blues_r', 'vrange': (0, 30)},
    },
    'Wa' : {
        'unit' : 'kg/m²',
        'description' : 'Atmospheric water content',
        'render' : {'cmap': 'YlGnBu', 'scale': 'linear'},
    },
    'Wc' : {
        'unit' : 'kg/m²',
        'description' : 'Cloud liquid water content',
        'render' : {'cmap': 'Blues', 'scale': 'linear'},
    },
    'Ws' : {
        'unit' : 'mm',
        'description' : 'Surface water (e.g. soil moisture, water bodies)',
        'render' : {'cmap': 'YlGn', 'scale': 'linear'},
    },
    'V' : {
        'requires_magnitude' : True,
        'unit' : 'm/s',
        'description' : 'Wind vector map (rows, cols, 2)',
    }
}

@dataclass
class DiagnosticClimateConfig:
    # Effective heat capacities for the whole vertical column (J/m2/K)
    c_air: float = 1.004e7   # 1004 J/kg/K * ~10000 kg/m2 of air column
    c_land: float = 2.0e6    # Land surface / ~1 m soil column capacity
    c_water: float = 4.184e7 # ~10m deep active mixing layer (1000kg/m3 * 10m * 4184 J/kg/K)
    
    sensible_heat_coef: float = 4.0 # Coefficient for sensible heat exchange
    stefan_boltzmann_constant: float = 5.670374419e-8 # (W/m2/K4)
    albedo_land: float = 0.25  # Average albedo for land
    albedo_water: float = 0.06 # Average albedo for water
    Lv: float = 2.5e6          # Latent heat of vaporization/condensation (J/kg)

    wind_friction: float = 0.012 # Friction coefficient for wind acceleration
    adv_sub_steps: int = 4 # Number of sub-steps for advection calculations to improve stability
    rho_air: float = 1.225 # Surface air density (kg/m3), used for wind acceleration calculations
    omega: float = 7.2921e-5 # Earth's angular velocity (rad/s)
    advection_scheme: str = 'semi_lagrangian' # Advection integration scheme

    lapse_rate: float = 0.0065 # Temperature drop per meter altitude (K/m)
    max_temperature_step: float = 40.0 # Maximum allowed temperature change per step (K)
    
    # Longwave radiation parameters
    greenhouse_base_emissivity: float = 0.0 # Baseline emissivity from well-mixed GHGs (CO2, etc)
    greenhouse_water_vapor_emissivity_multiplier: float = 1.0 # Extra emissivity scale from water vapor
    greenhouse_water_vapor_absorption_coef: float = 0.02 # Absorption coefficient for water vapor

    #random wind parameters
    wind_speed_scale : float = 0.31
    wind_speed_relaxation_time : float = 24.0 * 4
    wind_speed_target_sigma : float = 0.5 # Target standard deviation for wind speed fluctuations (m/s)
    

    wind_angle_target_sigma : float = 50.0 # Standard deviation for random wind direction changes (degrees)
    wind_angle_relaxation_time  : float = 24.0 * 8


def vm_relaxation_time_to_kappa(relaxation_time, dt):
    return dt / relaxation_time
def vm_sigma_target_to_sigma(sigma_target, kappa):
    return np.sqrt(2 * kappa) * sigma_target


class DiagnosticClimate(BaseModel):
    info = {
        'name':'diagnostic_climate',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.config = DiagnosticClimateConfig() 
        

        self.advection_engine = AdvectionEngine(
            wind_friction=self.config.wind_friction,
            latitude=self.world.latitude,
            cell_size=self.world.area.cell_size,
            rho_air=self.config.rho_air,
            omega=self.config.omega,
            scheme=self.config.advection_scheme
        )
        self.water_advection_engine = WaterAdvectionEngine(
            cell_size=self.world.area.cell_size,
            max_altitude=self.world.max_altitude,
        )
        self.erosion_engine = ErosionEngine() # Placeholder for future erosion logic
        self.prevailing_wind_angle, self.prevailing_wind_speed = prevailing_wind_degrees(self.world.latitude, self.world.longitude, self.world['time'].day_of_year)
        self.mrv_ou = Mr2OU(x0=self.prevailing_wind_speed, v0=0.0, dt=self.world['time'].dt, relaxation_time=self.config.wind_speed_relaxation_time, sigma_v=self.config.wind_speed_target_sigma)
        
        self.init() 

    def _compute_initial_temperature_base(self):
        """Returns baseline temperature map depending on latitude, day of year, and time of day."""
        time_obj = self.world['time']
        
        # Mean temp dependent on latitude
        lat_factor = np.cos(np.radians(self.world.latitude))
        T_mean = 30.0 * lat_factor - 15.0 * (1.0 - lat_factor) # Ex: 30C at equator, -15C at pole
        
        # Seasonal variation
        season_amplitude = 15.0 * (1.0 - lat_factor) # Stronger seasons near poles
        T_season_offset = season_amplitude * time_obj.season_phase
        
        # Diurnal variation
        hour_angle = (time_obj.fractional_hour - 14.0) / 24.0 * 2 * np.pi # Peak temp around 14:00
        T_diurnal_offset = 5.0 * np.cos(hour_angle) # +/- 5C swing
        
        return T_mean + T_season_offset + T_diurnal_offset

    def _compute_initial_wind(self, shape):
        V = np.zeros((*shape, 2), dtype=np.float32)
        V[..., 0] = self.prevailing_wind_speed * np.sin(np.radians(self.prevailing_wind_angle)) # u component (east-west)
        V[..., 1] = self.prevailing_wind_speed * np.cos(np.radians(self.prevailing_wind_angle)) # v component (north-south)
        return V

    def _get_random_wind_base_difference(self, lat, lon, day, dt):
        mu_angle, mu_speed = prevailing_wind_degrees(lat, lon, day)

        kappa = vm_relaxation_time_to_kappa(self.config.wind_angle_relaxation_time, dt)
        sigma_angle = vm_sigma_target_to_sigma(self.config.wind_angle_target_sigma, kappa)
        
        
        self.prevailing_wind_angle = vm_process(self.prevailing_wind_angle, mu_angle, kappa, sigma_angle)
        self.prevailing_wind_speed, _ = self.mrv_ou.step(mu_speed)
        self.prevailing_wind_speed = max(0.0, self.prevailing_wind_speed) # Ensure non-negative wind speed
        angle_rad = np.radians(self.prevailing_wind_angle)
        scale = self.config.wind_speed_scale / self.config.wind_angle_relaxation_time # Scale down the influence of random wind changes to prevent extreme spikes
        speed = self.prevailing_wind_speed * scale
        dV_i= speed * np.sin(angle_rad)
        dV_j = speed * np.cos(angle_rad)
        return dV_i, dV_j
    ## ========== Simulation & Generation ==========
    def init(self):
        """Bootstraps the initial climatic state with latitude and altitude-dependent values."""
        H = self.world.area['H']()
        M_sea = self.world.area['M_sea']()
        sea_level = self.world.area['sea_level']()
        Wa_max = self.world.area['Wa_max']()  # Populated by PrognosticClimate.init() at T=0

        # Height above sea level in metres
        H_m = np.maximum(0.0, H - sea_level) * self.world.max_altitude

        # Initialize Temperature
        T_base = self._compute_initial_temperature_base()
        Ta = (T_base - (self.config.lapse_rate/4) * H_m).astype(np.float32)
        Ts = Ta.copy()
        Tw = np.where(M_sea, np.float32(T_base), Ta).astype(np.float32)

        # Atmospheric water: 70 % of max capacity
        Wa = np.clip(0.7 * Wa_max, 0.0, None).astype(np.float32)

        # Clouds: Starts clear
        Wc = np.zeros_like(Wa)

        # Surface water: saturated ocean, modest soil moisture on land (mm)
        Ws = np.where(M_sea, np.float32(50.0), np.float32(50.0)).astype(np.float32)

        # Initialize Wind
        V = self._compute_initial_wind(H.shape)

        self.set_maps({'Ta': Ta, 'Ts': Ts, 'Tw': Tw, 'Wa': Wa, 'Wc': Wc, 'Ws': Ws, 'V': V})
    
    def step(self):
        """Advances the climate diagnostic state by a single time step (dt)."""
        H, sea_level, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip = self.get_maps()
        dt = self.world['time'].dt

        dTa, dTs, dTw = self._calculate_delta_T(M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc)
        
        # dTa, dTs, dTw are in K/s. dt is in hours.
        dt_sec = dt * 3600.0

        # Advection assumes dt in hours (acts quasi-statically for wind magnitude logic) to avoid Mach 200 winds.
        dV_i, dV_j = self._get_random_wind_base_difference(self.world.latitude, self.world.longitude, self.world['time'].day_of_year, dt)
        V[..., 0] += dV_i
        V[..., 1] += dV_j
        Ta = self._advect(H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt)
        
        # Apply integrations (Euler step)
        # We clip the maximal temperature step to +/- max_temperature_step per timestep to prevent numeric explosions
        # from aggressive explicit integration of Sensible Heat if Vspeed randomly spikes.
        Ta += np.clip(dTa * dt_sec, -self.config.max_temperature_step, self.config.max_temperature_step)
        Ts += np.clip(dTs * dt_sec, -self.config.max_temperature_step, self.config.max_temperature_step)
        Tw += np.clip(dTw * dt_sec, -self.config.max_temperature_step, self.config.max_temperature_step)

        # Apply mass balance for water phases
        Wa += (Evap - Condensation) * dt
        Wc += (Condensation - Precip) * dt
        Ws += (Precip - Evap) * dt

        # Convert saturation excess in atmospheric water to condensation (cloud formation)
        rH = Wa / Wa_max
        excess = np.maximum(rH - 1.0, 0.0)

        Condensation +=  excess * Wa
        Wa -= excess * Wa
        Wc += excess * Wa
        
        # Enforce constant saturation on pure sea cells
        Ws[M_sea == 1.0] = 50.0 # Or whatever baseline you use for ocean depth in mm

        self.set_maps({
            'Ta' : Ta,
            'Ts' : Ts,
            'Tw' : Tw,
            'Wa' : Wa,
            'Wc' : Wc,
            'Ws' : Ws,
            'Condensation' : Condensation,
            'V' : V,
            'H' : H
        })

    ## ========== Map Management ==========
    def get_maps(self):
        H = self.world.area['H']().copy()  # Terrain height
        M_sea = self.world.area['M_sea']()  # Sea mask
        Sun = self.world.area['Sun']() # Added to fetch solar radiation
        Ta = self.world.area['Ta']().copy()
        Ts = self.world.area['Ts']().copy()
        Tw = self.world.area['Tw']().copy()
        P = self.world.area['P']()
        Wa = self.world.area['Wa']().copy()
        Wc = self.world.area['Wc']().copy()
        Ws = self.world.area['Ws']().copy()
        V = self.world.area['V']().copy() # Added to keep vector state safe
        Vspeed = self.world.area['V_magnitude']()
        Evap = self.world.area['Evap']()
        Condensation = self.world.area['Condensation']().copy()
        Precip = self.world.area['Precip']()
        sea_level = self.world.area['sea_level']() # Added to fetch sea level for orographic effect
        Wa_max = self.world.area['Wa_max']() # Added to fetch max water capacity for humidity calculations
        
        return H, sea_level, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip

    ## ========== Core Calculations ==========
    def _calculate_delta_T(self, M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc):
        """Computes thermodynamic changes (dTa, dTs, dTw) tracking sensible/latent heat, solar, and evaporative cooling."""
        dT_air_from_land,  dT_land_loss  = self._calculate_sensible_heat(
            Ts, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_land)
        dT_air_from_water, dT_water_loss = self._calculate_sensible_heat(
            Tw, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_water)

        dT_air_latent        = self._calculate_atmosphere_latent_heat(Condensation, self.config.Lv, self.config.c_air)
        
        # Calculate Longwave / Greenhouse Radiation
        dT_air_lw, dT_land_lw = self._calculate_longwave_radiation(Ts, Ta, Wa, self.config.stefan_boltzmann_constant, self.config.c_air, self.config.c_land, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef)
        _, dT_water_lw = self._calculate_longwave_radiation(Tw, Ta, Wa, self.config.stefan_boltzmann_constant, self.config.c_air, self.config.c_water, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef)

        dT_land_solar_evap   = self._calculate_surface_heating(Sun, Evap, self.config.albedo_land, self.config.Lv, self.config.c_land)
        dT_water_solar_evap  = self._calculate_surface_heating(Sun, Evap, self.config.albedo_water, self.config.Lv, self.config.c_water)

        dT_air_sensible = (M_sea * dT_air_from_water) + ((1 - M_sea) * dT_air_from_land)
        dTa = dT_air_sensible + dT_air_latent + dT_air_lw
        dTs = (dT_land_solar_evap  + dT_land_lw  - dT_land_loss)  * (1 - M_sea)
        dTw = (dT_water_solar_evap + dT_water_lw - dT_water_loss) * M_sea
        return dTa, dTs, dTw

    def _advect(self, H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt):
        """Performs iterative sub-steps for advection to improve numeric stability."""
        sub_dt = dt / self.config.adv_sub_steps
        for _ in range(self.config.adv_sub_steps):
            Ta, Wa, Wc, V = self.advection_engine(H, sea_level, Ta, P, V, Wa, Wc, sub_dt)
            Ws = self.water_advection_engine(H, M_sea, Ws, sub_dt)
            H = self.erosion_engine(H, Ws, Ta, sub_dt) # Placeholder for future erosion logic
        return Ta

    ## ========== Vertical Thermodynamics ==========
    # Wind-driven turbulent heat exchange between surface and overlying air.
    def _calculate_sensible_heat(self, T_surface, Ta, Vspeed, sensible_heat_coef, c_air, c_surface):
        heat_transfer_coef = sensible_heat_coef * Vspeed
        flux = heat_transfer_coef * (T_surface - Ta)
        return flux / c_air, flux / c_surface   # (dT_air_gain, dT_surface_loss)

    # Net surface temperature change from solar gain and evaporative cooling.
    def _calculate_surface_heating(self, Sun, Evap, albedo, Lv, c_surface):
        # Evap is in mm/hr, convert to kg/m2/s by dividing by 3600
        evap_flux = (Evap / 3600.0) * Lv
        net = (Sun * (1 - albedo)) - evap_flux
        return net / c_surface
        
    def _calculate_longwave_radiation(self, T_surface, Ta, Wa, stefan_boltzmann_constant, c_air, c_surface, gh_base_eps, gh_wv_mult, gh_wv_coef):
        """Calculates longwave radiation exchange between surface, atmosphere (greenhouse effect), and space."""
        # Prevent negative Kelvin temperatures
        Tk_surf = np.maximum(T_surface + 273.15, 0.0)
        Tk_air = np.maximum(Ta + 273.15, 0.0)
        
        outgoing_surface = stefan_boltzmann_constant * (Tk_surf ** 4)
        
        # Emissivity of the atmosphere depends on baseline GHGs (CO2, etc) plus water vapor (Wa)
        eps_a = gh_base_eps + gh_wv_mult * (1.0 - np.exp(-gh_wv_coef * np.maximum(Wa, 0.0)))
        
        
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
    def _calculate_atmosphere_latent_heat(self, Condensation, Lv, c_air):
        # When water vapor condenses into clouds, it releases latent heat into the air.
        # Condensation is in mm/hr, convert to kg/m2/s
        heat_released = (Condensation / 3600.0) * Lv
        dT_air_latent = heat_released / c_air
        return dT_air_latent





