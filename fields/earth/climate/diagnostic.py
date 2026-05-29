import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

from .engines import AdvectionEngine, ErosionEngine, WaterAdvectionEngine
map_info = {
    'Ta' : {
        'unit' : 'C',
        'description' : 'Air temperature',
        'render' : {'cmap': 'RdYlBu_r'},
    },
    'Ts' : {
        'unit' : 'C',
        'description' : 'Surface (land) temperature',
        'render' : {'cmap': 'RdYlBu_r'},
    },
    'Tw' : {
        'unit' : 'C',
        'description' : 'Water (ocean/lake) temperature',
        'render' : {'cmap': 'Blues_r'},
    },
    'Wa' : {
        'unit' : 'kg/m2',
        'description' : 'Atmospheric water content',
        'render' : {'cmap': 'YlGnBu'},
    },
    'Wc' : {
        'unit' : 'kg/m2',
        'description' : 'Cloud liquid water content',
        'render' : {'cmap': 'Blues'},
    },
    'Ws' : {
        'unit' : 'mm',
        'description' : 'Surface water (e.g. soil moisture, water bodies)',
        'render' : {'cmap': 'YlGn'},
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
    c_land: float = 2.0e5    # Land surface skin/vegetation capacity
    c_water: float = 4.184e7 # ~10m deep active mixing layer (1000kg/m3 * 10m * 4184 J/kg/K)
    
    sensible_heat_coef: float = 1.2 # Coefficient for sensible heat exchange
    stefan_boltzmann_constant: float = 5.670374419e-8 # (W/m2/K4)
    albedo_land: float = 0.25  # Average albedo for land
    albedo_water: float = 0.06 # Average albedo for water
    Lv: float = 2.5e6          # Latent heat of vaporization/condensation (J/kg)

    wind_friction: float = 0.0015 # Friction coefficient for wind acceleration
    adv_sub_steps: int = 3 # Number of sub-steps for advection calculations to improve stability
    rho_air: float = 1.225 # Surface air density (kg/m3), used for wind acceleration calculations
    omega: float = 7.2921e-5 # Earth's angular velocity (rad/s)
    advection_scheme: str = 'semi_lagrangian' # Advection integration scheme

class DiagnosticClimate(BaseModel):
    info = {
        'name':'diagnostic_climate',
        'map_info' : map_info
    }

    """
    Improvements: Crucial Missing Link: The atmosphere currently does not absorb the outgoing longwave radiation (σT4)
     emitted by the earth, nor does it emit its own longwave radiation back to the surface.
      Adding a "Greenhouse Effect" term (where Wa​ traps longwave radiation) will prevent your simulation
       from freezing over too rapidly at night.
    """
    
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
        self.water_advection_engine = WaterAdvectionEngine() # Placeholder for future water-specific advection logic
        self.erosion_engine = ErosionEngine() # Placeholder for future erosion logic
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
        """Returns initial wind patterns approximating global circulation cells."""
        lat = self.world.latitude
        
        # Basic zonal flow: Trade winds (easterlies) near equator, Westerlies mid-lat, Polar easterlies
        # Sine wave proxy: sin(3 * latitude) gives roughly 3 cells
        zonal_wind = -5.0 * np.sin(3.0 * np.radians(lat))
        
        # Meridional flow (north/south) is weaker
        meridional_wind = 2.0 * np.cos(3.0 * np.radians(lat))
        
        V = np.zeros((*shape, 2), dtype=np.float32)
        V[..., 0] = zonal_wind
        V[..., 1] = meridional_wind

        # Add small perturbations 
        V += np.random.randn(*shape, 2).astype(np.float32) * 0.1
        return V

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
        lapse_rate = 0.0065  # K/m (ICAO standard atmosphere)
        Ta = (T_base - lapse_rate * H_m).astype(np.float32)
        Ts = Ta.copy()
        Tw = np.where(M_sea, np.float32(T_base), Ta).astype(np.float32)

        # Atmospheric water: 70 % of max capacity
        Wa = np.clip(0.7 * Wa_max, 0.0, None).astype(np.float32)

        # Clouds: Starts clear
        Wc = np.zeros_like(Wa)

        # Surface water: saturated ocean, modest soil moisture on land (mm)
        Ws = np.where(M_sea, np.float32(1000.0), np.float32(50.0)).astype(np.float32)

        # Initialize Wind
        V = self._compute_initial_wind(H.shape)

        self.set_maps({'Ta': Ta, 'Ts': Ts, 'Tw': Tw, 'Wa': Wa, 'Wc': Wc, 'Ws': Ws, 'V': V})
    
    def step(self):
        """Advances the climate diagnostic state by a single time step (dt)."""
        H, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wc, Ws, V, Vspeed, Evap, Condensation, Precip = self.get_maps()
        dt = self.world['time'].dt

        dTa, dTs, dTw = self._calculate_delta_T(M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc)

        Ta = self._advect(H, Ta, P, Wa, Wc, Ws, V, dt)
        # Apply integrations (Euler step)
        Ta += dTa * dt
        Ts += dTs * dt
        Tw += dTw * dt

        # Apply mass balance for water phases
        Wa += (Evap - Condensation) * dt
        Wc += (Condensation - Precip) * dt
        Ws += (Precip - Evap) * dt
        
        # Enforce constant saturation on pure sea cells
        Ws[M_sea == 1.0] = 1000.0 # Or whatever baseline you use for ocean depth in mm

        self.set_maps({
            'Ta' : Ta,
            'Ts' : Ts,
            'Tw' : Tw,
            'Wa' : Wa,
            'Wc' : Wc,
            'Ws' : Ws,
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
        Condensation = self.world.area['Condensation']()
        Precip = self.world.area['Precip']()
        
        return H, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wc, Ws, V, Vspeed, Evap, Condensation, Precip

    ## ========== Core Calculations ==========
    def _calculate_delta_T(self, M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc):
        """Computes thermodynamic changes (dTa, dTs, dTw) tracking sensible/latent heat, solar, and evaporative cooling."""
        dT_air_from_land,  dT_land_loss  = self._calculate_sensible_heat(
            Ts, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_land)
        dT_air_from_water, dT_water_loss = self._calculate_sensible_heat(
            Tw, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_water)

        dT_air_latent        = self._calculate_atmosphere_latent_heat(Condensation, self.config.Lv, self.config.c_air)
        
        # Calculate Longwave / Greenhouse Radiation
        dT_air_lw, dT_land_lw = self._calculate_longwave_radiation(Ts, Ta, Wa, self.config.stefan_boltzmann_constant, self.config.c_air, self.config.c_land)
        _, dT_water_lw = self._calculate_longwave_radiation(Tw, Ta, Wa, self.config.stefan_boltzmann_constant, self.config.c_air, self.config.c_water)

        dT_land_solar_evap   = self._calculate_surface_heating(Sun, Evap, self.config.albedo_land, self.config.Lv, self.config.c_land)
        dT_water_solar_evap  = self._calculate_surface_heating(Sun, Evap, self.config.albedo_water, self.config.Lv, self.config.c_water)

        dT_air_sensible = (M_sea * dT_air_from_water) + ((1 - M_sea) * dT_air_from_land)
        dTa = dT_air_sensible + dT_air_latent + dT_air_lw
        dTs = (dT_land_solar_evap  + dT_land_lw  - dT_land_loss)  * (1 - M_sea)
        dTw = (dT_water_solar_evap + dT_water_lw - dT_water_loss) * M_sea
        return dTa, dTs, dTw

    def _advect(self, H, Ta, P, Wa, Wc, Ws, V, dt):
        """Performs iterative sub-steps for advection to improve numeric stability."""
        sub_dt = dt / self.config.adv_sub_steps
        for _ in range(self.config.adv_sub_steps):
            Ta, Wa, Wc, V = self.advection_engine(H, Ta, P, V, Wa, Wc, sub_dt)
            Ws = self.water_advection_engine(H, Ws, sub_dt) # Placeholder for future water-specific advection logic
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
        net = (Sun * (1 - albedo)) - (Evap * Lv)
        return net / c_surface
        
    def _calculate_longwave_radiation(self, T_surface, Ta, Wa, stefan_boltzmann_constant, c_air, c_surface):
        """Calculates longwave radiation exchange between surface, atmosphere (greenhouse effect), and space."""
        Tk_surf = T_surface + 273.15
        Tk_air = Ta + 273.15
        
        outgoing_surface = stefan_boltzmann_constant * (Tk_surf ** 4)
        
        # Emissivity of the atmosphere depends heavily on water vapor content (Wa)
        # Using a simple exponential absorption model
        eps_a = 1.0 - np.exp(-0.015 * Wa)
        
        # Atmosphere emits both up to space and down to surface based on its emissivity
        downwelling_atmosphere = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
        upwelling_atmosphere = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
        
        # Atmosphere absorbs a fraction of outgoing surface radiation
        absorbed_by_atmosphere = eps_a * outgoing_surface
        
        # Net fluxes
        net_surface_lw = downwelling_atmosphere - outgoing_surface
        net_air_lw = absorbed_by_atmosphere - downwelling_atmosphere - upwelling_atmosphere
        
        return net_air_lw / c_air, net_surface_lw / c_surface

    # Air warming from latent heat released when water vapour condenses into clouds.
    def _calculate_atmosphere_latent_heat(self, Condensation, Lv, c_air):
        # When water vapor condenses into clouds, it releases latent heat into the air.
        heat_released = Condensation * Lv
        dT_air_latent = heat_released / c_air
        return dT_air_latent

