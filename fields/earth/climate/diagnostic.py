import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

from .engines import ErosionEngine, WaterAdvectionEngine
from .thermal import calculate_sensible_heat, calculate_surface_heating, calculate_longwave_radiation, calculate_atmosphere_latent_heat

from .wind import Wind
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
    
    # Radiation parameters
    sensible_heat_coef: float = 4.0 # Coefficient for sensible heat exchange
    stefan_boltzmann_constant: float = 5.670374419e-8 # (W/m2/K4)
    albedo_land: float = 0.25  # Average albedo for land
    albedo_water: float = 0.06 # Average albedo for water
    Lv: float = 2.5e6          # Latent heat of vaporization/condensation (J/kg)

    # Advection parameters
    adv_sub_steps: int = 4 # Number of sub-steps for advection calculations to improve stability
    rho_air: float = 1.225 # Surface air density (kg/m3), used for wind acceleration calculations
    omega: float = 7.2921e-5 # Earth's angular velocity (rad/s)
    advection_scheme: str = 'semi_lagrangian' # Advection integration scheme
    advection_poisson_iterations: int = 15 # Iterations for Poisson solver to enforce mass continuity in advection

    # Water advection parameters
    water_advection_slope_exponent: float = 2.0   # Weight steeper slopes more: 1=linear, 2=quadratic, …
    water_advection_flow_rate: float = 0.5        # Fraction of cell's water drained per hour at maximum weight
    water_advection_field_capacity: float = 20.0  # Soil moisture held by capillary forces [mm]; only excess above this routes

    # Thermodynamics parameters
    lapse_rate: float = 0.0065 # Temperature drop per meter altitude (K/m)
    max_temperature_step: float = 40.0 # Maximum allowed temperature change per step (K)
        
    # Longwave radiation parameters
    greenhouse_base_emissivity: float = 0.55  # Baseline emissivity from well-mixed GHGs (CO2, etc)
    greenhouse_water_vapor_emissivity_multiplier: float = 0.2 # Water vapor contribution; base + this must stay <= 1.0
    greenhouse_water_vapor_absorption_coef: float = 0.04  # Absorption coef; saturates around Wa~50 kg/m²


class DiagnosticClimate(BaseModel):
    info = {
        'name':'diagnostic_climate',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.config = DiagnosticClimateConfig() 
        
        self.wind_model = Wind(self.world, self.config.advection_scheme, self.config.advection_poisson_iterations)
        self.water_advection_engine = WaterAdvectionEngine(
            cell_size=self.world.area.cell_size,
            max_altitude=self.world.max_altitude,
            slope_exponent=self.config.water_advection_slope_exponent,
            flow_rate=self.config.water_advection_flow_rate,
            field_capacity=self.config.water_advection_field_capacity
        )
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
        V = np.zeros((*H.shape, 2), dtype=np.float32)
        V = self.wind_model.get_initial_wind(V)

        self.set_maps({'Ta': Ta, 'Ts': Ts, 'Tw': Tw, 'Wa': Wa, 'Wc': Wc, 'Ws': Ws, 'V': V})
    
    def step(self):
        """Advances the climate diagnostic state by a single time step (dt)."""
        H, sea_level, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip = self.get_maps()
        dt = self.world['time'].dt

        dTa, dTs, dTw = self._calculate_delta_T(M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc)
        
        # dTa, dTs, dTw are in K/s. dt is in hours.
        dt_sec = dt * 3600.0

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
        Ws = np.maximum(Ws + (Precip - Evap) * dt, 0.0)  # clamp: can't go below 0

        # Flush float32 subnormal values from Wc (can reach 1e-43 due to exponential
        # precipitation decay from tiny initial values).  Subnormal fp ops are ~100× slower.
        Wc = np.maximum(Wc, 0.0)   # guard against floating-point negatives
        Wc[Wc < 1e-10] = 0.0       # flush subnormals

        # Instantly condense any atmospheric water that exceeds saturation capacity.
        # This is a last-resort hard cap; with the soft RH threshold in prognostic it should
        # rarely trigger, but can catch numerical overshoots from large advection steps.
        excess_water = np.maximum(Wa - Wa_max, 0.0)  # kg/m² of actual overshoot
        if np.any(excess_water > 0):
            excess_condensation_rate = excess_water / dt  # kg/m²/hr = mm/hr
            # Apply the latent heat from this instant condensation to the air
            dTa_excess = calculate_atmosphere_latent_heat(excess_condensation_rate, self.config.Lv, self.config.c_air)
            Ta += np.clip(dTa_excess * dt_sec, -self.config.max_temperature_step, self.config.max_temperature_step)
            Wa -= excess_water
            Wc += excess_water
            # Add to Condensation so the plotter sees the full condensation budget this step.
            # Prognostic will recompute it next step from the updated (corrected) Wa anyway.
            Condensation += excess_condensation_rate
        
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
        dT_air_from_land,  dT_land_loss  = calculate_sensible_heat(
            Ts, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_land)
        dT_air_from_water, dT_water_loss = calculate_sensible_heat(
            Tw, Ta, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_water)

        dT_air_latent        = calculate_atmosphere_latent_heat(Condensation, self.config.Lv, self.config.c_air)
        
        # Calculate Longwave / Greenhouse Radiation
        dT_air_lw_land,  dT_land_lw  = calculate_longwave_radiation(Ts, Ta, Wa, self.config.stefan_boltzmann_constant, self.config.c_air, self.config.c_land, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef)
        dT_air_lw_water, dT_water_lw = calculate_longwave_radiation(Tw, Ta, Wa, self.config.stefan_boltzmann_constant, self.config.c_air, self.config.c_water, self.config.greenhouse_base_emissivity, self.config.greenhouse_water_vapor_emissivity_multiplier, self.config.greenhouse_water_vapor_absorption_coef)
        dT_air_lw = (1 - M_sea) * dT_air_lw_land + M_sea * dT_air_lw_water

        dT_land_solar_evap   = calculate_surface_heating(Sun, Evap, self.config.albedo_land, self.config.Lv, self.config.c_land)
        dT_water_solar_evap  = calculate_surface_heating(Sun, Evap, self.config.albedo_water, self.config.Lv, self.config.c_water)

        dT_air_sensible = (M_sea * dT_air_from_water) + ((1 - M_sea) * dT_air_from_land)
        dTa = dT_air_sensible + dT_air_latent + dT_air_lw
        dTs = (dT_land_solar_evap  + dT_land_lw  - dT_land_loss)  * (1 - M_sea)
        dTw = (dT_water_solar_evap + dT_water_lw - dT_water_loss) * M_sea

        return dTa, dTs, dTw

        #DEBUG PRINT
        def stats(name, arr):
            arr = np.asarray(arr)
            print(
                f"  {name:<15} "
                f"mean={arr.mean(): .4e} "
                f"min={arr.min(): .4e} "
                f"max={arr.max(): .4e}"
            )

        # DEBUG PRINT
        print("Air temp change")
        stats("Sensible", dT_air_sensible)
        stats("Latent", dT_air_latent)
        stats("Longwave", dT_air_lw)
        stats("Total", dTa)
        print()

        print("Land temp change")
        stats("Solar/Evap", dT_land_solar_evap)
        stats("Longwave", dT_land_lw)
        stats("Sensible Loss", dT_land_loss)
        stats("Total", dTs)
        print()

        print("Water temp change")
        stats("Solar/Evap", dT_water_solar_evap)
        stats("Longwave", dT_water_lw)
        stats("Sensible Loss", dT_water_loss)
        stats("Total", dTw)
        print()

        balance = self.config.c_air * dTa + self.config.c_land * dTs + self.config.c_water * dTw
        print("Balance check")
        stats("Balance", balance)
        
        return dTa, dTs, dTw

    def _advect(self, H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt):
        """Performs iterative sub-steps for advection to improve numeric stability."""
        sub_dt = dt / self.config.adv_sub_steps
        for _ in range(self.config.adv_sub_steps):
            Ta, Wa, Wc, V = self.wind_model(H, sea_level, Ta, P, V, Wa, Wc, sub_dt)
            Ws = self.water_advection_engine(H, M_sea, Ws, sub_dt)
            H = self.erosion_engine(H, Ws, Ta, sub_dt) # Placeholder for future erosion logic
        return Ta

