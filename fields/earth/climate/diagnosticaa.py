import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

from .engines import ErosionEngine, WaterAdvectionEngine
from . import Wind, Thermal


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
    # Advection parameters
    adv_sub_steps: int = 4 # Number of sub-steps for advection calculations to improve stability
    advection_scheme: str = 'semi_lagrangian' # Advection integration scheme
    advection_poisson_iterations: int = 15 # Iterations for Poisson solver to enforce mass continuity in advection

    # Water advection parameters
    water_advection_slope_exponent: float = 2.0   # Weight steeper slopes more: 1=linear, 2=quadratic, …
    water_advection_flow_rate: float = 0.5        # Fraction of cell's water drained per hour at maximum weight
    water_advection_field_capacity: float = 20.0  # Soil moisture held by capillary forces [mm]; only excess above this routes


class DiagnosticClimate(BaseModel):
    info = {
        'name':'diagnostic_climate',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.config = DiagnosticClimateConfig() 
        
        self.wind_model = Wind(self.world, self.config.advection_scheme, self.config.advection_poisson_iterations)
        self.thermal_model = Thermal(self.world)
        
        self.water_advection_engine = WaterAdvectionEngine(
            cell_size=self.world.area.cell_size,
            max_altitude=self.world.max_altitude,
            slope_exponent=self.config.water_advection_slope_exponent,
            flow_rate=self.config.water_advection_flow_rate,
            field_capacity=self.config.water_advection_field_capacity
        )
        self.erosion_engine = ErosionEngine() # Placeholder for future erosion logic
        self.init() 


    ## ========== Simulation & Generation ==========
    def init(self):
        """Bootstraps the initial climatic state with latitude and altitude-dependent values."""
        H = self.world.area['H']()
        M_sea = self.world.area['M_sea']()
        sea_level = self.world.area['sea_level']()
        Wa_max = self.world.area['Wa_max']()  # Populated by PrognosticClimate.init() at T=0

        
        # Initialize Temperature
        Ta, Ts, Tw = self.thermal_model.init(H, M_sea)

        # Atmospheric water: 70 % of max capacity
        Wa = np.clip(0.7 * Wa_max, 0.0, None).astype(np.float32)

        # Clouds: Starts clear
        Wc = np.zeros_like(Wa)

        # Surface water: saturated ocean, modest soil moisture on land (mm)
        Ws = np.where(M_sea, np.float32(50.0), np.float32(50.0)).astype(np.float32)

        # Initialize Wind
        V = np.zeros((*H.shape, 2), dtype=np.float32)
        V = self.wind_model.init(V)

        self.set_maps({'Ta': Ta, 'Ts': Ts, 'Tw': Tw, 'Wa': Wa, 'Wc': Wc, 'Ws': Ws, 'V': V})
    
    def step(self):
        """Advances the climate diagnostic state by a single time step (dt)."""
        H, sea_level, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip = self.get_maps()
        dt = self.world['time'].dt
        dt_sec = dt * 3600.0

        # 1. Compute thermodynamic changes from surface fluxes, radiation, and phase changes.
        #    S→L splitting: source terms act on state_n (the same state PrognosticClimate used
        #    to derive Evap/Condensation/Precip), so the fluxes are consistent with the fields
        #    they modify.  Advection then transports the updated state.
        dTa, dTs, dTw = self.thermal_model(M_sea, Sun, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc)
        Ta += dTa * dt_sec
        Ts += dTs * dt_sec
        Tw += dTw * dt_sec

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
            dTa_excess = self.thermal_model.calculate_atmosphere_latent_heat(
                excess_condensation_rate,
                self.thermal_model.config.Lv,
                self.thermal_model.config.c_air,
            )
            Ta += dTa_excess * dt_sec
            Wa -= excess_water
            Wc += excess_water
            # Add to Condensation so the plotter sees the full condensation budget this step.
            # Prognostic will recompute it next step from the updated (corrected) Wa anyway.
            Condensation += excess_condensation_rate

        # 2. Advect the post-physics state (S→L: transport last).
        H, Ta, Wa, Wc, Ws, V = self._advect(H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt)

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


    def _advect(self, H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt):
        """Performs iterative sub-steps for advection to improve numeric stability."""
        sub_dt = dt / self.config.adv_sub_steps
        for _ in range(self.config.adv_sub_steps):
            Ta, Wa, Wc, V = self.wind_model(H, sea_level, Ta, P, V, Wa, Wc, sub_dt)
            Ws = self.water_advection_engine(H, M_sea, Ws, sub_dt)
            H = self.erosion_engine(H, Ws, Ta, sub_dt) # Placeholder for future erosion logic
        return H, Ta, Wa, Wc, Ws, V
