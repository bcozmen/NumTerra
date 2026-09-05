import numpy as np
from dataclasses import dataclass, field

from fields import BaseModel


from . import Sun, Thermal, Pressure, Hydro, Wind,Water, Erosion

"""
map_info = {
    'H' : {
        'interp_order' : 3,
        'requires_grad' : True,
        'normalize_sea_level' : True,
        'unit' : 'm',
        'description' : 'Height map of the terrain',
        'render' : {'cmap': 'terrain'},  # composite: land=terrain, sea=Blues depth
    },
    'M_sea' : {
        'interp_order' : 0,
        'description' : 'Boolean mask indicating sea vs land',
    },
    'sea_level' : {
        'interp_order' : 0,
        'unit' : 'm',
        'description' : 'Height threshold for sea level',
    },
}
"""

map_info = {
    'Ta' : {
        'unit' : '°C',
        'description' : 'Air temperature',
        'render' : {'cmap': 'RdYlBu_r', 'vrange': (0, 40)},
    },
    'Ts' : {
        'unit' : '°C',
        'description' : 'Surface (land) temperature',
        'render' : {'cmap': 'RdYlBu_r', 'vrange': (0, 45)},
    },
    'Tw' : {
        'unit' : '°C',
        'description' : 'Water (ocean/lake) temperature',
        'render' : {'cmap': 'Blues_r', 'vrange': (-2, 32)},
    },
    'Wa' : {
        'unit' : 'kg/m²',
        'description' : 'Atmospheric water content',
        'render' : {'cmap': 'YlGnBu', 'scale': 'linear', 'vrange': (15, 40)},
    },
    'Wc' : {
        'unit' : 'kg/m²',
        'description' : 'Cloud liquid water content',
        'render' : {'cmap': 'Blues', 'scale': 'linear', 'vrange': (0, 0.6)},  # No upper limit; clouds can accumulate
    },
    'Ws' : {
        'unit' : 'mm',
        'description' : 'Surface water (e.g. soil moisture, water bodies)',
        'render' : {'cmap': 'YlGn', 'scale': 'linear', 'mask_sea': True, 'vrange': (0, 150)},  # No upper limit; surface water can accumulate
    },
    'V' : {
        'requires_magnitude' : True,
        'unit' : 'm/s',
        'description' : 'Wind vector map (rows, cols, 2)',
    },
    'Sun' : {
        'unit' : 'W/m²',
        'description' : 'Solar energy input to the surface',
        'render' : {'cmap': 'hot', 'vrange': (0, 1500)},
    },
    'Shadow' : {
        'unit' : 'bool/float',
        'description' : 'Shadow map (0.0 = completely shadowed, 1.0 = fully lit)',
        #'render' : {'cmap': 'gray'},
    },
    'P' : {
        'unit' : 'Pa',
        'description' : 'Atmospheric pressure',
        'render' : {'cmap': 'RdBu_r', 'scale': 'linear'},
    },
    'Wa_max' : {
        'unit' : 'kg/m²',
        'description' : 'Maximum atmospheric water capacity',
        'render' : {'cmap': 'YlGnBu', 'scale': 'linear', 'vrange': (15, 100)},
    },
    'Evap' : {
        'unit' : 'mm/hr',
        'description' : 'Evaporation rate',
        'render' : {'cmap': 'PuBu', 'scale': 'linear', 'vrange': (0, 0.3)},
    },
    'Condensation' : {
        'unit' : 'mm/hr',
        'description' : 'Condensation rate (vapor to cloud)',
        'render' : {'cmap': 'PuBuGn', 'scale': 'linear', 'vrange': (0, 0.3)},
    },
    'Precip' : {
        'unit' : 'mm/hr',
        'description' : 'Precipitation rate (cloud to surface)',
        'render' : {'cmap' : 'Blues', 'scale' : 'linear', 'vrange' : (0, 0.2)},
    },
}


@dataclass
class ClimateConfig:
    # Advection parameters
    atmospheric_layer_count: int = 3  # Number of layers to integrate for Wa_max estimation
    adv_sub_steps: int = 2 # Number of sub-steps for advection calculations to improve stability
    advection_scheme: str = 'semi_lagrangian' # Advection integration scheme
    advection_poisson_iterations: int = 10 # Iterations for Poisson solver to enforce mass continuity in advection


class Climate(BaseModel):
    info = {
        'name':'climate',
        'map_info' : map_info
     }
    def __init__(self, world):
        super().__init__(world)
        self.config = ClimateConfig()

        
        self.sun = Sun(self.world)
        self.thermal = Thermal(self.world)
        self.pressure = Pressure(self.world)
        self.hydro = Hydro(self.world, self.config.atmospheric_layer_count)

        self.wind = Wind(self.world, self.config.advection_scheme, self.config.advection_poisson_iterations)
        self.water = Water(self.world)
        self.erosion = Erosion(self.world) # Placeholder for future erosion logic
        
        H, H_grad_i, H_grad_j, sea_level, M_sea, _, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip = self.get_maps()
        self._init_prognostic(H, H_grad_i, H_grad_j, M_sea, Ta, Wa, Wc, Ws, Vspeed)
        self._init_diagnostic(H, sea_level, M_sea, Ta, Ts, Tw, Wa, Wa_max, Wc, Ws, V)

        ## ========== Map Management ==========
    def get_maps(self):
        H = self.world.area['H']().copy()  # Terrain height
        H_grad_i, H_grad_j = self.world.area['H_grad_i'](), self.world.area['H_grad_j']()  # Terrain gradients
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
        
        return H, H_grad_i, H_grad_j, sea_level, M_sea, Sun, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip

    def step(self):
        H, H_grad_i, H_grad_j, sea_level, M_sea, _, Ta, Ts, Tw, P, Wa, Wa_max, Wc, Ws, V, Vspeed, Evap, Condensation, Precip = self.get_maps()
        dt, dt_seconds = self.world['time'].dt, self.world['time'].dt * 3600.0
        
        Sun, Shadow, Sun_atm = self.sun(H, H_grad_i, H_grad_j, M_sea, Wa, Wc)

        # Keep thermal and moisture budgets in phase: use hydro fluxes from this same step.
        Wa_max, Evap, Condensation, Precip = self.hydro(P, Ta, Ts, Tw, Wa, Wc, M_sea, Ws, Vspeed)
        
        dTa, dTs, dTw = self.thermal(M_sea, Sun, Sun_atm, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc, dt_seconds)
        Ta, Ts, Tw = Ta + dTa, Ts + dTs, Tw + dTw

        Ta, Wa, Wc, Ws, Condensation = self.hydro.apply_mass_balance(
            Ta, Wa, Wc, Ws, Wa_max, Evap, Condensation, Precip, dt, self.thermal
        )

        P = self.pressure(H, Ta, Wa)  # ← now uses Ta_final and Wa_final

        H, Ta, Wa, Wc, Ws, V = self._advect(H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt)
        
        # Sea border: use actual water depth in Ws so advection correctly interacts with sea level
        Ws[M_sea == 1.0] = np.maximum(0.0, sea_level - H[M_sea == 1.0]) * self.world.max_altitude * 1000.0
        
        self.set_maps({
            'H' : H,
            'Sun' : Sun,
            'Shadow' : Shadow,
            'Ta' : Ta,
            'Ts' : Ts,
            'Tw' : Tw,
            'P' : P,
            'Wa' : Wa,
            'Wa_max' : Wa_max,
            'Wc' : Wc,
            'Ws' : Ws,
            'V' : V,
            'Evap' : Evap,
            'Condensation' : Condensation,
            'Precip' : Precip
        })
        

    def _advect(self, H, M_sea, sea_level, Ta, P, Wa, Wc, Ws, V, dt):
        """Performs iterative sub-steps for advection to improve numeric stability."""
        sub_dt = dt / self.config.adv_sub_steps
        for _ in range(self.config.adv_sub_steps):
            Ta, Wa, Wc, V = self.wind(H, sea_level, Ta, P, V, Wa, Wc, sub_dt)
            #Ws = self.water(H, M_sea, Ws, sub_dt)
            #H = self.erosion(H, Ws, Ta, sub_dt) # Placeholder for future erosion logic
        return H, Ta, Wa, Wc, Ws, V

    def _init_prognostic(self, H, H_grad_i, H_grad_j, M_sea, Ta, Wa, Wc, Ws, Vspeed):
        """Computes dependent maps (Sun, Pressure, Moisture Capacity, Evaporation, Precipitation) sequentially."""
        Sun, Shadow, _Sun_atm = self.sun(H, H_grad_i, H_grad_j, M_sea, Wa, Wc)
        # Temperature is diagnostic but initialized here since it's needed for pressure and hydro
        Ta, Ts, Tw = self.thermal.init(H, M_sea)
        P = self.pressure(H, Ta, Wa)
        Wa_max, Evap, Condensation, Precip = self.hydro(P, Ta, Ts, Tw, Wa, Wc, M_sea, Ws, Vspeed)

        self.set_maps({'Ta' : Ta, 'Ts' : Ts, 'Tw' : Tw, 'P' : P,
                        'Wa_max' : Wa_max, 'Evap' : Evap, 'Condensation' : Condensation, 'Precip' : Precip,
                        'Sun' : Sun, 'Shadow' : Shadow,})

    def _init_diagnostic(self, H, sea_level, M_sea, Ta, Ts, Tw, Wa, Wa_max, Wc, Ws, V):
        """Bootstraps the initial water, cloud, and wind state."""
        H = self.world.area['H']()
        M_sea = self.world.area['M_sea']()
        Wa_max = self.world.area['Wa_max']()  # Populated by _init_prognostic at T=0

        # Atmospheric water: 70 % of max capacity
        Wa = np.clip(0.7 * Wa_max, 0.0, None).astype(np.float32)

        # Clouds: Starts clear
        Wc = np.zeros_like(Wa)

        # Surface water: uniform initial moisture (mm) for land, actual depth for sea cells
        Ws = np.full(H.shape, 50.0, dtype=np.float32)
        Ws[M_sea == 1.0] = np.maximum(0.0, sea_level - H[M_sea == 1.0]) * self.world.max_altitude * 1000.0

        # Initialize Wind
        V = np.zeros((*H.shape, 2), dtype=np.float32)
        V = self.wind.init(V)

        self.set_maps({'Wa': Wa, 'Wc': Wc, 'Ws': Ws, 'V': V})
