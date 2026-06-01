import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel


from . import Sun, Pressure, Hydro

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

    



class PrognosticClimate(BaseModel):
    info = {
        'name':'prognostic_climate',
        'map_info' : map_info
    }
    def __init__(self, world):
        super().__init__(world)
        self.init()  # Run the simulation immediately to initialize maps

    ## ========== Simulation & Generation ==========
    def init(self):
        self.Sun = Sun(self.world)  # Initialize Sun model
        self.Pressure = Pressure(self.world)  # Initialize Pressure model
        self.Hydro = Hydro(self.world)  # Initialize Hydro model
        self._calculate()  # Bootstrap maps
    
    def step(self):
        self._calculate()  # Recalculate maps based on updated terrain and temperature

    def generate(self, area):
        pass


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

        Sun, Shadow = self.Sun(H, H_grad_i, H_grad_j, M_sea, Wa, Wc)
        P = self.Pressure(H, Ta, Wa)

        Wa_max, Evap, Condensation, Precip = self.Hydro(P, Ta, Wa, Wc, M_sea, Ws, Vspeed)
        
        # Updated call matching the new layered calculation signature
        self.set_maps({
            'Sun' : Sun,
            'Shadow' : Shadow,
            'P' : P,
            'Wa_max' : Wa_max,
            'Evap' : Evap,
            'Condensation' : Condensation,
            'Precip' : Precip
        })

