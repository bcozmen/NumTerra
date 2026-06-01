import numpy as np
from dataclasses import dataclass, field

from .numba import d8_water_routing

@dataclass
class WaterConfig:
    slope_exponent: float = 2.0     # Weight steeper slopes more: 1=linear, 2=quadratic, …
    flow_rate: float = 1.38e-4      # Fraction of cell's water drained per second at maximum weight (0.5 / 3600)
    field_capacity: float = 20.0    # Soil moisture held by capillary forces [mm]; only excess above this routes



class Water():
    def __init__(self, world):
        self.world = world
        self.config = WaterConfig()


    def __call__(self, H, M_sea, Ws, dt):
        """
        Parameters
        ----------
        H      : float32 (rows, cols)  normalised terrain height [0, 1]
        M_sea  : float32 (rows, cols)  sea mask (1 = ocean)
        Ws     : float32 (rows, cols)  surface water [mm]
        dt     : float                 timestep [hr]

        Returns
        -------
        Ws_new : float32 (rows, cols)
        """
        # Only route water above the field capacity.  Below this level, capillary
        # forces hold moisture in the soil — it doesn't flow downhill.
        # Sea cells (M_sea=1) contribute zero runoff; they are sinks, not sources.
        surface = H * self.world.max_altitude + Ws * 0.001  # Effective surface height (m), including water layer
        Ws_runoff  = np.maximum(Ws - self.config.field_capacity, 0.0) * (1.0 - M_sea)
        Ws_retained = Ws - Ws_runoff

        dt_sec = dt * 3600.0

        Ws_runoff_routed = d8_water_routing(
            surface, M_sea, Ws_runoff,
            self.world.max_altitude, self.world.area.cell_size[0], self.world.area.cell_size[1],
            dt_sec, self.config.slope_exponent, self.config.flow_rate,
        )
        return Ws_retained + Ws_runoff_routed