from .numba import d8_water_routing


class WaterAdvectionEngineConfig:
    slope_exponent: float = 2.0   # Weight steeper slopes more: 1=linear, 2=quadratic, …
    flow_rate: float = 0.5        # Fraction of cell's water drained per hour at maximum weight


class WaterAdvectionEngine():
    """
    D8-like multi-directional surface water routing.

    Water in each land cell drains proportionally to all downhill 8-connected
    neighbours, weighted by slope^slope_exponent.  Uses a pull formulation so
    the kernel is fully parallel (no race conditions).
    """

    def __init__(self, cell_size=(1000.0, 1000.0), max_altitude=1000.0,
                 slope_exponent=2.0, flow_rate=0.5):
        self.dx, self.dy = float(cell_size[0]), float(cell_size[1])
        self.max_altitude = float(max_altitude)
        self.slope_exponent = float(slope_exponent)
        self.flow_rate = float(flow_rate)

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
        return d8_water_routing(
            H, M_sea, Ws,
            self.max_altitude, self.dx, self.dy,
            dt, self.slope_exponent, self.flow_rate,
        )