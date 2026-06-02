from dataclasses import dataclass
import numpy as np

from .oscilator import WindOscillator
from .numba import (
    euler_advect, semi_lagrangian_advect,
    wind_accelerate, pressure_project, orographic_cooling,
)

@dataclass
class WindConfig:
    lapse_rate: float = 0.0098  # Dry adiabatic lapse rate (K/m)
    wind_friction: float = 0.001  # Friction coefficient for wind
    wind_nudge_timescale: float = 6.0 # Time scale for nudging towards prevailing wind (hours)
    wind_sub_steps: int = 10 # Number of sub-steps for wind acceleration to improve stability
    pressure_gradient_factor: float = 0.02  # Scaling factor for pressure gradient force

    v_sigma : float = 1.0 # Standard deviation of wind speed fluctuations (m/s)
    v_relaxation : float = 72.0  # Time scale for wind speed to relax back to prevailing speed (hours)

    theta_sigma : float = 10.0 # Standard deviation of wind direction fluctuations (degrees)
    theta_relaxation : float = 72.0 # Time scale for wind direction to relax back to prevailing direction (hours)


class Wind:
    def __init__(self, world, scheme, poisson_iterations):
        self.config = WindConfig()
        self.world = world
        self.dx, self.dy = world.area.cell_size
        self.advect3 = semi_lagrangian_advect if scheme == 'semi_lagrangian' else euler_advect
        self.poisson_iterations = poisson_iterations
        self.f = 2.0 * world.constants['Omega'] * np.sin(np.radians(world.latitude))
        self.oscillator = WindOscillator(
                        world, self.config.v_sigma, self.config.v_relaxation,
                        self.config.theta_sigma, self.config.theta_relaxation)

    def init(self, V):
        rad = np.radians(self.oscillator.theta_from)
        V[..., 0] = -self.oscillator.v * np.sin(rad)
        V[..., 1] = -self.oscillator.v * np.cos(rad)
        return V


    def __call__(self, H, sea_level, Ta, P, V, Wa, Wc, dt):
        cfg = self.config
        V = V.copy()  # don't mutate caller's array in-place
        v_x, v_y = V[..., 0], V[..., 1]
        dt_sec = dt * 3600.0

        # 1. Physical forces: PGF + Coriolis + Friction (in-place) 
        P_s = P.copy() * cfg.pressure_gradient_factor
        for _ in range(cfg.wind_sub_steps):
            wind_accelerate(P_s, v_x, v_y, self.dx, self.dy, self.world.constants['rho0'], self.f, cfg.wind_friction, dt_sec / cfg.wind_sub_steps)
        # 3. Mass conservation: divergence → Jacobi pressure solve → projection (in-place)
        pressure_project(v_x, v_y, self.dx, self.dy, self.poisson_iterations)
        # 2. Macro nudge: relax towards stochastic prevailing wind
        v_macro, theta_macro = self.oscillator.step(dt)
        rad = np.radians(theta_macro)
        nudge = min(dt / (cfg.wind_nudge_timescale), 1.0)
        v_x += (-v_macro * np.sin(rad) - v_x) * nudge
        v_y += (-v_macro * np.cos(rad) - v_y) * nudge

        

        # 4. Advect Ta / Wa / Wc together, then apply orographic cooling to Ta
        Ta, Wa, Wc = self.advect3(Ta, Wa, Wc, v_x, v_y, self.dx, self.dy, dt_sec)
        
        # 5. Orographic cooling: dTa = -Γ * dH, where Γ is the lapse rate and dH is the change in altitude from up/downwind terrain.
        H_m = H * self.world.max_altitude
        sea_level_m = sea_level * self.world.max_altitude
        dTa_oro = orographic_cooling(H_m, sea_level_m, v_x, v_y, self.dx, self.dy, cfg.lapse_rate)
        Ta += dTa_oro * dt * cfg.pressure_gradient_factor # Scale orographic cooling to prevent instability

        return Ta, Wa, Wc, V
