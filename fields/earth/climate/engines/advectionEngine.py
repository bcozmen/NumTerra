import numpy as np
from .numba import (fast_euler_advection, semi_lagrangian_advection, compute_wind_acceleration,
                    compute_divergence, solve_poisson_jacobi, project_velocity)


class AdvectionEngine():
    # If checkerboard pattern emerges, apply gaussian blur
    def __init__(self, wind_friction, latitude, cell_size, rho_air, omega, scheme, div_iterations):
        self.wind_friction = wind_friction
        self.latitude = latitude
        self.cell_size = cell_size  # Grid spacing in meters (tuple: (dx, dy))
        self.rho_air = rho_air     # Surface air density (kg/m3)
        self.omega = omega
        self.scheme = scheme
        self.div_iterations = div_iterations
        
        # Calculate the Coriolis parameter: f = 2 * Omega * sin(latitude)
        self.f = 2.0 * self.omega * np.sin(np.radians(self.latitude))
        
    def __call__(self, H, sea_level, Ta, P, V, Wa, Wc, dt):
        """
        Executes one advection tick, updating air temp, atmospheric water, clouds, and wind vector.
        """
        v_x = V[..., 0]
        v_y = V[..., 1]
        dx, dy = self.cell_size
        
        # Continuity correction: 
        # Calculate divergence and project it out to avoid air piling up into terrain barriers.
        div = compute_divergence(v_x, v_y, dx, dy)
        p_dyn = solve_poisson_jacobi(div, dx, dy, iterations=self.div_iterations)
        project_velocity(v_x, v_y, p_dyn, dx, dy)
        
        # Calculate wind acceleration
        dV = self.calculate_wind_acceleration(P, V)
        V += dV * dt
        
        # Advect scalars based on selected scheme
        if self.scheme in ('semi_lagrangian', 'fast_euler'):
            advect_func = semi_lagrangian_advection if self.scheme == 'semi_lagrangian' else fast_euler_advection
            
            new_Ta = advect_func(Ta, v_x, v_y, dx, dy, dt)
            Wa[:] = advect_func(Wa, v_x, v_y, dx, dy, dt)
            Wc[:] = advect_func(Wc, v_x, v_y, dx, dy, dt)
            
            dTa_oro = self.calculate_orographic_effect(H, sea_level, v_x, v_y, dx, dy)
            Ta[:] = new_Ta + dTa_oro * dt
        else: # Default numpy euler
            dTa_advect, dWa_advect, dWc_advect = self.calculate_advection(H, sea_level, Ta, V, Wa, Wc)
            Ta += dTa_advect * dt
            Wa += dWa_advect * dt
            Wc += dWc_advect * dt
            
        return Ta, Wa, Wc, V

    def calculate_wind_acceleration(self, P, V):
        """
        Calculates the acceleration of the wind vector (dV/dt) using PGF, Coriolis, and Friction.
        """
        dx, dy = self.cell_size
        
        if self.scheme in ['fast_euler', 'semi_lagrangian']:
            dV_x, dV_y = compute_wind_acceleration(P, V[..., 0], V[..., 1], dx, dy, self.rho_air, self.f, self.wind_friction)
            return np.stack([dV_x, dV_y], axis=-1)
            
        # fallback for numpy euler
        dP_dx, dP_dy = np.gradient(P, dx, dy)
        pgf_x, pgf_y = -dP_dx / self.rho_air, -dP_dy / self.rho_air
        v_x, v_y = V[..., 0], V[..., 1]
        coriolis_x, coriolis_y = self.f * v_y, -self.f * v_x
        fric_x, fric_y = -self.wind_friction * v_x, -self.wind_friction * v_y
        return np.stack([pgf_x + coriolis_x + fric_x, pgf_y + coriolis_y + fric_y], axis=-1)

    def calculate_orographic_effect(self, H, sea_level, v_x, v_y, dx, dy):
        """
        Computes the vertical adiabatic cooling/warming due to terrain slopes.
        """
        H = np.maximum(H - sea_level, 0.0)  # Relative height above sea level
        
        dH_dx, dH_dy = np.gradient(H, dx, dy)
        w = (v_x * dH_dx) + (v_y * dH_dy)
        lapse_rate = 0.0098 # Dry adiabatic lapse rate (K/m)
        return -w * lapse_rate

    def calculate_advection(self, H, sea_level, Ta, V, Wa, Wc):
        """
        Calculates the advection using purely vectorised numpy (Euler).
        """
        dx, dy = self.cell_size
        v_x, v_y = V[..., 0], V[..., 1]
        
        dTa_dx, dTa_dy = np.gradient(Ta, dx, dy)
        dWa_dx, dWa_dy = np.gradient(Wa, dx, dy)
        dWc_dx, dWc_dy = np.gradient(Wc, dx, dy)
        
        adv_Ta = -(v_x * dTa_dx + v_y * dTa_dy)
        adv_Wa = -(v_x * dWa_dx + v_y * dWa_dy)
        adv_Wc = -(v_x * dWc_dx + v_y * dWc_dy)
        
        oro_Ta = self.calculate_orographic_effect(H, sea_level, v_x, v_y, dx, dy)
        return adv_Ta + oro_Ta, adv_Wa, adv_Wc
