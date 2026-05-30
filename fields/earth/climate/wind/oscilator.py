import numpy as np

class WindOscillator:
    """Ornstein-Uhlenbeck stochastic oscillator for macro-scale prevailing wind."""

    def __init__(self, world, v_sigma, v_relaxation, theta_sigma, theta_relaxation):
        self.world = world
        self.v_sigma = v_sigma
        self.v_tau = 1.0 / v_relaxation
        self.theta_sigma = theta_sigma
        self.theta_tau = 1.0 / theta_relaxation
        self.theta_from, self.v = self.prevailing_wind(world.latitude, world.longitude, world['time'].day_of_year)

    def step(self, dt):
        theta_target, v_target = self.prevailing_wind(self.world.latitude, self.world.longitude, self.world['time'].day_of_year)
        sqrt_dt = np.sqrt(dt)
        # Speed: Ornstein-Uhlenbeck (relax to target + white noise)
        self.v = max(0.0, self.v
                     - self.v_tau * (self.v - v_target) * dt
                     + self.v_sigma * sqrt_dt * np.random.randn())
        # Direction: OU with modular wrap; high speed damps angular noise
        angle_diff = (theta_target - self.theta_from + 180) % 360 - 180
        speed_damping = max(0.5, self.v)
        
        dTheta_drift = self.theta_tau * angle_diff * dt
        dTheta_diffusion = self.theta_sigma * sqrt_dt * np.random.randn()
        dTheta = (dTheta_drift + dTheta_diffusion) / speed_damping
        self.theta_from = (self.theta_from + dTheta) % 360
        return self.v, self.theta_from

    def prevailing_wind(self, lat, lon, day):
        """Returns (azimuth_from_degrees, speed_m/s) for the prevailing circulation."""
        dlat = lat - self._itcz_latitude(lon, day)
        a = abs(dlat)
        if a < 5:
            u    = np.interp(a, [0, 5],  [0.0, -6.0])
            vmag = np.interp(a, [0, 5],  [0.0,  2.0])
            v_dir = -1
        elif a < 25:
            u, vmag, v_dir = -6.0, 2.0, -1
        elif a < 35:
            u    = np.interp(a, [25, 35], [-6.0, 6.0])
            vmag = np.interp(a, [25, 35], [ 2.0, 1.0])
            v_dir = -1 if a < 30 else 1
        elif a < 55:
            u, vmag, v_dir = 6.0, 1.0, 1
        elif a < 65:
            u    = np.interp(a, [55, 65], [6.0, -3.0])
            vmag = np.interp(a, [55, 65], [1.0,  0.5])
            v_dir = 1 if a < 60 else -1
        else:
            u, vmag, v_dir = -3.0, 0.5, -1
        v = 0.0 if dlat == 0 else np.sign(dlat) * v_dir * vmag
        return (np.degrees(np.arctan2(u, v)) + 180) % 360, np.hypot(u, v)

    def _itcz_latitude(self, lon, day):
        return 15 * np.sin(2 * np.pi * (day - 80) / 365.25) + 5 * np.sin(np.radians(lon * 2))

