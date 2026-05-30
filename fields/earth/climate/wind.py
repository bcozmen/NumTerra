import numpy as np

class Mr2OU:
    def __init__(self, x0=0.0, v0=0.0,
                 dt=1.0,
                 relaxation_time=12.0,   # velocity relaxation time
                 sigma_v=1.2):           # velocity noise scale

        self.x = x0
        self.v = v0
        self.dt = dt

        self.gamma = 1.0 / relaxation_time
        self.sigma_v = sigma_v

        # position mean-reversion (can tune separately or tie to same scale)
        self.theta_x = 1.0 / relaxation_time

    def step(self, mu_t):
        dt = self.dt

        # velocity OU
        self.v += -self.gamma * self.v * dt + self.sigma_v * np.sqrt(dt) * np.random.randn()

        # position mean reversion + inertia
        self.x += self.v * dt + self.theta_x * (mu_t - self.x) * dt

        if self.x < 0.0:
            self.x = 0.0
            self.v = 0.0

        return self.x, self.v

def ou_process(x, theta,mu, sigma):
    return x + (theta * (mu - x) + sigma * np.random.randn()) 

def vm_process(angle, mu, kappa, sigma):
    # shortest angular difference [-180, 180]
    diff = (mu - angle + 180) % 360 - 180

    # deterministic pull toward ITCZ regime
    angle += kappa * diff

    # stochastic weather variability (NOT regime change)
    angle += sigma * np.random.randn()

    return angle % 360

def itcz_latitude(day):
    return 15 * np.sin(2 * np.pi * (day - 80) / 365.25)

def prevailing_wind_degrees(lat, lon, day):
    itcz = itcz_latitude(day)
    dlat = lat - itcz
    a = abs(dlat)

    if a < 30:        # Hadley
        u, vmag = -6.0, 2.0
    elif a < 60:      # Ferrel
        u, vmag =  6.0, 1.0
    else:             # Polar
        u, vmag = -3.0, 0.5


    v = -np.sign(dlat) * vmag

    azimuth_to = np.degrees(np.arctan2(u, v)) % 360
    #azimuth_from = (azimuth_to + 180) % 360
    return azimuth_to, vmag