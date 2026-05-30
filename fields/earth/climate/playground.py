import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. Mock World Structures to match your API
# ==========================================
class MockTime:
    def __init__(self, day_of_year):
        self.day_of_year = day_of_year

class MockWorld:
    def __init__(self, latitude, longitude, day_of_year):
        self.latitude = latitude
        self.longitude = longitude
        self.time_obj = MockTime(day_of_year)
        
    def __getitem__(self, key):
        if key == 'time':
            return self.time_obj
        raise KeyError(key)

# ==========================================
# 2. The Upgraded Wind Oscillator Class
# ==========================================
class WindOscillator:
    def __init__(self, world, v_sigma, v_relaxation, theta_sigma, theta_relaxation):
        self.world = world

        angle_from, v = self.prevailing_wind_degrees(
            world.latitude, world.longitude, world['time'].day_of_year
        )
        self.v = v
        self.v_sigma = v_sigma
        self.v_tau = 1.0 / max(1e-3, v_relaxation)

        self.theta_from = angle_from
        self.theta_sigma = theta_sigma
        self.theta_tau = 1.0 / max(1e-3, theta_relaxation)
    
    def step(self, dt_hours):
        angle_target, v_target = self.prevailing_wind_degrees(
            self.world.latitude, self.world.longitude, self.world['time'].day_of_year
        )
        
        # Speed walk (Ornstein-Uhlenbeck process)
        dv_drift = -self.v_tau * (self.v - v_target) * dt_hours
        dv_diffusion = self.v_sigma * np.sqrt(dt_hours) * np.random.randn()
        self.v = max(0.0, self.v + dv_drift + dv_diffusion)

        # Angular walk with modular wrapping
        angle_diff = (angle_target - self.theta_from + 180) % 360 - 180
        dtheta_drift = self.theta_tau * angle_diff * dt_hours
        
        speed_damping = max(0.5, self.v)
        dtheta_diffusion = (self.theta_sigma * np.sqrt(dt_hours) * np.random.randn()) 
        dtheta = (dtheta_drift + dtheta_diffusion) / speed_damping
        self.theta_from = (self.theta_from + dtheta) % 360
    
        return self.v, self.theta_from, v_target, angle_target

    def itcz_latitude(self, lon, day):
        return 15 * np.sin(2 * np.pi * (day - 80) / 365.25) + 5 * np.sin(np.radians(lon * 2))

    def prevailing_wind_degrees(self, lat, lon, day):
        dlat = lat - self.itcz_latitude(lon, day)
        a = abs(dlat)

        if a < 5:
            u, vmag, v_dir = np.interp(a, [0, 5], [0.0, -6.0]), np.interp(a, [0, 5], [0.0, 2.0]), -1
        elif a < 25:
            u, vmag, v_dir = -6.0, 2.0, -1
        elif a < 35:
            u, vmag, v_dir = np.interp(a, [25, 35], [-6.0, 6.0]), np.interp(a, [25, 35], [2.0, 1.0]), -1 if a < 30 else 1
        elif a < 55:
            u, vmag, v_dir = 6.0, 1.0, 1
        elif a < 65:
            u, vmag, v_dir = np.interp(a, [55, 65], [6.0, -3.0]), np.interp(a, [55, 65], [1.0, 0.5]), 1 if a < 60 else -1
        else:
            u, vmag, v_dir = -3.0, 0.5, -1

        v = 0.0 if dlat == 0 else np.sign(dlat) * v_dir * vmag
        az_from = (np.degrees(np.arctan2(u, v)) + 180) % 360
        return az_from, np.hypot(u, v)

# ==========================================
# 3. Interactive Simulation Setup
# ==========================================
SIM_DAYS = 120       # Duration of simulation run
DT_HOURS = 3.0       # Your target physics step size
TOTAL_STEPS = int(SIM_DAYS * 24 / DT_HOURS)

# Base World Configurations
START_LAT = 15.0     # Trade winds zone
START_LON = 0.0

# Initialize layout
fig, (ax_speed, ax_dir) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
plt.subplots_adjust(bottom=0.38, hspace=0.3)

# Storage arrays for plotting
time_axis = np.arange(TOTAL_STEPS) * DT_HOURS / 24.0 # X axis in days

# Placeholders for data lines
line_v, = ax_speed.plot(time_axis, np.zeros(TOTAL_STEPS), label='Stochastic Wind Speed', color='royalblue')
line_v_target, = ax_speed.plot(time_axis, np.zeros(TOTAL_STEPS), label='ITCZ Baseline Target', color='orange', linestyle='--')
line_theta, = ax_dir.plot(time_axis, np.zeros(TOTAL_STEPS), label='Stochastic Direction', color='crimson')
line_theta_target, = ax_dir.plot(time_axis, np.zeros(TOTAL_STEPS), label='ITCZ Baseline Target', color='orange', linestyle='--')

# Decorate Axes
ax_speed.set_ylabel('Wind Speed (m/s)')
ax_speed.set_title('Wind Oscillator Interactive Lab')
ax_speed.grid(True, alpha=0.3)
ax_speed.legend(loc='upper right')

ax_dir.set_ylabel('Direction From (Degrees)')
ax_dir.set_xlabel('Simulation Time (Days)')
ax_dir.set_ylim(-10, 370)
ax_dir.set_yticks([0, 90, 180, 270, 360])
ax_dir.set_yticklabels(['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)', 'N (360°)'])
ax_dir.grid(True, alpha=0.3)
ax_dir.legend(loc='upper right')

# ==========================================
# 4. Adding UI Slider Controls
# ==========================================
ax_color = 'lightgoldenrodyellow'
ax_v_sig = plt.axes([0.15, 0.26, 0.3, 0.03], facecolor=ax_color)
ax_v_tau = plt.axes([0.15, 0.21, 0.3, 0.03], facecolor=ax_color)
ax_t_sig = plt.axes([0.60, 0.26, 0.3, 0.03], facecolor=ax_color)
ax_t_tau = plt.axes([0.60, 0.21, 0.3, 0.03], facecolor=ax_color)
ax_lat   = plt.axes([0.38, 0.12, 0.4, 0.03], facecolor=ax_color)

s_v_sigma = Slider(ax_v_sig, 'Speed Noise ($\sigma_v$)', 0.0, 10.0, valinit=1.0, valfmt='%1.1f m/s')
s_v_relax = Slider(ax_v_tau, 'Speed Memory (hours)', 6.0, 240.0, valinit=72.0, valfmt='%1.0f h')
s_t_sigma = Slider(ax_t_sig, 'Angle Noise ($\sigma_\\theta$)', 0.0, 40.0, valinit=10.0, valfmt='%1.1f°')
s_t_relax = Slider(ax_t_tau, 'Angle Memory (hours)', 6.0, 1000.0, valinit=72.0, valfmt='%1.0f h')
s_latitude = Slider(ax_lat, 'World Latitude', -60.0, 60.0, valinit=START_LAT, valfmt='%1.1f°')

# ==========================================
# 5. Simulation Runner / Callback Function
# ==========================================
def run_simulation(val=None):
    # Fix the random seed inside the loop to see exactly how parameters alter the SAME storm
    np.random.seed(42) 
    
    # Generate mock environment
    world = MockWorld(latitude=s_latitude.val, longitude=START_LON, day_of_year=1.0)
    
    # Initialize oscillator with slider data
    osc = WindOscillator(
        world=world, 
        v_sigma=s_v_sigma.val, 
        v_relaxation=s_v_relax.val, 
        theta_sigma=s_t_sigma.val, 
        theta_relaxation=s_t_relax.val
    )
    
    res_v = np.zeros(TOTAL_STEPS)
    res_v_target = np.zeros(TOTAL_STEPS)
    res_theta = np.zeros(TOTAL_STEPS)
    res_theta_target = np.zeros(TOTAL_STEPS)
    
    # Step simulation forward
    current_day = 1.0
    for step in range(TOTAL_STEPS):
        world.time_obj.day_of_year = current_day
        
        v, theta, v_targ, theta_targ = osc.step(DT_HOURS)
        
        res_v[step] = v
        res_v_target[step] = v_targ
        res_theta[step] = theta
        res_theta_target[step] = theta_targ
        
        current_day += DT_HOURS / 24.0

    # Update line visuals
    line_v.set_ydata(res_v)
    line_v_target.set_ydata(res_v_target)
    line_theta.set_ydata(res_theta)
    line_theta_target.set_ydata(res_theta_target)
    
    # Dynically re-adjust speed axis height limits based on noise output
    ax_speed.set_ylim(-0.5, max(12, np.max(res_v) + 2))
    
    fig.canvas.draw_idle()

# Register event listeners
s_v_sigma.on_changed(run_simulation)
s_v_relax.on_changed(run_simulation)
s_t_sigma.on_changed(run_simulation)
s_t_relax.on_changed(run_simulation)
s_latitude.on_changed(run_simulation)

# Initial execution triggers chart draw
run_simulation()
plt.show()