"""
Numba-Accelerated Earth Energy-Balance Model
- LIVE INFINITE MODE: Computes 10 days of physics per visual frame.
- Fixed Topography generation for accurate 'terrain' colormapping.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    GRID_SIZE       = 128
    DT              = 1800.0        # 30-minute ticks
    DAYS_PER_FRAME  = 10            # Run 10 days of physics before drawing
    START_DAY       = 172           # Summer Solstice
    LATITUDE        = 45.0
    AXIAL_TILT      = 23.44

    SOLAR_FLUX_MAX  = 900.0
    ALBEDO_LAND     = 0.25
    ALBEDO_WATER    = 0.06

    A_RAD           = 210.0
    B_RAD           = 2.2
    K_GREENHOUSE    = 0.45

    K_EVAP          = 130.0
    BETA_WIND       = 0.15

    C_LAND          = 1.8e6
    C_WATER         = 75.0e6

    DIFFUSION_SIGMA = 0.6           # Gaussian kernel half-width in pixels

# ─────────────────────────────────────────────────────────────────────────────
# 2. NUMBA KERNELS
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _physics_step(
    temperature, height_map, is_water, humidity, wind_speed, Nx, Ny, Nz,
    current_day, current_hour,
    LATITUDE, AXIAL_TILT, SOLAR_FLUX_MAX, ALBEDO_LAND, ALBEDO_WATER,
    A_RAD, B_RAD, K_GREENHOUSE, K_EVAP, BETA_WIND, C_LAND, C_WATER, DT
):
    N = temperature.shape[0]

    lat_rad     = LATITUDE * np.pi / 180.0
    declination = AXIAL_TILT * np.sin(2.0 * np.pi * (current_day - 80.0) / 365.0) * np.pi / 180.0
    hour_angle  = (current_hour - 12.0) * 15.0 * np.pi / 180.0

    # Fixed sun vector so it rises in the East (+X)
    sx = -np.cos(declination) * np.sin(hour_angle)
    sy = ( np.cos(lat_rad) * np.sin(declination)
         - np.sin(lat_rad) * np.cos(declination) * np.cos(hour_angle) )
    sz = ( np.sin(lat_rad) * np.sin(declination)
         + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle) )

    is_day = sz > 0.0
    out = np.empty_like(temperature)

    for i in prange(N):
        for j in range(N):
            T   = temperature[i, j]
            h   = height_map[i, j]
            wet = is_water[i, j]
            hum = humidity[i, j]
            ws  = wind_speed[i, j]

            # --- Solar ---
            if is_day:
                dot = sx * Nx[i, j] + sy * Ny[i, j] + sz * Nz[i, j]
                if dot < 0.0:
                    dot = 0.0
                albedo  = ALBEDO_WATER if wet else ALBEDO_LAND
                Q_solar = SOLAR_FLUX_MAX * dot * (1.0 - albedo)
            else:
                Q_solar = 0.0

            # --- OLR ---
            atm_density        = np.exp(-h / 8000.0)
            eff_greenhouse     = K_GREENHOUSE * hum * atm_density
            Q_radiation        = (A_RAD + B_RAD * T) * (1.0 - eff_greenhouse)

            # --- Evaporation ---
            vpd  = 1.0 - hum
            moist = 1.0 if wet else min(hum * 0.4, 0.4)
            Q_evap = K_EVAP * (1.0 + BETA_WIND * ws) * vpd * moist

            # --- Net heating ---
            Q_net  = Q_solar - Q_radiation - Q_evap
            C_cell = C_WATER if wet else C_LAND
            T_new  = T + (Q_net * DT) / C_cell

            # Freezing floor for ocean/lake
            if wet and T_new < -1.5:
                T_new = -1.5

            out[i, j] = T_new

    return out, sx, sy, sz

@njit(parallel=True, cache=True)
def _gaussian_blur_sep(arr, sigma):
    N  = arr.shape[0]
    radius = int(3.0 * sigma + 0.5)
    if radius < 1:
        return arr.copy()
    ksize = 2 * radius + 1

    kernel = np.empty(ksize)
    s2 = 2.0 * sigma * sigma
    ksum = 0.0
    for k in range(ksize):
        x = k - radius
        v = np.exp(-(x * x) / s2)
        kernel[k] = v
        ksum += v
    for k in range(ksize):
        kernel[k] /= ksum

    tmp = np.empty_like(arr)
    for i in prange(N):
        for j in range(N):
            acc = 0.0
            for k in range(ksize):
                jj = j + k - radius
                if jj < 0: jj = 0
                elif jj >= N: jj = N - 1
                acc += arr[i, jj] * kernel[k]
            tmp[i, j] = acc

    out = np.empty_like(arr)
    for j in prange(N):
        for i in range(N):
            acc = 0.0
            for k in range(ksize):
                ii = i + k - radius
                if ii < 0: ii = 0
                elif ii >= N: ii = N - 1
                acc += tmp[ii, j] * kernel[k]
            out[i, j] = acc
    return out

# ─────────────────────────────────────────────────────────────────────────────
# 3. WORLD GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_world(cfg):
    G = cfg.GRID_SIZE
    Y, X = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G), indexing='ij')

    dist  = np.sqrt(X**2 + Y**2)
    land  = dist < 0.72

    ridge   = np.exp(-((X - 0.45)**2) / 0.03) * 2600.0 * land
    volcano = np.exp(-((X + 0.35)**2 + (Y + 0.3)**2) / 0.03) * 1900.0 * land
    
    # FIX: Elevate base land and deepen water so 'terrain' colormap displays correctly
    hmap = np.maximum(ridge, volcano)
    hmap[land] += 150.0  

    sea  = ~land
    lake = (dist < 0.15) & (hmap < 400)
    
    hmap[sea] = -500.0
    hmap[lake] = -50.0

    is_water = sea | lake

    hum = 0.85 - 0.65 * np.exp(-((X)**2 + (Y)**2) / 0.25)
    hum[is_water] = 0.90
    hum = np.clip(hum, 0.05, 1.0)

    wind = np.ones((G, G), dtype=np.float32) * 4.0
    wind[hmap > 1500] = 7.0

    dx = 10000.0  # Widened spatial footprint for more realistic shading
    gy, gx = np.gradient(hmap.astype(np.float64), dx)
    norm = np.sqrt(gx**2 + gy**2 + 1.0)
    Nx = (-gx / norm).astype(np.float32)
    Ny = (-gy / norm).astype(np.float32)
    Nz = (1.0  / norm).astype(np.float32)

    return hmap.astype(np.float32), is_water.astype(np.bool_), hum.astype(np.float32), wind, Nx, Ny, Nz

# ─────────────────────────────────────────────────────────────────────────────
# 4. LIVE SIMULATION & VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def run_live_simulation(cfg):
    G = cfg.GRID_SIZE

    # FIX: Repositioned indices.
    idx_ocean    = (114, 12)  # Placed in deep water
    idx_desert   = (69, 76)   # Placed on flat plains
    idx_mountain = (44, 41)   # Centered on the circular volcano!

    hmap, is_water, humidity, wind, Nx, Ny, Nz = build_world(cfg)

    temperature  = np.ones((G, G), dtype=np.float32) * 15.0
    current_day  = float(cfg.START_DAY)
    current_hour = 12.0  # Force sample times to be at high noon for a brighter thermal map

    steps_per_frame = int(cfg.DAYS_PER_FRAME * 24 * 3600 / cfg.DT)

    time_hist, desert_hist, ocean_hist, mountain_hist, sun_z_hist = [], [], [], [], []

    # Warm-up JIT
    print("Warming up Numba JIT ...", end="", flush=True)
    _tiny = np.ones((4, 4), dtype=np.float32)
    _physics_step(_tiny, _tiny, _tiny.astype(np.bool_), _tiny, _tiny, _tiny, _tiny, _tiny,
                  current_day, current_hour, cfg.LATITUDE, cfg.AXIAL_TILT, cfg.SOLAR_FLUX_MAX,
                  cfg.ALBEDO_LAND, cfg.ALBEDO_WATER, cfg.A_RAD, cfg.B_RAD, cfg.K_GREENHOUSE,
                  cfg.K_EVAP, cfg.BETA_WIND, cfg.C_LAND, cfg.C_WATER, cfg.DT)
    _gaussian_blur_sep(_tiny, cfg.DIFFUSION_SIGMA)
    print(" done.")

    # ── Setup Matplotlib ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Numba Earth Model — Infinite Sim ({cfg.DAYS_PER_FRAME} Days / Frame)",
                 fontsize=14, fontweight='bold')
    plt.subplots_adjust(hspace=0.38, wspace=0.32)

    ax_temp, ax_topo = axes[0]
    ax_chart, ax_flux = axes[1]

    # Topography
    topo_img = ax_topo.imshow(hmap, cmap='terrain', origin='lower')
    fig.colorbar(topo_img, ax=ax_topo, label="Elevation (m)", fraction=0.046)
    ax_topo.set_title("World Topography")
    for (r, c), col, lbl in [(idx_ocean, 'cyan', 'Ocean'), (idx_desert, 'red', 'Desert'), (idx_mountain, 'white', 'Mountain')]:
        ax_topo.plot(c, r, 'o', color=col, markersize=8, markeredgecolor='k', label=lbl)
    ax_topo.legend(loc='upper left', fontsize=8)

    # Thermal Map
    temp_im = ax_temp.imshow(temperature, cmap='turbo', vmin=-30, vmax=50, origin='lower')
    fig.colorbar(temp_im, ax=ax_temp, label="Temp (°C)", fraction=0.046)
    for (r, c), col in [(idx_ocean, 'cyan'), (idx_desert, 'red'), (idx_mountain, 'white')]:
        ax_temp.plot(c, r, 'o', color=col, markersize=5, markeredgecolor='k')

    # Time-series Chart
    ax_chart.set_xlabel("Elapsed Simulation Time (Hours)")
    ax_chart.set_ylabel("Temperature (°C)")
    ax_chart.set_title("Station Temperature Histories")
    ax_chart.grid(True, linestyle='--', alpha=0.4)

    line_desert,   = ax_chart.plot([], [], color='#ff3333', linewidth=2, label='Desert Plain')
    line_ocean,    = ax_chart.plot([], [], color='#00aacc', linewidth=2, label='Maritime Ocean')
    line_mountain, = ax_chart.plot([], [], color='#555555', linewidth=2, label='Mountain Peak')
    ax_chart.legend(loc='upper right', fontsize=8)

    # Diagnostics Box
    ax_flux.axis('off')
    info_text = ax_flux.text(0.05, 0.97, "", transform=ax_flux.transAxes,
                             fontsize=10.5, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f4f8', alpha=0.85))

    frame_counter = [0]

    def update(frame):
        nonlocal temperature, current_day, current_hour
        
        # ── 1. Batch Physics Run ──
        for _ in range(steps_per_frame):
            temperature, sx, sy, sz = _physics_step(
                temperature, hmap, is_water, humidity, wind, Nx, Ny, Nz,
                current_day, current_hour,
                cfg.LATITUDE, cfg.AXIAL_TILT, cfg.SOLAR_FLUX_MAX,
                cfg.ALBEDO_LAND, cfg.ALBEDO_WATER,
                cfg.A_RAD, cfg.B_RAD, cfg.K_GREENHOUSE,
                cfg.K_EVAP, cfg.BETA_WIND, cfg.C_LAND, cfg.C_WATER, cfg.DT
            )
            temperature = _gaussian_blur_sep(temperature, cfg.DIFFUSION_SIGMA)

            current_hour += cfg.DT / 3600.0
            if current_hour >= 24.0:
                current_hour -= 24.0
                current_day += 1.0

        # ── 2. Log Diagnostics ──
        elapsed_h = (current_day - cfg.START_DAY) * 24.0 + current_hour
        time_hist.append(elapsed_h)
        desert_hist.append(temperature[idx_desert])
        ocean_hist.append(temperature[idx_ocean])
        mountain_hist.append(temperature[idx_mountain])
        sun_z_hist.append(sz)

        # ── 3. Update Visuals ──
        temp_im.set_array(temperature)
        ax_temp.set_title(f"Thermal Map — Day {int(current_day)},  {current_hour:04.1f}:00")

        line_desert.set_data(time_hist, desert_hist)
        line_ocean.set_data(time_hist, ocean_hist)
        line_mountain.set_data(time_hist, mountain_hist)

        # Dynamic Axis Rescaling
        ax_chart.set_xlim(max(0, time_hist[-1] - 365*24), time_hist[-1] + 24*10)
        current_temps = desert_hist + ocean_hist + mountain_hist
        if current_temps:
            ax_chart.set_ylim(min(current_temps) - 5, max(current_temps) + 5)

        text = (
            f"{'─'*28}\n"
            f" LIVE GRID DIAGNOSTICS\n"
            f"{'─'*28}\n"
            f" Current Day : {int(current_day)}\n"
            f" Sun Sz      : {sz:+.3f}  ({'DAY  ☀' if sz > 0 else 'NIGHT ☾'})\n"
            f"{'─'*28}\n"
            f" 🌊 Ocean    : {ocean_hist[-1]:+6.1f} °C\n"
            f" 🌵 Desert   : {desert_hist[-1]:+6.1f} °C\n"
            f" 🏔  Mountain : {mountain_hist[-1]:+6.1f} °C\n"
            f"{'─'*28}\n"
            f" Frame       : {frame_counter[0]}\n"
        )
        info_text.set_text(text)
        frame_counter[0] += 1
        
        return temp_im, line_desert, line_ocean, line_mountain, info_text

    # Blit is disabled so axes dynamically rescale infinitely 
    ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    cfg = Config()
    run_live_simulation(cfg)