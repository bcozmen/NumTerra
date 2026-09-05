import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_thermal_step(M_sea, Sun, Sun_atm, Ta, Ts, Tw, Vspeed, Evap, Condensation, Precip, Wa, Wc,
                         sensible_heat_coef, c_air, c_land, c_water,
                         Lv, sigma, gh_base_eps, gh_wv_mult, gh_wv_coef, gh_cloud_mult, gh_cloud_coeff,
                         albedo_land, albedo_water, external_heat_flux, dt):
    rows, cols = Ta.shape
    dTa = np.zeros_like(Ta)
    dTs = np.zeros_like(Ts)
    dTw = np.zeros_like(Tw)

    for i in prange(rows):
        for j in range(cols):
            # Condensation is in mm/hr, convert to kg/m2/s
            heat_released = (Condensation[i, j] / 3600.0) * Lv
            dT_air_external = external_heat_flux / c_air
            dT_air_latent = heat_released / c_air

            # True atmospheric solar absorption: energy that did not reach the surface (S0_toa - Sun).
            # Sun_atm is computed in sun.py as S0*effective_incidence*(1-transmission).
            dT_air_solar = Sun_atm[i, j] / c_air

            # Longwave radiation
            Tk_air = max(Ta[i, j] + 273.15, 0.0)
            
            # --- Smooth Multiplicative Overlap Math ---
            # Calculate the transparency (transmission) of each component
            tau_base  = 1.0 - gh_base_eps
            tau_wv    = 1.0 - (gh_wv_mult * (1.0 - np.exp(-gh_wv_coef * max(Wa[i, j], 0.0))))
            tau_cloud = 1.0 - (gh_cloud_mult * (1.0 - np.exp(-gh_cloud_coeff * max(Wc[i, j], 0.0))))

            # Combined transmission (energy that escapes directly to space)
            tau_total = tau_base * tau_wv * tau_cloud

            # Total Emissivity safely asymptotically approaches 1.0 smoothly!
            eps_a = 1.0 - tau_total
            
            atmos_emission = eps_a * sigma * (Tk_air ** 4)

            v_speed = Vspeed[i, j]
            m_sea = M_sea[i, j]
            evap_flux = (Evap[i, j] / 3600.0) * Lv
            
            if m_sea < 1.0:
                # Land
                heat_transfer_coef = sensible_heat_coef * v_speed
                sensible_flux_land = heat_transfer_coef * (Ts[i, j] - Ta[i, j])
                dT_air_from_land = sensible_flux_land / c_air
                dT_land_loss = sensible_flux_land / c_land

                Tk_surf = max(Ts[i, j] + 273.15, 0.0)
                outgoing_surface_land = sigma * (Tk_surf ** 4)
                absorbed_by_atmosphere_land = eps_a * outgoing_surface_land
                net_surface_lw_land = atmos_emission - outgoing_surface_land
                net_air_lw_land = absorbed_by_atmosphere_land - 2.0 * atmos_emission

                dT_air_lw_land = net_air_lw_land / c_air
                dT_land_lw = net_surface_lw_land / c_land

                net_solar_land = (Sun[i, j] * (1.0 - albedo_land)) - evap_flux
                dT_land_solar_evap = net_solar_land / c_land

                dTs[i, j] = (dT_land_solar_evap + dT_land_lw - dT_land_loss) * (1.0 - m_sea)
                
                # Acc
                dTa[i, j] += dT_air_from_land * (1.0 - m_sea) + dT_air_lw_land * (1.0 - m_sea)

            if m_sea > 0.0:
                # Water
                heat_transfer_coef = sensible_heat_coef * v_speed
                sensible_flux_water = heat_transfer_coef * (Tw[i, j] - Ta[i, j])
                dT_air_from_water = sensible_flux_water / c_air
                dT_water_loss = sensible_flux_water / c_water

                Tk_surf = max(Tw[i, j] + 273.15, 0.0)
                outgoing_surface_water = sigma * (Tk_surf ** 4)
                absorbed_by_atmosphere_water = eps_a * outgoing_surface_water
                net_surface_lw_water = atmos_emission - outgoing_surface_water
                net_air_lw_water = absorbed_by_atmosphere_water - 2.0 * atmos_emission

                dT_air_lw_water = net_air_lw_water / c_air
                dT_water_lw = net_surface_lw_water / c_water

                net_solar_water = (Sun[i, j] * (1.0 - albedo_water)) - evap_flux
                dT_water_solar_evap = net_solar_water / c_water

                dTw[i, j] = (dT_water_solar_evap + dT_water_lw - dT_water_loss) * m_sea
                
                # Acc
                dTa[i, j] += dT_air_from_water * m_sea + dT_air_lw_water * m_sea

            dTa[i, j] += dT_air_latent + dT_air_solar + dT_air_external

    return dTa * dt, dTs * dt, dTw * dt
