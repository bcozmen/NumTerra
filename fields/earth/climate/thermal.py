import numpy as np

## ========== Vertical Thermodynamics ==========
# Wind-driven turbulent heat exchange between surface and overlying air.
def calculate_sensible_heat(T_surface, Ta, Vspeed, sensible_heat_coef, c_air, c_surface):
    heat_transfer_coef = sensible_heat_coef * Vspeed
    flux = heat_transfer_coef * (T_surface - Ta)
    return flux / c_air, flux / c_surface   # (dT_air_gain, dT_surface_loss)

# Net surface temperature change from solar gain and evaporative cooling.
def calculate_surface_heating(Sun, Evap, albedo, Lv, c_surface):
    # Evap is in mm/hr, convert to kg/m2/s by dividing by 3600
    evap_flux = (Evap / 3600.0) * Lv
    net = (Sun * (1 - albedo)) - evap_flux
    return net / c_surface
    
def calculate_longwave_radiation(T_surface, Ta, Wa, stefan_boltzmann_constant, c_air, c_surface, gh_base_eps, gh_wv_mult, gh_wv_coef):
    """Calculates longwave radiation exchange between surface, atmosphere (greenhouse effect), and space."""
    # Prevent negative Kelvin temperatures
    Tk_surf = np.maximum(T_surface + 273.15, 0.0)
    Tk_air = np.maximum(Ta + 273.15, 0.0)
    
    outgoing_surface = stefan_boltzmann_constant * (Tk_surf ** 4)
    
    # Emissivity of the atmosphere depends on baseline GHGs (CO2, etc) plus water vapor (Wa)
    # Clamped to [0, 1] — physical emissivity cannot exceed 1
    eps_a = np.clip(gh_base_eps + gh_wv_mult * (1.0 - np.exp(-gh_wv_coef * np.maximum(Wa, 0.0))), 0.0, 1.0)
    
    
    # Atmosphere emits both up to space and down to surface based on its emissivity
    downwelling_atmosphere = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
    upwelling_atmosphere = eps_a * stefan_boltzmann_constant * (Tk_air ** 4)
    
    # Atmosphere absorbs a fraction of outgoing surface radiation
    absorbed_by_atmosphere = eps_a * outgoing_surface
    
    # Net fluxes
    net_surface_lw = downwelling_atmosphere - outgoing_surface
    # Atmosphere gains from absorbed surface LW; loses only what it emits upward to space.
    # The downwelling term belongs to the surface budget, not the atmospheric energy balance.
    # This gives net_air_lw = eps*sigma*(Ts^4 - Ta^4): 0 at equilibrium, stable.
    net_air_lw = absorbed_by_atmosphere - upwelling_atmosphere
    
    return net_air_lw / c_air, net_surface_lw / c_surface

# Air warming from latent heat released when water vapour condenses into clouds.
def calculate_atmosphere_latent_heat(Condensation, Lv, c_air):
    # When water vapor condenses into clouds, it releases latent heat into the air.
    # Condensation is in mm/hr, convert to kg/m2/s
    heat_released = (Condensation / 3600.0) * Lv
    dT_air_latent = heat_released / c_air
    return dT_air_latent





