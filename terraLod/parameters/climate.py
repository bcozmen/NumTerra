climate_params = {
    'latitude': 25.0,   # central latitude of the map in degrees (positive = N hemisphere)
    'precipitation_range' : (200, 2000), # mm per year at the wettest point (for scaling purposes)
    'wetness':   1.0,   # 0 = arid desert, 1 = very wet tropics
    'alpha': 0.85,      # wind field balance between geostrophic (0) and orographic (1)
    'orog_k_per_km': 0.05, # orographic precipitation increase per km of elevation (0.05 = 5% per km)
    'slope_beta': 0.025, # slope exponent for orographic precipitation (0.025 = mild)
    'moisture_diffusion_sigma_km': 20.0, # km, controls how much moisture is diffused across the map
    'plains_halflife_mult': 3.0,  # how many times longer moisture lives over flat terrain vs. mountains
                                  # 1.0 = no boost, 3.0 = 3× longer halflife on plains (more inland penetration)
    'plains_flat_slope': 0.01,    # slope threshold [m/m] defining "flat" (~1% grade); cells below this
                                  # receive the full plains_halflife_mult boost
}
