
hydrology_params = {
    'sea_level_percentile':      0.25,   # percentile of height map to set as sea level
    'init_lake_area_threshold':  10,     # km²; lakes smaller than this are removed (area in cells, not km²)
    'river_threshold':           10,     # NOTE: currently unused — river display uses river_threshold_norm in get_river_field
    'min_lake_river_acc':        10000,   # lakes whose peak inflow (mm·cells) is below this are evaporated
                                          # now wired up in Hydrology.run(); raise to remove more marginal lakes

    # --- precipitation-driven water budget (used by Hydrology.run) ---
    'infiltration_capacity':  0.55,   # fraction [0,1] of precipitation absorbed by soil
                                       # 0.05 was far too low: ~90% of rain became runoff, rivers overwhelmingly strong
    'land_evap_fraction':     0.60,   # max PET fraction [0,1] at 30 °C on land
                                       # 0.05 was too low: combined loss was only ~10%; now ~61% → realistic runoff
    'lake_open_evap_mm':      2000.0,  # open-water potential evaporation (mm/year) — lower → more overflow
    'spill_erosion_depth':    0.01, # normalised height eroded at lake spill point per iteration
                                       # 0.01 = 30 m/iter × 16 iters = up to 480 m total — saddles collapsed instantly
                                       # 0.003 = ~9 m/iter, much more geologically realistic
    'max_overflow_iterations': 40,    # cascade depth cap — more iterations → longer river chains
    'slope_exp':               2.5,   # MFD exponent: 1=dispersed, 2.5=channelised D8-like rivers
                                       # 1.5 was too dispersed; 1.8 produces narrower, more realistic channels
}
