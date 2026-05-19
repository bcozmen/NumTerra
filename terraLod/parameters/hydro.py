
hydrology_params = {
    'sea_level_percentile':      0.25,   # percentile of height map to set as sea level
    'init_lake_area_threshold':  10,    # km²; lakes smaller than this are removed (area in cells, not km²)
    'river_threshold':           10,     # min flow accumulation to define a river cell
    'min_lake_river_acc':        500,    # lakes below this peak inflow are evaporated

    # --- precipitation-driven water budget (used by Hydrology.run) ---
    'infiltration_capacity':  0.05,   # fraction [0,1] of precipitation absorbed by soil
    'land_evap_fraction':     0.05,   # max PET fraction [0,1] at 30 °C on land
    'lake_open_evap_mm':      100.0,  # open-water potential evaporation (mm/year) — lower → more overflow
    'spill_erosion_depth':    0.01,  # normalised height eroded at lake spill point
    'max_overflow_iterations': 16,     # cascade depth cap — more iterations → longer river chains
    'slope_exp':               1.5,   # MFD exponent: 1=dispersed, 2.5=channelised D8-like rivers
}
