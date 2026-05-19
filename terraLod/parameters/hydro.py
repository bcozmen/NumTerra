
hydrology_params = {
    'sea_level_percentile':      0.25,   # percentile of height map to set as sea level
    'init_lake_area_threshold':  10,    # km²; lakes smaller than this are removed (area in cells, not km²)
    'river_threshold':           10,     # min flow accumulation to define a river cell
    'min_lake_river_acc':        500,    # lakes below this peak inflow are evaporated

    # --- precipitation-driven water budget (used by Hydrology.run) ---
    'infiltration_capacity':  0.30,   # fraction [0,1] of precipitation absorbed by soil
    'land_evap_fraction':     0.30,   # max PET fraction [0,1] at 30 °C on land
    'lake_open_evap_mm':      900.0,  # open-water potential evaporation (mm/year)
    'spill_erosion_depth':    0.005,  # normalised height eroded at lake spill point
    'max_overflow_iterations': 5,     # cascade depth cap (1 = single pass, no cascade)
}
