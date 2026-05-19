
hydrology_params = {
    'sea_level_percentile': 0.25,  # percentile of height map to set as sea level
    'init_lake_area_threshold' : 100, #in km square minimum lake size to be kept for climate calculations (actual lakes will be calculated after climate)
    'river_threshold':      10,   # min flow accumulation to define a river cell
    'min_lake_river_acc':   500,   # lakes below this peak inflow are evaporated
}
