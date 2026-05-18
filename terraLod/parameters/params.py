from .macro import macro_params
from .micro import micro_params
from .erosion import erosion_params
from .hydro import hydro_params

ds_params = {
    'size_exponent': 11,
    'scale': 1.0,
    'roughness': 0.55,
}


plotter_params = {
    'shade_azim': 45,
    'shade_elev': 30,
    'azim': -180-30,
    'elev': 30,
    'light_dir': (0.25, 0.25, 5.0), #direction of light for shading
    'ambient': 0.6, #ambient light factor for shading
}

world_params = {
    'max_size': 100000.0, #world size in meters
    'max_altitude': 3000.0, #max altitude in meters
    'seed': 42, #random seed for noise generation
    'noise_exp_factor': 0.45, #exponent for noise contribution to final height map
    'sea_level_percentile': 0.25, #percentile of height map to set as sea level
    'ds_params': ds_params,
    'macro_params' : macro_params,
    'micro_params' : micro_params,
    'erosion_params': erosion_params,
    'hydro_params': hydro_params,
    'plotter_params': plotter_params,
}