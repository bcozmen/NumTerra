from .macro import macro_params
from .micro import micro_params
from .erosion import erosion_params
from .hydro import hydrology_params
from .climate import climate_params

ds_params = {
    'size_exponent': 10,
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
    'max_size': 200_000.0, #world size in meters
    'max_altitude': 3_000.0, #max altitude in meters
    'seed': 42, #random seed for noise generation
    'noise_exp_factor': 0.45, #exponent for noise contribution to final height map
    'ds_params': ds_params,
    'macro_params' : macro_params,
    'micro_params' : micro_params,
    'erosion_params': erosion_params,
    'hydrology_params': hydrology_params,
    'climate_params': climate_params,
    'plotter_params': plotter_params,
}