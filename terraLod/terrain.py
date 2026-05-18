import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

from .helper import normalize, get_grid, timeit, scale_erosion_params
from .noise import diamond_square, domain_warp, fbm
from .erosion import hydraulic_erosion, thermal_erosion, air_erosion
from .hydro import river_erosion
from .plotter import Plotter




DEBUG = True

class HMap():
    def __init__(self, height_map, river_mask, river_height, plotter, lim = (0, 1, 0, 1)):
        self.height_map = height_map
        self.river_mask = river_mask
        self.river_height = river_height
        self.lim = lim
        self.plotter = plotter

        self.shape = height_map.shape

    def plot(self, save_path = None, shade = True):
        self.plotter.plot(self.height_map, lim=self.lim, river_mask=self.river_mask, river_height=self.river_height, save_path=save_path,  shade=shade)


class Terrain():
    def __init__(self, world_params):
        self.world_params = world_params
        self._init_plotter()

        ds_base = self.build_ds()
        ds_base = normalize(ds_base)

        noise, weights = self.build_noise(ds_base, self.world_params['macro_params'], macro = True)
        combined_noise = self.combine_noise(noise, weights)
        #print("Combined noise:")
        #self.plotter.plot(combined_noise, shade=True, plot_slope_histogram=False)
        combined_noise = normalize(combined_noise, range =(-1, 1))

        

        #print("Base height map:")
        #self.plotter.plot(ds_base, shade=True, plot_slope_histogram=False)

        
        combined = ds_base * np.exp(self.world_params['noise_exp_factor'] * combined_noise)
        combined = normalize(combined)
        #print("Combined height map before erosion:")
        #self.plotter.plot(combined, shade=True, plot_slope_histogram=False)
        
        import time
        st = time.time()
        eroded = self.erode(combined)
        #eroded = normalize(eroded)
        et = time.time()
        print(f"Erosion took {et - st:.2f} seconds")

        #print("Final:")
        #self.plotter.plot(eroded, shade=True, plot_slope_histogram=False)

        self.base_map = self.get_interpolator(eroded)

    
    @timeit
    def generate(self, lim = (0, 1, 0, 1), shape = None):
        if shape is not None:
            self.world_params['shape'] = shape
        X, Y = get_grid(lim = lim, shape=self.world_params['shape'])
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)
        base_map = self.base_map(points).reshape(X.shape)

        noise, weights = self.build_noise(base_map, self.world_params['micro_params'], macro = False, lim = lim)
        combined_noise = self.combine_noise(noise, weights)
        combined = base_map + combined_noise 
        return HMap(combined, self._river_mask, self._river_height, self.plotter, lim = lim)
    @timeit
    def build_ds(self):
        ds_params = self.world_params['ds_params']
        ds_params['seed'] = self.world_params['seed']
        ds_params['size'] = 2 ** ds_params['size_exponent'] + 1
        ds_base = diamond_square(**ds_params)
        self.world_params['shape'] = ds_base.shape
        
        return ds_base

    @timeit
    def erode(self, height_map):
        h_params, t_params, a_params, r_params = scale_erosion_params(self.world_params)
        total_cells = height_map.shape[0] * height_map.shape[1]
        h_params['seed']       = self.world_params['seed'] + 2000
        h_params['iterations'] = int(total_cells * h_params['hydraulic_iterations_density'])

        eroded = thermal_erosion(height_map, **t_params)
        eroded = hydraulic_erosion(eroded, **h_params)

        print("After hydraulic and thermal erosion:")
        self.plotter.plot(eroded)
        eroded, river_mask, river_height, sea_mask, lake_mask, sea_level = river_erosion(eroded, **r_params)
        # Store river maps so __init__ can build interpolators from them.
        self._river_mask   = river_mask
        self._river_height = river_height
        print("After river erosion:")
        masks = {
            'river_mask': river_mask,
            'sea_mask': sea_mask,
            'lake_mask': lake_mask,
        }
        

        eroded = air_erosion(eroded, **a_params)
        eroded = normalize(eroded)
        eroded[eroded <= sea_level] = sea_level
        eroded -= sea_level
        self.plotter.plot(eroded, masks = masks)
        
        return eroded

    @timeit
    def build_noise(self, ds_base, parameters, macro = True, lim = (0, 1, 0, 1)):
        noise = np.zeros((*ds_base.shape, len(parameters)), dtype=ds_base.dtype)
        weights = self.weights_fn(parameters)
        for ix,params in enumerate(parameters):
            noise_dict = self._get_noise_dict(params, ix, macro = macro)
            
            noise[..., ix] = self._build_noise_layer(noise_dict, lim = lim) 
            if DEBUG and False:
                print(f"Noise layer {ix}")
                self.plotter.plot(noise[..., ix], lim=lim, shade=False, plot_slope_histogram=False)
        return noise, weights

    def combine_noise(self, noise, weights):
        #apply weights to noise layers and combine with ds_base in vectorised way
        noise = noise * weights[None, None, :]
        return np.sum(noise, axis=-1)
    def weights_fn(self, parameters):
        weights = np.array([p.get('weight', 0) for p in parameters])
        base_scales = np.array([p.get('base_freq', 0)  for p in parameters])
        weights = (2 * weights) / base_scales
        return weights
    @timeit
    def _build_noise_layer(self, noise_dict, lim = (0, 1, 0, 1)):
        #check if warp_x and warp_y exists and > 0
        X, Y = get_grid(lim = lim, shape=self.world_params['shape'])
        X, Y = domain_warp(X, Y, **noise_dict)

        noise = fbm(X, Y, **noise_dict)
        return noise
    def _get_noise_dict(self, noise_params, ix, macro = True):
        keys = ['octaves', 'persistence', 'lacunarity', 'base_freq', 'warp_x', 'warp_y', 'ridged']
        new_dict = {k: noise_params.get(k, 0) for k in keys}
        offset = 0 if macro else 1000
        new_dict['seed'] = self.world_params['seed'] + offset + ix * 174
        return new_dict
    @timeit
    def get_interpolator(self, grid):
        x = np.linspace(0,1,self.world_params['shape'][0])
        y = np.linspace(0,1,self.world_params['shape'][1])
        return RegularGridInterpolator((x, y), grid)
    def _init_plotter(self):
        plotter_params = self.world_params.get('plotter_params', {})
        plotter_params['max_size'] = self.world_params.get('max_size', 100000.0)
        plotter_params['max_altitude'] = self.world_params.get('max_altitude', 3000.0)
        self.plotter = Plotter(plotter_params)

    




        
        



