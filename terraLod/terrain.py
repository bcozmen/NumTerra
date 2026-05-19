import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from .helper import normalize, get_grid, scale_erosion_params, scale_hydro_params
from .noise import diamond_square, domain_warp, fbm
from .erosion import hydraulic_erosion, thermal_erosion, air_erosion
from .hydro import Hydrology
from .plotter import Plotter
from .climate.climate import Climate






DEBUG = True

class HMap():
    def __init__(self, height_map, masks, plotter, lim=(0, 1, 0, 1), extra_maps=None):
        self.height_map = height_map
        self.masks = masks
        self.lim = lim
        self.plotter = plotter

        self.shape = height_map.shape

        # All plottable maps keyed by name
        self.maps = {'height': height_map}
        if extra_maps:
            self.maps.update(extra_maps)

    def plot(self, key=None, save_path=None, shade=True, plot_slope_histogram=False):
        """
        Plot by key.  ``key=None`` (default) renders the full terrain view
        (2-D hillshaded + 3-D surface).  Any other key plots that scalar map
        as a 2-D colormap.

        Available keys: 'height', and any climate maps passed in, e.g.
        'temperature', 'humidity', 'precipitation'.
        """
        if key is None or key == 'height':
            self.plotter.plot(self.height_map, lim=self.lim, masks=self.masks,
                              save_path=save_path, shade=shade,
                              plot_slope_histogram=plot_slope_histogram)
        else:
            if key not in self.maps:
                raise KeyError(
                    f"Map '{key}' not found. Available keys: {list(self.maps.keys())}"
                )
            self.plotter.plot_overlay(self.height_map, self.maps[key], title=key,
                                      lim=self.lim, masks=self.masks, save_path=save_path)


class Terrain():
    def __init__(self, world_params):
        self.world_params = world_params
        self._init_plotter()

        ds_base = self.build_ds()
        ds_base = normalize(ds_base)

        noise, weights = self.build_noise(ds_base, self.world_params['macro_params'], macro = True)
        combined_noise = self.combine_noise(noise, weights)

        combined_noise = normalize(combined_noise, range =(-1, 1))


        
        combined = ds_base * np.exp(self.world_params['noise_exp_factor'] * combined_noise)
        combined = normalize(combined)
        
        eroded = self.erode(combined)

        eroded = gaussian_filter(eroded, sigma=1)
        eroded = normalize(eroded)

        self.hydro = self.init_hydro(eroded)
        self.climate = self.init_climate(eroded, self.hydro)
        eroded = self.hydro.run(self.climate)
        self.climate.run(init_run =False)


        self.base_map = eroded
        self.base_interpolator = self.get_interpolator(eroded)

    def init_hydro(self, height_map):
        hydro_params = self.world_params['hydrology_params']
        hydro_params = scale_hydro_params(self.world_params)
        hydro = Hydrology(height_map, **hydro_params)

        mask = {
            "sea_mask": hydro.base_sea_mask,
            "lake_mask": hydro.base_lake_mask,
        }
        self.plotter.plot(height_map, masks=mask, plot_slope_histogram=False)
        return hydro

    def init_climate(self, height_map, hydro):
        climate_params = self.world_params.get('climate_params', {})
        climate = Climate(height_map, hydro, self.world_params, **climate_params)
        climate.run()
        return climate

    
    def generate(self, lim = (0, 1, 0, 1), shape = None):
        output_shape = shape if shape is not None else self.world_params['shape']
        X, Y = get_grid(lim = lim, shape=output_shape)
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)
        this_map = self.base_interpolator(points).reshape(X.shape)

        noise, weights = self.build_noise(this_map, self.world_params['micro_params'], macro = False, lim = lim)
        combined_noise = self.combine_noise(noise, weights)
        combined = this_map + combined_noise
        masks = self.hydro.get_masks(combined, (X, Y))
        climate_maps = self.climate.get_maps() if self.climate is not None else {}
        return HMap(combined, masks, self.plotter, lim=lim, extra_maps=climate_maps)
    def build_ds(self):
        ds_params = self.world_params['ds_params']
        ds_params['seed'] = self.world_params['seed']
        ds_params['size'] = 2 ** ds_params['size_exponent'] + 1
        ds_base = diamond_square(**ds_params)
        self.world_params['shape'] = ds_base.shape
        
        return ds_base

    def erode(self, height_map):
        h_params, t_params, a_params = scale_erosion_params(self.world_params)
        total_cells = height_map.shape[0] * height_map.shape[1]
        h_params['seed']       = self.world_params['seed'] + 2000
        h_params['iterations'] = int(total_cells * h_params['hydraulic_iterations_density'])

        eroded = thermal_erosion(height_map, **t_params)
        eroded = hydraulic_erosion(eroded, **h_params)
        eroded = air_erosion(eroded, **a_params)

        return eroded



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
    def _build_noise_layer(self, noise_dict, lim = (0, 1, 0, 1)):
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
    def get_interpolator(self, grid):
        x = np.linspace(0,1,self.world_params['shape'][0])
        y = np.linspace(0,1,self.world_params['shape'][1])
        return RegularGridInterpolator((x, y), grid, bounds_error=False, fill_value=None, method='cubic')
    def _init_plotter(self):
        plotter_params = self.world_params.get('plotter_params', {})
        plotter_params['max_size'] = self.world_params.get('max_size', 100000.0)
        plotter_params['max_altitude'] = self.world_params.get('max_altitude', 3000.0)
        self.plotter = Plotter(plotter_params)

    




        
        

