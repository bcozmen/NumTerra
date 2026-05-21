from dataclasses import dataclass, field
import numpy as np

from .noise_params import macro_params, micro_params, ds_params

from terraLod.utils import domain_warp, fbm, diamond_square, normalize, timeit

@dataclass
class NoiseConfig:
    noise_exp_factor : float = 0.45
    macro_params : list = field(default_factory=lambda: macro_params)
    micro_params : list = field(default_factory=lambda: micro_params)
    ds_params : dict = field(default_factory=lambda: ds_params)

class NoiseGenerator:
    @timeit(label="Noise Generator Initialization")
    def __init__(self, seed, config = NoiseConfig()):
        self.seed = seed
        self.config = config
    
    @timeit(label="Diamond Square Generation")
    def build_ds(self, size):
        params = self.config.ds_params
        offset = 2000
        params['seed'] = self.seed + offset
        return normalize(diamond_square(size, **params))
    @timeit(label="Noise Generation")
    def get_noise(self, grid, macro = True):
        params = self.config.macro_params if macro else self.config.micro_params

        noise = np.zeros_like(grid[0])

        for noise_dict in params:
            offset = 0 if macro else 1000
            this_dict = {**noise_dict, 'seed': self.seed + offset + params.index(noise_dict) * 174}
            #noise_dict['seed'] = self.seed + offset + params.index(noise_dict) * 174

            noise_layer = self._build_noise_layer(grid, this_dict)
            weight = (2 * this_dict['weight']) / (this_dict['base_freq'])
            noise += noise_layer * weight
        
        if not macro:
            return noise
        noise = normalize(noise, range = (-1, 1)) 
        noise = np.exp(noise * self.config.noise_exp_factor)

        return noise
    
    def _build_noise_layer(self, grid, noise_dict):
        X, Y = grid
        X, Y = domain_warp(X, Y, **noise_dict)
        noise = fbm(X, Y, **noise_dict)
        return noise