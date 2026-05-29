import numpy as np

from fields.utils import get_cell_size, get_grid, get_grad
from .interpolator import Interpolator

class Area():
    def __init__(self, world, lim = (0, 1, 0, 1), size = (1024, 1024), can_interp = False):
        self.world = world
        self.lim = lim
        self.size = size
        self.can_interp = can_interp

        self.maps = {}

        self.cache = {
            "grid" : get_grid(lim = self.lim, shape=self.size),
            "cell_size" : get_cell_size(self.lim, self.size, self.world.max_size)
        }

    @property
    def cell_size(self):
        return self.cache["cell_size"]
    @property
    def grid(self):
        grid = self.cache["grid"]
        return (grid[0].copy(), grid[1].copy())

    def __getitem__(self, key):
        if key in self.maps:
            return self.maps[key]
        fake_arr = np.zeros(self.size, dtype = np.float32)
        fake_interp = Interpolator(fake_arr, order=0, can_interp=self.can_interp)
        return fake_interp

    def __setitem__(self, key, value):
        extra_maps = self._set_grad(key, value) + self._set_magnitude(key, value)
        order = self.world.map_info[key].get('interp_order', 1)
        
        if key in self.maps:
            self._update_item(key, value, extra_maps)
        else:
            self._add_item(key, value, extra_maps, order)


    def _update_item(self, key, value, extra_maps):
        self.maps[key].update(value)
        for grad_key, grad_value in extra_maps:
            self.maps[grad_key].update(grad_value)
    def _add_item(self, key, value, extra_maps, order):
        self.maps[key] = Interpolator(value, order=order, can_interp=self.can_interp)
        for grad_key, grad_value in extra_maps:
            self.maps[grad_key] = Interpolator(grad_value, order=order, can_interp=self.can_interp)
    def _set_grad(self, key, value):
        if self.world.map_info[key].get('requires_grad', False):
            grad_value = value.copy()
            if self.world.map_info[key].get('normalize_sea_level', False):
                sea_level = self['sea_level']()
                sea_mask = self['M_sea']()
                
                grad_value = grad_value - sea_level
                grad_value[sea_mask] = 0  # Set sea areas to zero
    
            grad_mag, grad_i, grad_j = get_grad(grad_value, self.cell_size, max_altitude=self.world.max_altitude)

            return [(key + "_grad_mag", grad_mag), (key + "_grad_i", grad_i), (key + "_grad_j", grad_j)]
        return []
    def _set_magnitude(self, key, value):
        if self.world.map_info[key].get('requires_magnitude', False):
            mag_value = np.linalg.norm(value, axis=-1)
            return [(key + "_magnitude", mag_value)]
        return []