import numpy as np
#from scipy.ndimage import map_coordinates, spline_filter


class Interpolator:
    def __init__(self, arr, order=3, can_interp=True):
        self.orig_arr = arr
        self.orig_arr.setflags(write=False)  # Ensure the original array is not modified
        self.dtype = arr.dtype

        self.interp_arr = None

        
        self.order = order
        self.can_interp = can_interp
        # number of channels: 1 for 2D arrays, C for (H, W, C)

    def update(self, arr):
        self.orig_arr = arr
        self.orig_arr.setflags(write=False)  # Ensure the original array is not modified
        self.interp_arr = None

    def __call__(self, pts=None, lim=(0, 1, 0, 1)):
        if pts is None:
            return self.orig_arr
        if not self.can_interp:
            raise ValueError("This Interpolator instance is not callable. Set can_interp=True to enable interpolation.")
        raise NotImplementedError("Interpolation not implemented yet. Call the Interpolator with pts=None to get the original array.")

