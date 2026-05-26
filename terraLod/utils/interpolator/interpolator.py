import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

class DummyInterpolator:
    def __init__(self,value):
        self.value = value
    def __call__(self):
        return self.value

class Interpolator:
    def __init__(self, arr, order=3, can_call=True):
        self.orig_arr = arr
        self.orig_arr.setflags(write=False)  # Ensure the original array is not modified
        self.dtype = arr.dtype

        self.interp_arr = None

        
        self.order = order
        self.can_call = can_call
        # number of channels: 1 for 2D arrays, C for (H, W, C)


    def copy(self):
        copy = Interpolator(self.orig_arr.copy(), order=self.order, can_call=self.can_call)
        if self.interp_arr is not None:
            copy.interp_arr = self.interp_arr.copy()
        return copy
    def update(self, arr):
        self.orig_arr = arr
        self.orig_arr.setflags(write=False)  # Ensure the original array is not modified
        self.interp_arr = None

    def _make_coords(self, pts, lim):
        pts = np.asarray(pts)
        xmin, xmax, ymin, ymax = lim
        x = (pts[:, 0] - xmin) / (xmax - xmin) * (self.shape[0] - 1)
        y = (pts[:, 1] - ymin) / (ymax - ymin) * (self.shape[1] - 1)
        return np.vstack([x, y])

    def __call__(self, pts=None, lim=(0, 1, 0, 1)):
        if pts is None:
            return self.orig_arr
        if not self.can_call:
            raise ValueError("This Interpolator instance is not callable. Set can_call=True to enable interpolation.")
        raise NotImplementedError("Interpolation not implemented yet. Call the Interpolator with pts=None to get the original array.")

