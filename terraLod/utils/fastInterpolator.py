import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

class FastInterpolator:
    def __init__(self, arr, order=3, can_call=True):
        self.orig_arr = arr
        self.dtype = arr.dtype
        self.interp_arr = None

        self.shape = arr.shape[:2]   # always (H, W)
        self.order = order
        self.can_call = can_call
        # number of channels: 1 for 2D arrays, C for (H, W, C)
        self.n_channels = arr.shape[2] if arr.ndim == 3 else None


    def _initialize(self):
        if self.n_channels is not None:
            # prefilter each channel independently
            if self.order > 1:
                self.interp_arr = np.stack(
                    [spline_filter(self.orig_arr[..., c], order=self.order)
                     for c in range(self.n_channels)],
                    axis=-1
                )
                self.prefilter = False
            else:
                self.interp_arr = self.orig_arr
                self.prefilter = True
        else:
            if self.order > 1:
                self.interp_arr = spline_filter(self.orig_arr, order=self.order)
                self.prefilter = False
            else:
                self.interp_arr = self.orig_arr
                self.prefilter = True

    def copy(self):
        copy = FastInterpolator(self.orig_arr.copy(), order=self.order, can_call=self.can_call)
        if self.interp_arr is not None:
            copy.interp_arr = self.interp_arr.copy()
        return copy
    def update(self):
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
            raise ValueError("This FastInterpolator instance is not callable. Set can_call=True to enable interpolation.")
        if self.interp_arr is None:
            self._initialize()

        coords = self._make_coords(pts, lim)

        if self.n_channels is not None:
            # interpolate each channel and stack into (..., C)
            return np.stack(
                [map_coordinates(
                    self.interp_arr[..., c],
                    coords,
                    order=self.order,
                    mode='nearest',
                    prefilter=self.prefilter
                ) for c in range(self.n_channels)],
                axis=-1
            ).astype(self.dtype)

        return map_coordinates(
            self.interp_arr,
            coords,
            order=self.order,
            mode='nearest',
            prefilter=self.prefilter
        ).astype(self.dtype)

