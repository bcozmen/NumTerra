import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates, spline_filter

def normalize(arr, axis = None,vmin = None, vmax = None, range = (0, 1)):
    if vmin is None:
        vmin = np.min(arr, axis=axis, keepdims=True)
    if vmax is None:
        vmax = np.max(arr, axis=axis, keepdims=True)
    return range[0] + (arr - vmin) * (range[1] - range[0]) / (vmax - vmin + 1e-8)

def get_grid(lim = (0, 1, 0, 1), shape = (2048, 2048)):
    if type(shape) == int:
        shape = (shape, shape)
    x = np.linspace(lim[0], lim[1], shape[0])
    y = np.linspace(lim[2], lim[3], shape[1])
    return np.meshgrid(x, y, indexing='ij')

def get_interpolator(map_base, *args, **kwargs):
    x = np.linspace(0, 1, map_base.shape[0])
    y = np.linspace(0, 1, map_base.shape[1])
    return RegularGridInterpolator((x, y), map_base, *args, **kwargs)

def get_slope(height_map, cell_sizes, scale_factor=1.0):
    """
    Returns:
        slope_rad : slope angle in RADIANS  (not degrees)
        grad_x    : dH/dx in units of (height_map * scale_factor) / cell_size
        grad_y    : dH/dy  "
    If height_map is normalized [0,1] and scale_factor=max_altitude (m),
    cell_sizes in metres → grad in m/m (dimensionless slope), slope in radians.
    """
    height_map = height_map * scale_factor
    grad_x, grad_y = np.gradient(height_map, *cell_sizes)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))   # radians
    return slope_rad, grad_x, grad_y

def get_cell_size(lim, size, max_size):
    range_x, range_y = lim[1] - lim[0], lim[3] - lim[2]
    cell_size_x, cell_size_y = range_x / (size[0] - 1), range_y / (size[1] - 1)
    return (cell_size_x * max_size, cell_size_y * max_size)

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

