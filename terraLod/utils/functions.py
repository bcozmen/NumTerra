import numpy as np

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

