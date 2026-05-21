import numpy as np

def get_cell_size(lim, max_size, height_map_shape):
    x_min, x_max, y_min, y_max = lim
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    cell_size_x = (max_size * x_range) / height_map_shape[0]
    cell_size_y = (max_size * y_range) / height_map_shape[1]

    return (cell_size_x, cell_size_y), max_range
    
def set_labels(ax, x_label='X', y_label='Y', z_label=None, title='3D Terrain', zlim = None):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    ax.set_title(title)
    if zlim is not None:
        ax.set_zlim(zlim)

def find_z_limits(height_map, lim, shape, max_size):
    h_range = height_map.max() - height_map.min()
    _, max_range = get_cell_size(lim, max_size, shape)
    max_range = 1 + 3 * np.sqrt(max_range)

    min_lim = height_map.min() - 0.01 * h_range
    max_lim = height_map.max() + (h_range * max_range)
    return (min_lim, max_lim)