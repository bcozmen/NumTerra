from .helper import get_cell_size
import numpy as np
def hillshade(height_map, gradients, lim , max_altitude, max_size, azim, elev):
    cell_size, max_range = get_cell_size(lim, max_size, height_map.shape)
    z = height_map * max_altitude
    
    dzdx, dzdy = gradients
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(-dzdy, dzdx)

    az = np.radians(azim)
    el = np.radians(elev)

    hs = (
        np.sin(el) * np.cos(slope) +
        np.cos(el) * np.sin(slope) * np.cos(az - aspect)
    )

    hillshade = np.clip(255 * hs, 0, 255)
    return hillshade

def get_3D_shade(self, height_map, gradients, lim, max_size, max_altitude, ambient):
    cell_size, max_range = get_cell_size(lim, max_size, height_map.shape)
    z_meters = height_map * max_altitude

    shade = lambert_shade(height_map, gradients, cell_size=cell_size)
    shade = shade ** 0.8  # gamma correction for better contrast
    shade = ambient + (1.0 - ambient) * shade
    shade = shade[..., None]
    return shade


def compute_normals(z, gradients, cell_size=(1.0, 1.0), z_scale=1.0):

    dzdx, dzdy = gradients

    normals = np.dstack((
        -dzdx,
        -dzdy,
        np.ones_like(z)
    ))

    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    return normals



def lambert_shade(z, gradients, light_dir=(0.25, 0.25, 5.0), cell_size=(1.0, 1.0)):
    n = compute_normals(z, gradients, cell_size=cell_size)

    L = np.array(light_dir, dtype=np.float32)
    L /= np.linalg.norm(L)

    shade = np.clip(np.sum(n * L, axis=-1), 0, 1)

    return shade