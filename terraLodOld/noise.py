from numba import njit, prange
import numpy as np

def diamond_square(size, scale, roughness, seed=0, **kwargs):
    return diamond_square_numba(size, scale, roughness, seed)

def domain_warp(X, Y, seed, warp_x, warp_y,octaves, persistence, lacunarity, base_freq, **kwargs):

    wx = np.zeros_like(X)
    if warp_x > 0:
        wx = fbm_numba(X, Y, seed + 101, octaves, persistence, lacunarity, base_freq / 4)

    wy = np.zeros_like(Y)
    if warp_y > 0:
        wy = fbm_numba(X, Y, seed + 211, octaves, persistence, lacunarity, base_freq / 4)
    warp_x_norm = warp_x * base_freq
    warp_y_norm = warp_y * base_freq
    return X + warp_x_norm * wx, Y + warp_y_norm * wy

def fbm(X, Y, seed, octaves, persistence, lacunarity, base_freq, **kwargs):
    ridge = kwargs.get('ridged', False)
    if ridge:
        return ridge_fbm_numba(X, Y, seed, octaves, persistence, lacunarity, base_freq)
    else:
        return fbm_numba(X, Y, seed, octaves, persistence, lacunarity, base_freq)

@njit(cache=True)
def diamond_square_numba(size, scale, roughness, seed=0):
    grid = np.zeros((size, size), dtype=np.float32)

    # Simple deterministic LCG RNG local to this function so behavior is reproducible
    # Uses 64-bit state
    state = np.uint64(seed if seed is not None else 0)

    def next_rand(s):
        # constants from numerical recipes
        s = np.uint64((s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)) & np.uint64(0xFFFFFFFFFFFFFFFF))
        # convert to float in [0,1)
        return s, (s >> np.uint64(11)) / float(np.uint64(1) << np.uint64(53))

    # corners
    state, r = next_rand(state)
    grid[0, 0] = r * scale
    state, r = next_rand(state)
    grid[0, size - 1] = r * scale
    state, r = next_rand(state)
    grid[size - 1, 0] = r * scale
    state, r = next_rand(state)
    grid[size - 1, size - 1] = r * scale

    step = size - 1
    current = scale

    while step > 1:
        half = step // 2

        # =========================
        # Square step
        # =========================
        for x in range(half, size - 1, step):
            for y in range(half, size - 1, step):

                avg = (
                    grid[x - half, y - half] +
                    grid[x - half, y + half] +
                    grid[x + half, y - half] +
                    grid[x + half, y + half]
                ) * 0.25

                state, r = next_rand(state)
                grid[x, y] = avg + (r - 0.5) * current

        # =========================
        # Diamond step
        # =========================
        for x in range(0, size, half):

            # alternating offset
            if ((x // half) % 2) == 0:
                start_y = half
            else:
                start_y = 0

            for y in range(start_y, size, step):

                total = 0.0
                count = 0

                if x - half >= 0:
                    total += grid[x - half, y]
                    count += 1

                if x + half < size:
                    total += grid[x + half, y]
                    count += 1

                if y - half >= 0:
                    total += grid[x, y - half]
                    count += 1

                if y + half < size:
                    total += grid[x, y + half]
                    count += 1

                state, r = next_rand(state)
                grid[x, y] = (
                    total / count +
                    (r - 0.5) * current
                )

        step //= 2
        current *= roughness

    return grid






@njit(parallel=True)
def fbm_numba(X, Y, seed, octaves, persistence, lacunarity, base_freq):
    out = np.zeros_like(X)

    rows, cols = X.shape

    for i in prange(rows):
        for j in range(cols):

            amplitude = 1.0
            frequency = base_freq
            total = 0.0
            norm = 0.0

            for o in range(octaves):
                nx = X[i, j] * frequency
                ny = Y[i, j] * frequency

                val = value_noise(nx, ny, seed + o * 17)

                total += val * amplitude
                norm += amplitude

                amplitude *= persistence
                frequency *= lacunarity

            out[i, j] = total / norm

    return out

@njit(parallel=True)
def ridge_fbm_numba(X, Y, seed, octaves, persistence, lacunarity, base_freq):
    """Ridged multi-fractal noise.

    Each octave contributes  ``1 - |2*noise - 1|``  (a "tent" function),
    which turns smooth hills into sharp ridgelines.  Successive octaves
    are weighted by the previous octave's value so ridges self-sharpen.

    Returns values roughly in [0, 1].
    """
    out = np.zeros_like(X)
    rows, cols = X.shape

    for i in prange(rows):
        for j in range(cols):
            amplitude = 1.0
            frequency = base_freq
            total = 0.0
            norm = 0.0
            prev = 1.0

            for o in range(octaves):
                nx = X[i, j] * frequency
                ny = Y[i, j] * frequency

                # Raw value in [0, 1] → ridge value in [0, 1]
                raw = value_noise(nx, ny, seed + o * 17)
                val = 1.0 - abs(raw * 2.0 - 1.0)
                val = val * val * prev   # sharpening feedback

                prev = val
                total += val * amplitude
                norm += amplitude

                amplitude *= persistence
                frequency *= lacunarity

            out[i, j] = total / norm

    return out

@njit
def hash2d(x, y, seed):
    n = x * 374761393 + y * 668265263 + seed * 1446647
    n = (n ^ (n >> 13)) * 1274126177
    return ((n ^ (n >> 16)) & 0xffffffff) / 0xffffffff

@njit
def fade(t):
    return t * t * (3.0 - 2.0 * t)

@njit
def value_noise(x, y, seed):
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = x0 + 1
    y1 = y0 + 1

    sx = fade(x - x0)
    sy = fade(y - y0)

    n00 = hash2d(x0, y0, seed)
    n10 = hash2d(x1, y0, seed)
    n01 = hash2d(x0, y1, seed)
    n11 = hash2d(x1, y1, seed)

    ix0 = n00 * (1 - sx) + n10 * sx
    ix1 = n01 * (1 - sx) + n11 * sx

    return ix0 * (1 - sy) + ix1 * sy





