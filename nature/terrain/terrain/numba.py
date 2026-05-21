from numba import njit
import numpy as np

@njit(cache=True)
def detect_sea(height_map, sea_level):
    xdim, ydim = height_map.shape
    yd = ydim  # local cache (faster access in Numba)

    n = xdim * ydim

    sea   = np.zeros((xdim, ydim), dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    head  = 0
    tail  = 0

    # ---------------------------------------------------------
    # initialize from borders
    # ---------------------------------------------------------

    # top + bottom
    for x in range(xdim):

        # top
        if height_map[x, 0] <= sea_level and not sea[x, 0]:
            sea[x, 0] = True
            queue[tail] = x * yd
            tail += 1

        # bottom
        y = yd - 1
        if height_map[x, y] <= sea_level and not sea[x, y]:
            sea[x, y] = True
            queue[tail] = x * yd + y
            tail += 1

    # left + right
    for y in range(yd):

        # left
        if height_map[0, y] <= sea_level and not sea[0, y]:
            sea[0, y] = True
            queue[tail] = y
            tail += 1

        # right
        x = xdim - 1
        if height_map[x, y] <= sea_level and not sea[x, y]:
            sea[x, y] = True
            queue[tail] = x * yd + y
            tail += 1

    # ---------------------------------------------------------
    # BFS flood fill
    # ---------------------------------------------------------

    while head < tail:

        idx = queue[head]
        head += 1

        x = idx // yd
        y = idx % yd

        # -----------------------------------------------------
        # 8-neighborhood (unchanged structure)
        # -----------------------------------------------------

        nx = x - 1
        ny = y - 1
        if nx >= 0 and ny >= 0:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x - 1
        ny = y
        if nx >= 0:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x - 1
        ny = y + 1
        if nx >= 0 and ny < yd:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x
        ny = y - 1
        if ny >= 0:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x
        ny = y + 1
        if ny < yd:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x + 1
        ny = y - 1
        if nx < xdim and ny >= 0:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x + 1
        ny = y
        if nx < xdim:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

        nx = x + 1
        ny = y + 1
        if nx < xdim and ny < yd:
            if not sea[nx, ny] and height_map[nx, ny] <= sea_level:
                sea[nx, ny] = True
                queue[tail] = nx * yd + ny
                tail += 1

    return sea