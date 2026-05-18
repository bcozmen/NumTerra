import numpy as np
from numba import njit, prange

def hydraulic_erosion(grid, iterations, erosion_rate, deposition_rate, evaporation,
                      min_slope, inertia, gravity, capacity_factor,
                      max_steps, seed, **kwargs):
    return hydraulic_erosion_numba(grid, int(iterations), erosion_rate, deposition_rate, evaporation,
                                  min_slope, inertia, gravity, capacity_factor,
                                  max_steps, seed)


@njit(cache=True, inline='always')
def _lcg(s):
    """64-bit LCG step.  Returns (new_state, uniform float in [0, 1))."""
    s = (
        s * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    ) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return s, float(s >> np.uint64(11)) / float(np.uint64(1) << np.uint64(53))
    
@njit(cache=True, parallel=True)
def hydraulic_erosion_numba(
    grid,
    iterations: int = 0,
    erosion_rate: float = 0.04,
    deposition_rate: float = 0.02,
    evaporation: float = 0.012,
    min_slope: float = 0.0005,
    inertia: float = 0.4,
    gravity: float = 10.0,
    capacity_factor: float = 12.0,
    max_steps: int = 600,
    seed: int = 0,
):
    h, w = grid.shape
    # Each parallel droplet reads from `out` (the live, accumulating map) so
    # that erosion by one droplet is visible to others.  Concurrent writes are
    # racy in theory, but the statistical result is a smooth blur that is
    # physically reasonable and prevents the blow-up caused by a frozen snapshot
    # where every droplet independently over-erodes the same source cells.
    out  = grid.copy()

    for i in prange(iterations):
        # Each droplet gets its own independent RNG state derived from seed + index.
        state = np.uint64(seed) ^ np.uint64(i) * np.uint64(2654435761)
        state, _ = _lcg(state)   # warm up

        # Random start position (at least 1 cell away from border)
        state, rx = _lcg(state)
        state, ry = _lcg(state)
        px = 1.0 + rx * (h - 3)
        py = 1.0 + ry * (w - 3)

        water = 1.0
        sediment = 0.0
        vx = 0.0
        vy = 0.0
        speed = 0.0

        for _step in range(max_steps):
            ix = int(px)
            iy = int(py)

            if ix < 1 or ix >= h - 1 or iy < 1 or iy >= w - 1:
                # Clamp to valid range before depositing (ix/iy can reach
                # h-1 / w-1 on the boundary check, which are valid indices,
                # but a large velocity step could push them one cell further).
                cix = max(1, min(ix, h - 2))
                ciy = max(1, min(iy, w - 2))
                out[cix, ciy] += sediment
                break

            fx = px - ix
            fy = py - iy

            # Bilinear sample from the live map.  Concurrent reads are racy
            # in prange but produce at worst a slight smoothing effect, which
            # is far preferable to the blow-up caused by a frozen snapshot.
            h00 = out[ix,     iy    ]
            h10 = out[ix + 1, iy    ]
            h01 = out[ix,     iy + 1]
            h11 = out[ix + 1, iy + 1]

            # Bilinear height at current position
            cur_h = (
                h00 * (1.0 - fx) * (1.0 - fy)
                + h10 * fx       * (1.0 - fy)
                + h01 * (1.0 - fx) * fy
                + h11 * fx       * fy
            )

            # Gradient (descent direction)
            gx = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx

            # Update velocity with inertia — steeper slopes accelerate more
            vx = vx * inertia - gx * (1.0 - inertia)
            vy = vy * inertia - gy * (1.0 - inertia)

            speed = (vx * vx + vy * vy) ** 0.5
            if speed < 1e-7:
                # Stalled droplet: deposit everything and stop
                out[ix,     iy    ] += sediment * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += sediment * fx         * (1.0 - fy)
                out[ix,     iy + 1] += sediment * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += sediment * fx         * fy
                break

            # Soft speed cap: allow physics-driven speed variation but
            # prevent runaway on steep cliffs.  Keeps erosion proportional
            # to actual slope rather than collapsing all droplets to speed=1.
            if speed > 4.0:
                inv = 4.0 / speed
                vx *= inv
                vy *= inv
                speed = 4.0

            # Step to next position
            nx = px + vx / speed   # unit-direction step so we always advance 1 cell
            ny = py + vy / speed
            nix = int(nx)
            niy = int(ny)
            if nix < 1 or nix >= h - 1 or niy < 1 or niy >= w - 1:
                out[ix, iy] += sediment
                break
            nfx = nx - nix
            nfy = ny - niy

            nh = (
                out[nix,     niy    ] * (1.0 - nfx) * (1.0 - nfy)
                + out[nix + 1, niy    ] * nfx         * (1.0 - nfy)
                + out[nix,     niy + 1] * (1.0 - nfx) * nfy
                + out[nix + 1, niy + 1] * nfx         * nfy
            )

            slope = cur_h - nh
            # Capacity scales with slope, speed AND gravity — physically motivated
            capacity = max(slope, min_slope) * speed * water * gravity * capacity_factor

            if sediment > capacity:
                # Deposit excess sediment bilinearly
                deposit = deposition_rate * (sediment - capacity)
                sediment -= deposit
                out[ix,     iy    ] += deposit * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += deposit * fx         * (1.0 - fy)
                out[ix,     iy + 1] += deposit * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += deposit * fx         * fy
            else:
                # Erode bilinearly
                erode = erosion_rate * (capacity - sediment)
                sediment += erode
                out[ix,     iy    ] -= erode * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] -= erode * fx         * (1.0 - fy)
                out[ix,     iy + 1] -= erode * (1.0 - fx) * fy
                out[ix + 1, iy + 1] -= erode * fx         * fy

            water *= 1.0 - evaporation
            if water < 0.01:
                # Evaporated: deposit remaining sediment at current position
                out[ix,     iy    ] += sediment * (1.0 - fx) * (1.0 - fy)
                out[ix + 1, iy    ] += sediment * fx         * (1.0 - fy)
                out[ix,     iy + 1] += sediment * (1.0 - fx) * fy
                out[ix + 1, iy + 1] += sediment * fx         * fy
                break

            px = nx
            py = ny

        else:
            # Droplet exhausted max_steps without any break — deposit remaining
            # sediment so mass is conserved.
            ix = int(px)
            iy = int(py)
            cix = max(1, min(ix, h - 2))
            ciy = max(1, min(iy, w - 2))
            out[cix, ciy] += sediment

    return out