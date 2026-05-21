import numpy as np
from nature.worldConfig import World
from nature.terrain import Terrain, Thermal, Wind, Humidity

def run_with_params(evap_scale, precip_scale, baseline_decay):
    print(f"\nTesting evap_scale={evap_scale}, precip_scale={precip_scale}, decay={baseline_decay}")
    from nature.terrain.humidity import numba
    
    orig_compute = numba.compute_rain_and_update_numba
    # Just running base World to trigger the compiled logic naturally is harder...
    # Let's just modify the source file and run it.

