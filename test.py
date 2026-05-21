import matplotlib.pyplot as plt
import numpy as np

from nature.worldConfig import World
from nature.terrain import Terrain, Thermal, Wind, Humidity
from nature.plotter import Plotter

print("--- Initialization ---")
world = World()
Terrain(world)
Thermal(world)
Wind(world)
Humidity(world)

def print_stats(w, stage):
    print(f"--- Stats {stage} ---")
    maps = ["height", "temperature", "humidity", "rain", "soil_moisture", "runoff"]
    for m in maps:
        if m in w.maps:
            arr = w[m]()
            print(f"{m:15s} | min: {arr.min():8.2f} | max: {arr.max():8.2f} | mean: {arr.mean():8.2f}")
            #print percentiles [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            print(f"{'':15s} | " + " | ".join([f"{p:5.1f}%" for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]))
            print(f"{'':15s} | " + " | ".join([f"{np.percentile(arr, p*100):8.2f}" for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]))
        else:
            print(f"{m} not found")
    
    if "wind" in w.maps:
        arr = w["wind"]()
        mag = np.sqrt(arr[..., 0]**2 + arr[..., 1]**2)
        print(f"wind            | min: {mag.min():8.2f} | max: {mag.max():8.2f} | mean: {mag.mean():8.2f}")
        print(f"{'':15s} | " + " | ".join([f"{p:5.1f}%" for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]))
        print(f"{'':15s} | " + " | ".join([f"{np.percentile(mag, p*100):8.2f}" for p in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]))

print_stats(world, "After Init")

print("\n--- Running Iterations ---")
for i in range(10):
    world()

print_stats(world, "After 10 Iterations")
