import numpy as np
from nature.worldConfig import World
from nature.terrain import Terrain, Thermal, Wind, Humidity
w = World()
Terrain(w)
Thermal(w)
Wind(w)
Humidity(w)

s = w["soil_moisture"]()
print("Total cells:", s.size)
print("Cells with 0 soil:", (s == 0).sum())
print("Cells > 0 and < 50:", ((s > 0) & (s < 50)).sum())
print("Cells > 50 and < 150:", ((s > 50) & (s < 150)).sum())
print("Cells == 200 (water/saturated):", (s == 200).sum())

r = w["rain"]()
print("Rain = 0:", (r == 0).sum())
