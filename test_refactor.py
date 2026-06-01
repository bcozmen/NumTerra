import numpy as np

# Let's test if np.nanmin ignores nans correctly
data = np.array([1.0, 2.0, np.nan])
print(np.nanmin(data), np.nanmax(data))
