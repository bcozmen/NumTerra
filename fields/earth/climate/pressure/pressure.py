import numpy as np

class Pressure:
    def __init__(self, world):
        self.world = world

    def __call__(self, H, Ta, Wa):
        """Calculates global atmospheric surface pressure maps."""
        P0, R, g = self.world.constants['P0'], self.world.constants['R_DRY_AIR'], self.world.constants['g']
        Tk = Ta + 273.15  # Convert to Kelvin
        
        H_m = H * self.world.max_altitude  # Scale normalized H to meters
        
        # Calculate local column mass by assuming initial standard pressure,
        # then iteratively adjusting to local conditions. 
        P_approx = P0 * np.exp(-g * H_m / (R * Tk))
        column_air_mass = P_approx / g
        q = Wa / np.maximum(column_air_mass, 1e-6)

        Tv = Tk * (1.0 + 0.61 * q)  # Virtual temperature
        P = P0 * np.exp(-g * H_m / (R * Tv))
        return P