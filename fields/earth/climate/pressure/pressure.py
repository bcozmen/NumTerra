import numpy as np

class Pressure:
    def __init__(self, world):
        self.world = world

    def __call__(self, H, Ta, Wa):
        """Calculates global atmospheric surface pressure maps with topography and dynamic weather."""
        P0 = self.world.constants['P0']             # Sea level standard pressure (e.g., 101325 Pa)
        R  = self.world.constants['R_DRY_AIR']      # Gas constant (287.05 J/kg·K)
        g  = self.world.constants['g']              # Gravity (9.81 m/s²)
        
        Tk = Ta + 273.15  # Convert Air Temp to Kelvin
        H_m = H * self.world.max_altitude  # Scale normalized H to meters

        # --- 1. Dynamic Weather Perturbations ---
        T_mean = np.mean(Tk)
        # Warm anomalies create low pressure systems; cold creates high pressure systems
        P_sea_level = P0 * (1.0 - 0.03 * ((Tk - T_mean) / T_mean))

        # --- 2. Accurate Column Moisture (q) ---
        total_column_mass = P0 / g
        q = Wa / np.maximum(total_column_mass, 1e-6)
        q = np.clip(q, 0.0, 0.04)  # Earth's atmosphere rarely exceeds 4% water vapor by mass

        # --- 3. Virtual Temperature Integration ---
        Tv = Tk * (1.0 + 0.61 * q)

        # --- 4. Hydrostatic Barometric Equation ---
        P = P_sea_level * np.exp(-g * H_m / (R * Tv))
        
        return P