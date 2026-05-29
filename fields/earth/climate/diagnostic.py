import numpy as np
from dataclasses import dataclass, field
from fields import BaseModel

map_info = {
    'T' : {
        'unit' : 'C',
        'description' : 'Temperature map (Ta: air temp, Ts: surface temp, Tw: water temp)'
    },
    'Wa' : {
        'unit' : 'kg/m2',
        'description' : 'Atmospheric water content'
    },
    'Ws' : {
        'unit' : 'mm',
        'description' : 'Surface water (e.g. soil moisture, water bodies)'
    },
    'V' : {
        'requires_magnitude' : True,
        'unit' : 'm/s',
        'description' : 'Wind vector map'
    }
}



class AdvectionEngine():
    #If checkerboard patter emerges, apply gaussian blur
    def __init__(self, sub_steps, wind_friction, latitude, cell_size=(1000,1000), rho_air=1.225, omega=7.2921e-5):
        self.sub_steps = sub_steps
        self.wind_friction = wind_friction
        self.latitude = latitude
        self.cell_size = cell_size  # Grid spacing in meters (tuple: (dx, dy))
        self.rho_air = rho_air     # Surface air density (kg/m3)
        self.omega = omega
        
        # Calculate the Coriolis parameter: f = 2 * Omega * sin(latitude)
        self.f = 2.0 * self.omega * np.sin(np.radians(self.latitude))

    def advect(self, H, Ta, P, V, Wa, dt):
        """
        V is a 3D array of shape (rows, cols, 2) 
        where V[..., 0] is u (i/y-wind) and V[..., 1] is v (j/x-wind).
        """
        for step in range(self.sub_steps):
            dt_sub = dt / self.sub_steps
            
            dTa_advect, dWa_advect = self.calculate_advection(H, Ta, V, Wa)
            dV = self.calculate_wind_acceleration(P, V)

            Ta += dTa_advect * dt_sub
            Wa += dWa_advect * dt_sub
            V += dV * dt_sub
            
        return Ta, Wa, V

    def calculate_wind_acceleration(self, P, V):
        """
        Calculates the acceleration of the wind vector (dV/dt) using
        Pressure Gradient Force, Coriolis Effect, and Friction.
        """
        dx, dy = self.cell_size
        
        # np.gradient returns gradients along (axis=0, axis=1) -> (rows/y, cols/x)
        dP_dy, dP_dx = np.gradient(P, dy, dx)
        
        # 1. Pressure Gradient Force (PGF)
        pgf_y = -dP_dy / self.rho_air
        pgf_x = -dP_dx / self.rho_air
        
        # Extract wind components. 
        # Based on your docstring mapping: V[..., 0] is y-wind, V[..., 1] is x-wind
        v_y = V[..., 0]
        v_x = V[..., 1]
        
        # 2. Coriolis Effect 
        # Deflects moving air right in Northern Hemisphere (lat > 0), left in Southern.
        coriolis_y = -self.f * v_x
        coriolis_x = self.f * v_y
        
        # 3. Rayleigh Surface Friction
        fric_y = -self.wind_friction * v_y
        fric_x = -self.wind_friction * v_x
        
        # Total acceleration per component
        dV_y = pgf_y + coriolis_y + fric_y
        dV_x = pgf_x + coriolis_x + fric_x
        
        # Restack into the (rows, cols, 2) shape
        return np.stack([dV_y, dV_x], axis=-1)


    def calculate_advection(self, H, Ta, V, Wa):
        """
        Calculates the advection (spatial transport) of Air Temperature (Ta) 
        and Atmospheric Water (Wa), including orographic temperature effects.
        """
        dx, dy = self.cell_size
        
        # Extract wind components
        v_x = V[..., 0]
        v_y = V[..., 1]
        
        # Compute spatial gradients for scalars
        dTa_dx, dTa_dy = np.gradient(Ta, dx, dy)
        dWa_dx, dWa_dy = np.gradient(Wa, dx, dy)
        dH_dx, dH_dy = np.gradient(H, dx, dy)
        
        # 1. Horizontal Advection
        adv_Ta = -(v_x * dTa_dx + v_y * dTa_dy)
        adv_Wa = -(v_x * dWa_dx + v_y * dWa_dy)
        
        # 2. Orographic (Vertical) Effects on Temperature
        # Vertical velocity (w) is induced by wind blowing over terrain slopes
        w = (v_x * dH_dx) + (v_y * dH_dy)
        
        # Dry adiabatic lapse rate (approx 9.8°C per 1000m -> 0.0098 K/m)
        lapse_rate = 0.0098
        
        # Cooling when forced up a mountain (w > 0), warming when forced down (w < 0)
        oro_Ta = -w * lapse_rate
        
        # Total advective and structural changes
        dTa_advect = adv_Ta + oro_Ta
        dWa_advect = adv_Wa
        
        return dTa_advect, dWa_advect


@dataclass
class DiagnosticClimateConfig:
    # Effective heat capacities for the whole vertical column (J/m2/K)
    c_air: float = 1.004e7   # 1004 J/kg/K * ~10000 kg/m2 of air column
    c_land: float = 2.0e5    # Land surface skin/vegetation capacity
    c_water: float = 4.184e7 # ~10m deep active mixing layer (1000kg/m3 * 10m * 4184 J/kg/K)
    
    sensible_heat_coef: float = 1.2 # Coefficient for sensible heat exchange
    stefan_boltzmann_constant: float = 5.670374419e-8 # (W/m2/K4)
    albedo_land: float = 0.25  # Average albedo for land
    albedo_water: float = 0.06 # Average albedo for water
    Lv: float = 2.5e6          # Latent heat of vaporization/condensation (J/kg)

    wind_friction: float = 0.0015 # Friction coefficient for wind acceleration
    adv_sub_steps: int = 10 # Number of sub-steps for advection calculations to improve stability
    rho_air: float = 1.225 # Surface air density (kg/m3), used for wind acceleration calculations
    omega: float = 7.2921e-5 # Earth's angular velocity (rad/s)

class DiagnosticClimate(BaseModel):
    info = {
        'name':'diagnostic_climate',
        'map_info' : map_info
    }
    
    def __init__(self, world):
        super().__init__(world)
        self.config = DiagnosticClimateConfig() 
        self.init() 

        self.advection_engine = AdvectionEngine(
            sub_steps=self.config.adv_sub_steps,
            wind_friction=self.config.wind_friction,
            latitude=self.world.latitude,
            cell_size=self.world.area.cell_size,
            rho_air=self.config.rho_air,
            omega=self.config.omega
        )

    ## ========== Simulation & Generation ==========
    def init(self):
        H = self.world.area['H']()  # Terrain height
        M_sea = self.world.area['M_sea']()  # Sea mask
        # Initialize temperature maps based on terrain and sea mask
    
    def step(self):
        H, M_sea, Sun, T, P, Wa, Ws, V, Vspeed, Evap, Precip = self.get_maps()
        dt = self.world['time'].dt

        # Sensible heat exchange calculations
        dT_air_from_land, dT_land_loss = self._calculate_sensible_heat_land(
            T, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_land
        )
        dT_air_from_water, dT_water_loss = self._calculate_sensible_heat_water(
            T, Vspeed, self.config.sensible_heat_coef, self.config.c_air, self.config.c_water
        )

        # Radiative and latent heat calculations
        dT_air_latent = self._calculate_atmosphere_latent_heat(
            Precip, self.config.Lv, self.config.c_air
        )
        dT_land_solar_and_evap = self._calculate_land_surface_heating(
            Sun, Evap, T, self.config.albedo_land, self.config.Lv, 
            self.config.stefan_boltzmann_constant, self.config.c_land
        )
        dT_water_solar_and_evap = self._calculate_water_surface_heating(
            Sun, Evap, T, self.config.albedo_water, self.config.Lv, 
            self.config.stefan_boltzmann_constant, self.config.c_water
        )

        # Advection skipped for now - leaving V unchanged
        # V = self.advec() 
        Ta, Wa, V = self.advection_engine.advect(H, T[0], P, V, Wa, dt)
        T[0] = Ta

        # Blend coastline calculations for sensible heat
        dT_air_sensible = (M_sea * dT_air_from_water) + ((1.0 - M_sea) * dT_air_from_land)

        # Apply integrations (Euler step)
        T[0] += (dT_air_sensible + dT_air_latent) * dt
        T[1] += (dT_land_solar_and_evap - dT_land_loss) * dt
        T[2] += (dT_water_solar_and_evap - dT_water_loss) * dt

        # Apply mass balance for water
        Wa += (Evap - Precip) * dt
        Ws += (Precip - Evap) * dt
        
        # Enforce constant saturation on pure sea cells
        Ws[M_sea == 1.0] = 1000.0 # Or whatever baseline you use for ocean depth in mm

        self.set_maps({
            'T' : T,
            'Wa' : Wa,
            'Ws' : Ws,
            'V' : V
        })

    ## ========== Map Management ==========
    def get_maps(self):
        H = self.world.area['H']()  # Terrain height
        M_sea = self.world.area['M_sea']()  # Sea mask
        Sun = self.world.area['Sun']() # Added to fetch solar radiation
        T = self.world.area['T']().copy()
        P = self.world.area['P']()
        Wa = self.world.area['Wa']().copy()
        Ws = self.world.area['Ws']().copy()
        V = self.world.area['V']().copy() # Added to keep vector state safe
        Vspeed = self.world.area['V_magnitude']()
        Evap = self.world.area['Evap']()
        Precip = self.world.area['Precip']()
        
        return H, M_sea, Sun, T, P, Wa, Ws, V, Vspeed, Evap, Precip

    ## ========== Vertical Thermodynamics ==========
    def _calculate_sensible_heat_land(self, T, Vspeed, sensible_heat_coef, c_air, c_land):
        Ta, Ts, Tw = T
        heat_transfer_coef = sensible_heat_coef * Vspeed
        flux = heat_transfer_coef * (Ts - Ta)

        dT_air_gain = flux / c_air
        dT_land_loss = flux / c_land
        return dT_air_gain, dT_land_loss

    def _calculate_sensible_heat_water(self, T, Vspeed, sensible_heat_coef, c_air, c_water):
        Ta, Ts, Tw = T
        # Same principle as land, but using Water Temperature (Tw)
        heat_transfer_coef = sensible_heat_coef * Vspeed
        flux = heat_transfer_coef * (Tw - Ta)

        dT_air_gain = flux / c_air
        dT_water_loss = flux / c_water
        return dT_air_gain, dT_water_loss

    def _calculate_atmosphere_latent_heat(self, Precip, Lv, c_air):
        # When water vapor condenses into precipitation, it releases latent heat into the air.
        # Assuming Precip is in kg/m2/s (or appropriately scaled by dt)
        heat_released = Precip * Lv
        dT_air_latent = heat_released / c_air
        return dT_air_latent

    def _calculate_land_surface_heating(self, Sun, Evap, T, albedo_land, Lv, stefan_boltzmann_constant, c_land):
        Ta, Ts, Tw = T
        Tk = Ts + 273.15
        outgoing_radiation = stefan_boltzmann_constant * (Tk**4)
        
        # Net flux = Solar In - Radiative Out - Latent Heat of Evaporation Out
        net_radiation = (Sun * (1 - albedo_land)) - outgoing_radiation - (Evap * Lv)
        return net_radiation / c_land

    def _calculate_water_surface_heating(self, Sun, Evap, T, albedo_water, Lv, stefan_boltzmann_constant, c_water):
        Ta, Ts, Tw = T
        Tk = Tw + 273.15
        outgoing_radiation = stefan_boltzmann_constant * (Tk**4)
        
        # Water bodies function identically to land for surface heating, 
        # but utilize water albedo, water temp, and water heat capacity.
        net_radiation = (Sun * (1 - albedo_water)) - outgoing_radiation - (Evap * Lv)
        return net_radiation / c_water