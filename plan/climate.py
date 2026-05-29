# =============================================================================
# SIMULATION CONFIGURATION & INITIALIZATION
# =============================================================================
# 200km x 200km grid with latitude/longitude
iteration = 1
dt = 1.0          # 1 hour master timestep
num_substeps = 10  # For CFL stability in spatial dynamics
sub_dt = dt / num_substeps

# Static / Base Inputs
H  = init_height()  # Height map (meters)
Ms = init_sea(H)    # Sea fraction map (0.0 = pure land, 1.0 = pure sea)

# Prognostic State Maps
Ta = init_air_temperature(H, Ms)   # Air Temp (C)
Ts = init_land_temperature(H, Ms)  # Land Surface Temp (C)
Tw = init_water_temperature(Ms)    # Water Temp (C)
V  = init_trade_winds()            # Wind vector map (m/s)
Wa = constant * carry_capacity(Ta) # Atmospheric water (kg/m2)
Ws = init_surface_water(Ms)        # Surface Water (mm)

# Diagnostics (Allocated arrays for hourly calculations)
S = P = E = R = Wa_max = None 

# Eco-Vegetation Maps
F   = init_forest_density()        # Forest density [0,1]
Veg = init_vegetation_density()    # Vegetation density [0,1]


# =============================================================================
# CLIMATE STEP FUNCTION
# =============================================================================
def climate_step():
    global Ta, Ts, Tw, V, Wa, Ws, H, Ms, iteration

    # ---------------------------------------------------------
    # STEP 1: DIAGNOSTICS (Hourly large-scale states)
    # ---------------------------------------------------------
    S = Calculate_Sun_Energy(H, Wa, latitude, time_of_day)
    P = Calculate_Pressure(H, Ta, Wa)
    Wa_max = Calculate_Max_Moisture(Ta) 

    # ---------------------------------------------------------
    # STEP 2: FLUXES & SOURCES/SINKS (Hourly vertical rates)
    # ---------------------------------------------------------
    E += Calculate_Evaporation(Ms, Wa_max, Wa, Ta, Ts, Tw, V, Ws) # from external vegetation step
    R = Calculate_Precipitation(Ta, Wa, Wa_max)
    
    # Sensible heat exchange
    dT_air_from_land, dT_land_loss   = Calculate_Sensible_Heat_Land(Ta, Ts, V)
    dT_air_from_water, dT_water_loss = Calculate_Sensible_Heat_Water(Ta, Tw, V)

    # Radiative & Latent heat changes
    dT_air_latent           = Calculate_Atmosphere_Latent_Heat(R)
    dT_land_solar_and_evap  = Calculate_Land_Surface_Heating(S, E, Ts)
    dT_water_solar_and_evap = Calculate_Water_Surface_Heating(S, E, Tw)

    # ---------------------------------------------------------
    # STEP 3: SPATIAL DYNAMICS (Sub-stepped PDEs for stability)
    # ---------------------------------------------------------
    for i in range(num_substeps):
        # Calculate velocities and horizontal gradients
        dT_advect, dWa_advect = Calculate_Advection(H, Ta, V, Wa)
        dV = Calculate_Wind_Acceleration(P, H, wind_friction, latitude)
        erosion, sediment, dWs_advect = Calculate_Erosion_And_Water_Advent(H, Ws) 
        
        # Apply time-decay to erosion
        erosion  *= (1.0 / iteration)
        sediment *= (1.0 / iteration)

        # Immediate micro-step integration
        Ta += dT_advect * sub_dt
        Wa += dWa_advect * sub_dt
        V  += dV * sub_dt
        Ws += dWs_advect * sub_dt
        H  += (sediment - erosion) * sub_dt

    # ---------------------------------------------------------
    # STEP 4: INTEGRATION (Apply Hourly Source/Sink Fluxes)
    # ---------------------------------------------------------
    # Blended coastline calculation (prevents sharp binary boundaries)
    dT_air_sensible = (Ms * dT_air_from_water) + ((1.0 - Ms) * dT_air_from_land)
    
    # Apply vertical thermodynamics
    Ta += (dT_air_sensible + dT_air_latent) * dt
    Ts += (dT_land_solar_and_evap - dT_land_loss) * dt
    Tw += (dT_water_solar_and_evap - dT_water_loss) * dt
    
    # Apply mass balances
    Wa += (E - R) * dt
    Ws += (R - E) * dt 
    
    # Enforce constant sea level on marine cells
    Ws[Ms == 1.0] = sea_level 
    
    # ---------------------------------------------------------
    # STEP 5: TERRAIN & MASK HANDOFF
    # ---------------------------------------------------------
    old_Ms = Ms.copy()
    Ms.update(H) 
    
    # Ghost Temperature Fix: Initialize new sea cells with their old land temp
    new_sea_cells = (Ms > 0.0) & (old_Ms == 0.0)
    Tw[new_sea_cells] = Ts[new_sea_cells]

    E = 0.0  # Reset evaporation for next iteration


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================
while True:
    # Ecosystem directly updates Ws, Wa, and cools Ts via transpiration
    external_vegetation_step()  
    
    # Run core climate mechanics
    climate_step()
    
    # Other decoupled external steps
    externals() 

    # Safely plot/save maps here for total internal consistency
    # plot_climate_state() 

    iteration += 1