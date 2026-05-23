import numpy as np

# ==========================================
# MAIN ORCHESTRATOR
# ==========================================

def compute_rain_and_update(
    humidity, humidity_cap, evap_frac,
    wind_i, wind_j, grad_i, grad_j,
    sea_mask, lake_mask, river_mask,
    soil_moisture, rain_accum, runoff_accum,
    condensation_rate, orographic_factor,
    soil_capacity, soil_evap_rate, itcz_factor,
    rain_shadow_fraction, wilting_point, hpa_to_mm,
    inv_iterations
):
    """
    Main loop step for weather and hydrology simulation.
    Orchestrates atmospheric moisture, rain events, and ground water distribution.
    """
    # 0. Set up bounds and masks
    max_humidity = humidity_cap + 1e-8
    is_water_body = sea_mask | lake_mask | river_mask

    # 1. Atmospheric Evaporation
    local_humidity, evaporation_hpa = _compute_evaporation(
        humidity, max_humidity, evap_frac, sea_mask
    )

    # 2. Convective / Frontal Rain
    convective_rain_hpa = _compute_convective_rain(
        local_humidity, max_humidity, condensation_rate, itcz_factor
    )

    # 3. Orographic (Terrain) Rain & Shadow
    orographic_rain_hpa, local_humidity = _compute_orographic_effects(
        local_humidity, max_humidity, wind_i, wind_j, grad_i, grad_j, 
        orographic_factor, itcz_factor, rain_shadow_fraction
    )

    # 4. Vapor Pressure Allocation (Deplete rain from atmosphere)
    total_precip_hpa = np.minimum(convective_rain_hpa + orographic_rain_hpa, local_humidity)
    local_humidity = np.maximum(local_humidity - total_precip_hpa, 0.0)
    
    total_precip_mm = total_precip_hpa * hpa_to_mm
    evaporation_mm = evaporation_hpa * hpa_to_mm

    # 5. Ground & Soil Hydrology
    final_soil_moisture, final_runoff = _compute_hydrology(
        soil_moisture, soil_capacity, total_precip_mm, evaporation_mm, 
        is_water_body, soil_evap_rate, inv_iterations
    )

    # 6. Final Outputs
    return (
        local_humidity, 
        final_soil_moisture, 
        rain_accum + total_precip_mm, 
        runoff_accum + final_runoff
    )

# ==========================================
# INTERNAL SIMULATION MODULES
# ==========================================

def _compute_evaporation(humidity, max_humidity, evap_frac, sea_mask):
    """Calculates atmospheric evaporation and caps moisture over oceans."""
    evaporation_hpa = evap_frac * max_humidity
    local_humidity = humidity + evaporation_hpa
    
    # Cap the marine boundary layer to prevent infinite moisture loops
    #ocean_humidity_cap = 0.88 * max_humidity
    #local_humidity = np.where(
    #    sea_mask, 
    #    np.minimum(local_humidity, ocean_humidity_cap), 
    #    local_humidity
    #)
    
    return local_humidity, evaporation_hpa


def _compute_convective_rain(local_humidity, max_humidity, condensation_rate, itcz_factor, threshold=0.75):
    """Calculates ambient rainfall triggered by high relative humidity."""
    relative_humidity = local_humidity / max_humidity
    
    # Rain triggers exponentially as relative humidity exceeds the threshold
    is_raining = relative_humidity > threshold
    rain_intensity = np.where(is_raining, ((relative_humidity - threshold) / (1.0 - threshold))**2, 0.0)
    
    return rain_intensity * condensation_rate * itcz_factor * local_humidity


def _compute_orographic_effects(local_humidity, max_humidity, wind_i, wind_j, 
                                grad_i, grad_j, orographic_factor, 
                                itcz_factor, rain_shadow_fraction):
    """Calculates terrain-driven rain (windward) and drying effects (leeward)."""
    wind_speed = np.hypot(wind_i, wind_j) + 1e-8
    
    # Calculate vertical uplift via dot product
    wind_dir_i = wind_i / wind_speed
    wind_dir_j = wind_j / wind_speed
    terrain_uplift = (wind_dir_i * grad_i) + (wind_dir_j * grad_j)
    
    # Scale and squash the uplift values 
    scaled_uplift = np.clip(3.0 * terrain_uplift, -10.0, 10.0)
    orographic_modifier = np.tanh(scaled_uplift) 
    
    is_windward_slope = orographic_modifier > 0.0
    
    # Windward slopes generate rain
    orographic_rain_hpa = np.where(
        is_windward_slope, 
        orographic_modifier * orographic_factor * itcz_factor * local_humidity, 
        0.0
    )
    
    # Leeward slopes simulate rain shadow by lowering the moisture capacity
    shadow_humidity_cap = (1.0 + orographic_modifier * (1.0 - rain_shadow_fraction)) * max_humidity
    adjusted_humidity = np.where(
        ~is_windward_slope, 
        np.minimum(local_humidity, shadow_humidity_cap), 
        local_humidity
    )
    
    return orographic_rain_hpa, adjusted_humidity


def _compute_hydrology(soil_moisture, soil_capacity, total_precip_mm, 
                       evaporation_mm, is_water_body, soil_evap_rate):
    """Calculates ground infiltration, soil moisture depletion, and runoff."""
    # Scale base evaporation by iterations
    iter_evap_depletion = (evaporation_mm * 0.2) 
    iter_soil_evap_rate = (soil_evap_rate * 0.25) 
    
    # 1. Deduct natural soil evaporation
    land_moisture = np.maximum(soil_moisture - iter_evap_depletion, 0.0)
    
    # 2. Split precipitation into infiltration and immediate runoff
    soil_saturation_ratio = np.minimum(land_moisture / soil_capacity, 1.0)
    immediate_runoff_mm = total_precip_mm * (soil_saturation_ratio ** 2)
    infiltrated_water_mm = total_precip_mm - immediate_runoff_mm
    
    # 3. Add infiltrated water and apply evaporation decay
    land_moisture += infiltrated_water_mm
    land_moisture *= (1.0 - iter_soil_evap_rate)
    
    # 4. Handle over-saturation (excess becomes runoff)
    excess_water = np.maximum(land_moisture - soil_capacity, 0.0)
    land_runoff_mm = immediate_runoff_mm + excess_water
    land_moisture = np.clip(land_moisture - excess_water, 0.0, soil_capacity)
    
    # 5. Apply permanent water body rules
    final_soil_moisture = np.where(is_water_body, soil_capacity, land_moisture)
    final_runoff = np.where(is_water_body, total_precip_mm, land_runoff_mm)
    
    return final_soil_moisture, final_runoff


