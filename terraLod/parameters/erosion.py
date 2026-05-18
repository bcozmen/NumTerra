# ---------------------------------------------------------------------------
# Erosion parameters expressed in PHYSICAL units.
# Conversion to normalised grid units happens in Terrain._scale_erosion_params()
# so that changes to world_size, max_altitude, or size_exponent are respected.
# ---------------------------------------------------------------------------

hydraulic_params = {
    'hydraulic_iterations_density': 1.0,    # droplets per cell
    # Base rates — calibrated for size_exponent=11 (dx ≈ 48.8 m).
    # Terrain._scale_erosion_params() multiplies these by (dx_m / ref_dx_m)
    # so that each droplet step erodes the same physical volume at any resolution.
    'erosion_rate_base':             0.1,
    'deposition_rate_base':          0.05,
    'evaporation':                   0.032,
    'inertia':                       0.35,
    'gravity':                       10.0,
    'capacity_factor':               12.0,
    # Physical quantities — converted at runtime by Terrain._scale_erosion_params()
    'min_slope_angle_deg':           0.35,   # minimum slope floor in degrees
    'max_travel_m':                  50_000, # maximum droplet travel distance in metres
}

thermal_params = {
    # Number of passes = thermal_reach_m / dx_m.
    # At dx≈48.8 m: 250 m → ~5 passes.  Enough to smooth loose scree at the
    # base of slopes without grinding down cliff faces.
    'thermal_reach_m':  250,
    # Angle of repose for loose scree.  Must be below terrain max (~25°) to
    # fire at all, but close enough that solid rock faces (near-vertical) are
    # left untouched.  22° hits only the steepest unstable debris slopes.
    'talus_angle_deg':  22.0,
}

# ---------------------------------------------------------------------------
# Aeolian (wind) erosion parameters — physical units where possible.
# scale_erosion_params() converts saltation_base_m → saltation_cells and
# avalanche_talus_deg → normalised talus before passing to air_erosion().
# ---------------------------------------------------------------------------
air_params = {
    # Number of full wind passes (saltation + avalanche per pass).
    # ~8 passes produces visible dune/ridge shaping without over-eroding.
    'air_iterations':       8,

    # Wind direction vector (need not be unit; will be normalised at runtime).
    # (1, 0.3) → roughly WSW prevailing wind.
    'wind_x':               1.0,
    'wind_y':               0.3,

    # Shear coefficient: multiplied by the windward slope (normalised units).
    # Larger → more aggressive entrainment on exposed ridges.
    'wind_strength':        0.08,

    # Minimum shear to mobilise a grain (Shields-like threshold).
    # Must be below the normalised windward slope of the terrain.
    # At dx≈48.8 m, dz=3000 m: max slope ≈ tan(25°)×(48.8/3000) ≈ 0.0076.
    # Setting threshold to 0.003 ensures most windward faces are erodible.
    'threshold':            0.003,

    # Physical saltation hop distance in metres.
    # Converted to grid cells at runtime: saltation_cells = saltation_base_m / dx_m.
    # 150 m is typical mid-range saltation for sandy terrain.
    'saltation_base_m':     150.0,

    # Fraction of eroded volume entrained per unit excess shear.
    'erosion_rate':         0.015,

    # Fraction of entrained grains deposited at the landing cell (1.0 → mass conserved).
    'deposition_rate':      1.0,

    # Maximum stable slope for slip-face avalanching, in degrees.
    # Must be below terrain max slope (~25°) to trigger; 20° is a good
    # starting point for visible dune shaping.
    'avalanche_talus_deg':  20.0,
}



erosion_params = {
    "hydraulic": hydraulic_params,
    "thermal":   thermal_params,
    "air":        air_params,
}
