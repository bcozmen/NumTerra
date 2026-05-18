# ---------------------------------------------------------------------------
# River erosion parameters.
# Rivers are identified by flow accumulation (D8 routing on a pit-filled DEM)
# and then carved into the landscape with smoothed valley walls.
# ---------------------------------------------------------------------------
hydro_params = {
    # Fraction of total cells that must drain through a cell for it to be
    # treated as a river.  Smaller → more rivers (fine headwater streams).
    # 0.003 ≈ 0.3 % of cells, which at 2049×2049 ≈ 12 500 cells → large rivers.
    # 0.0005 captures medium-sized tributaries as well.
    # Resolution-invariant: threshold = fraction × n_cells, which scales
    # proportionally to accumulation at any grid resolution.
    'accumulation_threshold': 0.0008, # 

    # Maximum carve depth expressed in PHYSICAL METRES.
    # Converted to normalised [0,1] units at runtime via carve_strength_m / max_altitude.
    # Applied at the highest-accumulation cell; actual depth scales as
    # log1p(acc / threshold) / log1p(total_cells / threshold).
    # 120 m → 4 % of max_altitude = 3000 m.
    'carve_strength_m': 120.0,

    # Half-width of the valley-wall smoothing kernel in physical metres.
    # Converted to grid cells at runtime.  500 m produces gentle V-valleys
    # at 48.8 m/cell resolution; reduce for narrow gorges.
    'valley_width_m': 500.0,

    # Extra hydraulic erosion passes run only along river cells after carving.
    # Currently unused by the carving kernel (reserved for future use).
    'river_iterations': 0,

    # Maximum number of distinct river networks to keep.  Set to None to use
    # ``accumulation_threshold`` directly.  When set, the threshold is
    # auto-raised (binary search) until ≤ max_rivers connected channel
    # components remain.  Start with ~5-15 for a continental-scale map.
    'max_rivers': 30,

    # Lake density controls
    # ---------------------
    # Maximum area of a single lake as a fraction of the total map.
    # Basins larger than this are discarded (they are typically large flat
    # plains that have flooded entirely rather than genuine crater/alpine lakes).
    # 0.01 → a lake may cover at most 1 % of the map.  Set to 1.0 to disable.
    'max_lake_area_fraction': 0.01,

    # Minimum lake depth in PHYSICAL METRES.
    # Basins shallower than this are discarded — they represent nearly-flat
    # depressions that flood entire valleys.  Converted to normalised units
    # at runtime via min_lake_depth_m / max_altitude.
    # 15 m kills the shallow flooded-valley artefact without removing true lakes.
    'min_lake_depth_m': 15.0,
}