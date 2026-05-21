You’re very close to a standard “procedural hydrology” pipeline used in terrain generation systems. The key shift is this:

You don’t actually need full atmospheric realism. You need a stable moisture + precipitation field that behaves plausibly with elevation, coast distance, and latitude.

So the problem becomes: generate consistent water input → route it downhill → accumulate it into rivers/lakes.

0. High-level pipeline (what you want)

Given:

heightmap
sea mask
latitude

You want to compute:

Step 1: climate fields
temperature
humidity / moisture availability
precipitation (rainfall map)
Step 2: hydrology
runoff per cell
water flow direction (downhill graph)
river accumulation
lake formation (sinks / basins)
1. Temperature model (keep it simple but structured)

You don’t need physics simulation. Use a deterministic function:

A. Latitude effect
temp_lat = 1 - abs(lat) / max_lat
B. Elevation lapse rate
temp = temp_lat - (height * lapse_rate)

Typical:

lapse_rate ≈ 0.6–1.0 per 1000m (scaled to your map units)
C. Optional sea moderation (very useful)

Distance to ocean smooths extremes:

near sea → milder temps
inland → hotter summers, colder winters (if you simulate seasons)

Simple version:

temp += ocean_proximity_factor
2. Moisture / humidity field (this is where rivers are born)

You don’t need full atmospheric simulation. Instead:

Moisture sources:
A. Ocean evaporation (primary)
all sea cells = humidity source = 1.0
B. Wind transport (optional but powerful)

You can approximate prevailing wind:

e.g. west → east or lat-dependent circulation bands

Then advect moisture inland:

humidity[x+1] += humidity[x] * decay
C. Terrain dampening

Mountains remove moisture:

humidity decreases when air rises over elevation

Simple rule:

humidity -= elevation_loss_factor
3. Precipitation model (the key step)

This is the most important simplification:

Rain = moist air + uplift

So compute precipitation like:

A. Orographic (mountains)

If windward slope:

uplift = max(0, height[x] - height[prev_x])
rain += humidity * uplift_factor

This creates:

wet mountains
dry rain shadows
B. Temperature constraint (optional realism)

Cold air holds less moisture:

rain *= clamp(temp, 0, 1)
C. Baseline evaporation recycling (optional)

Some rain returns moisture locally:

small feedback loop improves realism
4. Important trick: you don’t need full atmosphere

Most good terrain generators do:

precipitation = function(altitude, moisture, wind direction, ocean distance)

Not full fluid simulation.

If you only do one thing well, do this:

👉 wind + mountains + ocean moisture

That alone produces believable Earth-like rivers.

5. Now hydrology: turning rain into rivers

This is where most systems succeed or fail.

Step 1: compute flow direction (D8 or similar)

For each cell:

water flows to steepest downhill neighbor
flow_dir[x] = argmin(neighbor_height)
Step 2: rainfall becomes runoff

Each cell starts with:

water[x] = precipitation[x]
Step 3: accumulate flow (this creates rivers)

Process cells from high → low:

water[downstream] += water[upstream]

This naturally forms:

streams
rivers
deltas
Step 4: river threshold

A river appears when flow exceeds threshold:

if water[x] > river_threshold:
    mark as river

This is the key to visual results.

6. Lakes (this is the missing piece many people struggle with)

Lakes form when water cannot flow out.

Detect sinks:

A cell is a lake if:

all neighbors are higher OR equal
no downhill exit exists

Then:

Fill basin until overflow:
flood fill upward in elevation
water rises until it finds an outlet saddle point
overflow becomes river outlet

This is basically:

“water level equalization in a depression”

7. Putting it together (correct order)
Phase 1: terrain inputs
heightmap
sea mask
latitude
Phase 2: climate fields
temperature map
humidity map (from ocean + wind)
precipitation map (humidity + uplift + terrain)
Phase 3: hydrology
runoff initialization = precipitation
flow direction (downhill graph)
accumulation (river network)
lake filling (basins)
8. What actually matters most (priority order)

If you want “good-looking Earth-like results”, focus on:

1. Elevation-based flow routing (critical)
2. Moisture from oceans (critical)
3. Mountain uplift rain (very important)
4. River accumulation threshold (critical)
5. Lake basin filling (critical)

Everything else is polish.

9. Common mistake to avoid

Don’t try to simulate:

full atmospheric physics
pressure systems
detailed cloud dynamics

You’ll get instability and noise.

Instead:

fake the atmosphere, simulate only water transport correctly

10. If you want next step

I can help you design:

a clean algorithm architecture (data structures + passes)
a GPU-friendly version for large maps
or a fully deterministic noise-based climate system
or even pseudocode for Unity / Unreal / Python

Just tell me your target engine and map scale.