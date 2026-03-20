# Map Matching Guide

## Overview
Map matching aligns noisy GPS observations to the most plausible road geometry in a driveable road network. In this repository, the implementation targets DOBACK routes and produces corrected coordinates for downstream terrain and stability analysis.

Input:
- GPS trajectory points (`lat`, `lon`, timestamp)
- Road network (OSM-derived graph, cached in GraphML)

Output:
- Corrected route points on-road (`lat_corrected`, `lon_corrected`)
- Metadata such as edge identifier, road type, and matching quality indicators

## Algorithm
The matcher is a three-layer approach:

1. Candidate generation and geometric scoring
- Search nearby road candidates within `max_dist`.
- Orthogonally project GPS point onto each candidate edge.
- Compute a combined score from distance and directional alignment.

2. Temporal consistency and hysteresis
- Reuse short trajectory history to reduce jitter between parallel roads.
- Add continuity bonus for staying on the same edge.
- Apply switch threshold so edge transitions require meaningful improvement.

3. Post-processing smoothing
- Remove isolated one or two point spikes.
- Use local voting to stabilize short oscillations.

## Scoring Model
Core geometric score (lower is better):

`score = d_norm^2 * (1 - dir_weight) + dir_penalty * dir_weight`

Where:
- `d_norm`: normalized point-to-edge distance
- `dir_penalty`: directional disagreement term
- `dir_weight`: tradeoff between proximity and heading alignment

Temporal transition constraints penalize implausible transitions requiring unrealistic speed or abrupt changes.

## Key Parameters
- `max_dist` (default around 150 m): candidate search radius
- `dir_weight` (default around 0.35): heading importance
- `HISTORY_WINDOW`: number of previous points used for temporal memory
- `HISTORY_DECAY`: exponential decay applied to older history
- `HISTORY_BONUS`: continuity incentive for same-edge matches
- `SWITCH_THRESHOLD`: minimum relative gain needed to switch edge
- `SMOOTH_RADIUS`: neighborhood radius for post-process voting
- `SMOOTH_MIN_RUN`: minimum run length preserved from smoothing

## Operational Tuning
If route is unstable (zig-zag):
- Increase `SWITCH_THRESHOLD`
- Increase `HISTORY_BONUS`
- Increase `HISTORY_WINDOW`

If route is too sticky and misses turns:
- Decrease `SWITCH_THRESHOLD`
- Decrease `HISTORY_WINDOW`
- Decrease `HISTORY_BONUS`

If many unmatched points (`edge_id = -1`):
- Increase `max_dist` (for example, 150 to 200)
- Verify road network coverage and cache freshness

## Pipeline Integration
Map matching stage is executed before terrain feature extraction:

1. `batch_processor.py` generates processed route CSV files.
2. `map_matching.py` snaps points to road network.
3. `compute_route_terrain_features.py` enriches matched points.
4. ML and visualization pipelines consume the enriched data.

## Runtime Notes
- Initial network build can be slower; subsequent runs reuse cache.
- Performance is sensitive to search radius and route sampling density.
- For production runs, keep cache under `output/cached_networks` versioned by area/config.

## Validation Checklist
- Compare raw vs matched trajectory overlay in map viewer.
- Confirm continuity through roundabouts and junctions.
- Track unmatched rate and mean correction distance.
- Re-test on at least one noisy urban and one peri-urban route.
