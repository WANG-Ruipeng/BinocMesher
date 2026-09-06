# Fail-fast geometry/window pilot — 2026-09-05

Outcome: **stage 1 passed; stage 2 stopped at the first failed boundary probe;
stage 3 was not started.** No replacement plans were emitted, no production
code or cached mesh was modified, and no renders were launched.

## Fixed experiment and independent reference

The selected input was index 0 (`event-00-6d2d6dc2cdd7`) from the existing
four-canonical-event demo campaign, chosen before measuring quality. Its exact
root is `120/11` in the cache's internal time units, not seconds. Inputs were
saved baseline and critical-patch meshes; this was not a newly built scene.

The independent reference is the original static height field used to generate
the demo: `terrain(x,y,z) = z - 5*Noise2(x/10,y/10,octaves=4)`. This function
does not depend on the BEB1 compiler or its proposed replacement geometry.
The WSL and working-copy profile scripts were compared and agree after CRLF
normalization. Source-file and mesh hashes are recorded in `result.json`.

Both patches were evaluated at identical XY midpoint-grid locations with equal
sampled coverage. All saved whole-mesh safety checks and OMP 1/8 array equality
were required first. Three fixed grid sizes (32, 64, 128) were tested. At each,
mean absolute vertical error must decrease beyond the numerical tolerance and
maximum sampled error must not increase. The gate is fail-fast, not a search
over events or thresholds.

| Grid | Common samples | Baseline mean height error | Replacement mean height error | Relative reduction |
| --- | ---: | ---: | ---: | ---: |
| 32 | 460 | 0.212476 | 0.190010 | 10.5733% |
| 64 | 1,834 | 0.213601 | 0.191050 | 10.5577% |
| 128 | 7,339 | 0.213472 | 0.190940 | 10.5551% |

At grid 128, maximum sampled vertical error decreased from 0.455862 to
0.436467. These are scene-coordinate units, not pixels or necessarily meters.
This is a local demo root-slice fidelity result, not Hausdorff distance,
SSIM improvement, paper-scene coverage, or a topology-repair result.

## Stage 2 counterexample

The preflight slices the *existing admitted triangulated mapping-cylinder side
wall* halfway between its lower and root levels. At exact internal time
`117/11`, a side-wall intersection point is not on the corresponding ordinary
source-labeled patch boundary segment:

- Side-wall triangle: `[0, 1, 6]`.
- Intersection point: `[0.69580078125, 0.10986328125, 0.42724609375]`.
- Boundary SourceVIDs: `133:4|139:2` and `179:4|302:0`.
- Distance to the ordinary boundary segment: `0.02370484342888187`.
- Numerical screening tolerance: `2.015070214538707e-7`.

The full segment endpoints and input hashes are in `window_boundary.json`.
The ordinary patch was independently extracted from the original processed
cache at this exact time. No approximate welding or vertex movement was used.
The tolerance only distinguishes clear incompatibility from floating-point
roundoff; passing these finite probes would not certify an entire interval.

Consequently, the recorded root admission and three-level checks do not by
themselves establish that the current side-wall construction can be glued to
the ordinary mesh at intermediate times. This is a **window-extension blocker**,
not evidence that the already-tested root-only SSP1 run introduced a crack.
Execution stopped after the lower-slab midpoint failure; the upper slab,
whole-mesh window execution, and rendering were not attempted.

## Reproduction

Run from the repository root in WSL using the existing Miniforge environment.
The outputs below must be new paths. Do not overwrite the recorded results.

```bash
PY=/home/warpwang/miniforge3/envs/binoc-exp/bin/python
CAMPAIGN=/home/warpwang/binoc-runs/full-ubuntu26-smoke/all_canonical_beb1
EVENT="$CAMPAIGN/event-00-6d2d6dc2cdd7"
CACHE=/home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache
PROFILE=/home/warpwang/src/BinocMesher/experiments/tv0_tv4/run_lightweight_profile.py

"$PY" experiments/source_splice/test_geometry_gate.py
"$PY" experiments/source_splice/test_window_boundary.py

"$PY" experiments/source_splice/run_geometry_gate.py \
  --event-root "$EVENT" --profile-source "$PROFILE" \
  --output /tmp/NEW_geometry_gate

# Run only if stage 1 returns 0. The recorded stage 2 run returns 2 (STOP).
"$PY" experiments/source_splice/check_window_boundary.py \
  --event-ir "$EVENT/critical_beb1_event_ir.json" --cache-root "$CACHE" \
  --geometry-result /tmp/NEW_geometry_gate/result.json \
  --output /tmp/NEW_geometry_gate/window_boundary.json
```

Four geometry-gate tests and three boundary-preflight tests passed. The scripts
contain no renderer or automatic repair path. Further work requires repairing
and admitting the intermediate-time boundary construction before resuming
stages 2 and 3; no such repair was made during this stopped experiment.
