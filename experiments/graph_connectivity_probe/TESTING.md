# Pre-campaign verification

2026-09-08: 12 tests PASS (10.718 s unittest runtime), existing WSL Miniforge
Python environment, no packages installed. This includes numerical preflight
bridges on two fixed prior cases (indices 3 and 15), not selected by results.
No preflight artifacts or parameter tuning were used to choose campaign inputs.

- Six connectivity tests: edge and face insertion, fixed boundary, immutable
  vertices, true Delaunay flips, cocircular tie handling, refused invalid
  insertions and enumeration cap.
- Three independent enumeration tests: polygon Catalan counts and independent
  interior-neighbor/pocket enumeration match the flip orbit for every one of
  the 12 XY configurations used by this campaign; square center is singleton.
- Three experiment tests: real query counts, common XYZ across all flips,
  square aliases, normal/distance screening and frozen metric bridges.

Observed finite enumeration counts (development validity, not error outcomes):
square centers and edge midpoints: 1; square face barycenters: 2;
hexagon centers and face barycenters: 36; hexagon edge midpoints: 30.
All are below the registered 256-state cap.

Independent preflight review found no blocking implementation issue. It
identified the need to distinguish normals at distance-selected oracle states
from normal-optimal/fixed-mesh results. The campaign only screens distance for
oracle comparisons; normal values remain labeled diagnostics for those states.

```bash
/home/warpwang/miniforge3/envs/binoc-exp/bin/python -m unittest discover \
  -s /mnt/e/BinocMesher/experiments/graph_connectivity_probe -p 'test_*.py' -v
```
