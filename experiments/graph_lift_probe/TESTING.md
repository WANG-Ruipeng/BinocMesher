# Pre-campaign development verification

2026-09-07: all 15 unit/integration tests passed under existing WSL
Miniforge environment `/home/warpwang/miniforge3/envs/binoc-exp`.
Final prelaunch run: 15 tests, 0.287 s reported by unittest, OK.

Coverage: analytic triangle distances; independent scalar seven-region
brute-force nearest-face comparison; a long-triangle centroid-pruning
counterexample; immutable metric inputs; deterministic face ties; area-weighted
statistics; same-PL subdivision invariance; graph derivatives; rank-deficient
spatial source with valid slice; invalid active-domain rejection; asymmetric
centroid distinction; mesh budgets/topology; complete analytic-fixture metric
integration; near-zero percentage suppression and unresolved comparisons.

Independent preflight review checked model construction, protocol, query
accounting and runner safeguards. Before launch, input snapshots were moved
before construction and expanded to include the polygon; the scale diagnostic
reuses existing boundary vertices; greedy scoring queries are explicitly a
subset of total interior model queries. Neither development tests nor this
review constitutes historical GL replication or production certification.

Command:

```bash
/home/warpwang/miniforge3/envs/binoc-exp/bin/python -m unittest discover \
  -s /mnt/e/BinocMesher/experiments/graph_lift_probe -p 'test_*.py' -v
```
