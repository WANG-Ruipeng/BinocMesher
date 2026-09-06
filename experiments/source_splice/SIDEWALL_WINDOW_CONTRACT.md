# Side-wall window contract repair — 2026-09-05

Status: the unsafe inference from root/three-level admission to continuous-window
readiness is now blocked. **The continuous replacement geometry is not repaired.**
Rendering and production whole-mesh window execution remain stopped.

## Root cause

For one slab let the ordinary edge be `[a(s), b(s)]`, with affine endpoint
trajectories and `s` in `[0,1]`. Put `e0=b(0)-a(0)` and `e1=b(1)-a(1)`.
Slicing the planar triangle spanning `a(0), b(0), b(1)` introduces the point
`q(s)=a(s)+s*e1` on its spacetime diagonal. The ordinary edge direction is
`e(s)=(1-s)*e0+s*e1`, so

```
(q(s)-a(s)) x e(s) = s*(1-s) * (e1 x e0).
```

If `e0 x e1 != 0`, the intermediate diagonal point is off the ordinary edge.
Changing the triangulation diagonal does not remove this obstruction. An
affine trajectory for every vertex is insufficient to make the swept edge a
planar spacetime quadrilateral. For parallel endpoint edges, positive dot
product additionally excludes edge collapse/reversal inside the slab.

The original witness remains saved at
`artifacts/geometry_gate_20260905/window_boundary.json`: at internal time
`117/11`, the boundary distance was `0.02370484342888187`. This is not a
roundoff-sized error and has **not** been reduced to zero by this repair.

## Changes

- `sidewall_contract.py` performs exact rational collinearity and noncollapse
  checks on the serialized binary64 endpoints of all eight edge/slab pairs.
  It uses no tolerance and never welds or moves geometry. Exact predicates on
  rounded endpoints can conservatively reject nearly parallel sides; the
  primary witness above is a macroscopic mismatch, not such a near-zero case.
- The compiler now labels its old mapping-cylinder PASS as three-level root
  support, records the separate side-wall audit, and explicitly sets
  `continuous_window_ready=false`. Existing `whole_mesh_splice_ready` and
  `mapping_cylinder_ready` retain **root-only** meaning for compatibility.
- `--require-continuous-window` exits 4, writes a rejected IR and never emits
  an SSP1 plan. No continuous-window route is admitted, even if the necessary
  side-wall test passes: source interval, embedding, and runtime contracts
  would still require certification.
- The preflight recomputes the side-wall predicate, including for old IRs;
  absent or forged positive audit fields cannot authorize a window. A rebuilt
  IR is compared by event identity and unchanged measured root geometry, not
  by its output directory name.

## Verification

Results: `artifacts/sidewall_guard_fix_20260905/verification.json`.

- Seven new predicate tests passed, including translation/stretch positives,
  rotating/collapsing negatives, tiny nonzero rotations, legacy/forged PASS
  rejection, and invalid time layout.
- Existing closure-combinatorics regression, three boundary tests and four
  height-field gate tests passed.
- The first demo event was recompiled from the same cache/theory. The emitted
  root SSP1 is byte-identical to the old plan, SHA-256
  `df0beda686136630ad2615d5e1f3122e12ebbf99a11b3287e69576d1ec42309e`.
- Explicit window compilation was rejected without creating a plan. Both
  old and rebuilt IRs were rejected by preflight before numerical probes,
  runtime plan emission, or rendering.
- No new OMP runtime campaign was run; the unchanged root plan and unchanged
  production core preserve the inputs to the previously saved OMP checks.

Run the focused tests in the existing environment:

```bash
python experiments/source_splice/test_sidewall_contract.py
python experiments/source_splice/test_critical_beb1_event_ir.py
python experiments/source_splice/test_window_boundary.py
python experiments/source_splice/test_geometry_gate.py
```

## Remaining decision

With the ordinary outside mesh fixed, the current nonplanar swept boundary
cannot be exactly represented by this finite planar-triangle side wall.
A genuinely continuous solution requires either a changed parameterized
boundary/interior representation, or a redesigned patch/interface that is
compatible with linear 4D pieces. Neither has been implemented here. A
parameterized time-local construction would change the pure linear-4D
tetrahedral slicing model; it must not be silently substituted and reported
as the original method.
