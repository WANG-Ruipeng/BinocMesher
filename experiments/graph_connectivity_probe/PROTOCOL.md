# Connectivity-controlled local geometry probe

Registered before numeric campaign execution on 2026-09-08. User request:
continue the graph-lift probe and test whether connectivity explains its gains.

## Scope and budgets

Only a stage-1 connectivity attribution extension is authorized. Reuse the
24 frozen patches (12 model instances on two domains) from
../graph_lift_probe/artifacts/run_20260907_130141/inputs.json. No new seeds,
target surfaces, production meshes, rendering, time sequences, WMTK,
publication, or edits to previous evidence/production code.

Campaign limits: 1800 s wall, 4 GiB worker address space, 100 MiB total new
directory, repository logical size <400,000,000,000 bytes. OMP/BLAS one thread.
At most 256 triangulations per fixed vertex set. Development unit tests and
independent completeness checks precede the campaign. Unexpected exceptions,
input changes, invalid mesh/invariant failures or resource/state-cap violations
stop the campaign, preserve a STOP and do not auto-repair/restart. Geometric
regressions are measured outcomes, not errors to tune away.

## Fixed points, multiple connectivity rules

All arms keep the same original boundary, one interior vertex and F=m faces.
Seven fixed point rules, determined without evaluation samples:

1. XYZ mean of boundary vertices (zero interior model queries).
2. Parameter boundary-vertex mean lifted to the model (one query).
3. Parameter polygon-area centroid lifted to the model (one query).
4-5. Largest-parameter-area face barycenter of original root-0/root-1
     boundary triangulation, lifted once; ties use existing face order.
6-7. Longest-parameter-length interior-edge midpoint of root-0/root-1
     triangulation, lifted once; ties use lexicographic edge IDs.

For each EXACT fixed XYZ vertex array measure:
- Initial rule: center fan for 1-3; 1-to-3 face insertion for 4-5;
  2-to-4 interior-edge insertion for 6-7.
- Universal fan: connect that same interior vertex to every boundary edge.
- Common deterministic XY Delaunay rule, using legal interior flips only;
  geometric predicates and ties depend only on XY and vertex IDs, not z,
  model residuals or reference samples. Boundary and vertices are immutable.

Initial vs fan/Delaunay at fixed XYZ isolates a connection change. Comparing
different points with the common FAN has the same combinatorial edge layout;
comparing them with the common DELAUNAY uses the same rule but may yield
different layouts. Neither says geometry approximation is Delaunay-optimal.
Every method pays its genuine point-model queries; flips cost zero additional
model queries but have measured CPU time and operation counts. Boundary setup
is m shared model queries per patch; report sharing rather than charging it
again secretly. Score/evaluation samples never select a deployable arm.

The square's lifted diagonal midpoint and lifted parameter centers coincide;
assert identical geometry and four faces up to ordering. Treat these as aliases,
not independent algorithm wins. XYZ and lifted-vertex-mean have identical XY,
so assert that their common Delaunay face sets also coincide.

## Fixed-vertex connectivity envelope (evaluation-only oracle)

Enumerate the legal flip orbit for each point set with a hard 256-state cap.
Independently validate the used parameter point configurations in development:
enumerate the interior vertex's boundary-neighbor subsets and triangulate each
remaining convex boundary pocket. Compare both enumerations' canonical face
sets. This is a finite floating-point audit on the selected configurations,
not an exact-arithmetic theorem for arbitrary degeneracies.

Measure every enumerated connectivity with the frozen surface metrics; record
state IDs, all three levels, and per-level minimum/maximum. Best-of-evaluation
is explicitly an ORACLE DIAGNOSTIC, not a one-query construction algorithm.
Its geometry evaluation cost and state count are separate from deployable
arm construction. Compare oracle envelopes only as conditional fixed-point
diagnostics; do not present them as new production baselines or claim new GL.

## Metrics and validity

Import (do not modify) the prior graph model, common-parameter surface
quadrature, actual point-to-triangle distance and area-weighted metrics.
Reuse levels (reference, quadrature)=(16,12),(32,12),(32,24).
Import prior source parameters rather than regenerating an altered distribution.
Reproduce original center-fan and root-0 face-barycenter metrics before
interpreting the extension (distance <=1e-10 scale, normal <=1e-8 degrees).

Preserve exact input XYZ before/after connectivity functions; check disk
Euler, m boundary edges, all m+1 vertices used, F=m, positive nondegenerate
parameter triangles and matching area. Shared convex XY domain implies graph
validity here; no spatial rank-3 condition or old production certificate is used.
Planar controls and alias distance equality must pass. Face ordering at a
crease can change a one-sided normal at zero-area edge samples; do not use
such ambiguity as a geometry gain. Use canonical oriented face ordering for
this campaign. The frozen bridge separately measures original face ordering
on the original quadrature, and compares its means within the stated tolerances;
it is not compared against the newly canonical ordering at crease samples.

Primary metric: mean of the two directed area-weighted mean distances
(length, not squared). Reference proxy approximation and quadrature changes
are screened empirically: tolerance = twice the sum of both arms' two
refinement deltas +1e-10*max(1,boundary bbox diagonal). Near-zero baseline
(<=1e-9*scale) has no percentage. Report wins/ties/losses/unresolved plus
absolute and relative distances. This is not an error bound or significance
test. Also screen normal mean with the analogous empirical rule, absolute
floor 1e-8 degrees; keep P95/sample-max levels as diagnostics, not exact
Hausdorff claims. Report all cases, domains and geometry families separately.

## Planned comparisons and interpretation

- Old center-area fan vs root-0 face insertion (bridge to prior 62.841%).
- Fixed face-barycenter: fan/XY Delaunay vs initial insertion (pure connectivity).
- Center-area vs face-barycenter under common fan, and under common Delaunay.
- Center-area vs edge-midpoint under common fan and common Delaunay.
- Vertex-mean vs area-centroid under common Delaunay.
- Lifted vertex-mean vs XYZ mean under the identical Delaunay connectivity.
- Repeat placement comparisons for root-1 ordinary candidates.
- Fixed-point oracle envelopes for center, face and edge choices; no algorithmic
  speed/accuracy claim from choosing a state with evaluation data. Best-state
selection is separately performed at each resolution; normals stored at the
distance-selected state are not a normal-optimal envelope or fixed-mesh
convergence sequence. Only distance is screened for oracle comparisons.

If ordinary edge insertion/reconnection matches the center construction,
record equivalence. If a point difference remains under common connectivity,
it is a conditional placement effect on these inputs, not evidence of an
original GL strategy. If Delaunay worsens error, preserve that result; do not
tune it into an error-driven optimizer in this campaign. Stop after the report.
