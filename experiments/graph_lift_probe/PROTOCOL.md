# Graph-lift identity and geometric attribution probe

Registered before numeric experiment execution on 2026-09-07.
User specification: attachment 557a24d3-b695-41d2-ad08-98f7ff79648b.

## Authorized scope and limits

Only stage 0 (historical algorithm identity) and stage 1 (local geometric
attribution) are in scope. No independent-target campaign, time sequence,
rendering, scene generation, production integration, WMTK installation,
certificate framework, commit or push. Existing evidence is not modified.

The numeric campaign has a 1,800-second wall limit, 4 GiB address-space
limit, and 100 MiB combined new-directory storage limit. Repository logical
size must remain below 400,000,000,000 bytes. BLAS and OMP use one thread.
Development unit tests precede the registered campaign. An unexpected
invariant failure, nonfinite value, input mutation, or resource limit stops
the campaign with a preserved STOP receipt; do not silently repair/relaunch
the same campaign. Predetermined unsupported-domain negative controls are
not unexpected failures. No full-mesh dumps or disposable render caches.

## Stage 0

Search bounded local and git history locations for original GL implementation,
parameter-center rule, case generator and per-case data. A report/formula alone
does not establish whether the original used parameter vertex mean or area
centroid. If unavailable, identify this probe as a NEW RECONSTRUCTION OF SIMPLE
MODEL SAMPLING, not replication of the historical 120 cases or 91.72% figure.
Do not create an `ours` row or an extra old-GL strategy without evidence.

## Stage 1 input contract

Use a bilinear-corner spacetime completion, queried at tau=0:
G_tau(q) = (u,v,zL(q)+(tau-L(q))/(U(q)-L(q))*(zU(q)-zL(q))).
Admit only inputs whose containing unit-square corners satisfy L<tau<U.
Bilinearity then gives full-square D>0 and active-domain containment.
The XY identity makes the 2D slice a valid graph even when the spatial 3D
source map is rank deficient. Do not impose a spatial rank-3 condition.

Two fixed parameter domains are used: the entire unit square (m=4), and
an asymmetric convex hexagonal CROPPED ROI inside that square (m=6).
The ROI is not a recovered native six-sided cell. All methods share the
same straight mesh boundary, disk topology and analytic model on the same
parameter domain. The analytic boundary can be curved in XYZ: its common
straight-boundary approximation error is reported, not hidden by changing
the reference to a method's own surface.

Use 24 deterministic synthetic cases: two domains x four model families
(affine/planar, bilinear, variable-denominator rational, stronger rational)
x three predeclared seeds. These are new controlled cases, not natural
scene observations or the old random-embedding distribution. All cases,
including zeros and regressions, are retained. No point-selection tuning
using evaluation samples or independent target queries.

The seeds are 1103, 2207, 3301, reused across the two domains for paired
domain diagnostics. The 24 cases are therefore 12 source models on two
domains, not 24 independent random source models. All queries use tau=0.

## Methods and budget accounting

- Boundary triangulation rooted at polygon vertex 0; also rooted at vertex 1
  to expose diagonal dependence. Both have m-2 faces.
- Geometry-preserving PL subdivision: insert existing PL triangle centroids,
  split each chosen triangle into three, preserve the geometric point set.
- XYZ boundary-vertex mean fan, m faces, no interior model evaluation.
- Graph lift of parameter vertex mean, m faces, one model point evaluation.
- Graph lift of parameter polygon area centroid, same face/evaluation budget.
- Uniform model refinement: repeatedly choose the largest parameter-area
  triangle and insert its model-lifted barycenter.
- Greedy model refinement: score current face barycenters by model-vs-PL
  height residual times parameter area; insert the largest-scoring one.
  Charge every new scoring query and reuse stored unchanged-face scores.

Refinement budgets k=1,2,4 internal vertices; F=m+2k-2. Start both refiners
from the same root-0 triangulation; no boundary subdivision. All actual
model calls (shared boundary setup and method-local calls), candidate
evaluations, faces, vertices and construction times are recorded. No dummy
faces. Greedy and center methods do not have equal evaluation counts:
compare their measured budgets, never label a same-face comparison equal-cost.
Original-target construction queries are zero; stage 1 has no such target.

## Metrics and guards

Use common deterministic parameter-domain quadrature, lifted onto each
actual PL mesh or the analytic graph and weighted by each surface's area
Jacobian. Compute point-to-triangle distance, not vertex-cloud distance.
Use a refined analytic-graph triangle mesh for the mesh-to-model direction;
report its discretization convergence separately. Use normals of triangle
surfaces and analytic graph derivatives. Report both directional and
symmetric means, area-weighted P95 and sampled maxima in absolute units;
sampled maxima and P95 are not exact Hausdorff bounds. Relative percentages
are undefined for near-zero baselines and never pooled as huge gains.

Check pure-subdivision equality on identical physical sample locations and
weights, planar zero error, common mesh boundary, disk Euler relation,
positive oriented parameter triangle area and coverage, finite outputs,
active-domain validity, and no evaluation-query leakage into construction.
Use increasing reference and quadrature resolutions. If convergence is
insufficient to resolve a comparison, label it UNRESOLVED rather than PASS
or an algorithm win. Retain every per-case result and failure.

The registered (reference subdivision, quadrature subdivision) levels are
(16,12), (32,12), (32,24). Parameter triangle centroids supply quadrature;
uniform reference triangles sample the exact graph. Compare paired mean
distance reductions to twice the sum of each arm's reference-refinement
and quadrature-refinement changes, plus 1e-10 times max(1,boundary bbox
diagonal). This is an EMPIRICAL resolution screen, not a certified bound
or statistical confidence interval. Differences below the numeric term are
ties; remaining signs that do not clear this screen are UNRESOLVED.
Near-zero baseline means (<=1e-9 times the same scale) have no relative
percentage. Other metrics retain all three levels without inheriting the
mean-distance classification. Normals use same-parameter correspondence,
not nearest-point correspondence. This is a local graph metric, not a
production rendering metric. Boundary same-parameter chord gaps are an
evaluation-only diagnostic, not a certified lower bound on mesh error.

## Interpretation and stopping point

Separate density/metric effects, boundary fan geometry changes, internal
model information, center definitions and ordinary refinement. Historical
GL identity remains UNKNOWN without its original implementation. No result
here proves original-scene accuracy, temporal behavior, production safety,
WMTK superiority, or novelty relative to published isosurface methods.
Stop after the stage-1 report; stages 2 and 3 require separate direction.
