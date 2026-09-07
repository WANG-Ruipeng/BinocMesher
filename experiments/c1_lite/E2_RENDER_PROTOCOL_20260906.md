# Fixed E2 minimal natural-schedule render protocol — 2026-09-06

User authorized this small experiment after freeze `000028e` / tag
`e2-runtime-freeze-20260906`. No modification to the frozen runtime, original
source/cache, window, anchors or prior evidence. No long video, new scene,
environment installation, GPU path tracing, commit or push in this experiment.

## Fixed temporal and method design

- Natural zero-based frames 13, 14, 15, 16, 17 from the original camera clock
  `(i+0.5)/24`; ordinary physical input is the actual binary64 clock minus the
  original minimum camera time, exactly as core.py. Only frame 15 hits E2.
- Separate exact-rational diagnostic at tau=104/5. It is not an extra natural
  sample and is excluded from natural-frame averages/counts.
- Methods: ordinary raw; extra_smooth; actual legacy C0 root-only SSP1;
  centroid-window; BEB1-anchor window. For centroid keep both original endpoint
  anchors and replace only the root anchor with the arithmetic mean of the
  four original root boundary positions, with the same exact-then-RNE32 center
  evaluation and actual-geometry acceptance checks as BEB1.
- Each window method first validates its complete five-query physical batch,
  then its separate exact diagnostic. No images are produced until every mesh
  needed by this experiment has passed its applicable runtime checks. A refusal
  is STOP, not a silently relabelled successful intervention.
- Native actual arrays are rendered. C0 is genuinely executed through SSP1 at
  root, not reconstructed by copying BEB1. Other C0 frames equal ordinary raw.
- One OMP 1 execution only; existing OMP 1/8 evidence is not repeated. All five
  methods render all faces, without in_view-tag filtering. The C0 center tag
  and the window center tag can differ; this is recorded, not hidden.

## Cameras fixed before method-quality evaluation

Original view: the original 640x360 cameras, fx=fy=1000; exact-root pose is the
linear interpolation along that same original translation-only trajectory.
E2 is outside this frustum. Preserve that zero-visible-support outcome.

Diagnostic view: compute the 3D bounding box of all four original boundary
vertices at every original source breakpoint. Let target be its midpoint and
D be its largest XY extent. Eye = target + 5D normalize((1,-1.5,2.5)). Look at
target with world +Z as up, OpenCV right/down/forward camera convention and the
same 640x360, fx=fy=1000, principal point (320,180). Keep this camera fixed for
all methods/times. No view-dependent mesh rebuilding: this is a controlled
observation of the existing mesh, not an original-camera performance claim.

The common diagnostic ROI is the projected bounding rectangle of the frozen
window boundary union, padded by four pixels. For the natural hit and root,
also report the ordinary source-pair footprint as a method-independent local
ROI. Do not choose a camera/ROI after observing a favorable treatment result.

## Rasterization and independent reference

Deterministic CPU pinhole rasterization at pixel centers; near clipping 1e-6,
perspective-correct camera-Z depth, original oriented flat world-space face
normals, no backface culling. Degenerate triangles may have zero raster coverage;
they are counted, never deleted from the input mesh or its correctness audit.

Independent static terrain: use the original analytic procedural height
function, not any candidate mesh. Reference XY domain is the original window
XY bounding box padded by D/2 on each side. Mesh resolutions 128 and 256
vertices per axis; allow 512 only if their refinement check fails. Reference
normals use central finite differences of the original height function at
unprojected reference hits, with steps 1e-4 and 5e-5 as a consistency check.

Reference checks within the fixed ROI: no coverage discrepancy, mean depth
difference <=1e-4 scene units, mean reference-normal difference <=0.1 degrees;
if needed compare 256/512 under the same conditions. A failed final check is
STOP_REFERENCE_UNRESOLVED. Separately compare an apparent method gain against
twice the reference-refinement change in that metric before calling its sign
resolved; a small or negative gain is data, not a reason to retune the setup.

Report coverage/missing/extra counts with explicit denominators, depth mean /
P95 / RMSE and normal-angle mean / P95 against reference. Report absolute
pixel differences versus raw and centroid, not just relative percentages.
Silhouette/boundary changes are a coverage diagnostic; unchanged patch boundary
does not need to produce a nonzero improvement. Wireframe subdivision is not
a silhouette improvement.

RGB is only a cheap flat-shaded Lambertian teaser: fixed world light
normalize((-0.4,-0.6,1)), ambient .25, diffuse .75, albedo (.65,.70,.60), black
background, no textures/shadows/AO and a common display transfer. Show ordinary
unamplified RGB separately from labelled amplified differences; a colored
heatmap alone is not evidence of unaided visible improvement. No warped SSIM
or statistical/general-scene superiority claim from these six time points.

## Bounds, safety and archival

New artifact root: `artifacts/e2_render_20260906`, fresh directory only. Limit
new persistent output to 200 MiB, live temporary/cache/build work to 1 GiB, and
the full local project including WSL disk/backups to the user's 400 GB ceiling.
Only the approximately 4.26 MB fixed cache is copied to an owned temporary
directory, removed automatically. No new mesh cache or .blend files are kept.
Store compact JSON, preview PNGs and compressed active diagnostic buffers only.
The driver checks its 20-minute budget between stages (cooperative, not a hard
watchdog). STOP and partial diagnostics remain explicitly labelled if a stage
fails. Original input hashes are checked before/after; safety errors stop,
unfavorable scientific results are retained.

Historical effect sizes must not be conflated: root 10.55% was E0, not E2.
For E2 the prior ideal grid128 root-height reductions were about 4.06% vs raw
and 1.75% vs centroid; window-integral reductions were 2.11% and 0.93%.
This new experiment measures actual output images, not those old percentages.
