# Revised schedule / E2 images / Forest eligibility protocol

This protocol supersedes the unexecuted E2_RENDER_PROTOCOL_20260906.md after
the user's scope correction. Frozen commit/tag and all prior evidence stay
unchanged. No RGB, long video, new scene/cache build, environment installation,
commit or push. A continuous-real-time theorem is NOT a prerequisite, and a
requested-schedule result is NOT promoted into such a theorem.

## Deliverables

1. E2 schedule-level structural-coherence manifest: bind source/cache/library,
   actual time conversion/phase, five natural queries (zero-based 13..17) and
   a separately labelled exact root 104/5; stable semantic source/owner/boundary
   structures and anchor prescription; actual per-query identity mappings,
   complete source-face retirement, and same-query unchanged baseline exterior;
   declared transitions and atomic schedule fallback. Do not confuse actual
   index stability with semantic identity or per-query safety with full-time
   coherence. Same-root E3 is not silently admitted by an E2-only manifest.
2. E2 natural depth / world geometric normal / silhouette-boundary measurements.
   Primary methods raw, centroid-window, BEB1-anchor window; no RGB teaser.
   Only original natural frame 15 hits the event. Exact root is a separate
   diagnostic, excluded from natural-frame rates. Show original-camera results
   (E2 is out of view) and a separately labelled fixed diagnostic camera. Never
   call the diagnostic view a naturally visible improvement.
3. Every one of the 131 Forest canonical registry events receives an eligibility
   record. Separate PASS, geometric REJECT, numeric UNKNOWN, implementation
   UNSUPPORTED, NOT_REACHED and runtime NOT_ATTEMPTED. Registry saddle events
   are not automatically BEB1 events. Exhaust safe available classification,
   support/owner/interface, same-root and schedule checks without expanding
   bounded demo code into an untested general runtime or loading the full
   Forest cache into unconstrained Python objects.
4. Report denominators: all events, classified eligible events, eligible
   windows with a natural hit, actually modified natural frames, and visible
   support separately. Unknown/not-attempted is not a geometric rejection or a
   measured zero success rate. Report measured stage wall/CPU times, peak RSS,
   output bytes; historical cache build is separate and not free.

## E2 measurement controls

Reuse the camera construction in e2_render_views.py, fixed from the original
window boundary union before any method-quality result: original cameras
640x360, fx=fy=1000; diagnostic target=bbox midpoint, D=max XY extent,
eye=target+5D*normalize((1,-1.5,2.5)), world +Z up. Same existing meshes, no
camera-dependent LOD rebuilding. Render all faces without in_view filtering
or backface culling. Flat oriented geometric normals, camera-Z depth,
deterministic pixel-center rasterization and near=1e-6.

The common diagnostic ROI is the projected original window boundary bbox,
padded by four pixels. For the active natural frame and exact root, also use
the projection of the two original source faces as a fixed local footprint,
shared by every method. Do not hide missing coverage by intersection-only
scoring: record missing/extra/boundary pixels and both-valid denominators.

Independent reference is the original procedural terrain, sampled on the
window XY bbox padded by D/2. Compare 128/256 vertices per axis, with a single
512 refinement allowed if needed. Reference normal is the original height
gradient via central differences at reference world hits (steps1e-4/5e-5).
Reference refinement requires identical coverage inside the common ROI,
mean camera-Z difference <=1e-4 scene units, mean normal-angle difference
<=0.1 degrees. If 256/512 still fails, STOP_REFERENCE_UNRESOLVED; preserve
diagnostics, do not retune camera or thresholds. Report whether apparent gains
exceed twice the corresponding reference-refinement metric change.

Depth mean/P95/RMSE and normal mean/P95 errors use full-precision buffers;
preview PNG normalization is shared and does not determine metric values.
Image-change thresholds: depth1e-9 scene units, normal angle1e-5 degrees;
report counts separately from nonzero quantized display changes. No image
SSIM, statistical significance, perceptual visibility, or six-scene claim.
Unfavorable quality outcomes are data; safety/input/reference failures stop
their affected stage and remain labelled, not silently counted as acceptance.

## Resource and artifact controls

Fresh artifact root artifacts/schedule_validation_20260906. Keep E2/Forest
outputs separate so an unsupported Forest stage cannot masquerade as an E2
failure or vice versa. Persistent new output <=100 MiB (Forest<=20 MiB), E2
temporary cache/build work <=1 GiB; Forest reader target peak<=2 GiB. Entire
workspace plus WSL/backups stays below400 GB. No large arrays/meshes retained;
compact JSON/CSV and selected geometry PNGs only. Own E2 cache copies are
removed automatically; original Forest cache remains read-only. Stage budgets
20 minutes are cooperative checks, not hard process watchdog promises.

The Forest131 registry is from the limited 64-frame pre-displacement pilot,
not completed post-displacement geometry and not six scenes. Input-stage
eligibility must retain that scope, even if all evaluated gates pass.
