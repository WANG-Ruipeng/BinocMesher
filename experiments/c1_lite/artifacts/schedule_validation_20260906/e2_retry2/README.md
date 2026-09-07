# E2: fixed-schedule structure and geometry images

Completed on 2026-09-06 with frozen production commit
`000028e0fa5bc4922aeb51aa9fb9ee96b799301e`. No production changes, RGB,
long video, scene regeneration or all-real-time theorem claim.

## What passed

`structural_coherence.json` binds the actual library, original source/cache,
binary64 time conversion, semantic SourceVID/owner relations, and six queries:
zero-based natural frames 13..17 plus the separately requested exact root 104/5.
All three affine source branches and four source singletons passed exact
relation/owner checks. Both window controls committed their five-query physical
batch and separate one-query exact transaction. Their only changed input is
the root anchor; endpoint anchors, source support and boundary are shared.

Only natural frame 15 is active. Its original exterior arrays remain bit-exact;
the four inactive neighbors remain entirely bit-exact relative to their own
raw baseline. Complete source retirement and actual patch geometry checks
passed at the requested active queries. A deliberately injected final-input
recheck failure returned the entire five-query baseline batch; the next batch
then reproduced its successful result. This negative control is not an
observed spontaneous algorithm failure.

This is **requested-schedule structural coherence**, not arbitrary-time
floating-point geometry, continuous-window admission, or E2/E3 group admission.

## Natural schedule and views

- Original camera schedule temporal hit: 1/24 = 4.1667% (frame 15).
- Selected natural frames modified: 1/5 = 20%.
- Original-camera visible support at frame 15: 0 pixels. All three original
  depth/normal/mask images agree. No naturally visible improvement is claimed.
- A fixed diagnostic camera, selected from the original boundary bbox before
  quality measurements, sees the same existing mesh. It is not a new natural
  camera, and camera-dependent LOD was not rebuilt.
- Root-aligned images are diagnostic and excluded from all natural-frame rates.

## Independent-reference error

The table uses a method-independent local footprint: the separately projected
two original source faces. Its 16,034 natural-frame pixels are also visible in
the full raw mesh (no footprint occlusion ambiguity in this case).
Depth is mean absolute camera-Z error in scene units; normal is mean angular
error in degrees against the original terrain gradient. Lower is better.

| Natural frame 15 | Depth MAE | Normal error |
| --- | ---: | ---: |
| ordinary raw | 0.1726222644 | 21.78937387 |
| centroid-window | 0.1683017260 | 21.76577193 |
| BEB1-anchor window | 0.1652722846 | 21.82798891 |

BEB1 depth error decreases by 4.25784% versus raw and 1.80001% versus centroid.
Normal error instead increases by 0.038615 degrees (0.17722%) versus raw and
0.062217 degrees (0.28585%) versus centroid. Both directions exceed twice the
measured reference-refinement sensitivity; that is not statistical significance
or a complete numerical error bound. Do not report all-metric superiority.

The larger, predeclared 40,420-pixel window-bbox ROI is also fully reported in
`result.json`, including unmodified surrounding geometry:

| Natural frame 15, bbox ROI | Depth MAE | Normal error |
| --- | ---: | ---: |
| ordinary raw | 0.1938013941 | 22.92635300 |
| centroid-window | 0.1920875021 | 22.91699047 |
| BEB1-anchor window | 0.1908857688 | 22.94167100 |

At the separate exact root, the original-source footprint is 16,284 pixels.
Depth MAE is raw 0.1746035988, centroid 0.1697454703, BEB1 0.1662197993:
4.80162% lower than raw and 2.07703% lower than centroid. Normal error is
21.98128977, 21.97579047 and 22.08094551 degrees, respectively, so it worsens
there too. The earlier quoted 10.55% height-error example was E0, not E2, and
is not the image metric measured here.

## Actual image changes and reference checks

At natural frame 15, the diagnostic depth and normal buffers each change on
16,034 pixels versus raw (6.9592% of the 640x360 frame). Under fixed display
normalization, 7,861 depth PNG pixels and 16,034 normal PNG pixels change.
Silhouette and pixel-boundary XOR are zero. These are numerical/code-value
changes, not a human-perceptual visibility study. Pixel-boundary agreement is
not itself a mesh-topology certificate.

The original procedural terrain reference uses 128 and 256 grid vertices per
axis. Their shared ROI has complete, identical coverage; mean depth change is
0.00007185525 and mean normal change is 0.00143165 degrees, passing the
predeclared 0.0001 / 0.1-degree refinement limits. No 512 refinement or camera
retuning was needed. Terrain-gradient steps 1e-4 and 5e-5 were also checked.
The reference mesh covers a finite local domain: its black exterior in the
overview is outside that reference domain, not a hole in the candidate mesh.

View the [original camera](original_natural_15_overview.png) and the
[diagnostic camera](diagnostic_natural_15_overview.png). These sheets are
half-resolution overviews; individual method PNGs retain 640x360 resolution.
`diagnostic_roi.png` and per-case `source_footprint.png` show the two metric
domains. Signed difference maps are explicitly amplified at a fixed 0.01
scene-unit scale and must not be presented as unamplified output.

## Measured cost and preservation

- Successful worker: 25.3869 s wall, peak process RSS 286,310,400 bytes.
- Six-query mesh/manifest preparation: 8.2504 s; source-relation/plan checking
  6.0879 s. This is additional experimental certification work, not raw slicing.
- Six ordinary slices: 0.04966 s total; six identity-instrumented slices:
  0.05585 s. Separate timings are noisy, not an isolated ledger microbenchmark.
- BEB1 five-natural-query transaction: 0.12771 s; separate exact-root
  transaction: 0.09310 s. Extra independent validators are reported separately.
- Core build and historical cache generation are reused and not timed here.
  Execution uses OMP=1 / OpenBLAS=1 in the existing WSL environment.
- Original cache and source are unchanged. Only the disposable cache copy's
  `log.txt` was appended; all its other files and source timestamps remained
  unchanged. The entire temporary copy was automatically removed.
- No meshes or full-precision image arrays are persisted. The worker produced
  1,421,012 bytes / 131 files before this explanatory report.

Sibling `e2/result.json` and `e2_retry1/result.json` preserve two early harness
stops (7.9169 s and 8.0423 s) caused by an overly strict temporary-log check.
They are not geometric rejection/admission observations. A real prepare-only
preflight passed before this final run. No camera, reference thresholds,
method anchors or frozen runtime logic changed between attempts.
