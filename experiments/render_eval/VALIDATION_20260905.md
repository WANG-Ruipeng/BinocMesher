# Forest render repair validation - 2026-09-05

## Scope and fixes

- Both OcMesher wrappers pass the spatial four-tuple, excluding Binoc timestamps.
- Binoc beauty and GT tasks use one frame per task. Multi-frame tasks fail before slicing.
- Runner reports FINISHED only after every selected method and required SSIM pair succeeds.
- SSIM schema v2 converts this fork's Blender flow to forward image displacement.
- Completed geometry can be copied into a new run; old images and scores are not overwritten.

These runs validate the baseline rendering pipeline, not BEB1/SSP1 coverage or
the full six-scene, five-method paper experiment.

## Completed checks

WSL paths below are under /home/warpwang/runs/.

| Check | Result |
| --- | --- |
| test_render_fixes.py | 5 tests passed: native 24/96-camera adapters, required flows, complete pairs, failure status/output protection, synthetic flow direction |
| render-fix-oc-smoke-20260905 | OcMesher-96 and OcMesher-24 each produced 2 images, 2 flows, 1 SSIM pair; both manager/scorer return codes 0 |
| render-fix-binoc-smoke-20260905 | Cached Binoc frames 25/26 produced 2 images, 2 flows, 1 pair; manager/scorer return codes 0 |
| Binoc multi-frame guard | Invalid 25-26 render rejected before slicing |
| Binoc frame audit | Beauty and GT each sliced frame 25 at 1.0208333333333333 s and frame 26 at 1.0625 s |
| Binoc flow geometry, frame 25 | Median endpoint error 0.002468 px across 469342 pixels |
| Spherical flow geometry, frame 25 | Median endpoint error 0.002458 px across 469030 pixels |

Two-frame Oc tests use ppc=30 and 8 samples: execution checks only.
Cached Binoc smoke uses ppc=6 and 8 samples: do not use its noisy images for quality claims.
Geometry checks use static-scene depth and saved camera K/T, with a 0.05 px median-error threshold.

## Existing Spherical-8 rescored

Input: forest96-four-baseline-pilot-20260905/spherical8/0
Output: forest96-recovery-v2-20260905/spherical8/ssim

- Frames 1-96, 95 pairs, schema binoc-warped-ssim-v2.
- Full-image warped SSIM mean: 0.9064150033208461.
- Valid-support warped SSIM mean: 0.916932342697744.
- Valid support excludes warp boundaries, not occlusions.
- Earlier v1 scores used the wrong flow x sign and should not be compared with v2.

## Recovery run launched

Output: /home/warpwang/runs/forest96-recovery-v2-20260905
Started: 2026-09-05 10:37:57 +08:00
Order: OcMesher-96, OcMesher-24, then Binoc.
Settings: Forest seed 0, 96 frames, 960x540, ppc=6, 3072 samples, OMP8.
Binoc cam_block_size=1; one active pipeline task at a time.

The launch reused completed coarse/populate scenes and will reuse Binoc's 4D
cache from the original run. It regenerates all three methods' images and GT.
At this checkpoint OcMesher-96 was actively building its mesh, not yet finished.
Use status.tsv, per-method logs, and FINISHED/FAILED to determine the later result.
Spherical is rescored separately and is not a row in this three-method queue.

This is a pilot, not a final paper-quality result. The old Binoc renders froze
slices for 24-frame blocks and are not a valid temporal baseline.
