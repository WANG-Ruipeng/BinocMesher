# Forest rendering pilot

Run from WSL using the Miniforge `binoc-exp` environment. Defaults:
Forest seed 0, frames 1-96, 960x540, ppc=6, 3072 beauty samples,
OMP8, one active pipeline task. These are pilot settings.

The local Infinigen compatibility changes live in its WSL checkout. They are
also preserved in `infinigen_compat.patch` so another checkout can reproduce them:

```bash
cd /home/warpwang/src/BinocMesher/infinigen_binocmesher
git apply --recount --check ../experiments/render_eval/infinigen_compat.patch
git apply --recount ../experiments/render_eval/infinigen_compat.patch
```

Do not apply the patch twice. The changes adapt both OcMesher wrappers to the
four-component spatial camera tuple and reject multi-frame Binoc render tasks.
Binoc must slice independently for each frame; `cam_block_size=1` is required
for both beauty and GT. Keeping the 4D build in the global stage reuses its cache.

## Fresh batch

```bash
bash experiments/render_eval/run_forest96_pilot.sh /home/warpwang/runs/NEW_RUN
```

Optional trailing arguments select methods: `binoc ocmesher96 ocmesher24 spherical8`.
The output directory must not already exist. Existing results are never overwritten.
Only all-success batches receive `FINISHED`; failed batches receive `FAILED`
and exit nonzero. Inspect `status.tsv` for individual method return codes.

The older 20260905 overnight runner wrote FINISHED even after failures. Its Binoc
images used 24-frame frozen slices and must not be used as the correct baseline.

## Recover geometry, regenerate images

```bash
PILOT_REUSE_ROOT=/home/warpwang/runs/forest96-four-baseline-pilot-20260905 \
  bash experiments/render_eval/run_forest96_pilot.sh \
  /home/warpwang/runs/NEW_RECOVERY ocmesher96 ocmesher24 binoc
```

The preparation step copies completed coarse scenes and, for Binoc, the completed
fine scene and 4D cache. It validates the source frame range, resolution and Binoc
ppc. Rendered frames, render completion markers and previous scores are not reused.
The existing Spherical result remains in the original run.

For a cached Binoc verification at frames 25 and 26, additionally set
`PILOT_RENDER_START=25 PILOT_RENDER_END=26 PILOT_SAMPLES=8` and select only `binoc`.
For a tiny fresh OcMesher test use `PILOT_FRAMES=2 PILOT_PPC=30 PILOT_SAMPLES=8`.
Those two-frame tests verify execution; they do not measure 24/96-frame block behavior.
Constructor regression tests separately exercise both adapters with 24/96 cameras.

## Checks and scoring

```bash
python experiments/render_eval/test_render_fixes.py
python experiments/render_eval/verify_binoc_guard.py CACHED_BINOC_TEST_ROOT
python experiments/render_eval/verify_binoc_guard.py CACHED_BINOC_TEST_ROOT --audit-only
```

The guard check uses a deliberately invalid multi-frame render and requires an
explicit rejection before slicing. The audit checks both beauty and GT logs for
single-frame tasks and the expected slice time `(frame - 0.5) / 24`.

`warped_ssim.py` defaults to this fork's Blender Flow.npy convention: it samples
the next image at `(x - raw_flow_x, y + raw_flow_y)`. The saved x component has
the opposite sign to forward image displacement. This was checked using saved
camera intrinsics/extrinsics and depth reprojection, not selected by maximizing SSIM.
For standard forward `(dx, dy)` input, use `--flow-format forward` instead.
Output schema `binoc-warped-ssim-v2` records the convention. Earlier v1 scores
used the wrong x sign and must be recomputed; the original image/flow files remain valid.

It reports full-image and valid-support scores.
The latter excludes out-of-bounds warp support, not occlusion. An explicit expected
frame count requires every image and every necessary forward-flow frame; the last
image does not need forward flow. Output includes per-frame CSV and summary JSON.

For a static-scene flow check (requires consecutive camera files and depth):

```bash
python experiments/render_eval/verify_flow_geometry.py SCENE_ROOT \
  --frame 25 --output flow_geometry_check.json
```

See `VALIDATION_20260905.md` for the repair checks and the active recovery run.
