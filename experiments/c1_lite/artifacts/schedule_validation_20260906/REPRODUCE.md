# Reproduce this bounded experiment

Use the existing WSL Ubuntu environment, Python 3.10.21
(`/home/warpwang/miniforge3/envs/binoc-exp/bin/python`), NumPy 1.26.4 and Pillow
12.3.0. No new environments or software were installed. The production source
remains at `000028e0fa5bc4922aeb51aa9fb9ee96b799301e`.

The new scripts are experiment code, not part of the earlier freeze. Their
executed hashes are in the per-run JSON records. The original installed WSL
library is not overwritten: E2 explicitly loads the independently built frozen
library under `e2-runtime-loop-20260906/build` and verifies its SHA256.

## E2

Run inside WSL, using a fresh output directory. This command is provided for
reproduction; `e2_reproduce` was not created as part of the reported run.

```bash
cd /mnt/e/BinocMesher/experiments/c1_lite
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/warpwang/miniforge3/envs/binoc-exp/bin/python -u run_schedule_e2_images.py \
  --repo /home/warpwang/binoc-runs/e2-runtime-loop-20260906/build \
  --cache-root /home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache \
  --source-report artifacts/ab_20260906/window_audit/event-02-88ade47aa4fa/source.json \
  --output artifacts/schedule_validation_20260906/e2_reproduce \
  --max-seconds 1200
```

The reported success is in `e2_retry2`, not the two earlier harness-stop
directories. Those stops are preserved to show the log-audit correction.
Do not remove or overwrite existing output directories to reuse their names.

The worker copies only the 4.26 MB cache into an automatically cleaned WSL
temporary directory. It generates all meshes in memory, validates the declared
schedule, checks the original terrain reference refinement, then writes compact
JSON and PNGs. No RGB, video, full-precision image arrays or mesh arrays persist.

## Forest front-end and direct-face-quad probe

Important: this command reproduces the **restricted direct-face-quad support
probe**, not complete closure-derived C1 eligibility. Read the top-level report
and `forest/scope_correction.json` before interpreting any candidate counts.
The frozen E2 source support itself differs from the direct critical-face quad.

```bash
cd /mnt/e/BinocMesher/experiments/c1_lite
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/warpwang/miniforge3/envs/binoc-exp/bin/python -u forest_eligibility_funnel.py \
  --cache-root /home/warpwang/binoc-runs/forest-census64-repair-20260906/HyperMesh/OpaqueTerrain \
  --stage64 artifacts/repair_20260906/forest/stage64 \
  --output artifacts/schedule_validation_20260906/forest_reproduce \
  --max-seconds 1200
```

This reads the original Forest cache; it neither copies nor rebuilds it. It
retains only a bounded boundary-incidence halo. The 64-frame, 6px input cache
stops before the Infinigen displacement stage and is not paper-scale geometry.

## Tests

```bash
cd /mnt/e/BinocMesher
OPENBLAS_NUM_THREADS=1 /home/warpwang/miniforge3/envs/binoc-exp/bin/python \
  -m unittest discover -s experiments/c1_lite -p 'test_*.py'
/home/warpwang/miniforge3/envs/binoc-exp/bin/python \
  -m unittest discover -s experiments/source_splice -p 'test_*.py'
```

Observed final results: 331 C1-lite tests and 14 source-splice tests passed,
345 total. This includes 14 synthetic Forest-reader tests, and does not turn
the unsupported closure-derived Forest stage into a passing one.

The resource checks are cooperative, not hard process watchdogs. The actual
successful workers were far below the 20-minute / 100-MiB persistent-output
limits. Reused library/cache build time, Python development time, and historical
experiments are not included in reported worker timings; no controlled cold-
versus-warm performance benchmark was performed.
