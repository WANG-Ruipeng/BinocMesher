# Final visibility gate: approved final resource retry

This is the continuation of `../README.md`, not a replacement of its historical
results or STOP receipts. The user approved the attached final-gate plan.

The new authoritative budget receipt is [budget_amendment01.json](../budget_amendment01.json),
SHA-256 `39ce0a8088c5c72a4b6595c8d054becf6c909410d4e70e1460a042f8488c2ac0`.
Only the native build ceilings change: 4 to 8 GiB, 2700 to 5400 seconds.
The original protocol and 176 method/input bindings, old 18 adapter Python files,
and Mountain OMP1/8 result remain byte-identical. This is the last resource increase.

The original campaign deadline is **2026-09-06 19:28:39 UTC** (2026-09-07 03:28:39
Asia/Shanghai), not nine additional hours. The source 1200-second budget, screen
3600-second budget, native 1 GiB per-file guard, memory guards, 512 MiB compact
artifact budget and 400 GB total storage budget remain unchanged.

## Fixed order and inputs

1. Cave seed 1, absolute frames 1–64: reuse the saved coarse input from
   `/home/warpwang/binoc-runs/visibility-screen-20260906/cave_attempt01/coarse`.
2. Forest B seed 0, absolute frames 97–160: reuse the saved original coarse input
   `/home/warpwang/runs/forest96-four-baseline-pilot-20260905/binoc/0/coarse`.

The second attempts have fresh native data directories named `cave_attempt02`
and `forest_b_attempt02`. Do not overwrite or try to resume the deleted, incomplete
first-attempt native caches. Coarse generation is not rerun.

Cave is followed by Forest B regardless of Cave's result. One heavy worker at a
time. The stage order is native cache, complete registry/source population,
component admission, actual requested sequence, original-camera visibility,
and OMP1/8 confirmation. No displacement, RGB, SSIM, or five-method quality study
is permitted before both fixed retry outcomes are known.

## Commands

Use the existing Miniforge Python. The examples show the Cave paths; change only
the registered segment/path names for Forest B. A directory listed here may
already exist; each stage requires a genuinely fresh output directory, and must
not be rerun blindly. The commands are operational documentation, not completion
claims.

```bash
cd /mnt/e/BinocMesher/experiments/visibility_screen
export PYTHONDONTWRITEBYTECODE=1
PY=/home/warpwang/miniforge3/envs/binoc-exp/bin/python
P=protocol_20260906.json
S=artifacts/screen_20260906/preregistration_seal.json
A=artifacts/screen_20260906/budget_amendment01.json
R=artifacts/screen_20260906/final_gate01/cave

$PY -B final_gate_build.py --protocol "$P" --seal "$S" --budget-amendment "$A" \
  --segment cave --output /home/warpwang/binoc-runs/visibility-screen-20260906/cave_attempt02 \
  --report-dir "$R/build02"

$PY -B final_gate_source.py --protocol "$P" --seal "$S" --budget-amendment "$A" \
  --segment cave --build-report "$R/build02" --output "$R/source02"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B final_gate_screen.py \
  --protocol "$P" --seal "$S" --budget-amendment "$A" --segment cave \
  --build-report "$R/build02" --source "$R/source02" --output "$R/screen01_omp1"

OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B final_gate_screen.py \
  --protocol "$P" --seal "$S" --budget-amendment "$A" --segment cave \
  --build-report "$R/build02" --source "$R/source02" \
  --replay-from "$R/screen01_omp1" --output "$R/screen02_omp8"
```

The byte-identical original build worker still consumes the original protocol
for scene/algorithm inputs. Its new supervisor alone applies the approved limits;
the build summary/contract binds this budget envelope. The original worker's
completion receipt is not relabeled as if it had read the amendment.

## Result interpretation

Use the unchanged coverage aggregator with the confirmed Mountain summary and
both final retry outcomes, preserving previous resource stops. Then use
`final_gate_verdict.py` to bind the coverage and both second-build summaries.

- A qualifying original-camera component permits only the bounded, separately
  certified attribute/displacement and quality follow-up. Original pixel/frame
  thresholds remain in force; visibility is not improvement.
- Visible candidates without visible admitted replacement permit one fixed
  rejection diagnosis, not automatic Method expansion.
- Three fully measured new segments with zero visible candidates yield
  `STOP_ORIGINAL_CAMERA_VISUAL_HEADLINE`, not `STOP_METHOD`.
- Resource or evidence failures remain unknown. No third budget increase or
  outcome-selected camera/seed/segment/LOD search is authorized.

The runbook is not a result receipt. Read each stage summary and the final
coverage/verdict before describing any segment as complete.
