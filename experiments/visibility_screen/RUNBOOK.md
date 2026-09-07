# Preregistered visibility screening

This experiment implements priorities 1–4 of the supplied research plan. It does
not extend the frozen event construction, contact policy, or theorem domain.

## Fixed inputs

The experiment protocol is `protocol_20260906.json`, SHA-256
`9f8ec007632e763383d87ad544d4be2d0069bc506a89d3f38734e3c1bb6d2ae1`.
The pre-result source/native/evidence seal is
`artifacts/screen_20260906/preregistration_seal.json`.
This is a file-level seal, **not a new Git commit**.

| Segment | Official scene / seed | Absolute frames | Geometry |
| --- | --- | --- | --- |
| Forest B | Forest / 0, saved original scene | 97–160 | Pre-surface-displacement opaque |
| Cave | Cave / 1 | 1–64 | Pre-surface-displacement opaque |
| Mountain | Snowy Mountain / 1 | 1–64 | Pre-surface-displacement opaque |

All use 24 FPS, 960×540 and 6 px LOD. LOD camera relaxation and the original
image camera are recorded separately. Forest B evaluates the saved original
Blender camera; it does not linearly interpolate the trajectory text file.
Seeds, segments, thresholds and budgets must not change in response to results.

## Environment and execution

The native/Infinigen environment is WSL Ubuntu. Use the existing Miniforge
environment at `/home/warpwang/miniforge3/envs/binoc-exp/bin/python` and original
project at `/home/warpwang/src/BinocMesher`; Anaconda and Miniconda are not used.
The experiment scripts and compact reports are Windows-visible under
`/mnt/e/BinocMesher/experiments/visibility_screen`.

Run **one heavy scene worker at a time**, always into a fresh attempt directory.
The commands below use Forest B as an example; the other two fixed IDs are
`cave` and `mountain`. These are actual execution commands, not completed-result
claims.

```bash
cd /mnt/e/BinocMesher/experiments/visibility_screen
export PYTHONDONTWRITEBYTECODE=1
PY=/home/warpwang/miniforge3/envs/binoc-exp/bin/python

$PY -B scene_build.py \
  --protocol protocol_20260906.json --segment forest_b \
  --output /home/warpwang/binoc-runs/visibility-screen-20260906/forest_b_attempt01 \
  --report-dir artifacts/screen_20260906/forest_b/build01

$PY -B source_stage.py \
  --protocol protocol_20260906.json \
  --seal artifacts/screen_20260906/preregistration_seal.json \
  --segment forest_b --build-report artifacts/screen_20260906/forest_b/build01 \
  --output artifacts/screen_20260906/forest_b/source01

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B run_screen.py \
  --protocol protocol_20260906.json \
  --seal artifacts/screen_20260906/preregistration_seal.json --segment forest_b \
  --build-report artifacts/screen_20260906/forest_b/build01 \
  --source artifacts/screen_20260906/forest_b/source01 \
  --output artifacts/screen_20260906/forest_b/screen01_omp1

OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -B run_screen.py \
  --protocol protocol_20260906.json \
  --seal artifacts/screen_20260906/preregistration_seal.json --segment forest_b \
  --build-report artifacts/screen_20260906/forest_b/build01 \
  --source artifacts/screen_20260906/forest_b/source01 \
  --replay-from artifacts/screen_20260906/forest_b/screen01_omp1 \
  --output artifacts/screen_20260906/forest_b/screen02_omp8
```

Each later command requires the preceding stage's completion receipt. A source
compiler success is not runtime admission. An OMP1 screen is not yet OMP1/8
confirmation. The confirmation produces a separate `combined_summary.json`;
the original OMP1 evidence is never overwritten.

The screen uses every canonical registry event, including failed and unsupported
candidates, to build conservative support components. It constructs the actual
64 natural-query outputs plus separate exact-root diagnostics. Mesh arrays are
held in memory, with compact identity/hash receipts rather than mesh videos.

## Selection and interpretation

A selected component must be jointly admitted and have at least 16 original-
camera baseline support pixels and at least one actual replacement pixel on
each of at least two natural frames. Selection is ordered by segment ID,
earliest qualifying absolute frame, then component ID, never by quality delta.
Wait for all three fixed segment outcomes before selecting a pilot.

Unknown visibility is not zero. Candidate-halo visibility is marked as a proxy
when an exact compiled source is unavailable. A zero visible denominator has
an undefined rate. Geometry or image differences are not, by themselves,
quality improvements. Failed build/certificate attempts stay in the evidence
ledger and are not silently counted as zero events.

The previous Forest A result is preserved: 131 events, 79 components, 20 jointly
admitted components / 25 events, 16 modified natural frames, and zero visible
replacement pixels. It remains a negative row in the cross-scene report.

## Conditional priorities 2 and 3

Only a qualifying visible component can enter the separately versioned
post-displacement pilot. It requires actual owner-element attribute evaluation
at the new center, identical displacement semantics, and renewed boundary,
interface, nondegeneracy, contact and outside-support checks. No interpolation,
boundary clamping or support enlargement is an admissible shortcut.

`post_displacement_pilot.py` currently supplies **pure array audits**, not an
executed Blender worker. Synthetic tests are not actual post-displacement
evidence. A real worker and bound inputs are required before Gate B can pass.

Only a passing post-displacement component can enter the small visual study.
Concrete controls and a quality reference must be bound before interpreting
depth/normal differences as errors or improvements. An undefined control or
reference remains explicitly unsupported.

If all three completed new segments have zero visible admitted events, stop
the visual-improvement experiment; do not add outcome-selected segments or
replace the original camera with a diagnostic view.

## Resources and reproducibility

The protocol caps the campaign at 9 hours, each coarse build at 1 hour, each
native cache build at 45 minutes, new persistent data at 32 GB, compact reports
and images at 512 MiB, and total existing plus new storage at 400 GB.
Budget failures are infrastructure stops, not method rejection rates.
Input caches, original environments and prior evidence are preserved. Only
validated run-owned temporary copies and disposable intermediates may be
removed. Certification/image/audit wall times are not production playback cost.

Fast adapter regression suite:

```bash
PYTHONDONTWRITEBYTECODE=1 $PY -B -m unittest discover
```

For actual stage statuses and results, read the versioned JSON receipts and the
campaign results report; this runbook deliberately contains no unmeasured
cross-scene results.
