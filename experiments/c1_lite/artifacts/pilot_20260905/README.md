# C1-lite pilot: stopped at the prospective endpoint contract

Date: 2026-09-05. Local branch: `experiment/c1-lite-pilot`.

This is a small offline research screen, not a new production method, a
continuous-window certificate, or a paper-quality comparison. No rendering,
runtime overlay, full-scene meshing, commit, or push was performed in this pilot.
Earlier uncommitted source-splice guard changes were left separate.

## Completed

1. Froze the existing C0 campaign, relevant current WSL source, and available
   core library. Original files were retained and source hashes checked.
2. Ran a read-only census of the four existing canonical **demo** events.
3. Implemented an isolated event-driven fan and same-cost naive-fan preflight.
4. Ran 14 synthetic unit tests successfully.
5. Ran the fixed first event's endpoint admission check. It stopped at the
   lower endpoint; no terrain-benefit or natural-frame evaluation followed.

## C0 provenance

- Source checkout and recorded historical validation HEAD:
  `ae81991c5d2cd8788f4340bbac4e3a934f71ef9c`.
- Archive: `tmp/c1_lite_c0_freeze_20260905/c0_snapshot.tar.gz` (4,203,916 bytes).
- SHA-256: `22413ebdbe5a1bfb9fc5ab16fcc3d4cba105332dc7b779db9439359b6848e680`.
- Full per-file hashes: `tmp/c1_lite_c0_freeze_20260905/manifest.json`.
- The captured library is the currently available binary, **not attested as
  the historical validation binary**. Environments and Forest renders were
  not copied. The archive is intentionally ignored by Git.

## Census

Details: [census.json](census.json).

| Fixed demo event | Window width (ms) | Native 24 FPS hits | Zero-phase 24/48/96/192 FPS hits | Old-wall sampled maximum gap |
| --- | ---: | ---: | --- | ---: |
| 00 | 32.6708 | 1 | 1 / 2 / 3 / 6 | 0.0237048434 |
| 01 | 13.3103 | 0 | 0 / 0 / 1 / 2 | approximately 2.29e-16 |
| 02 | 23.9586 | 1 | 0 / 1 / 2 / 4 | 0.0108860166 |
| 03 | 23.9586 | 1 | 0 / 1 / 2 / 4 | 0 |

The reconstructed mapping is `global_seconds = origin + tau * delta`, with
`origin` the binary64 value of `1/48` and `delta = 0.029948229166666663` seconds.
It agrees with the demo profile and cache temporal dimensions. It is not a
recording of the production slicer's final floating-point time calculation.
Native phase is `(frame_index + 0.5)/24`, not zero phase.

The first window includes native frame index 8, at ideal global time `17/48`
seconds. The exact inverse under those binary64 mapping parameters is
`288230376151711745/25895968073357991` in internal units. This time was **not
evaluated** by the reference preflight after its endpoint rejection.

All interval-stability and continuous-embedding certificates remain UNKNOWN.
The first event has a candidate associated-record breakpoint at `tau=11`, but
it is not a breakpoint of the currently selected suppressed owners. Six
matching sampled owner lists neither prove instability nor certify stability.
Its boundary hypervertices have `in_view=0`; actual camera ROI visibility is
unverified. These four demos are not a six-scene event-coverage census.

Old-wall gaps use six fixed interior samples and directed distances from
sliced wall vertices to source edges. They are neither Hausdorff distances nor
continuous-time maxima. Zero sampled gap does not imply an exact certificate.

## Why the first reference stopped

Details: [reference/result.json](reference/result.json).

Fixed event: `event-00-6d2d6dc2cdd7`, root `120/11`, original window
`[114/11, 126/11]`. The first tested endpoint was `114/11`.

The current implementation computes the midpoint of the parser's binary64
source coordinates, then rounds that new center to binary32. The prospective
runtime check compares it against the **also rounded** source diagonal:

| Quantity | Coordinates |
| --- | --- |
| Rounded source endpoint A | `(0.8056640625, -0.6591796875, 0.2685546875)` |
| Rounded source endpoint B | `(0.37841796875, 0.9480794072151184, 0.37841796875)` |
| Current rounded center | `(0.592041015625, 0.1444498747587204, 0.323486328125)` |
| Exact midpoint of rounded A/B | `(0.592041015625, 0.1444498598575592, 0.323486328125)` |

The current center fails an exact rational collinearity test. Its computed
distance to the rounded diagonal is approximately **3.9441e-9** in model units.
The two center candidates differ by one binary32 ULP in y at this scale.
For this particular endpoint, the midpoint of rounded A/B is itself exactly
representable in binary32. This is an evaluation-order mismatch, not evidence
that no binary32 construction can work.

This prospective check did **not** execute the C++ slicer. It also does not
establish a visible artifact, loss of topology, or a failure of the ideal-real
fan construction. It must not be conflated with the old approximately 0.0237
sidewall mismatch. We stopped rather than changing the prescribed coordinate
contract, tuning the window, or switching to a favorable event.

## What has not been established

- C1-lite improvement over either the baseline or the same-cost naive fan:
  **NOT MEASURED in this run**.
- Upper endpoint, root C0 identity, and interior preflight checks: not reached.
- Whole-window source/owner stability, embedding and exterior compatibility:
  not certified.
- Actual runtime vertex reuse, rendering attributes, OMP behavior, and visual
  temporal quality: not tested by this offline pilot.

Review also identified limits in the currently unexecuted PASS path: the
binary64 fan needs its own endpoint check, separately from the binary32 gate;
and demanding improvement at every grid at every interior sample is a stronger
screening rule than demonstrating a repeatable scientific benefit. That rule
must not be interpreted as a theorem or changed after observing results to
manufacture a PASS. The omitted raw/extra_smooth comparison and camera
visibility check are prerequisites for any later visual claim.

## Suggested next decision

Review the endpoint representation contract first. A narrowly scoped candidate
is to derive the endpoint center from the exact coordinates actually reused
by the baseline, then rerun the **same event and same window**. This was not
implemented here, and this endpoint's representability does not establish the
upper endpoint or the whole interval. Do not start broad engine changes,
continuous certificates, or long renders merely to bypass this stop.

## Reproduction

In the existing WSL environment, using a fresh output directory:

```bash
/home/warpwang/miniforge3/envs/binoc-exp/bin/python -B -m unittest discover \
  -s /mnt/e/BinocMesher/experiments/c1_lite -p 'test_*.py' -v

/home/warpwang/miniforge3/envs/binoc-exp/bin/python -B \
  /mnt/e/BinocMesher/experiments/c1_lite/reference.py \
  --event-root /home/warpwang/binoc-runs/full-ubuntu26-smoke/all_canonical_beb1/event-00-6d2d6dc2cdd7 \
  --cache-root /home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache \
  --profile-source /home/warpwang/src/BinocMesher/experiments/tv0_tv4/run_lightweight_profile.py \
  --output /mnt/e/BinocMesher/tmp/c1_lite_reference_rerun_NEW
```

The expected semantic result for this implementation and input is
`STOP_PREFLIGHT`, with an empty measurements list. Scripts refuse to overwrite
an existing output directory/file.
