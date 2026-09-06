# Reproduce the bounded A/B study

These commands run in **WSL Ubuntu**, using the existing permitted Miniforge
environment. They assume the preserved local demo campaign/cache still exist;
this compact result directory is not a full raw-input backup. No Blender,
scene generation, network access, or production C1 window integration is used.

Use fresh output paths. The scripts deliberately refuse overwriting completed
outputs. The original demo cache is read-only; the C++ diagnostic creates its
own temporary cache copy because the ordinary wrapper writes manifests/logs.

## Fixed inputs and tests

```bash
experiment_python=/home/warpwang/miniforge3/envs/binoc-exp/bin/python
experiment_code=/mnt/e/BinocMesher/experiments/c1_lite
experiment_cache=/home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache
experiment_campaign=/home/warpwang/binoc-runs/full-ubuntu26-smoke/all_canonical_beb1
experiment_replay=/home/warpwang/binoc-runs/c1-ab-replay-UNUSED

"$experiment_python" -B -m unittest discover \
  -s "$experiment_code" -p 'test_*.py' -v
```

The completed run passed 76 synthetic/regression tests. These are software
checks, not 76 independent research events. The B plotting script additionally
received five input/layout assertions and an output-refusal check.

## A: continuous ideal-source audit

```bash
"$experiment_python" -B "$experiment_code/run_window_audit.py" \
  --campaign-root "$experiment_campaign" \
  --cache-root "$experiment_cache" \
  --output "$experiment_replay/window_audit" \
  --max-seconds 7200
```

The script preserves four fixed events, every source branch and every actual
breakpoint singleton. Its disk cap is 10 MiB. An exit code of 2 can mean a
completed audit with an UNKNOWN/REJECT certificate; inspect `summary.json`
instead of treating all nonzero exits as crashes. Runtime admission remains
false, including when conditional ideal-geometry certificates pass.

## B: matched local anchor comparison

```bash
"$experiment_python" -B "$experiment_code/anchor_ablation.py" \
  --campaign-root "$experiment_campaign" \
  --cache-root "$experiment_cache" \
  --profile-source /home/warpwang/src/BinocMesher/experiments/tv0_tv4/run_lightweight_profile.py \
  --output "$experiment_replay/anchor_ablation"

"$experiment_python" -B "$experiment_code/plot_anchor_ablation.py" \
  --input "$experiment_replay/anchor_ablation/result.json" \
  --output "$experiment_replay/anchor_ablation_plots"
```

The primary design is 33 exact times per event and spatial grids 32/64/128.
All events, failed contracts, zero gains, and adverse quality differences stay
in the result. The script caps its JSON at 16 MiB; plotting caps images at 5 MiB.
Height-error integrals are finite quadrature estimates, not exact continuous
error bounds or SSIM. C0 root-only does not acquire nonzero-duration gain from
the quadrature weight at one isolated root.

## Actual C++ baseline diagnostics

```bash
for experiment_event in \
  event-00-6d2d6dc2cdd7 event-01-e40703e638f3 \
  event-02-88ade47aa4fa event-03-0c61aa0982ef
do
  for experiment_threads in 1 8
  do
    "$experiment_python" -B "$experiment_code/runtime_baseline_audit.py" \
      --repo /home/warpwang/src/BinocMesher \
      --cache-root "$experiment_cache" \
      --event-root "$experiment_campaign/$experiment_event" \
      --omp "$experiment_threads" \
      --output "$experiment_replay/runtime_${experiment_event}_omp${experiment_threads}" \
      || break 2
  done
done
```

This is 80 small slices: four events, five times, two smoothing flags, and
OMP 1/8. Mesh arrays stay in RAM. Keep the eight `result.json` files after
completion; temporary `mutable_cache_copy` folders are reproducible copies,
not original inputs. The original run archived those eight JSONs under
`runtime_baseline/`, verified their hashes, and removed its temporary copies.

Record source-script and `core.so` hashes from the JSONs when comparing reruns.
The Windows and WSL checkouts are different local revisions; the original
execution deliberately records the actual loaded WSL binary, rather than
assuming a Git commit identifies all uncommitted changes.
