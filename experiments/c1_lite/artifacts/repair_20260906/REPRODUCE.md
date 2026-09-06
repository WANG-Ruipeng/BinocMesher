# Reproduce the baseline-relative repair experiments

Use the existing permitted Miniforge `binoc-exp` environment inside WSL. No
Blender render or new environment installation is needed for the demo audits.
Commands require the preserved local demo inputs; compact JSON is not a raw
scene backup. Use fresh output directories: completed results are never overwritten.

```bash
repair_python=/home/warpwang/miniforge3/envs/binoc-exp/bin/python
repair_code=/mnt/e/BinocMesher/experiments/c1_lite
repair_cache=/home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache
repair_source=$repair_code/artifacts/ab_20260906/window_audit
repair_b=$repair_code/artifacts/ab_20260906/anchor_ablation/result.json
repair_output=/home/warpwang/binoc-runs/c1-repair-replay-UNUSED

"$repair_python" -B -m unittest discover -s "$repair_code" -p 'test_*.py' -v

"$repair_python" -B "$repair_code/runtime_retained.py" \
  --source-root "$repair_source" --cache-root "$repair_cache" \
  --output "$repair_output/raw_retained"

"$repair_python" -B "$repair_code/run_window_audit_v2.py" \
  --source-root "$repair_source" --cache-root "$repair_cache" \
  --ab-result "$repair_b" --output "$repair_output/window_audit_v2" \
  --max-seconds 900

"$repair_python" -B "$repair_code/run_window_audit_v3.py" \
  --source-root "$repair_source" --cache-root "$repair_cache" \
  --ab-result "$repair_b" --output "$repair_output/window_audit_v3" \
  --max-seconds 900

"$repair_python" -B "$repair_code/runtime_breakpoint_crosscheck.py" \
  --repo /home/warpwang/src/BinocMesher \
  --cache-root "$repair_cache" --source-root "$repair_source" \
  --retained-root "$repair_output/raw_retained" \
  --output "$repair_output/runtime_crosscheck" --max-seconds 240

"$repair_python" -B "$repair_code/runtime_rounding_probe.py" \
  --repo /home/warpwang/src/BinocMesher --cache-root "$repair_cache" \
  --source-report "$repair_source/event-02-88ade47aa4fa/source.json" \
  --output "$repair_output/runtime_rounding_probe"

"$repair_python" -B "$repair_code/audit_relative_plane_feasibility.py" \
  --cache-root "$repair_cache" \
  --source-report "$repair_source/event-02-88ade47aa4fa/source.json" \
  --output "$repair_output/e2_relative_plane/result.json" --max-seconds 90
```

The breakpoint driver launches separate OMP 1 and 8 processes, evaluates raw
and extra_smooth at the 11 distinct frozen breakpoints, and removes its owned
temporary cache copies automatically. Mesh arrays stay in RAM. Source-coordinate
matches and matching face counts do not establish final SourceVID/owner identity.

The v2/v3 drivers return successful process exit on a completed audit even when
the scientific outcome is REJECT or UNKNOWN. Read each event's `admission`
and `all_units_scanned` fields. No driver emits a production replacement plan.
All gates must pass before any later production integration; the current gates
explicitly retain unproven binary32, actual identity and event isolation checks.

The old B result is input evidence, not a newly measured treatment gain. It is
bound by SHA-256 `8b17fd6a61f3f7f0abe25c39638e1cb01a1ff610f6b66351d7253d234ed15605`.
The mesh construction/anchor code was not changed by these contact repairs.

## Forest

Forest has different assets, timing, cache shape and mesher budgets from the
toy demo. Do not use the demo loader or its time conversion on a Forest cache.
The first 24-frame run is a retained resource STOP, not a completed zero-event
census. The fixed 64-frame follow-up has its own pre-execution protocol under
`forest/`, uses a private frozen driver snapshot and a fresh cache, and never
renders or overwrites the original scene/cache. Consult those protocols and
the final Forest report rather than launching the old full rendering pipeline.

The source and input hashes recorded in each report identify the actual scripts
and loaded `core.so`; a Git commit alone does not identify both local checkouts.
The Windows and WSL checkouts intentionally remain separate.
