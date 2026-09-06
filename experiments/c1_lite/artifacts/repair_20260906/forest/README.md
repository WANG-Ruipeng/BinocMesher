# Forest census pilot: native registry available, wrapper stopped

The fixed first 64 frames of Forest seed 0 produced complete native BinocMesher caches and registries, but the Infinigen wrapper then stopped with `KeyError: 'eroded'`. Overall status remains **STOP**, not successful end-to-end scene generation. No renderer ran and no Forest census worker remained at the final process check.

The reusable cache is **pre-surface-displacement** geometry. Disabling attribute generation did not disable the downstream displacement kernel; that kernel requires `eroded` and can change vertices. Therefore this cache is not a complete Infinigen mesh or rendering reference. The older “only material/tag evaluation” note in `effective_inputs.json` is superseded by this diagnosis; the execution evidence remains unchanged.

## Verified native denominator

| Cache | Raw observations | Logical incidences | Canonical saddle events | Exact internal roots |
| --- | ---: | ---: | ---: | ---: |
| OpaqueTerrain | 440 | 268 | 131 | 2 |
| atmosphere | 0 | 0 | 0 | 0 |

OpaqueTerrain root `3/2` has 66 canonical events (184 raw, 132 logical); root `5/2` has 65 (256 raw, 136 logical). All 440 observations have the temporal-neighbour role. Direct CSV counts agree with the native summaries. **These are bilinear-saddle events, not certified BEB1 constructions.** BEB1 classification, complete closure/support, atomic compatibility, window/exterior admission and runtime contracts remain uncompiled or untested; no BEB1 coverage percentage is available.

This is one predeclared 64-frame, 24 fps, 960×540, 6 px pilot with two time groups, not the paper's full settings or a six-scene census. Selection was based on the time-tree structure, not event count or benefit. Future natural-frame work must use the saved camera inputs and this cache's time model, not the four-demo conversion.

## Evidence and retained data

- [Read-only native verification](stage64/native_registry_verification.json), [raw supervisor STOP summary](stage64/summary.json), [full wrapper traceback](stage64/worker_full.log).
- [OpaqueTerrain native summary](stage64/OpaqueTerrain/event_registry_p1_summary.json), [CSV](stage64/OpaqueTerrain/event_registry_p1.csv); [atmosphere native summary](stage64/atmosphere/event_registry_p1_summary.json), [CSV](stage64/atmosphere/event_registry_p1.csv).
- [Executed driver snapshot](stage64/driver_snapshot.py), [stage64 protocol](stage64/protocol.json), [predeclared phase protocol](phase2_64_protocol.json), [actual camera inputs](stage64/camera_inputs.json).
- [Sidecar availability checks and SHA256](stage64/provenance_file_checks.json), with [the explicit valid-empty-header correction](stage64/provenance_empty_header_check.json). These check format/length and BPM2 first-record identity, not every owner record or a geometry certificate.

Driver snapshot SHA256: `b87e5a6f3e4d888def5bba72c3d914c5b038d861a044f6c0fb450d98afbec058`. Worker wall time was 1,203.26 seconds; peak observed output was 1,783,308,839 bytes and final output before the supervisor summary was 1,149,146,053 bytes. Original scene/assets/cache fingerprints were unchanged. All stage64 caches plus 48,000,835 bytes of private assets are retained under `/home/warpwang/binoc-runs/forest-census64-repair-20260906`.

Stage 1 (24 frames, one time group) stopped at the pilot's own 128 MiB per-file cap before a registry existed: its counts are **unknown, not zero**. Its generated cache and copied assets were removed after preserving hashes/logs: 488,586,831 bytes reclaimed, no trash copy, reproducible from the unchanged source. See [diagnosis](stage1_diagnosis.json), [file inventory](failed_cache_inventory.json) and [cleanup record](stage1_cleanup.json). Do not add the overlapping stages as independent scene samples.

## Read-only recheck; no rebuild needed

In WSL, this command reads only the retained native summary (no mesher, slicer or renderer):

```bash
/home/warpwang/miniforge3/envs/binoc-exp/bin/python -m json.tool /home/warpwang/binoc-runs/forest-census64-repair-20260906/HyperMesh/OpaqueTerrain/event_registry_p1_summary.json
```

The next useful experiment is a separately bounded, read-only/offline eligibility compiler over the **entire frozen set of 131 events**, retaining unsupported cases and grouping by `3/2` and `5/2`; it must consume this cache's source/provenance and actual time mapping. The existing demo campaign CLI is not a drop-in Forest experiment (demo profile and expected-count assumptions). Classification must precede any claim of BEB1 support or runtime admission. Forest's terrain is not guaranteed to be a height field, so the demo same-XY error metric is not automatically valid. Restore a well-defined full surface-displacement path before making Infinigen rendering-quality claims. None of these next steps was executed in this pilot.
