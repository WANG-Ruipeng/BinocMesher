# Reproduce the fixed E2 runtime experiment

Run these commands inside the existing Ubuntu WSL distribution. They use the
existing permitted Miniforge environment, not Anaconda or Miniconda. No Blender,
rendering, environment installation or source-cache rebuilding is involved.

## Tested isolated build

The tested package is at:

`/home/warpwang/binoc-runs/e2-runtime-loop-20260906/build/binocmesher`

It is a copy of the Windows workspace package sources, Python modules, utilities
and makefile, with no old library copied in. It was compiled using:

```bash
make -C /home/warpwang/binoc-runs/e2-runtime-loop-20260906/build/binocmesher -j 4 \
  CXX=/home/warpwang/miniforge3/envs/binoc-exp/bin/x86_64-conda-linux-gnu-g++
```

Compiler: conda-forge GCC 11.4.0. Production flags:
`-w -O3 -std=c++17 -fPIC -fopenmp`; linker `-shared -fopenmp`, no fast-math.
Do not substitute the original installed library: the new Python opt-in API
requires the new identity/discard C symbols. Do not overwrite the WSL checkout
with this snapshot: that checkout has unrelated local changes to preserve.
For changed sources, use a fresh isolated build; the existing makefile does not
track all header dependencies automatically.

Frozen SHA256 bindings:

| Input | SHA256 |
|---|---|
| Original WSL core.so | `b2bf1b5f1847087645d88507446b012dd4b77324269e45bf893de0e17d330796` |
| Tested new core.so | `f4263a2f47ba5283175a921e49b8867998bdd8124aac810793b34242ec43a3c9` |
| E2 source.json bytes | `6b7f08fb7689ef8e62002c787e301a321fb343cda16d1ce37c54021b58c92dc2` |
| Source-cache inventory | `329c7681d849938a86e1f8812a1d05e3018070a06fb4bd070b81fd6ffb6cd3f6` |
| Runtime experiment driver | `306abf2601525ecc3a9f7e8b77f61d34bf03138eb7ce1ecec60cd8ac2220f72d` |

Inventory digest is SHA256 of ordered relative file names plus each file's
SHA256 bytes for processed hyperpolys, hypervertices and event_registry_p1.csv;
it is not a tar-file digest. Original full cache size: 4,255,152 bytes.
The actual initialized `deltaT` is binary64 `0x1.eaabfa360338dp-6`;
`tsize` is `0x1.eaabfa360338dp-1`, with 16 time groups.

## Experiment commands

Each output must be fresh. The suffix below is deliberately different from the
completed evidence directories; change it for further reruns. The driver creates
and removes its own temporary cache copy and refuses an altered source report.

```bash
e2_python=/home/warpwang/miniforge3/envs/binoc-exp/bin/python
e2_workspace=/mnt/e/BinocMesher
e2_source=$e2_workspace/experiments/c1_lite/artifacts/ab_20260906/window_audit/event-02-88ade47aa4fa/source.json
e2_cache=/home/warpwang/binoc-runs/full-ubuntu26-smoke/tv0_tv4/cache
e2_build=/home/warpwang/binoc-runs/e2-runtime-loop-20260906/build
e2_output=$e2_workspace/experiments/c1_lite/artifacts/e2_runtime_repeat

"$e2_python" -B "$e2_workspace/experiments/c1_lite/run_e2_runtime.py" \
  --repo /home/warpwang/src/BinocMesher --cache-root "$e2_cache" \
  --source-report "$e2_source" --output "$e2_output/reference" \
  --omp 1 --reference-only --max-seconds 600

"$e2_python" -B "$e2_workspace/experiments/c1_lite/run_e2_runtime.py" \
  --repo "$e2_build" --cache-root "$e2_cache" \
  --source-report "$e2_source" --output "$e2_output/omp1" \
  --omp 1 --max-seconds 600

"$e2_python" -B "$e2_workspace/experiments/c1_lite/run_e2_runtime.py" \
  --repo "$e2_build" --cache-root "$e2_cache" \
  --source-report "$e2_source" --output "$e2_output/omp8" \
  --omp 8 --max-seconds 600
```

`--max-seconds` is a cooperative bound checked between schedules, not a hard
process watchdog. The fixed small experiment completed in about 15 seconds per
new-library worker; do not extrapolate that bound to other caches or scenes.
Add `--smoke` for 5 exact and 14 physical queries. Full mode uses 33 and 41.
For each mode compare the three `baseline_hashes` lists; compare new OMP 1/8
`result_hashes` and `identity_ledger_hashes`. Require public API equality, every
independent validator and negative test to pass, original_cache_unchanged and
temporary_cache_removed to be true. Stored comparison is in `comparison.json`.

## Public API

After initializing the new-library mesher normally with one element and a
provenance-enabled raw cache, the actual production call is:

```python
from fractions import Fraction
import json

# source_report is the frozen E2 source.json path listed above.
with open(source_report, encoding="utf-8") as handle:
    source_contract = json.load(handle)
times = [Fraction(102, 5), Fraction(103, 5), Fraction(104, 5),
         Fraction(21), Fraction(106, 5)]
meshes, report = mesher.slice_window_batch(
    times, source_contract, time_mode="exact", extra_smooth=False
)
# Each mesh is (vertices, faces, tags). Consume after the whole call returns.
# COMMITTED_REQUESTED_SCHEDULE means these queries passed, not all real times.
# BASELINE_ENTIRE_SCHEDULE means no query in this batch was replaced.
```

Exact inputs are discrete rational time coordinates, not physical seconds.
For seconds use `time_mode="physical"`; the wrapper records the actual double
conversion and checks that query's actual output. Unsupported proposals or
unknown geometry fall back atomically. A failure of ordinary slicing or a memory
limit raises without publishing a partial batch. Do not combine with an old
`BINOC_SOURCE_SPLICE_PLAN`, smooth mode intervention or concurrent mesher calls.

## Regression tests

```bash
"$e2_python" -B -m unittest discover \
  -s "$e2_workspace/experiments/c1_lite" -p 'test_*.py' -q
"$e2_python" -B -m unittest discover \
  -s "$e2_workspace/experiments/source_splice" -p 'test_*.py' -q
"$e2_python" -B "$e2_workspace/experiments/source_splice/test_critical_beb1_event_ir.py"
```

See verification.json for final counts. Unit-test success is not an additional
scene census or an arbitrary-time numerical certificate.
