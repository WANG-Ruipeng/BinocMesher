# Optional offline mixed batching

This opt-in host facility consumes distinct tasks that are already ready and
share one live Field, K and numerical profile. It preserves D1 field mathematics,
coordinate precision, CSR endpoint order and node-local witness rank.
The single-task path remains available and batching defaults to off. Offline
batching immediately consumes the caller's complete group: no queue, waiting
to form a batch, or manufacturing repeated jobs.

Published is the packaged backend. The initial recommended scope is K6
ready-host Published, where historical performance passed its stability rule.
K3 correctness is retained without a K3 performance claim. Graph is
**OPTIONAL_BACKEND_NOT_PACKAGED**: explicit requests are rejected, never silently
changed to Published. The historical Graph positive result remains below.

## Data and lifecycle boundary

Task IDs are stable and unique. Prefix sums use each task's actual N/M. CSR
offsets omit each original final sentinel, shift by endpoint prefix, and append
one total-M sentinel. Owners shift by node prefix. Witness remains node-local
rank and never receives a task/global endpoint offset.

Five outputs split by node range. SDF/aux/xyz/sign split the center stripe and
each endpoint-round stripe separately; bounds traces split each node-round
stripe. Per-task returned byte buffers have independent lifetime. Compatibility
includes a live Field instance/generation and CUDA device/context: equal
parameter bytes or a recycled pointer cannot authorize cross-Field reuse.

Published field creation preserves the historical SdfTrees/LandTiles
header/schema and coordinate-domain guards in
batching/native/validated_domain.h, migrated from field_admission.h.
This first version requires the original tree seed and all 12 tree parameter floats,
all 8 LandTiles integer parameters and its first 18 float header values bitwise.
It accepts the corresponding meta/header, 2048 grid and
coordinate envelope with 1..256 lattices. Heightmap and parameters are explicit
external inputs, not compiled-in assets or a hard-coded scene hash. Out-of-domain
inputs are rejected. Arbitrary field/domain safety remains outside this scope.

A new geometry requires a new solver: this ABI has no set-geometry update.
Each run performs a real cold reset, complete closed-loop solve, necessary
error checks/synchronization and requested output readback. Failed or stale
results cannot be reused. CUDA errors and numerical mismatches do not trigger
a hidden retry. Pre-submit compatibility/budget rejection is a separate explicit
fallback decision. Successful results are dispatched only after the whole group
and required checks finish.

Destroy solvers before their immutable shared Field. The global allocation cap
covers all active handles and Field assets, is no larger than 2 GiB, and is
tightened by initial free memory. New Published has zero Graph reservation.
Planner estimates, tracked allocation, conservative reservation, point-in-time
cudaMemGetInfo observations and unknown device peak are distinct quantities.

## Historical performance, not a new interface benchmark

The sealed source is the 256-arm distinct-task experiment: all arms OK, 6,306
application solves, two timer boundaries, K3/K6 each using five different
original tasks. There is one scene. Corresponding K3/K6 tasks share the same
five geometries at different K; this is not ten independent scenes. The earlier
224-arm / 15,001-solve replicated-scale study is not this evidence.

For five different K6 tasks already ready on the host, Published
ENTRY_READY_HOST had a direct-pair median of **2.744365x**, with required A/A
passing. Historical Graph RESIDENT_IO had **3.488316x**, also passing its own
A/A. The other six comparisons remain timing-inconclusive. These are limited
local throughput signals, not full-mesher speedups, single-request latency
guarantees or production promotion.

| Historical method | K | Boundary | Pooled direct ratio | Required A/A stable | Verdict |
| --- | ---: | --- | ---: | --- | --- |
| D1_GRAPH | 3 | ENTRY_READY_HOST | 2.670360 | False | INCONCLUSIVE_TIMING |
| D1_PUBLISHED | 3 | ENTRY_READY_HOST | 2.641399 | False | INCONCLUSIVE_TIMING |
| D1_GRAPH | 3 | RESIDENT_IO | 3.783865 | False | INCONCLUSIVE_TIMING |
| D1_PUBLISHED | 3 | RESIDENT_IO | 3.763667 | False | INCONCLUSIVE_TIMING |
| D1_GRAPH | 6 | ENTRY_READY_HOST | 2.736520 | False | INCONCLUSIVE_TIMING |
| D1_PUBLISHED | 6 | ENTRY_READY_HOST | 2.744365 | True | LOCAL_MIXED_BATCH_THROUGHPUT_SIGNAL_ONLY |
| D1_GRAPH | 6 | RESIDENT_IO | 3.488316 | True | LOCAL_MIXED_BATCH_THROUGHPUT_SIGNAL_ONLY |
| D1_PUBLISHED | 6 | RESIDENT_IO | 3.521300 | False | INCONCLUSIVE_TIMING |

Preserved records include [all eight results](benchmarks/mixed_batch_20260918/SUMMARY.json),
[all 32 A/A controls](benchmarks/mixed_batch_20260918/AA.json),
[all 256 timing rows](benchmarks/mixed_batch_20260918/TIMINGS.csv),
[the complete protocol/schedule](benchmarks/mixed_batch_20260918/PROTOCOL.json),
[per-process calibration](benchmarks/mixed_batch_20260918/CALIBRATION_R.json) and
[task identities](benchmarks/mixed_batch_20260918/TASK_IDENTITIES.json).
Twenty-three A/A controls passed; nine failed. No failed control, inconclusive
comparison or planned arm was removed.

Each quartet uses sqrt(a1*a2/(b1*b2)), SERIAL over MIXED for the same five tasks.
Each process takes the median of two opposed ABBA/BAAB quartets; pooled takes
the median of the four raw ratios. Both styles' own A/A in both processes must
be within [0.95,1.05], then both process medians and pooled must reach 1.05.
R uses SERIAL D1_GRAPH only: three actual equal-work units, 50 ms target,
256 cap; all four K/boundary values freeze before that process's first arm.
R repeats are not independent statistical samples.

## What the two historical timers included

| Boundary | Included | Excluded |
| --- | --- | --- |
| RESIDENT_IO | Reset/solve/error checks, five-output D2H, synchronization, split and dispatch | Geometry pack, allocation/H2D/prepare, two actual warmups per alias, destruction, final CPU comparisons/hashes |
| ENTRY_READY_HOST | From loaded but uncombined host tasks: pack/admission, allocation/H2D/prepare, solve/error/readback, split/dispatch and destroy each unit, including Graph capture/instantiate when applicable | File reading, shared Field admission/upload, final CPU comparisons/hashes; two separate factory warmups precede formal units |

Formal ENTRY handles were new, with no hidden handle warmup. Shared Field cost
is outside both historical timers. SERIAL performs five solves and MIXED one
per equal-work unit. Independent stage medians cannot be added to reconstruct
full solve time; different paired ratios are not chained. Python/ctypes
transport is included where specified, not native C++ production timing.
Online queue wait, batch-formation latency and production simultaneous readiness
were not measured and are not assumed to cost zero.

The 6,306 solves comprise 42 correctness + 16 limited memcheck + 3,268 process-1
benchmark + 2,980 process-2 benchmark. Each process uses 484 nonformal solves
plus 96*sum(four R). Historical observed maxima were 121,217,640 tracked bytes
and 671,088,640 Graph budget bytes, 792,306,280 combined, including ten live
Graph handles in resident A/A. Reservation is not measured Graph storage.
Observed free memory ranged from 15,438,184,448 to 15,681,454,080 bytes; these
samples do not measure a device peak.

## Provenance and integration regression

[EVIDENCE_MANIFEST.json](benchmarks/mixed_batch_20260918/EVIDENCE_MANIFEST.json)
records original byte lengths/SHA256 separately from public-derived identities.
Large/raw evidence, libraries, scenes and third-party trees stay outside Git.
The owner supplies external_evidence_root, external_backend_root and
external_build_root. No personal machine path is required. The CPU-only
[export script](benchmarks/mixed_batch_20260918/export_history.py) validates the
original seal and regenerates the records into a new output directory.

[REFERENCE_HASHES.json](benchmarks/mixed_batch_20260918/REFERENCE_HASHES.json)
is the unedited per-task eleven-output reference manifest. New-build checking
may use these hashes only in its comparator; hashes/expected values are never
solver inputs, and the checker need not load the historical library.

Historical and newly built libraries have separate identities. New clean-source
build, CPU tests, finite GPU smoke and memcheck are integration regressions, not
a rerun or recertification of the historical 256 arms. A new binary hash is
recorded with numerical regression, not a fabricated identity claim. Historical
A/B sanitizer failures and SPEC2 coverage gaps remain unchanged.

Graph and the A/B/C research routes, W4, rooted and compaction are not packaged.
See [SOURCE_MIGRATION.md](benchmarks/mixed_batch_20260918/SOURCE_MIGRATION.md)
for the wrapper/ABI/build dependency chain. Ordinary package import and help
do not create CUDA objects. Build/run examples use explicit external inputs,
toolchain and Infinigen dependencies.

## Build and call explicitly

Run from the repository root with Python 3.10+ on the existing Linux/WSL CUDA
setup. This does not install a toolchain. Supply the inner Infinigen include
root and driver library directory explicitly; choose the architecture of the
selected local device. Build output must be a new directory outside the checkout.

```bash
python -B gpu_accelerate/build_batching.py \
  --nvcc "$CUDA_ROOT/bin/nvcc" --host-compiler "$CXX" \
  --infinigen-include "$INFINIGEN_INCLUDE" \
  --driver-library-dir "$CUDA_DRIVER_LIB" --arch sm_120 \
  --output "$BATCH_BUILD"

python -B -m unittest discover -s gpu_accelerate/tests -p 'test_batching*.py' -v
```

`--dry-run` prints the build plan without compiling. The resulting
`libbatching_published.so` contains the required Field bridge, original D1
variant1 and typed adapter together. Compiler versions, input/include hashes,
strict flags, commands, raw logs and resulting library hash are saved beside it.
There is no runtime dependency on an experimental library or helper script.

For the supplied external binary fixtures, the following example deliberately
chooses K6. Callers with ready geometry can construct immutable `Task` objects
instead, using `field.binding(k, trace=..., audit=...)`; geometry is float64,
CSR/owners int32. Byte buffers use the documented little-endian fixture layout.

```python
from gpu_accelerate.batching.backend import PublishedBackend
from gpu_accelerate.batching.inputs import load_inputs
from gpu_accelerate.batching.runner import solve_many

parameters, jobs, identity = load_inputs(fields_path, jobs_path,
    fields_sha256=expected_fields_sha256, jobs_sha256=expected_jobs_sha256)
with PublishedBackend(library_path, expected_device_uuid=device_uuid) as backend:
    with backend.create_field(parameters.meta, parameters.tree_ip,
            parameters.tree_fp, parameters.land_ip, parameters.land_fp,
            parameters_sha256=parameters.sha256) as field:
        tasks = [job.task(field, trace=False, audit=False)
                 for job in jobs if job.k == 6]
        original = solve_many(field, tasks)  # batching='off', backend='published'
        mixed = solve_many(field, tasks, batching='offline', backend='published')
        # mixed.outputs[task_id]: owned position/witness/valid/left/right bytes
```

`trace=True` additionally returns complete SDF, aux, xyz, sign and left/right
round traces. `audit=True` enables the original solver audit. These settings are
part of compatibility. An incompatible offline group raises by default;
`fallback='serial'` explicitly permits sequential execution only after a
pre-submit compatibility or successfully cleaned budget rejection. It does not
permit cross-Field tasks or retry a failed solve. Returned `selected_path` and
`fallback_reason` expose that choice. Only one backend can be live per process;
its calls are serialized and its Field resources are caller-owned and reusable.

The native `mb_*` ABI is defined in `batching/native/published_api.h`. It uses
integer tokens, typed pointer/extent arguments and explicit status codes. Old
raw handles are not interchangeable with these tokens. Do not bypass the Python
binding's lifecycle and thread serialization when using the C ABI directly.

## Finite numerical regression

Input paths, their expected SHA256 and the device UUID are required explicitly.
The committed eleven-array manifest is a comparator only. Without `--execute`,
this command prints its scope without loading a library, input or CUDA context.

```bash
python -B gpu_accelerate/check_batching_gpu.py \
  --library "$BATCH_BUILD/libbatching_published.so" \
  --fields "$FIELDS_BIN" --jobs "$NATURAL_JOBS_BIN" \
  --fields-sha256 "$FIELDS_SHA256" --jobs-sha256 "$JOBS_SHA256" \
  --historical-hashes gpu_accelerate/benchmarks/mixed_batch_20260918/REFERENCE_HASHES.json \
  --expected-device-uuid "$GPU_UUID" \
  --output "$NEW_REGRESSION_DIR" --phase regression --execute
```

`--phase regression` performs 48 successful application solves: ten original
single-task references, twenty default serial solves, sixteen mixed forward /
reverse repeated solves, and two cold resets of one K6 handle. It also exercises
invalid native/public runs and stale/destroyed readback rejection. Production
uses the required five outputs; audit/trace compares all eleven arrays bitwise.
`--phase smoke` is a separate limited 14-solve check. Neither collects paired
performance samples.

For finite memcheck, run that same explicit command under the existing
`compute-sanitizer --tool memcheck --error-exitcode 86 --target-processes all`,
change phase to `memcheck`, and run `--audit 0` and `--audit 1` separately into
new output directories. Each mode performs four mixed solves (K3/K6, repeated).
The caller must supervise GPU work serially within the integration task's time
budget. Preserve nonzero exits and raw sanitizer logs; `RESULT.json` alone is
not a sanitizer verdict. A clean source export must rebuild a new library and
run its own finite smoke; copying a previously built library is insufficient.

For disabling this feature, leave `batching='off'` or keep using the preexisting
single-task D1 API. Nothing in the production mesher enables mixed batching.
## This engineering integration

The finite Published integration passed on the existing local RTX 5080 / CUDA
13.1 toolchain: 46 CPU tests, 48 real GPU regression solves, four production
memcheck solves, four audit/trace memcheck solves, and a separate clean-build
14-solve smoke. Both memcheck runs reported zero errors. The fresh source
export also passed all 46 CPU tests. These are 70 new successful application
solves, not another performance matrix.

See [INTEGRATION_VALIDATION.json](benchmarks/mixed_batch_20260918/INTEGRATION_VALIDATION.json)
for exact binary/tree identities, coverage and untested scope, and
[INTEGRATION_EVIDENCE_MANIFEST.json](benchmarks/mixed_batch_20260918/INTEGRATION_EVIDENCE_MANIFEST.json)
for sizes and hashes of the external raw build/regression logs. Actual supervised
GPU/sanitizer worker wall was 7.627123 seconds within the 600-second cap; this is
supervision accounting, not a throughput measurement. Candidate-to-final changes
after the clean source export are documentation/evidence only. Published kernels
and preexisting single-task entry points remain unchanged.