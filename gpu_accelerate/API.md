# GPU solver C ABI

The declarations are in [include/solver_api.h](include/solver_api.h), which also
includes the field bridge declarations from [src/field_types.h](src/field_types.h).
The implementation operates on an ordered CSR list of nodes belonging to one
field. It returns node outputs; assembling the original mesher's cumulative
vertex arrays is a separate caller/adapter responsibility.

## Input and ownership

Call `bm_field_create` and `bm_field_get_view` to obtain an immutable `FieldView`.
Its pointer members are CUDA device addresses. The solver borrows these buffers;
destroy every solver that uses them before destroying the field. The field bridge
copies the supplied parameter blobs. The caller must validate the complete real
field parameter schema; the bridge checks basic header lengths and size bounds.

`bm_solver_create` copies these host buffers and prepares private device storage:

| Input | Type and layout |
|---|---|
| `host_offsets` | `int[n+1]`, monotonic, first value 0, final value m |
| `host_centers` | `double[n*3]`, contiguous xyz per node |
| `host_endpoints` | `double[m*3]`, stable per-node endpoint order |
| `max_k` | Integer in [0,64]; later calls require 0 <= k <= max_k |

The interface uses 32-bit `int`, 32-bit `float` and 64-bit `double` on the
supported CUDA host platform. Coordinates must be finite. `n=0` requires `m=0`
and still requires one offset. Nonempty nodes may have zero endpoints. Geometry
and field buffers cannot be updated in place through this API; create a new
solver when they change. A handle owns its stream and must be used serially in
the CUDA context/device where it was created.

## Execution and numerical contract

| Mode | Operation |
|---|---|
| 0 / R0 | GPU-resident staged kernels for reset, field queries and node updates |
| 1 / R1 | The same staged kernels captured in a CUDA Graph |
| 2 / L0 | One thread performs the complete solve for a node |
| 3 / L1_8 | One aligned eight-thread group per node |
| 4 / L1_32 | One aligned warp per node |

`BM_L132_BLOCK_THREADS` selects 64, 128 or 256 threads per block for mode 4
(default 256). It does not change the 32-thread node group. Other modes retain
256-thread blocks. `BM_SOLVER_DEBUG_GUARDS=1` enables separate validation builds.

Before mode 1, call `bm_solver_prepare_graph(handle,k,trace)`. An unchanged key
reuses the prepared graph; changing k or trace destroys and rebuilds it. An empty
node list records the configuration without launching a zero-sized grid.
`bm_solver_run` resets the solve state, executes one complete solve and checks
stream completion. Repeating a run recomputes the solution rather than reusing
previous answers.

Each solve evaluates every center, every endpoint for each of k bisection rounds,
and every endpoint once at the final right bound. There is no early exit that
skips the remaining field queries. Bounds and interpolation use double precision;
`center*(1-t) + endpoint*t` is converted to float before the field call. The field
and auxiliaries remain float32. Signs use `sdf >= 0`, including exact zero. The
witness is the least original endpoint rank with a different sign. Final geometry
is recomputed from the double inputs and final right bound.

Build both CUDA translation units with the same compiler/architecture and strict
floating-point options: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`,
with host fast-math and FP contraction disabled. Do not substitute an algebraic
interpolation rewrite, reduced precision or approximate field evaluation while
claiming this numerical contract.

## Readback

After a successful run or measurement, `bm_solver_readback` copies selected
outputs to host buffers and synchronizes the solver stream. Let
`Q = n + (k+1)*m`, using k from the last successful solve. Each pointer may be
NULL to omit that copy; supplied buffers must have at least these capacities:

| Output | Elements | Meaning |
|---|---|---|
| `position` | `float[n*3]` | xyz per node; zero for invalid/empty nodes |
| `witness` | `int[n]` | Local endpoint rank, or -1 when absent |
| `valid` | `int[n]` | 1: witness; 0: empty node; -1: no witness; -2: nonfinite field SDF |
| `left`, `right` | `double[n]` each | Final bounds |
| `aux3` | `float[Q*3]` | All three observable auxiliary columns per query |
| `trace_sdf` | `float[Q]` | Queried SDF values |
| `trace_xyz` | `float[Q*3]` | Actual float coordinates supplied to the field |
| `trace_sign` | `int[Q]` | Query sign |
| `trace_left`, `trace_right` | `double[(k+1)*n]` each | Initial bounds then bounds after each round |

For `trace=0`, the five trace destination pointers must be NULL; the six ordinary
outputs, including every auxiliary, are still produced. Nonfinite SDF overrides
other validity states. Query order is all centers, then all endpoints in CSR
order for each midpoint round, then all final-right endpoints. Bounds traces are
round-major. SdfTrees uses its native tree-ID auxiliary in column 0 and zero
padding in columns 1 and 2. LandTiles retains its three auxiliary columns.

## Timing and diagnostics

Normal create/prepare/run/readback operations do not create or record timing
events. In an ordinary build, explicit `bm_solver_timing_prepare(...,1,...)`
creates reusable events and reports their host setup time. Both timing entry
points return `cudaErrorNotSupported` in a debug-guard build. Omitting the timing
opt-in returns `cudaErrorNotPermitted` in an otherwise valid ordinary call.

`bm_solver_measure_resident` requires those events, positive repeats and
`allow_timing=1`. It enqueues that many complete trace=0 solves between two stream
events and synchronizes. `event_ms` is the total CUDA-event interval and `wall_ms`
is host submission plus synchronization time; divide by repeats for per-solve
values. The event interval may include device idle gaps awaiting host submissions;
it is not a sum of isolated kernel durations. Allocation, uploads, field setup,
graph preparation, event creation and output readback are outside this interval.
Readback exposes the last solve. Repeats are not independent inputs.

`bm_solver_get_info` writes 40 uint64 values with schema version 1 at index 0.
Important indices are: 1 debug guards; 2/3 n/m; 4/5 max_k/requested k; 6/7
allocated/current Q; 21 graph configuration ready; 24 graph generation count;
31 timing prepared; 32 event records; 33 normal run attempts; 34 completed solves;
35 graph instantiations; 38 ordinary-build indicator; 39 allocation count.
Indices 27..30 contain process-local device pointer values and must not be reused
as portable buffer identifiers. Remaining words describe resource accounting and
field metadata; this is diagnostic metadata, not measured peak device usage.

In a guard build, call `bm_solver_debug_check` only after successful completion.
Its eight values are: schema version, allocations checked, corrupted guard-byte
count, incorrect position/witness/valid write-count totals, incorrect active-query
auxiliary write-count total, and writes outside the current query range. Success
requires schema 1, complete allocation coverage and zero values at indices 2..7.
The function returns `cudaErrorAssert` for a detected violation. Ordinary builds
return `cudaErrorNotSupported`.

## Errors and lifecycle

Every export returns a CUDA runtime error code; zero is success.
`bm_field_error_string` describes a code. Check every return value. A failed create
may return a partially owned handle. Failures return immediately without retrying
or silently issuing cleanup that could obscure the first error. On success,
destroy the solver before its field; destroying NULL is permitted.

The caller owns the failure policy. Preserve the first error and available host
evidence. Do not continue solving or timing in a failed CUDA context. The review
harness exits that worker process on failure; process teardown releases remaining
resources. If an explicitly managed cleanup is appropriate, check its result too:
a failing destroy preserves the remaining handle/resources rather than pretending
that destruction completed. Once destroy succeeds, the handle is invalid.
