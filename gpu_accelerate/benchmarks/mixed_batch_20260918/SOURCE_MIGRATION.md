# Source migration and backend closure

The integration starts from the measured distinct-task host transform, then
removes experiment-directory coupling. It does not claim that the entire old
and new host adapters are byte-identical. Original source hashes and sizes are
in SOURCE_MIGRATION.json and EVIDENCE_MANIFEST.json. All original paths there
are relative to explicit external roots.

## Historical executed chain

mixed_worker.py PreparedGroup/entry_unit
-> mixed_common.py pack_jobs/split_outputs
-> external typed_api.py bindings
-> ti_field_create / ti_solver_create / ti_solver_prepare / ti_solver_run /
ti_solver_readback / ti_solver_trace / ti_solver_validation /
ti_solver_counts / ti_solver_info / ti_solver_destroy
-> libthree_ideas_production.so, SHA256
e79a7395cbf0b19e04b0a626bcbf5df2a22f908d0983bef6ee431e36031f9dea
-> original graph_fix_r2 BUILD_PLAN/BUILD_STATUS/SOURCE_RECHECK/INCLUDE_CLOSURE
-> three_ideas_core.cu including the fixed published D4 numerical source.

The historical D1_PUBLISHED method used variant 1 and the original enqueue.
D1_GRAPH added a captured execution path with historical Graph lifecycle and
query helpers. Historical ti_* added checked token/context/memory/result
ownership around the published source; the name D1_PUBLISHED did not imply
a line-identical published host wrapper. The historical worker also used
input_loader.py for real binary inputs, pilot.py for existing gate/hash
helpers, and experiment_supervisor.py for process supervision. Those old
experiment modules are not runtime imports of the packaged feature.

## New packaged Published chain

batching.runner.solve_many (batching=off by default; offline is explicit)
-> batching.schema task/binding validation
-> batching.pack.pack_tasks / split_outputs, or the independent serial path
-> batching.backend.PublishedBackend / Field / Solver
-> mb_field_* / mb_solver_* in batching/native/published_api.h
-> libbatching_published.so from build_batching.py
-> batching/native/published_adapter.cu and field_bridge_guarded.cu
-> repository src/d4_solver.cu / d4_pipeline.cuh / d4_field_eval.cuh,
   original gpu_solver/field_eval/field_types/field_bridge source.

mb_solver_run calls d4_solver_run with variant=1, checks its return and the
original device validation, and exposes outputs only after success.
Preparation/run attempts invalidate previous result identity. mb_solver_* and
the Python wrapper check capacities, ownership/context, generations and
cleanup. The Field's device parameters stay immutable while any solver borrows
them. New native wrappers are separate files; numerical kernel/field bodies
remain unchanged. The shared published D4 source still includes its preexisting
D2/D3 internals, but this feature exports only the Published variant-1 entry.
It does not import the historical A/B/C research route sources.

Memory is accounted through native/memory_guard.h across the Field and every
active solver. Per-handle plans and global tracked values are distinct.
Graph reservation is zero for this packaged Published backend. Domain
admission from the original field_admission.h is migrated into
native/validated_domain.h; dropping it would broaden unsupported field/geometry
domains and is not part of this integration.

Explicit build dependencies are the existing nvcc, host C++ compiler, CUDA
runtime/CUB and driver library, and external Infinigen headers. Scene parameters
and jobs are caller-supplied runtime inputs. The new builder writes a new DSO,
build commands, source/include identities, compiler output and binary hash.
The old DSO is provenance only, not a substitute for the new clean build.

## Graph boundary

Graph is OPTIONAL_BACKEND_NOT_PACKAGED. An explicit Graph request raises a
clear unsupported-backend error rather than falling back silently. The
historical Graph positive result, A/A and all timing rows remain in the public
archive; it does not authorize a new Graph API or substitute for Published
ready-host performance.

## New validation versus historical measurements

The candidate Git tree, clean build logs and finite regression records provide
new source/binary identities. REFERENCE_HASHES.json permits a comparator-only
numerical bridge to the original ten tasks without loading an old library.
Neither those hashes nor expected arrays may enter a solver. Clean-tree smoke
and new production/audit memcheck do not rerun 256 timing arms and do not
recertify the historical speed ratios. Historical failed research algorithms
retain their prior status.
