#ifndef BINOCMESHER_GPU_SOLVER_API_H
#define BINOCMESHER_GPU_SOLVER_API_H

#include <stddef.h>
#include <stdint.h>
#include "../src/field_types.h"

/* Return values are CUDA runtime error codes; zero means success.
 * Handles are opaque, serially used, and bound to their creation CUDA context.
 * FieldView borrows device parameter storage owned by the field bridge.
 */
enum BMSolverMode {
    BM_SOLVER_MODE_R0 = 0,
    BM_SOLVER_MODE_R1 = 1,
    BM_SOLVER_MODE_L0 = 2,
    BM_SOLVER_MODE_L1_8 = 3,
    BM_SOLVER_MODE_L1_32 = 4
};

enum {
    BM_SOLVER_INFO_WORDS = 40,
    BM_SOLVER_DEBUG_WORDS = 8
};

#ifdef __cplusplus
extern "C" {
#endif

/* Copy CSR geometry and allocate scratch for 0 <= k <= max_k <= 64.
 * A partially initialized, owned handle can be returned on allocation failure.
 * Destroy the solver before destroying the field that owns field's pointers.
 */
int bm_solver_create(FieldView field, int n, int m,
    const int* host_offsets, const double* host_centers,
    const double* host_endpoints, int max_k, void** out_handle);

/* Required before R1 with the same k and trace. Geometry/field stay immutable. */
int bm_solver_prepare_graph(void* handle, int k, int trace);

/* Exactly one complete solve; synchronizes the solver's private CUDA stream.
 * trace is 0 or 1. This entry does not create or record timing events.
 */
int bm_solver_run(void* handle, int mode, int k, int trace);

/* Copy selected outputs of the last successful solve to caller-owned host
 * buffers. Each destination may be NULL; trace destinations must be NULL when
 * the last solve used trace=0. Capacities and element types are in API.md.
 */
int bm_solver_readback(void* handle,
    float* host_position, int* host_witness, int* host_valid,
    double* host_left, double* host_right, float* host_aux3,
    float* host_trace_sdf, float* host_trace_xyz, int* host_trace_sign,
    double* host_trace_left, double* host_trace_right);

/* Host metadata snapshot, schema 1; capacity >= BM_SOLVER_INFO_WORDS. */
int bm_solver_get_info(void* handle, int k, uint64_t* values, size_t capacity);

/* Debug build only; after a successful solve, capacity >= BM_SOLVER_DEBUG_WORDS.
 * Checks allocation redzones and final-output production counts on the device.
 * Ordinary builds return cudaErrorNotSupported.
 */
int bm_solver_debug_check(void* handle, uint64_t* values, size_t capacity);

/* Explicit instrumentation opt-in; allow_timing must be 1. Ordinary build only.
 * Prepare creates reusable events; repeat preparation returns setup_wall_ms=0.
 */
int bm_solver_timing_prepare(void* handle, int allow_timing, double* setup_wall_ms);

/* Execute repeats complete trace=0 solves in one CUDA-event interval and
 * synchronize. Returns total event and host submission/synchronization times
 * in milliseconds, not per-solve averages. repeats must be positive.
 * R1 requires an already prepared graph for (k, trace=0).
 */
int bm_solver_measure_resident(void* handle, int mode, int k,
    int repeats, int allow_timing, double* event_ms, double* wall_ms);

/* NULL is accepted. On a cleanup error, returns immediately; no automatic retry.
 * A successful destroy releases the handle, but not its borrowed FieldView.
 */
int bm_solver_destroy(void* handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BINOCMESHER_GPU_SOLVER_API_H */
