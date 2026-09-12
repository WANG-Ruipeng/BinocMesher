// Timing is an explicit opt-in API. Correctness APIs never record timing events.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <limits>
#include <new>
#include <vector>
#include "field_eval.cuh"

#ifndef BM_L132_BLOCK_THREADS
#define BM_L132_BLOCK_THREADS 256
#endif
static_assert(BM_L132_BLOCK_THREADS == 64 || BM_L132_BLOCK_THREADS == 128 || BM_L132_BLOCK_THREADS == 256, "Frozen block candidates");

#ifndef BM_SOLVER_DEBUG_GUARDS
#define BM_SOLVER_DEBUG_GUARDS 0
#endif

namespace {
struct SolverDevice {
    FieldView field{};
    int n = 0;
    int m = 0;
    int* offsets = nullptr;
    int* owners = nullptr;
    double* centers = nullptr;
    double* endpoints = nullptr;
    double* left = nullptr;
    double* right = nullptr;
    float* center_sdf = nullptr;
    float* query_xyz = nullptr;
    float* query_sdf = nullptr;
    float* position = nullptr;
    int* witness = nullptr;
    int* valid = nullptr;
    float* aux = nullptr;
    float* trace_sdf = nullptr;
    float* trace_xyz = nullptr;
    int* trace_sign = nullptr;
    double* trace_left = nullptr;
    double* trace_right = nullptr;
#if BM_SOLVER_DEBUG_GUARDS
    unsigned* position_writes = nullptr;
    unsigned* witness_writes = nullptr;
    unsigned* valid_writes = nullptr;
    unsigned* aux_writes = nullptr;
#endif
};
struct Allocation {
    void* base = nullptr;
    size_t payload_bytes = 0;
};
constexpr size_t kMaxAllocations = 32;
constexpr size_t kGuardBytes = BM_SOLVER_DEBUG_GUARDS ? 256 : 0;
constexpr unsigned char kGuardValue = 0xa5;
constexpr size_t kInfoWords = 40;
constexpr size_t kDebugWords = 8;
struct SolverHandle {
    SolverDevice d{};
    int max_k = 0;
    bool ready = false;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    int graph_k = -1;
    int graph_trace = -1;
    int last_k = -1;
    int last_trace = 0;
    Allocation allocations[kMaxAllocations]{};
    size_t allocation_count = 0;
    size_t ordinary_bytes = 0;
    size_t trace_bytes = 0;
    size_t count_bytes = 0;
    size_t guard_bytes = 0;
    size_t requested_bytes = 0;
    size_t free_before = 0;
    size_t total_device = 0;
    size_t memory_budget = 0;
    size_t graph_free_before = 0;
    size_t graph_free_after = 0;
    uint64_t graph_generation = 0;
    uint64_t graph_instantiations = 0;
    uint64_t event_records = 0;
    uint64_t normal_runs = 0;
    uint64_t completed_solves = 0;
    cudaEvent_t timing_begin = nullptr;
    cudaEvent_t timing_end = nullptr;
    bool timing_ready = false;
};

#define BM_SOLVER_CUDA(expr) do { \
    const cudaError_t status = (expr); \
    if (status != cudaSuccess) return static_cast<int>(status); \
} while (0)
#define BM_SOLVER_CALL(expr) do { const int status = (expr); if (status != 0) return status; } while (0)

static __device__ __forceinline__ void store_query(SolverDevice d, size_t q,
    float x, float y, float z, float sdf, const float* aux, int trace) {
    d.aux[3 * q] = aux[0];
    d.aux[3 * q + 1] = aux[1];
    d.aux[3 * q + 2] = aux[2];
#if BM_SOLVER_DEBUG_GUARDS
    atomicAdd(d.aux_writes + q, 1u);
#endif
    if (trace) {
        d.trace_xyz[3 * q] = x;
        d.trace_xyz[3 * q + 1] = y;
        d.trace_xyz[3 * q + 2] = z;
        d.trace_sdf[q] = sdf;
        d.trace_sign[q] = sdf >= 0.0f;
    }
}
static __device__ __forceinline__ void store_bounds(SolverDevice d, int i,
    int round, double left, double right, int trace) {
    if (trace) {
        const size_t q = static_cast<size_t>(round) * d.n + i;
        d.trace_left[q] = left;
        d.trace_right[q] = right;
    }
}
static __device__ __forceinline__ float field_at_center(SolverDevice d, int i, int trace) {
    const float x = static_cast<float>(d.centers[3 * static_cast<size_t>(i)]);
    const float y = static_cast<float>(d.centers[3 * static_cast<size_t>(i) + 1]);
    const float z = static_cast<float>(d.centers[3 * static_cast<size_t>(i) + 2]);
    float aux[3];
    const float sdf = bm_eval(d.field, x, y, z, aux);
    store_query(d, i, x, y, z, sdf, aux, trace);
    return sdf;
}
static __device__ __forceinline__ void point_xyz(SolverDevice d, int i, int p,
    double t, float& x, float& y, float& z) {
    const size_t c = 3 * static_cast<size_t>(i);
    const size_t e = 3 * static_cast<size_t>(p);
    // Original bisection.cpp: double center*(1-middle)+endpoint*middle,
    // then the wrapper's float32 conversion. Do not rewrite as center+t*delta.
    x = static_cast<float>(d.centers[c] * (1.0 - t) + d.endpoints[e] * t);
    y = static_cast<float>(d.centers[c + 1] * (1.0 - t) + d.endpoints[e + 1] * t);
    z = static_cast<float>(d.centers[c + 2] * (1.0 - t) + d.endpoints[e + 2] * t);
}
static __device__ __forceinline__ float field_at_endpoint(SolverDevice d, int i,
    int p, double t, int round, int trace) {
    float x, y, z, aux[3];
    point_xyz(d, i, p, t, x, y, z);
    const float sdf = bm_eval(d.field, x, y, z, aux);
    const size_t q = d.n + static_cast<size_t>(round) * d.m + p;
    store_query(d, q, x, y, z, sdf, aux, trace);
    return sdf;
}
static __device__ __forceinline__ void finish_node(SolverDevice d, int i,
    double left, double right, int witness, bool bad) {
    d.left[i] = left;
    d.right[i] = right;
    d.witness[i] = witness;
    d.valid[i] = bad ? -2 : (d.offsets[i] == d.offsets[i + 1] ? 0 : (witness < 0 ? -1 : 1));
#if BM_SOLVER_DEBUG_GUARDS
    // Count final production, not the original reset/intermediate status stores.
    atomicAdd(d.witness_writes + i, 1u);
    atomicAdd(d.valid_writes + i, 1u);
    atomicAdd(d.position_writes + i, 1u);
#endif
    const size_t c = 3 * static_cast<size_t>(i);
    if (witness < 0 || bad) {
        d.position[c] = 0.0f;
        d.position[c + 1] = 0.0f;
        d.position[c + 2] = 0.0f;
        return;
    }
    const size_t e = 3 * static_cast<size_t>(d.offsets[i] + witness);
    // Recompute from double source geometry and final right. The endpoint
    // query's rounded float coordinates are not reused as geometry state.
    for (int axis = 0; axis < 3; ++axis)
        d.position[c + axis] = static_cast<float>(
            d.centers[c + axis] * (1.0 - right) + d.endpoints[e + axis] * right);
}

__global__ void reset_kernel(SolverDevice d, int trace) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.n)) return;
    const int i = static_cast<int>(index);
    d.left[i] = 0.0;
    d.right[i] = 1.0;
    d.witness[i] = -1;
    d.valid[i] = 0;
    for (int axis = 0; axis < 3; ++axis) d.position[3 * static_cast<size_t>(i) + axis] = 0.0f;
    store_bounds(d, i, 0, 0.0, 1.0, trace);
}
__global__ void center_kernel(SolverDevice d, int trace) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.n)) return;
    const int i = static_cast<int>(index);
    const float sdf = field_at_center(d, i, trace);
    d.center_sdf[i] = sdf;
    if (!isfinite(sdf)) d.valid[i] = -2;
}
__global__ void xyz_kernel(SolverDevice d, int finishing) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.m)) return;
    const int p = static_cast<int>(index);
    const int i = d.owners[p];
    const double t = finishing ? d.right[i] : (d.left[i] + d.right[i]) / 2.0;
    float x, y, z;
    point_xyz(d, i, p, t, x, y, z);
    d.query_xyz[3 * static_cast<size_t>(p)] = x;
    d.query_xyz[3 * static_cast<size_t>(p) + 1] = y;
    d.query_xyz[3 * static_cast<size_t>(p) + 2] = z;
}
__global__ void query_field_kernel(SolverDevice d, int round, int trace) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.m)) return;
    const int p = static_cast<int>(index);
    const float x = d.query_xyz[3 * static_cast<size_t>(p)];
    const float y = d.query_xyz[3 * static_cast<size_t>(p) + 1];
    const float z = d.query_xyz[3 * static_cast<size_t>(p) + 2];
    float aux[3];
    const float sdf = bm_eval(d.field, x, y, z, aux);
    d.query_sdf[p] = sdf;
    store_query(d, d.n + static_cast<size_t>(round) * d.m + p, x, y, z, sdf, aux, trace);
}
__global__ void update_kernel(SolverDevice d, int round, int trace) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.n)) return;
    const int i = static_cast<int>(index);
    bool diff = false;
    for (int p = d.offsets[i]; p < d.offsets[i + 1]; ++p) {
        const float sdf = d.query_sdf[p];
        if (!isfinite(sdf)) d.valid[i] = -2;
        diff |= (sdf >= 0.0f) != (d.center_sdf[i] >= 0.0f);
    }
    const double middle = (d.left[i] + d.right[i]) / 2.0;
    if (diff) d.right[i] = middle;
    else d.left[i] = middle;
    store_bounds(d, i, round + 1, d.left[i], d.right[i], trace);
}
__global__ void final_kernel(SolverDevice d) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.n)) return;
    const int i = static_cast<int>(index);
    int witness = -1;
    bool bad = d.valid[i] == -2;
    for (int p = d.offsets[i]; p < d.offsets[i + 1]; ++p) {
        const float sdf = d.query_sdf[p];
        bad |= !isfinite(sdf);
        if (witness == -1 && ((sdf >= 0.0f) != (d.center_sdf[i] >= 0.0f)))
            witness = p - d.offsets[i];
    }
    finish_node(d, i, d.left[i], d.right[i], witness, bad);
}

__global__ void local_thread_kernel(SolverDevice d, int k, int trace) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(d.n)) return;
    const int i = static_cast<int>(index);
    double left = 0.0, right = 1.0;
    store_bounds(d, i, 0, left, right, trace);
    const float center = field_at_center(d, i, trace);
    d.center_sdf[i] = center;
    bool bad = !isfinite(center);
    for (int round = 0; round < k; ++round) {
        const double middle = (left + right) / 2.0;
        bool diff = false;
        for (int p = d.offsets[i]; p < d.offsets[i + 1]; ++p) {
            const float sdf = field_at_endpoint(d, i, p, middle, round, trace);
            bad |= !isfinite(sdf);
            diff |= (sdf >= 0.0f) != (center >= 0.0f);
        }
        if (diff) right = middle;
        else left = middle;
        store_bounds(d, i, round + 1, left, right, trace);
    }
    int witness = -1;
    for (int p = d.offsets[i]; p < d.offsets[i + 1]; ++p) {
        const float sdf = field_at_endpoint(d, i, p, right, k, trace);
        bad |= !isfinite(sdf);
        if (witness == -1 && ((sdf >= 0.0f) != (center >= 0.0f)))
            witness = p - d.offsets[i];
    }
    finish_node(d, i, left, right, witness, bad);
}

template <int GROUP>
__global__ void local_group_kernel(SolverDevice d, int k, int trace) {
    static_assert(GROUP == 8 || GROUP == 32, "Supported node groups are 8 and 32");
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t node_index = tid / GROUP;
    const int lane = threadIdx.x % GROUP;
    // Every node has a whole aligned subgroup. Inactive tail nodes return as
    // complete subgroups; no block barrier and no partial subgroup participate.
    if (node_index >= static_cast<size_t>(d.n)) return;
    const int i = static_cast<int>(node_index);
    const int warp_base = (threadIdx.x % 32) / GROUP * GROUP;
    const unsigned mask = GROUP == 32 ? 0xffffffffu : (0xffu << warp_base);
    double left = 0.0, right = 1.0;
    float center = 0.0f;
    if (lane == 0) {
        store_bounds(d, i, 0, left, right, trace);
        center = field_at_center(d, i, trace);
        d.center_sdf[i] = center;
    }
    center = __shfl_sync(mask, center, 0, GROUP);
    bool bad = !isfinite(center);
    const int begin = d.offsets[i];
    const int count = d.offsets[i + 1] - begin;
    for (int round = 0; round < k; ++round) {
        const double middle = (left + right) / 2.0;
        bool diff = false;
        bool round_bad = false;
        for (size_t rank = lane; rank < static_cast<size_t>(count); rank += GROUP) {
            const float sdf = field_at_endpoint(d, i, static_cast<int>(begin + rank), middle, round, trace);
            round_bad |= !isfinite(sdf);
            diff |= (sdf >= 0.0f) != (center >= 0.0f);
        }
        const bool any_diff = __any_sync(mask, diff);
        bad |= __any_sync(mask, round_bad);
        if (any_diff) right = middle;
        else left = middle;
        if (lane == 0) store_bounds(d, i, round + 1, left, right, trace);
    }
    int witness = 2147483647;
    bool final_bad = false;
    for (size_t rank = lane; rank < static_cast<size_t>(count); rank += GROUP) {
        const float sdf = field_at_endpoint(d, i, static_cast<int>(begin + rank), right, k, trace);
        final_bad |= !isfinite(sdf);
        if ((sdf >= 0.0f) != (center >= 0.0f)) witness = witness < static_cast<int>(rank) ? witness : static_cast<int>(rank);
    }
    bad |= __any_sync(mask, final_bad);
    for (int delta = GROUP / 2; delta > 0; delta /= 2) {
        const int other = __shfl_down_sync(mask, witness, delta, GROUP);
        witness = witness < other ? witness : other;
    }
    if (lane == 0)
        finish_node(d, i, left, right, witness == 2147483647 ? -1 : witness, bad);
}

int enqueue_resident(SolverHandle* handle, int k, int trace) {
    const SolverDevice d = handle->d;
    if (d.n == 0) return static_cast<int>(cudaSuccess);
    const unsigned nb = static_cast<unsigned>((static_cast<size_t>(d.n) + 255) / 256);
    const unsigned mb = static_cast<unsigned>((static_cast<size_t>(d.m) + 255) / 256);
    reset_kernel<<<nb, 256, 0, handle->stream>>>(d, trace);
    BM_SOLVER_CUDA(cudaPeekAtLastError());
    center_kernel<<<nb, 256, 0, handle->stream>>>(d, trace);
    BM_SOLVER_CUDA(cudaPeekAtLastError());
    for (int round = 0; round < k; ++round) {
        if (d.m != 0) {
            xyz_kernel<<<mb, 256, 0, handle->stream>>>(d, 0);
            BM_SOLVER_CUDA(cudaPeekAtLastError());
            query_field_kernel<<<mb, 256, 0, handle->stream>>>(d, round, trace);
            BM_SOLVER_CUDA(cudaPeekAtLastError());
        }
        update_kernel<<<nb, 256, 0, handle->stream>>>(d, round, trace);
        BM_SOLVER_CUDA(cudaPeekAtLastError());
    }
    if (d.m != 0) {
        xyz_kernel<<<mb, 256, 0, handle->stream>>>(d, 1);
        BM_SOLVER_CUDA(cudaPeekAtLastError());
        query_field_kernel<<<mb, 256, 0, handle->stream>>>(d, k, trace);
        BM_SOLVER_CUDA(cudaPeekAtLastError());
    }
    final_kernel<<<nb, 256, 0, handle->stream>>>(d);
    BM_SOLVER_CUDA(cudaPeekAtLastError());
    return static_cast<int>(cudaSuccess);
}

bool checked_add(size_t& sum, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - sum) return false;
    sum += value;
    return true;
}
int plan_allocation(SolverHandle* handle, size_t count, size_t item_bytes,
    bool trace = false, bool debug_count = false) {
    if (count == 0) return static_cast<int>(cudaSuccess);
    if (count > std::numeric_limits<size_t>::max() / item_bytes)
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t bytes = count * item_bytes;
    size_t& category = debug_count ? handle->count_bytes : handle->ordinary_bytes;
    if (!checked_add(category, bytes) ||
        (trace && !checked_add(handle->trace_bytes, bytes)) ||
        !checked_add(handle->guard_bytes, 2 * kGuardBytes) ||
        !checked_add(handle->requested_bytes, bytes) ||
        !checked_add(handle->requested_bytes, 2 * kGuardBytes))
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaSuccess);
}
int plan_memory(SolverHandle* handle, size_t q, size_t b) {
    const size_t n = handle->d.n, m = handle->d.m;
    BM_SOLVER_CALL(plan_allocation(handle, n + 1, sizeof(int)));
    BM_SOLVER_CALL(plan_allocation(handle, m, sizeof(int)));
    BM_SOLVER_CALL(plan_allocation(handle, 3 * n, sizeof(double)));
    BM_SOLVER_CALL(plan_allocation(handle, 3 * m, sizeof(double)));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(double)));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(double)));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(float)));
    BM_SOLVER_CALL(plan_allocation(handle, 3 * m, sizeof(float)));
    BM_SOLVER_CALL(plan_allocation(handle, m, sizeof(float)));
    BM_SOLVER_CALL(plan_allocation(handle, 3 * n, sizeof(float)));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(int)));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(int)));
    BM_SOLVER_CALL(plan_allocation(handle, 3 * q, sizeof(float)));
    BM_SOLVER_CALL(plan_allocation(handle, q, sizeof(float), true));
    BM_SOLVER_CALL(plan_allocation(handle, 3 * q, sizeof(float), true));
    BM_SOLVER_CALL(plan_allocation(handle, q, sizeof(int), true));
    BM_SOLVER_CALL(plan_allocation(handle, b, sizeof(double), true));
    BM_SOLVER_CALL(plan_allocation(handle, b, sizeof(double), true));
#if BM_SOLVER_DEBUG_GUARDS
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(unsigned), false, true));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(unsigned), false, true));
    BM_SOLVER_CALL(plan_allocation(handle, n, sizeof(unsigned), false, true));
    BM_SOLVER_CALL(plan_allocation(handle, q, sizeof(unsigned), false, true));
#endif
    // This preflight precedes stream creation and every solver cudaMalloc.
    // Field buffers already owned by the caller are reflected in free_before.
    BM_SOLVER_CUDA(cudaMemGetInfo(&handle->free_before, &handle->total_device));
    handle->memory_budget = (handle->free_before / 10) * 7 +
                            ((handle->free_before % 10) * 7) / 10;
    if (handle->requested_bytes > handle->memory_budget)
        return static_cast<int>(cudaErrorMemoryAllocation);
    return static_cast<int>(cudaSuccess);
}
template <class T> int allocate(SolverHandle* handle, T** out, size_t count) {
    if (count == 0) return static_cast<int>(cudaSuccess);
    if (count > std::numeric_limits<size_t>::max() / sizeof(T) ||
        handle->allocation_count >= kMaxAllocations)
        return static_cast<int>(cudaErrorInvalidValue);
    const size_t bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<size_t>::max() - 2 * kGuardBytes)
        return static_cast<int>(cudaErrorInvalidValue);
    void* base = nullptr;
    BM_SOLVER_CUDA(cudaMalloc(&base, bytes + 2 * kGuardBytes));
    Allocation& allocation = handle->allocations[handle->allocation_count++];
    allocation.base = base;
    allocation.payload_bytes = bytes;
    unsigned char* payload = static_cast<unsigned char*>(base) + kGuardBytes;
    *out = reinterpret_cast<T*>(payload);
#if BM_SOLVER_DEBUG_GUARDS
    BM_SOLVER_CUDA(cudaMemsetAsync(base, kGuardValue, kGuardBytes, handle->stream));
    BM_SOLVER_CUDA(cudaMemsetAsync(payload + bytes, kGuardValue, kGuardBytes, handle->stream));
#endif
    return static_cast<int>(cudaSuccess);
}
#if BM_SOLVER_DEBUG_GUARDS
int debug_begin(SolverHandle* handle) {
    const SolverDevice& d = handle->d;
    const size_t n = d.n;
    const size_t q = n + (static_cast<size_t>(handle->max_k) + 1) * d.m;
    if (n != 0) {
        BM_SOLVER_CUDA(cudaMemsetAsync(d.position_writes, 0, n * sizeof(unsigned), handle->stream));
        BM_SOLVER_CUDA(cudaMemsetAsync(d.witness_writes, 0, n * sizeof(unsigned), handle->stream));
        BM_SOLVER_CUDA(cudaMemsetAsync(d.valid_writes, 0, n * sizeof(unsigned), handle->stream));
        BM_SOLVER_CUDA(cudaMemsetAsync(d.position, 0xcd, 3 * n * sizeof(float), handle->stream));
        BM_SOLVER_CUDA(cudaMemsetAsync(d.witness, 0xcd, n * sizeof(int), handle->stream));
        BM_SOLVER_CUDA(cudaMemsetAsync(d.valid, 0xcd, n * sizeof(int), handle->stream));
    }
    if (q != 0) {
        BM_SOLVER_CUDA(cudaMemsetAsync(d.aux_writes, 0, q * sizeof(unsigned), handle->stream));
        BM_SOLVER_CUDA(cudaMemsetAsync(d.aux, 0xcd, 3 * q * sizeof(float), handle->stream));
    }
    // Redzones are initialized once at allocation, never repaired between runs.
    return static_cast<int>(cudaSuccess);
}
#endif
int enqueue_mode(SolverHandle* handle, int mode, int k, int trace) {
    if (mode == 0) return enqueue_resident(handle, k, trace);
    if (mode == 1) {
        if (handle->d.n != 0)
            BM_SOLVER_CUDA(cudaGraphLaunch(handle->graph_exec, handle->stream));
    } else if (handle->d.n != 0) {
        const size_t threads = static_cast<size_t>(handle->d.n) * (mode == 3 ? 8 : mode == 4 ? 32 : 1);
        const unsigned block_threads = mode == 4 ? BM_L132_BLOCK_THREADS : 256;
        const unsigned blocks = static_cast<unsigned>((threads + block_threads - 1) / block_threads);
        if (mode == 2) local_thread_kernel<<<blocks, 256, 0, handle->stream>>>(handle->d, k, trace);
        else if (mode == 3) local_group_kernel<8><<<blocks, 256, 0, handle->stream>>>(handle->d, k, trace);
        else local_group_kernel<32><<<blocks, block_threads, 0, handle->stream>>>(handle->d, k, trace);
        BM_SOLVER_CUDA(cudaPeekAtLastError());
    }
    return static_cast<int>(cudaSuccess);
}
template <class T> int upload(T* dest, const T* source, size_t count, cudaStream_t stream) {
    if (count == 0) return static_cast<int>(cudaSuccess);
    BM_SOLVER_CUDA(cudaMemcpyAsync(dest, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
    return static_cast<int>(cudaSuccess);
}
template <class T> int download(T* dest, const T* source, size_t count, cudaStream_t stream) {
    if (dest == nullptr || count == 0) return static_cast<int>(cudaSuccess);
    BM_SOLVER_CUDA(cudaMemcpyAsync(dest, source, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    return static_cast<int>(cudaSuccess);
}
bool valid_run(SolverHandle* handle, int k, int trace) {
    return handle != nullptr && handle->ready && k >= 0 && k <= handle->max_k &&
           (trace == 0 || trace == 1);
}
}  // anonymous namespace

extern "C" int bm_solver_create(FieldView field, int n, int m,
    const int* offsets, const double* centers, const double* endpoints,
    int max_k, void** out_handle) {
    if (out_handle == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    *out_handle = nullptr;
    if (n < 0 || m < 0 || max_k < 0 || max_k > 64 || offsets == nullptr ||
        (n == 0 && m != 0) || (n != 0 && centers == nullptr) || (m != 0 && endpoints == nullptr) ||
        field.kind < 0 || field.kind > 4 || field.fp == nullptr ||
        (field.kind == 3 && field.ip == nullptr) ||
        (field.kind == 4 && (field.ip == nullptr || field.ip2 == nullptr || field.fp2 == nullptr)) ||
        (field.kind == 3 && field.meta != 0 && (field.ip2 == nullptr || field.fp2 == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (offsets[0] != 0 || offsets[n] != m) return static_cast<int>(cudaErrorInvalidValue);
    for (int i = 0; i < n; ++i)
        if (offsets[i] < 0 || offsets[i] > offsets[i + 1]) return static_cast<int>(cudaErrorInvalidValue);
    for (size_t i = 0; i < 3 * static_cast<size_t>(n); ++i)
        if (!std::isfinite(centers[i])) return static_cast<int>(cudaErrorInvalidValue);
    for (size_t i = 0; i < 3 * static_cast<size_t>(m); ++i)
        if (!std::isfinite(endpoints[i])) return static_cast<int>(cudaErrorInvalidValue);
    const size_t q = static_cast<size_t>(n) + (static_cast<size_t>(max_k) + 1) * m;
    const size_t b = (static_cast<size_t>(max_k) + 1) * n;
    if (q > std::numeric_limits<size_t>::max() / (3 * sizeof(float)) ||
        b > std::numeric_limits<size_t>::max() / sizeof(double))
        return static_cast<int>(cudaErrorInvalidValue);
    SolverHandle* handle = new (std::nothrow) SolverHandle;
    if (handle == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    *out_handle = handle;
    handle->d.field = field;
    handle->d.n = n;
    handle->d.m = m;
    handle->max_k = max_k;
    BM_SOLVER_CALL(plan_memory(handle, q, b));
    BM_SOLVER_CUDA(cudaStreamCreateWithFlags(&handle->stream, cudaStreamNonBlocking));
    SolverDevice& d = handle->d;
    BM_SOLVER_CALL(allocate(handle, &d.offsets, static_cast<size_t>(n) + 1));
    BM_SOLVER_CALL(allocate(handle, &d.owners, m));
    BM_SOLVER_CALL(allocate(handle, &d.centers, 3 * static_cast<size_t>(n)));
    BM_SOLVER_CALL(allocate(handle, &d.endpoints, 3 * static_cast<size_t>(m)));
    BM_SOLVER_CALL(allocate(handle, &d.left, n));
    BM_SOLVER_CALL(allocate(handle, &d.right, n));
    BM_SOLVER_CALL(allocate(handle, &d.center_sdf, n));
    BM_SOLVER_CALL(allocate(handle, &d.query_xyz, 3 * static_cast<size_t>(m)));
    BM_SOLVER_CALL(allocate(handle, &d.query_sdf, m));
    BM_SOLVER_CALL(allocate(handle, &d.position, 3 * static_cast<size_t>(n)));
    BM_SOLVER_CALL(allocate(handle, &d.witness, n));
    BM_SOLVER_CALL(allocate(handle, &d.valid, n));
    BM_SOLVER_CALL(allocate(handle, &d.aux, 3 * q));
    BM_SOLVER_CALL(allocate(handle, &d.trace_sdf, q));
    BM_SOLVER_CALL(allocate(handle, &d.trace_xyz, 3 * q));
    BM_SOLVER_CALL(allocate(handle, &d.trace_sign, q));
    BM_SOLVER_CALL(allocate(handle, &d.trace_left, b));
    BM_SOLVER_CALL(allocate(handle, &d.trace_right, b));
#if BM_SOLVER_DEBUG_GUARDS
    BM_SOLVER_CALL(allocate(handle, &d.position_writes, n));
    BM_SOLVER_CALL(allocate(handle, &d.witness_writes, n));
    BM_SOLVER_CALL(allocate(handle, &d.valid_writes, n));
    BM_SOLVER_CALL(allocate(handle, &d.aux_writes, q));
#endif
    std::vector<int> owners;
    try { owners.resize(m); }
    catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
    for (int i = 0; i < n; ++i)
        for (int p = offsets[i]; p < offsets[i + 1]; ++p) owners[p] = i;
    BM_SOLVER_CALL(upload(d.offsets, offsets, static_cast<size_t>(n) + 1, handle->stream));
    BM_SOLVER_CALL(upload(d.owners, owners.data(), m, handle->stream));
    BM_SOLVER_CALL(upload(d.centers, centers, 3 * static_cast<size_t>(n), handle->stream));
    BM_SOLVER_CALL(upload(d.endpoints, endpoints, 3 * static_cast<size_t>(m), handle->stream));
    BM_SOLVER_CUDA(cudaStreamSynchronize(handle->stream));
    handle->ready = true;
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_solver_prepare_graph(void* opaque, int k, int trace) {
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (!valid_run(handle, k, trace)) return static_cast<int>(cudaErrorInvalidValue);
    if (handle->graph_k == k && handle->graph_trace == trace) return static_cast<int>(cudaSuccess);
    if (handle->graph_exec != nullptr) {
        BM_SOLVER_CUDA(cudaGraphExecDestroy(handle->graph_exec));
        handle->graph_exec = nullptr;
    }
    if (handle->graph != nullptr) {
        BM_SOLVER_CUDA(cudaGraphDestroy(handle->graph));
        handle->graph = nullptr;
    }
    handle->graph_k = -1;
    handle->graph_trace = -1;
    size_t total_memory = 0;
    BM_SOLVER_CUDA(cudaMemGetInfo(&handle->graph_free_before, &total_memory));
    if (handle->d.n != 0) {
        BM_SOLVER_CUDA(cudaStreamBeginCapture(handle->stream, cudaStreamCaptureModeThreadLocal));
        BM_SOLVER_CALL(enqueue_resident(handle, k, trace));
        BM_SOLVER_CUDA(cudaStreamEndCapture(handle->stream, &handle->graph));
        BM_SOLVER_CUDA(cudaGraphInstantiate(&handle->graph_exec, handle->graph, 0));
        ++handle->graph_instantiations;
    }
    BM_SOLVER_CUDA(cudaMemGetInfo(&handle->graph_free_after, &total_memory));
    ++handle->graph_generation;
    handle->graph_k = k;
    handle->graph_trace = trace;
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_solver_run(void* opaque, int mode, int k, int trace) {
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (!valid_run(handle, k, trace) || mode < 0 || mode > 4)
        return static_cast<int>(cudaErrorInvalidValue);
    if (mode == 1 && (handle->graph_k != k || handle->graph_trace != trace))
        return static_cast<int>(cudaErrorInvalidValue);
    handle->last_k = -1;
    ++handle->normal_runs;
#if BM_SOLVER_DEBUG_GUARDS
    BM_SOLVER_CALL(debug_begin(handle));
#endif
    BM_SOLVER_CALL(enqueue_mode(handle, mode, k, trace));
    BM_SOLVER_CUDA(cudaStreamSynchronize(handle->stream));
    handle->last_k = k;
    handle->last_trace = trace;
    ++handle->completed_solves;
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_solver_get_info(void* opaque, int k, uint64_t* values, size_t capacity) {
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (handle == nullptr || k < 0 || k > handle->max_k ||
        values == nullptr || capacity < kInfoWords)
        return static_cast<int>(cudaErrorInvalidValue);
    const SolverDevice& d = handle->d;
    const uint64_t qcap = static_cast<uint64_t>(d.n) +
                         (static_cast<uint64_t>(handle->max_k) + 1) * d.m;
    const uint64_t q = static_cast<uint64_t>(d.n) + (static_cast<uint64_t>(k) + 1) * d.m;
    const uint64_t native_columns = d.field.kind == 3 ? 3 : d.field.kind == 4 ? 1 : 0;
    const uint64_t absent = std::numeric_limits<uint64_t>::max();
    const uint64_t info[kInfoWords] = {
        1, BM_SOLVER_DEBUG_GUARDS ? 1u : 0u,
        static_cast<uint64_t>(d.n), static_cast<uint64_t>(d.m),
        static_cast<uint64_t>(handle->max_k), static_cast<uint64_t>(k),
        qcap, q, handle->ordinary_bytes, handle->trace_bytes, handle->count_bytes,
        handle->guard_bytes, handle->requested_bytes, handle->free_before,
        handle->total_device, handle->memory_budget,
        3 * sizeof(float) * qcap, 3 * sizeof(float) * q, native_columns,
        native_columns * sizeof(float) * q,
        (3 - native_columns) * sizeof(float) * q,
        handle->graph_k >= 0 ? 1u : 0u,
        handle->graph_k < 0 ? absent : static_cast<uint64_t>(handle->graph_k),
        handle->graph_trace < 0 ? absent : static_cast<uint64_t>(handle->graph_trace),
        handle->graph_generation, static_cast<uint64_t>(d.field.kind),
        static_cast<uint64_t>(d.field.meta),
        reinterpret_cast<uintptr_t>(d.field.ip), reinterpret_cast<uintptr_t>(d.field.fp),
        reinterpret_cast<uintptr_t>(d.field.ip2), reinterpret_cast<uintptr_t>(d.field.fp2),
        handle->timing_ready ? 1u : 0u, handle->event_records, handle->normal_runs,
        handle->completed_solves, handle->graph_instantiations,
        handle->graph_free_before, handle->graph_free_after,
        BM_SOLVER_DEBUG_GUARDS ? 0u : 1u, handle->allocation_count
    };
    std::copy(info, info + kInfoWords, values);
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_solver_debug_check(void* opaque, uint64_t* values, size_t capacity) {
    if (values == nullptr || capacity < kDebugWords)
        return static_cast<int>(cudaErrorInvalidValue);
    std::fill(values, values + kDebugWords, uint64_t(0));
#if !BM_SOLVER_DEBUG_GUARDS
    (void)opaque;
    return static_cast<int>(cudaErrorNotSupported);
#else
    values[0] = 1;
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (handle == nullptr || !handle->ready || handle->last_k < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    unsigned char guard[2 * kGuardBytes];
    for (size_t i = 0; i < handle->allocation_count; ++i) {
        const Allocation& a = handle->allocations[i];
        const unsigned char* base = static_cast<const unsigned char*>(a.base);
        BM_SOLVER_CUDA(cudaMemcpy(guard, base, kGuardBytes, cudaMemcpyDeviceToHost));
        BM_SOLVER_CUDA(cudaMemcpy(guard + kGuardBytes,
            base + kGuardBytes + a.payload_bytes, kGuardBytes, cudaMemcpyDeviceToHost));
        ++values[1];
        for (size_t j = 0; j < 2 * kGuardBytes; ++j)
            values[2] += guard[j] != kGuardValue;
    }
    const size_t n = handle->d.n;
    const size_t qcap = n + (static_cast<size_t>(handle->max_k) + 1) * handle->d.m;
    const size_t q = n + (static_cast<size_t>(handle->last_k) + 1) * handle->d.m;
    std::vector<unsigned> counts;
    try { counts.resize(qcap); }
    catch (const std::bad_alloc&) { return static_cast<int>(cudaErrorMemoryAllocation); }
    const unsigned* node_counts[3] = {
        handle->d.position_writes, handle->d.witness_writes, handle->d.valid_writes
    };
    if (n != 0) {
        for (size_t j = 0; j < 3; ++j) {
            BM_SOLVER_CUDA(cudaMemcpy(counts.data(), node_counts[j],
                                     n * sizeof(unsigned), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i) values[3 + j] += counts[i] != 1u;
        }
    }
    if (qcap != 0) {
        BM_SOLVER_CUDA(cudaMemcpy(counts.data(), handle->d.aux_writes,
                                 qcap * sizeof(unsigned), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < q; ++i) values[6] += counts[i] != 1u;
        for (size_t i = q; i < qcap; ++i) values[7] += counts[i] != 0u;
    }
    for (size_t i = 2; i < kDebugWords; ++i)
        if (values[i] != 0) return static_cast<int>(cudaErrorAssert);
    return static_cast<int>(cudaSuccess);
#endif
}

// Explicit opt-in only. Normal create/prepare_graph/run/debug/readback paths
// contain neither chrono::now nor cudaEventCreate/Record/ElapsedTime calls.
extern "C" int bm_solver_timing_prepare(void* opaque, int allow_timing, double* setup_wall_ms) {
    if (setup_wall_ms == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    *setup_wall_ms = std::numeric_limits<double>::quiet_NaN();
#if BM_SOLVER_DEBUG_GUARDS
    (void)opaque; (void)allow_timing;
    return static_cast<int>(cudaErrorNotSupported);
#else
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (handle == nullptr || !handle->ready) return static_cast<int>(cudaErrorInvalidValue);
    if (allow_timing != 1) return static_cast<int>(cudaErrorNotPermitted);
    if (handle->timing_ready) {
        *setup_wall_ms = 0.0;
        return static_cast<int>(cudaSuccess);
    }
    const auto begin = std::chrono::steady_clock::now();
    if (handle->timing_begin == nullptr)
        BM_SOLVER_CUDA(cudaEventCreate(&handle->timing_begin));
    if (handle->timing_end == nullptr)
        BM_SOLVER_CUDA(cudaEventCreate(&handle->timing_end));
    handle->timing_ready = true;
    const auto end = std::chrono::steady_clock::now();
    *setup_wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return static_cast<int>(cudaSuccess);
#endif
}

extern "C" int bm_solver_measure_resident(void* opaque, int mode, int k,
    int repeats, int allow_timing, double* event_ms, double* wall_ms) {
    if (event_ms == nullptr || wall_ms == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    *event_ms = std::numeric_limits<double>::quiet_NaN();
    *wall_ms = std::numeric_limits<double>::quiet_NaN();
#if BM_SOLVER_DEBUG_GUARDS
    (void)opaque; (void)mode; (void)k; (void)repeats; (void)allow_timing;
    return static_cast<int>(cudaErrorNotSupported);
#else
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (!valid_run(handle, k, 0) || mode < 0 || mode > 4 || repeats <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (allow_timing != 1) return static_cast<int>(cudaErrorNotPermitted);
    if (!handle->timing_ready ||
        (mode == 1 && (handle->graph_k != k || handle->graph_trace != 0)) ||
        static_cast<uint64_t>(repeats) >
            std::numeric_limits<uint64_t>::max() - handle->completed_solves)
        return static_cast<int>(cudaErrorInvalidValue);
    handle->last_k = -1;
    const auto begin = std::chrono::steady_clock::now();
    BM_SOLVER_CUDA(cudaEventRecord(handle->timing_begin, handle->stream));
    ++handle->event_records;
    for (int repeat = 0; repeat < repeats; ++repeat)
        BM_SOLVER_CALL(enqueue_mode(handle, mode, k, 0));
    BM_SOLVER_CUDA(cudaEventRecord(handle->timing_end, handle->stream));
    ++handle->event_records;
    BM_SOLVER_CUDA(cudaStreamSynchronize(handle->stream));
    const auto end = std::chrono::steady_clock::now();
    float elapsed_ms = 0.0f;
    BM_SOLVER_CUDA(cudaEventElapsedTime(&elapsed_ms, handle->timing_begin, handle->timing_end));
    *event_ms = static_cast<double>(elapsed_ms);
    *wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    handle->last_k = k;
    handle->last_trace = 0;
    handle->completed_solves += static_cast<uint64_t>(repeats);
    return static_cast<int>(cudaSuccess);
#endif
}

extern "C" int bm_solver_readback(void* opaque, float* position, int* witness,
    int* valid, double* left, double* right, float* aux3, float* trace_sdf,
    float* trace_xyz, int* trace_sign, double* trace_left, double* trace_right) {
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (handle == nullptr || !handle->ready || handle->last_k < 0 ||
        (!handle->last_trace && (trace_sdf != nullptr || trace_xyz != nullptr || trace_sign != nullptr ||
                                trace_left != nullptr || trace_right != nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    const SolverDevice& d = handle->d;
    const size_t q = d.n + (static_cast<size_t>(handle->last_k) + 1) * d.m;
    const size_t b = (static_cast<size_t>(handle->last_k) + 1) * d.n;
    BM_SOLVER_CALL(download(position, d.position, 3 * static_cast<size_t>(d.n), handle->stream));
    BM_SOLVER_CALL(download(witness, d.witness, d.n, handle->stream));
    BM_SOLVER_CALL(download(valid, d.valid, d.n, handle->stream));
    BM_SOLVER_CALL(download(left, d.left, d.n, handle->stream));
    BM_SOLVER_CALL(download(right, d.right, d.n, handle->stream));
    BM_SOLVER_CALL(download(aux3, d.aux, 3 * q, handle->stream));
    BM_SOLVER_CALL(download(trace_sdf, d.trace_sdf, q, handle->stream));
    BM_SOLVER_CALL(download(trace_xyz, d.trace_xyz, 3 * q, handle->stream));
    BM_SOLVER_CALL(download(trace_sign, d.trace_sign, q, handle->stream));
    BM_SOLVER_CALL(download(trace_left, d.trace_left, b, handle->stream));
    BM_SOLVER_CALL(download(trace_right, d.trace_right, b, handle->stream));
    BM_SOLVER_CUDA(cudaStreamSynchronize(handle->stream));
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_solver_destroy(void* opaque) {
    SolverHandle* handle = static_cast<SolverHandle*>(opaque);
    if (handle == nullptr) return static_cast<int>(cudaSuccess);
    handle->ready = false;
    if (handle->stream != nullptr) BM_SOLVER_CUDA(cudaStreamSynchronize(handle->stream));
    if (handle->timing_begin != nullptr) {
        BM_SOLVER_CUDA(cudaEventDestroy(handle->timing_begin));
        handle->timing_begin = nullptr;
    }
    if (handle->timing_end != nullptr) {
        BM_SOLVER_CUDA(cudaEventDestroy(handle->timing_end));
        handle->timing_end = nullptr;
    }
    if (handle->graph_exec != nullptr) {
        BM_SOLVER_CUDA(cudaGraphExecDestroy(handle->graph_exec));
        handle->graph_exec = nullptr;
    }
    if (handle->graph != nullptr) {
        BM_SOLVER_CUDA(cudaGraphDestroy(handle->graph));
        handle->graph = nullptr;
    }
    for (size_t i = 0; i < handle->allocation_count; ++i) {
        Allocation& allocation = handle->allocations[i];
        if (allocation.base != nullptr) {
            BM_SOLVER_CUDA(cudaFree(allocation.base));
            allocation.base = nullptr;
        }
    }
    if (handle->stream != nullptr) {
        BM_SOLVER_CUDA(cudaStreamDestroy(handle->stream));
        handle->stream = nullptr;
    }
    delete handle;
    return static_cast<int>(cudaSuccess);
}

#undef BM_SOLVER_CALL
#undef BM_SOLVER_CUDA