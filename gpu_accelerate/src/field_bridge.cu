#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include "field_eval.cuh"

namespace {
struct FieldHandle {
    FieldView field{};
    bool ready = false;
    // Retain partially allocated host-bridge scratch after any failure. The
    // supervisor may call destroy explicitly; no failure is silently retried.
    float* transient = nullptr;
};

#define BM_CUDA_OR_RETURN(expr) do { \
    const cudaError_t bm_status = (expr); \
    if (bm_status != cudaSuccess) return static_cast<int>(bm_status); \
} while (0)

bool valid_blob(size_t count, const void* ptr, size_t item_size) {
    return (count == 0 || ptr != nullptr) &&
           count <= std::numeric_limits<size_t>::max() / item_size;
}

bool valid_point_count(size_t n) {
    // Grid x is bounded by the CUDA architecture limit 2^31-1 blocks.
    return n <= std::numeric_limits<size_t>::max() / (7 * sizeof(float)) &&
           (n / 256 + (n % 256 != 0)) <= 2147483647ULL;
}

template <class T>
int copy_parameter(size_t count, const T* source, T** target) {
    if (count == 0) return static_cast<int>(cudaSuccess);
    BM_CUDA_OR_RETURN(cudaMalloc(reinterpret_cast<void**>(target), count * sizeof(T)));
    BM_CUDA_OR_RETURN(cudaMemcpy(*target, source, count * sizeof(T), cudaMemcpyHostToDevice));
    return static_cast<int>(cudaSuccess);
}

template <class T>
int free_parameter(T** pointer) {
    if (*pointer == nullptr) return static_cast<int>(cudaSuccess);
    BM_CUDA_OR_RETURN(cudaFree(*pointer));
    *pointer = nullptr;
    return static_cast<int>(cudaSuccess);
}

__global__ void bm_field_batch_kernel(FieldView field, size_t n,
                                     const float* xyz, float* sdf, float* aux3) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float aux[3];
    sdf[p] = bm_eval(field, xyz[3 * p], xyz[3 * p + 1], xyz[3 * p + 2], aux);
    aux3[3 * p] = aux[0];
    aux3[3 * p + 1] = aux[1];
    aux3[3 * p + 2] = aux[2];
}
}  // anonymous namespace

extern "C" int bm_field_create(int kind, int meta,
                                 size_t ni, const int* ip, size_t nf, const float* fp,
                                 size_t ni2, const int* ip2, size_t nf2, const float* fp2,
                                 void** out_handle) {
    if (out_handle == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    *out_handle = nullptr;
    if (kind < 0 || kind > 4 ||
        !valid_blob(ni, ip, sizeof(int)) || !valid_blob(nf, fp, sizeof(float)) ||
        !valid_blob(ni2, ip2, sizeof(int)) || !valid_blob(nf2, fp2, sizeof(float)))
        return static_cast<int>(cudaErrorInvalidValue);
    if ((kind <= 1 && nf < 4) || (kind == 2 && nf < 8) ||
        (kind == 3 && (ni < 8 || nf < 18)) ||
        (kind == 3 && meta != 0 && (ni2 < 5 || nf2 < 9)) ||
        (kind == 4 && (ni < 2 || nf < 12 || ni2 < 8 || nf2 < 18)))
        return static_cast<int>(cudaErrorInvalidValue);
    FieldHandle* handle = new (std::nothrow) FieldHandle;
    if (handle == nullptr) return static_cast<int>(cudaErrorMemoryAllocation);
    *out_handle = handle;
    handle->field.kind = kind;
    handle->field.meta = meta;
    // Fail-fast, leaving a partial owned handle for explicit cleanup.
    int result = copy_parameter(ni, ip, &handle->field.ip);
    if (result != 0) return result;
    result = copy_parameter(nf, fp, &handle->field.fp);
    if (result != 0) return result;
    result = copy_parameter(ni2, ip2, &handle->field.ip2);
    if (result != 0) return result;
    result = copy_parameter(nf2, fp2, &handle->field.fp2);
    if (result != 0) return result;
    handle->ready = true;
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_field_get_view(void* opaque, FieldView* out_view) {
    FieldHandle* handle = static_cast<FieldHandle*>(opaque);
    if (handle == nullptr || !handle->ready || out_view == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    *out_view = handle->field;
    return static_cast<int>(cudaSuccess);
}

extern "C" int bm_field_eval_device(void* opaque, size_t n, const float* device_xyz,
                                      float* device_sdf, float* device_aux3, void* stream) {
    FieldHandle* handle = static_cast<FieldHandle*>(opaque);
    if (handle == nullptr || !handle->ready || !valid_point_count(n) ||
        (n != 0 && (device_xyz == nullptr || device_sdf == nullptr || device_aux3 == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return static_cast<int>(cudaSuccess);
    const unsigned int blocks = static_cast<unsigned int>(n / 256 + (n % 256 != 0));
    bm_field_batch_kernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        handle->field, n, device_xyz, device_sdf, device_aux3);
    // This reports launch errors only. The caller must check the same stream
    // at its explicit completion boundary to detect asynchronous errors.
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int bm_field_eval_host(void* opaque, size_t n, const float* host_xyz,
                                    float* host_sdf, float* host_aux3, void* stream) {
    FieldHandle* handle = static_cast<FieldHandle*>(opaque);
    if (handle == nullptr || !handle->ready || handle->transient != nullptr ||
        !valid_point_count(n) ||
        (n != 0 && (host_xyz == nullptr || host_sdf == nullptr || host_aux3 == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return static_cast<int>(cudaSuccess);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    BM_CUDA_OR_RETURN(cudaMalloc(reinterpret_cast<void**>(&handle->transient),
                                 7 * n * sizeof(float)));
    float* device_xyz = handle->transient;
    float* device_sdf = device_xyz + 3 * n;
    float* device_aux3 = device_sdf + n;
    BM_CUDA_OR_RETURN(cudaMemcpyAsync(device_xyz, host_xyz, 3 * n * sizeof(float),
                                      cudaMemcpyHostToDevice, cuda_stream));
    const int launch_status = bm_field_eval_device(opaque, n, device_xyz,
                                                   device_sdf, device_aux3, stream);
    if (launch_status != 0) return launch_status;
    BM_CUDA_OR_RETURN(cudaMemcpyAsync(host_sdf, device_sdf, n * sizeof(float),
                                      cudaMemcpyDeviceToHost, cuda_stream));
    BM_CUDA_OR_RETURN(cudaMemcpyAsync(host_aux3, device_aux3, 3 * n * sizeof(float),
                                      cudaMemcpyDeviceToHost, cuda_stream));
    BM_CUDA_OR_RETURN(cudaStreamSynchronize(cuda_stream));
    return free_parameter(&handle->transient);
}

extern "C" int bm_field_destroy(void* opaque) {
    FieldHandle* handle = static_cast<FieldHandle*>(opaque);
    if (handle == nullptr) return static_cast<int>(cudaSuccess);
    // On a cleanup error return immediately, preserving the remaining owned
    // pointers and handle. The supervisor records the error, not a fake success.
    int result = free_parameter(&handle->transient);
    if (result != 0) return result;
    result = free_parameter(&handle->field.ip);
    if (result != 0) return result;
    result = free_parameter(&handle->field.fp);
    if (result != 0) return result;
    result = free_parameter(&handle->field.ip2);
    if (result != 0) return result;
    result = free_parameter(&handle->field.fp2);
    if (result != 0) return result;
    delete handle;
    return static_cast<int>(cudaSuccess);
}

extern "C" const char* bm_field_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

#undef BM_CUDA_OR_RETURN