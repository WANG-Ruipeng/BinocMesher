#ifndef BM_BATCHING_MEMORY_GUARD_H
#define BM_BATCHING_MEMORY_GUARD_H
#include <cuda_runtime.h>
#include <stddef.h>
cudaError_t mb_cuda_malloc(void**,size_t);
cudaError_t mb_cuda_free(void*);
#endif
