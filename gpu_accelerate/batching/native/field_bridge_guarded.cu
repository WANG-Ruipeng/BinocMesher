// Published field implementation unchanged; allocations join the adapter budget.
#include "memory_guard.h"
#define cudaMalloc mb_cuda_malloc
#define cudaFree mb_cuda_free
#include "field_bridge.cu"
#undef cudaMalloc
#undef cudaFree
