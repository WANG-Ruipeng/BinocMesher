#ifndef BM_BATCHING_PUBLISHED_API_H
#define BM_BATCHING_PUBLISHED_API_H
#include <stddef.h>
#include <stdint.h>
#if defined(__GNUC__)
#define MB_API __attribute__((visibility("default")))
#else
#define MB_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Serial use in one live CUDA context. Tokens are identities, never pointers.
   Every failed run/prepare invalidates previous outputs. Create can return a
   partial owned token on failure: destroy it and check cleanup before fallback.
   No Graph backend or timing API is exported. All statuses are CUDA codes. */
MB_API int mb_abi(uint64_t*,size_t);
MB_API int mb_configure(size_t); /* <=2 GiB; can only tighten global limit */
/* 16 words: schema,ordinal,cc_major,cc_minor,warp,driver,runtime,free,total,
   global_limit,tracked,initial_free,driver_context_id,graph_reserved(0),fields,solvers. */
MB_API int mb_context(uint64_t*,size_t,char*,size_t,char*,size_t);
MB_API int mb_field_create(int,size_t,const int*,size_t,const float*,size_t,const int*,size_t,const float*,uint64_t*);
MB_API int mb_field_invalidate(uint64_t);
MB_API int mb_field_destroy(uint64_t);
MB_API int mb_solver_create(uint64_t,int,int,const int*,const double*,const double*,int,uint64_t*);
MB_API int mb_solver_prepare(uint64_t,int,int,int);
MB_API int mb_solver_run(uint64_t,int,int,int);
MB_API int mb_solver_readback(uint64_t,float*,size_t,int*,size_t,int*,size_t,double*,size_t,double*,size_t);
MB_API int mb_solver_trace(uint64_t,float*,float*,float*,int*,double*,double*,size_t,size_t);
MB_API int mb_solver_counts(uint64_t,uint64_t*,size_t);
MB_API int mb_solver_validation(uint64_t,uint64_t*,size_t);
/* 24 words preserve historical host-adapter layout; graph/profile words are 0.
   0 schema;1 method(0);2 N;3 M;4 maxK;5 field token;6 solver token;7 generation;
   8 successful attempt;9 lastK/UINT64_MAX;10 trace;11 audit;12 generation;
   13/14 graph nodes/edges(0);15 base bytes;16 pipeline bytes;17 global tracked;
   18 global limit;19 graph reserve(0);20 successful solves;21 invalidation/attempt;
   22 guards;23 diagnostic build(0). Planned bytes are not measured device peak. */
MB_API int mb_solver_info(uint64_t,uint64_t*,size_t);
MB_API int mb_solver_destroy(uint64_t);
#ifdef __cplusplus
}
#endif
#endif
