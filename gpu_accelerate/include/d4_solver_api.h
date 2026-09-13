#ifndef BM_D4_SOLVER_API_H
#define BM_D4_SOLVER_API_H
#include "solver_api.h"
#ifdef __cplusplus
extern "C" {
#endif
// Independent immutable handle: kind=4 only, 1..256 lattices, 0<=max_k<=6.
// Caller destroys solver before borrowed field, same CUDA context, serial use.
// Every normal solve resets the logical table; errors are sticky until destroy.
int d4_solver_create(FieldView,int,int,const int*,const double*,const double*,int,void**);
// variant: 1 split / 2 call-local grouping / 3 solve-local cold runs.
int d4_solver_run(void*,int variant,int k,int trace,int diagnostic);
int d4_solver_readback(void*,float*,int*,int*,double*,double*,float*,float*,float*,int*,double*,double*);
int d4_solver_get_info(void*,int k,uint64_t*,size_t);
int d4_solver_debug_check(void*,uint64_t*,size_t); // 8 words, base schema
int d4_solver_read_counts(void*,uint64_t*,size_t); // (max_k+2)*20 words
int d4_solver_read_memory(void*,uint64_t*,size_t); // 16 words
int d4_solver_read_validation(void*,uint64_t*,size_t); // 12 words, total first
int d4_solver_read_diagnostics(void*,uint32_t*,uint32_t*,uint32_t*,uint32_t*,float*,float*,float*,size_t,size_t);
int d4_solver_key_fixture(void*,uint64_t*,size_t); // 8 words; disposable diagnostic handle only
int d4_solver_timing_prepare(void*,int allow_timing,double* setup_ms);
int d4_solver_measure_resident(void*,int variant,int k,int repeats,int allow_timing,double* event_ms,double* wall_ms);
int d4_solver_destroy(void*);
#ifdef __cplusplus
}
#endif
#endif
