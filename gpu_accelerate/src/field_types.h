#ifndef BM_REVIEWED_FIELD_TYPES_H
#define BM_REVIEWED_FIELD_TYPES_H

#include <stddef.h>

/* The pointer members refer to CUDA device allocations. This POD is passed by
 * value to every solver kernel. No library-global field state is consulted. */
typedef struct FieldView {
    int kind;
    int meta;
    int* ip;
    float* fp;
    int* ip2;
    float* fp2;
} FieldView;

#ifdef __cplusplus
extern "C" {
#endif
int bm_field_create(int kind, int meta,
                    size_t ni, const int* ip, size_t nf, const float* fp,
                    size_t ni2, const int* ip2, size_t nf2, const float* fp2,
                    void** out_handle);
int bm_field_get_view(void* handle, FieldView* out_view);
int bm_field_eval_device(void* handle, size_t n, const float* device_xyz,
                         float* device_sdf, float* device_aux3, void* stream);
int bm_field_eval_host(void* handle, size_t n, const float* host_xyz,
                       float* host_sdf, float* host_aux3, void* stream);
int bm_field_destroy(void* handle);
const char* bm_field_error_string(int code);
#ifdef __cplusplus
}
#endif

#endif