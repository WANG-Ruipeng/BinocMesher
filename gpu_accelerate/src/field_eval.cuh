#ifndef BM_REVIEWED_FIELD_EVAL_CUH
#define BM_REVIEWED_FIELD_EVAL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "field_types.h"

/* Include the immutable original common device implementations, in their
 * original dependency order, without cuda/elements/header.h: that file also
 * includes core.cu and its old host init/cleanup exports. Namespace-local
 * linkage allows this header to be used by more than one CUDA translation
 * unit without duplicating externally linked constant arrays. */
namespace {
namespace bm_upstream {
using namespace std;
// Preserve the scalar overloads visible in the original global namespace.
using ::floor;
#define DEVICE_FUNC __device__
#define CONSTANT_ARRAY __device__ __constant__
#include "terrain/source/common/utils/vectors.h"
#include "infinigen_gpl/bnodes/utils/nodes_util.h"
#include "infinigen_gpl/bnodes/utils/blender_noise.h"
#include "terrain/source/common/utils/elements_util.h"
#include "terrain/source/common/utils/FastNoiseLite.h"
#include "terrain/source/common/utils/smooth_bool_ops.h"
#include "terrain/source/common/elements/caves.h"
#include "terrain/source/common/elements/landtiles.h"
#include "terrain/source/common/elements/sdf_trees.h"
#undef CONSTANT_ARRAY
#undef DEVICE_FUNC
}  // namespace bm_upstream
}  // anonymous namespace

/* All callers must supply three writable floats, and make their values
 * observable (the batch bridge writes all three to global output). The real
 * field branches keep the complete top-level auxiliary computation. */
static __device__ __forceinline__ float bm_eval(FieldView field,
                                               float x, float y, float z,
                                               float* aux) {
    aux[0] = 0.0f;
    aux[1] = 0.0f;
    aux[2] = 0.0f;
    if (field.kind == 3) {
        float sdf;
        bm_upstream::landtiles(bm_upstream::float3_nonbuiltin(x, y, z),
                               &sdf, aux, field.meta, field.ip, field.fp,
                               field.ip2, field.fp2);
        return sdf;
    }
    if (field.kind == 4) {
        float sdf;
        // sdf_trees.h keeps NULL aux in its internal landtiles calls.
        bm_upstream::sdf_trees(bm_upstream::float3_nonbuiltin(x, y, z),
                               &sdf, aux, field.ip, field.fp,
                               field.ip2, field.fp2);
        return sdf;
    }
    if (field.kind == 0) {
        return field.fp[0] * x + field.fp[1] * y + field.fp[2] * z + field.fp[3];
    }
    float dx = x - field.fp[0];
    float dy = y - field.fp[1];
    float dz = z - field.fp[2];
    float sdf = dx * dx + dy * dy + dz * dz - field.fp[3] * field.fp[3];
    if (field.kind == 2) {
        dx = x - field.fp[4];
        dy = y - field.fp[5];
        dz = z - field.fp[6];
        const float sdf2 = dx * dx + dy * dy + dz * dz - field.fp[7] * field.fp[7];
        sdf = fminf(sdf, sdf2);
    }
    return sdf;
}

#endif