#ifndef BM_D4_FIELD_EVAL_CUH
#define BM_D4_FIELD_EVAL_CUH
#include "field_eval.cuh"
#include <cstdint>

// Experimental decomposition of the immutable SdfTrees implementation.
// Build with exactly the frozen reference floating-point options.
namespace d4 {
struct Key { int lattice, hash; uint32_t x, y, z; };
struct Descriptor { uint32_t enabled; float x, y, z, radius, ratio; };
static_assert(sizeof(Key)==20 && sizeof(Descriptor)==24, "D4 wire sizes");
struct Less {
    __host__ __device__ bool operator()(const Key& a,const Key& b) const {
        if(a.lattice!=b.lattice) return a.lattice<b.lattice;
        if(a.hash!=b.hash) return a.hash<b.hash;
        if(a.x!=b.x) return a.x<b.x;
        if(a.y!=b.y) return a.y<b.y;
        return a.z<b.z;
    }
};
__host__ __device__ inline bool equal(const Key& a,const Key& b) {
    return a.lattice==b.lattice && a.hash==b.hash && a.x==b.x && a.y==b.y && a.z==b.z;
}
static __device__ __forceinline__ Key classify(FieldView field,
    bm_upstream::float3_nonbuiltin position,int i) {
    float center[3],distance; int hash;
    bm_upstream::Voronoi(position.x,position.y,0,bm_upstream::myhash(field.ip[0],i),
        1,field.fp[0],1,center,&distance,&hash);
    return Key{i,hash,__float_as_uint(center[0]),__float_as_uint(center[1]),__float_as_uint(center[2])};
}
static __device__ __forceinline__ Descriptor construct(FieldView field,const Key& key) {
    using namespace bm_upstream;
    const int seed=field.ip[0];
    float density=field.fp[0],r_short=field.fp[1],ratio=field.fp[2];
    float r_randomness=field.fp[3],ratio_randomness=field.fp[4],height_offset=field.fp[5];
    float mask_octaves=field.fp[9],mask_freq=field.fp[10],mask_shift=field.fp[11];
    r_short/=density;
    float3_nonbuiltin cell_center(__uint_as_float(key.x),__uint_as_float(key.y),__uint_as_float(key.z));
    float mask=Perlin(cell_center.x,cell_center.y,cell_center.z,myhash(1,seed),mask_octaves,mask_freq)+mask_shift<0;
    Descriptor result{uint32_t(mask!=0),0,0,0,0,1};
    if(mask) {
        float sdf_a;
        landtiles(cell_center,&sdf_a,NULL,0,field.ip2,field.fp2,NULL,NULL);
        float scale=1-r_randomness+r_randomness*2*hash_to_float(myhash(0,myhash(key.hash,key.lattice)));
        float r_short_cell=r_short*scale;
        scale=1-ratio_randomness+ratio_randomness*2*hash_to_float(myhash(1,myhash(key.hash,key.lattice)));
        float ratio_cell=ratio*scale;
        cell_center.z=cell_center.z-sdf_a+height_offset*r_short_cell*ratio_cell;
        result.x=cell_center.x; result.y=cell_center.y; result.z=cell_center.z;
        result.radius=r_short_cell; result.ratio=ratio_cell;
    }
    return result;
}
static __device__ __forceinline__ float evaluate(FieldView field,
    bm_upstream::float3_nonbuiltin position,const Key* keys,const uint64_t* mapping,
    const Descriptor* descriptors,float* aux) {
    using namespace bm_upstream;
    int seed=field.ip[0],n_lattice=field.ip[1];
    float density=field.fp[0],r_short=field.fp[1];
    float noise_octaves=field.fp[6],noise_scale=field.fp[7],noise_freq=field.fp[8];
    noise_freq*=density; noise_scale/=density; r_short/=density;
    float sdf_=1e9,sdf_a; int tree_id=0;
    for(int i=0;i<n_lattice;++i) {
        const Descriptor& d=descriptors[mapping[i]];
        if(d.enabled) {
            float3_nonbuiltin cell_center(d.x,d.y,d.z);
            float3_nonbuiltin vec=position-cell_center;
            float ratio_cell=d.ratio,r_short_cell=d.radius;
            float tmp=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z/(ratio_cell*ratio_cell))-r_short_cell;
            if(tmp<sdf_) { sdf_=tmp; tree_id=keys[i].hash; }
        }
    }
    landtiles(position,&sdf_a,NULL,0,field.ip2,field.fp2,NULL,NULL);
    float sdf=min(sdf_+Perlin(position.x,position.y,position.z,seed,noise_octaves,noise_freq)*noise_scale,sdf_a+r_short*0.1f);
    aux[0]=(float)tree_id; aux[1]=0.0f; aux[2]=0.0f;
    return sdf;
}

// Separate diagnostic copy of the original monolithic loop. It deliberately
// does not call classify/construct/evaluate, so candidate descriptors cannot
// validate themselves. The immutable bm_eval is additionally checked against
// this loop by the pipeline, outside every timing interval.
static __device__ __forceinline__ float original_diagnostic(FieldView field,
    bm_upstream::float3_nonbuiltin position,Key* keys,Descriptor* descriptors,float* aux) {
    using namespace bm_upstream;
    int seed=field.ip[0],n_lattice=field.ip[1];
    float density=field.fp[0],r_short=field.fp[1],ratio=field.fp[2];
    float r_randomness=field.fp[3],ratio_randomness=field.fp[4],height_offset=field.fp[5];
    float noise_octaves=field.fp[6],noise_scale=field.fp[7],noise_freq=field.fp[8];
    float mask_octaves=field.fp[9],mask_freq=field.fp[10],mask_shift=field.fp[11];
    noise_freq*=density; noise_scale/=density; r_short/=density;
    float cell_center_[3],distance,sdf_=1e9,sdf_a; int tree_id=0;
    for(int i=0;i<n_lattice;++i) {
        int hash;
        Voronoi(position.x,position.y,0,myhash(seed,i),1,density,1,&cell_center_[0],&distance,&hash);
        keys[i]=Key{i,hash,__float_as_uint(cell_center_[0]),__float_as_uint(cell_center_[1]),__float_as_uint(cell_center_[2])};
        float mask=Perlin(cell_center_[0],cell_center_[1],cell_center_[2],myhash(1,seed),mask_octaves,mask_freq)+mask_shift<0;
        descriptors[i]=Descriptor{uint32_t(mask!=0),0,0,0,0,1};
        if(mask) {
            float3_nonbuiltin cell_center(cell_center_[0],cell_center_[1],cell_center_[2]);
            landtiles(cell_center,&sdf_a,NULL,0,field.ip2,field.fp2,NULL,NULL);
            float scale=1-r_randomness+r_randomness*2*hash_to_float(myhash(0,myhash(hash,i)));
            float r_short_cell=r_short*scale;
            scale=1-ratio_randomness+ratio_randomness*2*hash_to_float(myhash(1,myhash(hash,i)));
            float ratio_cell=ratio*scale;
            cell_center.z=cell_center.z-sdf_a+height_offset*r_short_cell*ratio_cell;
            descriptors[i]=Descriptor{1,cell_center.x,cell_center.y,cell_center.z,r_short_cell,ratio_cell};
            float3_nonbuiltin vec=position-cell_center;
            float tmp=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z/(ratio_cell*ratio_cell))-r_short_cell;
            if(tmp<sdf_) { sdf_=tmp; tree_id=hash; }
        }
    }
    landtiles(position,&sdf_a,NULL,0,field.ip2,field.fp2,NULL,NULL);
    float sdf=min(sdf_+Perlin(position.x,position.y,position.z,seed,noise_octaves,noise_freq)*noise_scale,sdf_a+r_short*0.1f);
    aux[0]=(float)tree_id; aux[1]=0.0f; aux[2]=0.0f;
    return sdf;
}
static __device__ __forceinline__ bool equal_descriptor(const Descriptor& a,const Descriptor& b) {
    return a.enabled==b.enabled && __float_as_uint(a.x)==__float_as_uint(b.x) &&
        __float_as_uint(a.y)==__float_as_uint(b.y) && __float_as_uint(a.z)==__float_as_uint(b.z) &&
        __float_as_uint(a.radius)==__float_as_uint(b.radius) && __float_as_uint(a.ratio)==__float_as_uint(b.ratio);
}
} // namespace d4
#endif
