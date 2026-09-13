#ifndef BM_D4_PIPELINE_CUH
#define BM_D4_PIPELINE_CUH
#include "d4_field_eval.cuh"
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/version.cuh>
#include <vector>
#include <limits>
#include <algorithm>
#include <climits>
#include <new>
#ifndef BM_SOLVER_DEBUG_GUARDS
#define BM_SOLVER_DEBUG_GUARDS 0
#endif
#ifndef D4_MUTANT_SKIP_RESET
#define D4_MUTANT_SKIP_RESET 0
#endif
#ifndef D4_MUTANT_KEY_HASH_ONLY
#define D4_MUTANT_KEY_HASH_ONLY 0
#endif
namespace d4 {
constexpr int block_threads=128, stats_words=20, error_words=12;
constexpr size_t redzone_bytes=BM_SOLVER_DEBUG_GUARDS ? 256 : 0;
struct Stats { uint64_t v[stats_words]; };
struct Device {
    FieldView field{};
    int capacity=0,lattices=0,calls=0,max_queries=0;
    Key *keys=nullptr,*sorted=nullptr,*unique=nullptr,*run_keys=nullptr;
    Descriptor* descriptors=nullptr;
    int *request_ids=nullptr,*flags=nullptr,*scan=nullptr,*group=nullptr;
    int *miss_flags=nullptr,*miss_scan=nullptr,*counts=nullptr,*unique_counts=nullptr;
    uint64_t *mapping=nullptr,*unique_mapping=nullptr,*ready=nullptr,*epoch=nullptr;
    int* next_call=nullptr;
    unsigned* mapping_writes=nullptr;
    uint64_t* error=nullptr;
    Stats* stats=nullptr;
    Key *reference_keys=nullptr,*candidate_keys=nullptr;
    Descriptor *reference_descriptors=nullptr,*candidate_descriptors=nullptr;
    float *reference_sdf=nullptr,*reference_aux=nullptr,*diagnostic_xyz=nullptr;
};
struct Allocation { unsigned char* base; size_t bytes; };
struct State {
    Device d{};
    FieldView field{};
    cudaStream_t stream=nullptr;
    int request_capacity=0,max_queries=0,lattices=0,max_calls=0;
    size_t bytes=0,payload_bytes=0,guard_bytes=0,sort_bytes=0,scan_bytes=0;
    size_t free_before=0,total_device=0,memory_budget=0,external_budget=SIZE_MAX;
    void *sort_temp=nullptr,*scan_temp=nullptr;
    uint64_t* error=nullptr;
    uint64_t resets=0;
    std::vector<Allocation> allocations;
};
#define D4_CUDA(expr) do { cudaError_t rc=(expr); if(rc!=cudaSuccess) return int(rc); } while(0)
#define D4_CALL(expr) do { int rc=(expr); if(rc!=0) return rc; } while(0)
static __device__ __forceinline__ void add(uint64_t* p,uint64_t n=1) {
    atomicAdd(reinterpret_cast<unsigned long long*>(p),static_cast<unsigned long long>(n));
}
static __device__ __forceinline__ void fail(Device d,int slot,uint64_t n=1) { add(d.error,n);add(d.error+slot,n); }
struct GroupLess {
    __host__ __device__ bool operator()(const Key& a,const Key& b) const {
#if D4_MUTANT_KEY_HASH_ONLY
        return a.hash<b.hash;
#else
        return Less{}(a,b);
#endif
    }
};
__host__ __device__ inline bool group_equal(const Key& a,const Key& b) {
#if D4_MUTANT_KEY_HASH_ONLY
    return a.hash==b.hash;
#else
    return equal(a,b);
#endif
}
static __global__ void reset_kernel(Device d) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d.calls) {d.counts[j]=0;d.unique_counts[j]=0;}
    if(j<d.calls*stats_words) reinterpret_cast<uint64_t*>(d.stats)[j]=0;
    // Failure flags remain sticky across solve repeats. init zeroes them;
    // a worker with any failure must exit, so later cold resets cannot hide it.
    if(j==0) { *d.epoch+=1; *d.next_call=0; }
}
static __global__ void begin_kernel(Device d,int q,int call,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(diagnostic && j<d.capacity) d.mapping_writes[j]=0;
    if(j==0) {
        if(*d.next_call!=call) fail(d,11);
        *d.next_call=call+1;
        if(diagnostic) {d.stats[call].v[0]=q;d.stats[call].v[1]=uint64_t(q)*d.lattices;}
    }
}
static __global__ void classify_kernel(Device d,const float* xyz,int count,int call,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count) return;
    int p=j/d.lattices,i=j%d.lattices;
    d.keys[j]=classify(d.field,bm_upstream::float3_nonbuiltin(xyz[3*p],xyz[3*p+1],xyz[3*p+2]),i);
    d.sorted[j]=d.keys[j];d.request_ids[j]=j;
    if(diagnostic) add(d.stats[call].v+8);
}
static __global__ void flag_kernel(Device d,int count) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<count) d.flags[j]=(j==0 || !group_equal(d.sorted[j],d.sorted[j-1])) ? 1 : 0;
}
static __global__ void group_kernel(Device d,int count,int call,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count) return;
    int id=d.scan[j]-1;
    if(id<0 || id>=count) { fail(d,10);return; }
    if(d.flags[j]) d.unique[id]=d.sorted[j];
    int request=d.request_ids[j];
    if(request<0 || request>=count) { fail(d,6);return; }
    d.group[request]=id;
    if(j==count-1) {d.unique_counts[call]=d.scan[j];if(diagnostic)d.stats[call].v[2]=d.scan[j];}
}
static __global__ void lookup_kernel(Device d,int count,int call,int mode,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count) return;
    d.miss_flags[j]=0;
    if(j>=d.unique_counts[call]) return;
    uint64_t found=UINT64_MAX;
    if(mode==3) {
        for(int r=0;r<call && found==UINT64_MAX;++r) {
            int lo=0,hi=d.counts[r];
            if(hi<0 || hi>d.capacity) {fail(d,10);return;}
            size_t base=size_t(r)*d.capacity;
            while(lo<hi) {int mid=lo+(hi-lo)/2; if(GroupLess{}(d.run_keys[base+mid],d.unique[j]))lo=mid+1;else hi=mid;}
            if(lo<d.counts[r] && group_equal(d.run_keys[base+lo],d.unique[j])) found=base+lo;
        }
    }
    d.unique_mapping[j]=found;
    d.miss_flags[j]=(found==UINT64_MAX);
    if(diagnostic && found!=UINT64_MAX) add(d.stats[call].v+3);
}
static __global__ void publish_keys_kernel(Device d,int count,int call,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count) return;
    if(j<d.unique_counts[call] && d.miss_flags[j]) {
        int id=d.miss_scan[j]-1;
        if(id<0 || id>=d.capacity) {fail(d,10);return;}
        size_t offset=size_t(call)*d.capacity+id;
        d.run_keys[offset]=d.unique[j];
        d.unique_mapping[j]=offset;
    }
    if(j==count-1) {d.counts[call]=d.miss_scan[j];if(diagnostic)d.stats[call].v[4]=d.miss_scan[j];}
}
static __global__ void construct_kernel(Device d,int count,int call,int mode,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count) return;
    if(mode!=1 && j>=d.counts[call])return;
    size_t offset=size_t(call)*d.capacity+j;
    Key key=mode==1 ? d.keys[j] : d.run_keys[offset];
    if(mode==1) {d.run_keys[offset]=key;d.mapping[j]=offset;if(diagnostic){atomicAdd(d.mapping_writes+j,1u);add(d.stats[call].v+16);}}
    Descriptor value=construct(d.field,key);
    d.descriptors[offset]=value;
    d.ready[offset]=*d.epoch;
    if(diagnostic) {
        add(d.stats[call].v+5);
        if(value.enabled) {add(d.stats[call].v+6);add(d.stats[call].v+7);}
    }
    if(j==0 && mode==1) {d.counts[call]=count;if(diagnostic)d.stats[call].v[4]=count;}
}
static __global__ void restore_kernel(Device d,int count,int call,int diagnostic) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count)return;
    int u=d.group[j];
    if(u<0 || u>=d.unique_counts[call]) {fail(d,6);return;}
    d.mapping[j]=d.unique_mapping[u];
    if(diagnostic){atomicAdd(d.mapping_writes+j,1u);add(d.stats[call].v+16);}
}
static __global__ void evaluate_kernel(Device d,const float* xyz,int q,int call,float* sdf,float* aux,int diagnostic) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=q)return;
    size_t base=size_t(p)*d.lattices;
    bool bad=false;
    for(int i=0;i<d.lattices;++i) {
        uint64_t index=d.mapping[base+i];
        if(index>=uint64_t(d.capacity)*d.calls) {fail(d,6);bad=true;continue;}
        if(d.ready[index]!=*d.epoch) {fail(d,7);bad=true;}
        if(diagnostic && d.descriptors[index].enabled)add(d.stats[call].v+12);
    }
    if(bad){sdf[p]=__uint_as_float(0x7fc00000);aux[3*p]=aux[3*p+1]=aux[3*p+2]=0;return;}
    float a[3];
    sdf[p]=evaluate(d.field,bm_upstream::float3_nonbuiltin(xyz[3*p],xyz[3*p+1],xyz[3*p+2]),d.keys+base,d.mapping+base,d.descriptors,a);
    for(int i=0;i<3;++i)aux[3*p+i]=a[i];
    if(diagnostic){add(d.stats[call].v+9);add(d.stats[call].v+10);add(d.stats[call].v+11);}
}
static __global__ void diagnostic_kernel(Device d,const float* xyz,int q,int call,const float* sdf,const float* aux) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=q)return;
    size_t point=size_t(call)*d.max_queries+p;
    size_t base=size_t(call)*d.capacity+size_t(p)*d.lattices;
    bm_upstream::float3_nonbuiltin position(xyz[3*p],xyz[3*p+1],xyz[3*p+2]);
    float a[3],b[3];
    float original=bm_eval(d.field,position.x,position.y,position.z,a);
    float copied=original_diagnostic(d.field,position,d.reference_keys+base,d.reference_descriptors+base,b);
    d.reference_sdf[point]=original;
    for(int i=0;i<3;++i){d.reference_aux[3*point+i]=a[i];d.diagnostic_xyz[3*point+i]=xyz[3*p+i];}
    if(__float_as_uint(original)!=__float_as_uint(copied))fail(d,5);
    for(int i=0;i<3;++i)if(__float_as_uint(a[i])!=__float_as_uint(b[i]))fail(d,5);
    if(__float_as_uint(original)!=__float_as_uint(sdf[p]))fail(d,3);
    for(int i=0;i<3;++i)if(__float_as_uint(a[i])!=__float_as_uint(aux[3*p+i]))fail(d,4);
    add(d.stats[call].v+17);
    for(int i=0;i<d.lattices;++i) {
        size_t request=size_t(p)*d.lattices+i,trace=base+i;
        d.candidate_keys[trace]=d.keys[request];
        add(d.stats[call].v+14);
        if(d.reference_descriptors[trace].enabled)add(d.stats[call].v+13);
        if(!equal(d.keys[request],d.reference_keys[trace]))fail(d,1);else add(d.stats[call].v+18);
        uint64_t id=d.mapping[request];
        if(id>=uint64_t(d.capacity)*d.calls){fail(d,6);continue;}
        // Compare the mapped key too: disabled descriptors can be identical.
        if(!equal(d.run_keys[id],d.reference_keys[trace]))fail(d,1);
        d.candidate_keys[trace]=d.run_keys[id];
        d.candidate_descriptors[trace]=d.descriptors[id];
        if(!equal_descriptor(d.descriptors[id],d.reference_descriptors[trace]))fail(d,2);else add(d.stats[call].v+19);
    }
}
static __global__ void production_kernel(Device d,int count,int call,int mode) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d.capacity && d.mapping_writes[j]!=(j<count ? 1u:0u))fail(d,9);
    if(j==0) {
        uint64_t total=0;for(int r=0;r<=call;++r)total+=d.counts[r];
        d.stats[call].v[15]=(mode==3) ? total : d.counts[call];
        if(d.stats[call].v[11]!=d.stats[call].v[0])fail(d,9);
        if(d.stats[call].v[16]!=uint64_t(count))fail(d,9);
    }
}
static __global__ void guard_kernel(unsigned char* base,size_t bytes,uint64_t* error) {
    unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<redzone_bytes && (base[j]!=0xa5 || base[redzone_bytes+bytes+j]!=0xa5)) {add(error);add(error+8);}
}
inline bool size_add(size_t& total,size_t value) {if(value>SIZE_MAX-total)return false;total+=value;return true;}
inline bool size_mul(size_t a,size_t b,size_t& result) {if(b && a>SIZE_MAX/b)return false;result=a*b;return true;}
template<class T> inline int plan(State& s,size_t count) {
    size_t n;
    if(!size_mul(count,sizeof(T),n) || !size_add(s.payload_bytes,n) ||
       (count && (!size_add(s.guard_bytes,2*redzone_bytes))))return int(cudaErrorInvalidValue);
    return 0;
}
template<class T> inline int alloc(State& s,T*& ptr,size_t count) {
    ptr=nullptr;if(!count)return 0;
    size_t n;if(!size_mul(count,sizeof(T),n) || n>SIZE_MAX-2*redzone_bytes)return int(cudaErrorInvalidValue);
    unsigned char* raw=nullptr;D4_CUDA(cudaMalloc(reinterpret_cast<void**>(&raw),n+2*redzone_bytes));
    s.allocations.push_back({raw,n});ptr=reinterpret_cast<T*>(raw+redzone_bytes);
    if(redzone_bytes){D4_CUDA(cudaMemsetAsync(raw,0xa5,redzone_bytes,s.stream));D4_CUDA(cudaMemsetAsync(raw+redzone_bytes+n,0xa5,redzone_bytes,s.stream));}
    D4_CUDA(cudaMemsetAsync(ptr,0,n,s.stream));
    return 0;
}
inline int init(State& s,FieldView field,int max_queries,int lattice_count,int max_calls,cudaStream_t stream) {
    if(field.kind!=4 || max_queries<0 || lattice_count<1 || max_calls<1 || max_calls>66 ||
       !field.ip || !field.fp || !field.ip2 || !field.fp2 || !s.allocations.empty())return int(cudaErrorInvalidValue);
    size_t capacity,total_requests,total_queries;
    if(!size_mul(size_t(max_queries),size_t(lattice_count),capacity) || capacity>INT_MAX-127 ||
       !size_mul(capacity,size_t(max_calls),total_requests) || !size_mul(size_t(max_queries),size_t(max_calls),total_queries))return int(cudaErrorInvalidValue);
    int actual_lattices=0;
    D4_CUDA(cudaMemcpyAsync(&actual_lattices,field.ip+1,sizeof(int),cudaMemcpyDeviceToHost,stream));
    D4_CUDA(cudaStreamSynchronize(stream));
    if(actual_lattices!=lattice_count)return int(cudaErrorInvalidValue);
    try {s.allocations.reserve(40);} catch(const std::bad_alloc&) {return int(cudaErrorMemoryAllocation);}
    s.field=field;s.stream=stream;s.max_queries=max_queries;s.lattices=lattice_count;s.max_calls=max_calls;s.request_capacity=int(capacity);
    s.d.field=field;s.d.capacity=int(capacity);s.d.lattices=lattice_count;s.d.calls=max_calls;s.d.max_queries=max_queries;
    if(capacity) {
        D4_CUDA(cub::DeviceMergeSort::SortPairs(nullptr,s.sort_bytes,static_cast<Key*>(nullptr),static_cast<int*>(nullptr),int(capacity),GroupLess{},stream));
        D4_CUDA(cub::DeviceScan::InclusiveSum(nullptr,s.scan_bytes,static_cast<int*>(nullptr),static_cast<int*>(nullptr),int(capacity),stream));
    }
    // The plan counts every physical allocation, including CUB temporary
    // storage and full diagnostic traces, before the first cudaMalloc.
    for(int j=0;j<3;++j)D4_CALL(plan<Key>(s,capacity));
    D4_CALL(plan<Key>(s,total_requests));
    D4_CALL(plan<Descriptor>(s,total_requests));
    for(int j=0;j<6;++j)D4_CALL(plan<int>(s,capacity));
    for(int j=0;j<2;++j)D4_CALL(plan<int>(s,max_calls));
    for(int j=0;j<2;++j)D4_CALL(plan<uint64_t>(s,capacity));
    D4_CALL(plan<uint64_t>(s,total_requests));D4_CALL(plan<uint64_t>(s,1));D4_CALL(plan<int>(s,1));
    D4_CALL(plan<unsigned>(s,capacity));D4_CALL(plan<uint64_t>(s,error_words));D4_CALL(plan<Stats>(s,max_calls));
    for(int j=0;j<2;++j){D4_CALL(plan<Key>(s,total_requests));D4_CALL(plan<Descriptor>(s,total_requests));}
    D4_CALL(plan<float>(s,total_queries));
    size_t triple_queries;if(!size_mul(total_queries,3,triple_queries))return int(cudaErrorInvalidValue);
    for(int j=0;j<2;++j)D4_CALL(plan<float>(s,triple_queries));
    D4_CALL(plan<unsigned char>(s,s.sort_bytes));D4_CALL(plan<unsigned char>(s,s.scan_bytes));
    s.bytes=s.payload_bytes;if(!size_add(s.bytes,s.guard_bytes))return int(cudaErrorInvalidValue);
    D4_CUDA(cudaMemGetInfo(&s.free_before,&s.total_device));
    s.memory_budget=std::min((s.free_before/10)*7+((s.free_before%10)*7)/10,s.external_budget);
    if(s.bytes>s.memory_budget)return int(cudaErrorMemoryAllocation);
    Device& d=s.d;
    D4_CALL(alloc(s,d.keys,capacity));D4_CALL(alloc(s,d.sorted,capacity));D4_CALL(alloc(s,d.unique,capacity));
    D4_CALL(alloc(s,d.run_keys,total_requests));D4_CALL(alloc(s,d.descriptors,total_requests));
    D4_CALL(alloc(s,d.request_ids,capacity));D4_CALL(alloc(s,d.flags,capacity));D4_CALL(alloc(s,d.scan,capacity));D4_CALL(alloc(s,d.group,capacity));
    D4_CALL(alloc(s,d.miss_flags,capacity));D4_CALL(alloc(s,d.miss_scan,capacity));
    D4_CALL(alloc(s,d.counts,max_calls));D4_CALL(alloc(s,d.unique_counts,max_calls));
    D4_CALL(alloc(s,d.mapping,capacity));D4_CALL(alloc(s,d.unique_mapping,capacity));D4_CALL(alloc(s,d.ready,total_requests));
    D4_CALL(alloc(s,d.epoch,1));D4_CALL(alloc(s,d.next_call,1));D4_CALL(alloc(s,d.mapping_writes,capacity));
    D4_CALL(alloc(s,d.error,error_words));s.error=d.error;D4_CALL(alloc(s,d.stats,max_calls));
    D4_CALL(alloc(s,d.reference_keys,total_requests));D4_CALL(alloc(s,d.reference_descriptors,total_requests));
    D4_CALL(alloc(s,d.candidate_keys,total_requests));D4_CALL(alloc(s,d.candidate_descriptors,total_requests));
    D4_CALL(alloc(s,d.reference_sdf,total_queries));D4_CALL(alloc(s,d.reference_aux,triple_queries));D4_CALL(alloc(s,d.diagnostic_xyz,triple_queries));
    unsigned char* temporary=nullptr;D4_CALL(alloc(s,temporary,s.sort_bytes));s.sort_temp=temporary;
    D4_CALL(alloc(s,temporary,s.scan_bytes));s.scan_temp=temporary;
    D4_CUDA(cudaStreamSynchronize(stream));
    return 0;
}
inline int reset(State& s) {
    if(!s.error)return int(cudaErrorInvalidValue);
#if D4_MUTANT_SKIP_RESET
    if(s.resets++)return 0;
#else
    ++s.resets;
#endif
    int jobs=std::max(s.max_calls*stats_words,error_words);
    reset_kernel<<<(jobs+127)/128,128,0,s.stream>>>(s.d);
    D4_CUDA(cudaPeekAtLastError());return 0;
}
inline int group_requests(State& s,int count,int call,int diagnostic) {
    D4_CUDA(cub::DeviceMergeSort::SortPairs(s.sort_temp,s.sort_bytes,s.d.sorted,s.d.request_ids,count,GroupLess{},s.stream));
    flag_kernel<<<(count+127)/128,128,0,s.stream>>>(s.d,count);D4_CUDA(cudaPeekAtLastError());
    D4_CUDA(cub::DeviceScan::InclusiveSum(s.scan_temp,s.scan_bytes,s.d.flags,s.d.scan,count,s.stream));
    group_kernel<<<(count+127)/128,128,0,s.stream>>>(s.d,count,call,diagnostic);D4_CUDA(cudaPeekAtLastError());return 0;
}
// All work counts stay on the device. Host loop bounds use only immutable
// geometry/parameter upper bounds; no sort/unique/miss D2H is performed here.
inline int eval(State& s,const float* xyz,int q,int call,int mode,float* sdf,float* aux,int diagnostic) {
    if(q<0 || q>s.max_queries || call<0 || call>=s.max_calls || mode<1 || mode>3 ||
       (diagnostic!=0 && diagnostic!=1) || (q && (!xyz || !sdf || !aux)))return int(cudaErrorInvalidValue);
    int count=q*s.lattices;
    begin_kernel<<<std::max(1,(s.request_capacity+127)/128),128,0,s.stream>>>(s.d,q,call,diagnostic);D4_CUDA(cudaPeekAtLastError());
    if(!q)return 0;
    int rb=(count+127)/128,qb=(q+127)/128;
    classify_kernel<<<rb,128,0,s.stream>>>(s.d,xyz,count,call,diagnostic);D4_CUDA(cudaPeekAtLastError());
    if(mode!=1 || diagnostic)D4_CALL(group_requests(s,count,call,diagnostic));
    if(mode!=1) {
        lookup_kernel<<<rb,128,0,s.stream>>>(s.d,count,call,mode,diagnostic);D4_CUDA(cudaPeekAtLastError());
        D4_CUDA(cub::DeviceScan::InclusiveSum(s.scan_temp,s.scan_bytes,s.d.miss_flags,s.d.miss_scan,count,s.stream));
        publish_keys_kernel<<<rb,128,0,s.stream>>>(s.d,count,call,diagnostic);D4_CUDA(cudaPeekAtLastError());
    }
    construct_kernel<<<rb,128,0,s.stream>>>(s.d,count,call,mode,diagnostic);D4_CUDA(cudaPeekAtLastError());
    if(mode!=1){restore_kernel<<<rb,128,0,s.stream>>>(s.d,count,call,diagnostic);D4_CUDA(cudaPeekAtLastError());}
    evaluate_kernel<<<qb,128,0,s.stream>>>(s.d,xyz,q,call,sdf,aux,diagnostic);D4_CUDA(cudaPeekAtLastError());
    if(diagnostic) {
        diagnostic_kernel<<<qb,128,0,s.stream>>>(s.d,xyz,q,call,sdf,aux);D4_CUDA(cudaPeekAtLastError());
        production_kernel<<<std::max(1,(s.request_capacity+127)/128),128,0,s.stream>>>(s.d,count,call,mode);D4_CUDA(cudaPeekAtLastError());
    }
    return 0;
}
inline int read_stats(State& s,std::vector<Stats>& output) {
    output.resize(s.max_calls);D4_CUDA(cudaMemcpyAsync(output.data(),s.d.stats,output.size()*sizeof(Stats),cudaMemcpyDeviceToHost,s.stream));
    D4_CUDA(cudaStreamSynchronize(s.stream));return 0;
}
inline int check(State& s,uint64_t* errors) {
    if(!errors || !s.error)return int(cudaErrorInvalidValue);
    if(redzone_bytes)for(const auto& a:s.allocations){guard_kernel<<<2,128,0,s.stream>>>(a.base,a.bytes,s.error);D4_CUDA(cudaPeekAtLastError());}
    D4_CUDA(cudaMemcpyAsync(errors,s.error,error_words*sizeof(uint64_t),cudaMemcpyDeviceToHost,s.stream));
    D4_CUDA(cudaStreamSynchronize(s.stream));return 0;
}
inline int destroy(State& s) {
    D4_CUDA(cudaStreamSynchronize(s.stream));
    for(auto& a:s.allocations)if(a.base){D4_CUDA(cudaFree(a.base));a.base=nullptr;}
    s.allocations.clear();s.error=nullptr;return 0;
}
// Focused externally observable comparator mutation fixture: synthetic keys,
// never presented as natural scene coverage. No field is replaced by them.
static __global__ void key_fixture_kernel(uint64_t* out) {
    if(blockIdx.x || threadIdx.x)return;
    Key a{0,123,0x3f800000,0,0},b=a,c=a;b.x=0x3f800001;c.lattice=1;
    out[0]=equal(a,a);out[1]=group_equal(a,b);out[2]=group_equal(a,c);
    out[3]=uint64_t(GroupLess{}(a,b) || GroupLess{}(b,a));
    out[4]=uint64_t(GroupLess{}(a,c) || GroupLess{}(c,a));
}
inline int key_fixture(State& s,uint64_t* host_output) {
    if(!host_output)return int(cudaErrorInvalidValue);
    // Error storage is temporary fixture output; reset is required afterwards.
    key_fixture_kernel<<<1,128,0,s.stream>>>(s.error);D4_CUDA(cudaPeekAtLastError());
    D4_CUDA(cudaMemcpyAsync(host_output,s.error,5*sizeof(uint64_t),cudaMemcpyDeviceToHost,s.stream));
    D4_CUDA(cudaStreamSynchronize(s.stream));return 0;
}
#undef D4_CUDA
#undef D4_CALL
} // namespace d4
#endif
