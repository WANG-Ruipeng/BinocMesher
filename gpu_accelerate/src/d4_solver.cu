// Independent D4 handle. The frozen solver below is included without edits;
// symbol renaming prevents reference/candidate symbol interposition.
#define bm_solver_create d4_base_create
#define bm_solver_prepare_graph d4_base_prepare_graph
#define bm_solver_run d4_base_run
#define bm_solver_readback d4_base_readback
#define bm_solver_get_info d4_base_get_info
#define bm_solver_debug_check d4_base_debug_check
#define bm_solver_timing_prepare d4_base_timing_prepare
#define bm_solver_measure_resident d4_base_measure_resident
#define bm_solver_destroy d4_base_destroy
#include "gpu_solver.cu"
#undef bm_solver_create
#undef bm_solver_prepare_graph
#undef bm_solver_run
#undef bm_solver_readback
#undef bm_solver_get_info
#undef bm_solver_debug_check
#undef bm_solver_timing_prepare
#undef bm_solver_measure_resident
#undef bm_solver_destroy
#include "d4_pipeline.cuh"
#include <cstring>

#define D4_TRY(x) do { int d4_code=(x); if(d4_code) return d4_code; } while(0)
namespace {
struct D4Handle {
    SolverHandle* base=nullptr;
    d4::State pipeline{};
    float* center_xyz=nullptr;
    int device=-1;
    bool ready=false;
    int last_mode=0, last_diagnostic=0;
};
__global__ void d4_center_xyz(SolverDevice d,float* xyz) {
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(p>=size_t(d.n))return;
    for(int a=0;a<3;a++)xyz[3*p+a]=static_cast<float>(d.centers[3*p+a]);
}
__global__ void d4_store(SolverDevice d,const float* xyz,const float* sdf,
                         size_t count,size_t offset,int center,int trace) {
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(p>=count)return;
    float aux[3]={d.aux[3*(offset+p)],d.aux[3*(offset+p)+1],d.aux[3*(offset+p)+2]};
    store_query(d,offset+p,xyz[3*p],xyz[3*p+1],xyz[3*p+2],sdf[p],aux,trace);
    if(center && !isfinite(sdf[p]))d.valid[p]=-2;
}
int d4_enqueue(D4Handle* h,int mode,int k,int trace,int diagnostic) {
    auto* b=h->base; const auto d=b->d;auto s=b->stream;
    D4_TRY(d4::reset(h->pipeline)); // Mandatory logical cold table for EVERY solve.
    if(d.n==0)return 0;
    const unsigned nb=unsigned((size_t(d.n)+255)/256),mb=unsigned((size_t(d.m)+255)/256);
    reset_kernel<<<nb,256,0,s>>>(d,trace);
    d4_center_xyz<<<(d.n+127)/128,128,0,s>>>(d,h->center_xyz);
    D4_TRY(cudaPeekAtLastError());
    D4_TRY(d4::eval(h->pipeline,h->center_xyz,d.n,0,mode,d.center_sdf,d.aux,diagnostic));
    d4_store<<<(d.n+127)/128,128,0,s>>>(d,h->center_xyz,d.center_sdf,d.n,0,1,trace);
    D4_TRY(cudaPeekAtLastError());
    for(int round=0;round<=k;round++) {
        if(d.m) {
            xyz_kernel<<<mb,256,0,s>>>(d,round==k);
            D4_TRY(cudaPeekAtLastError());
            size_t offset=size_t(d.n)+size_t(round)*d.m;
            D4_TRY(d4::eval(h->pipeline,d.query_xyz,d.m,round+1,mode,d.query_sdf,d.aux+3*offset,diagnostic));
            d4_store<<<(d.m+127)/128,128,0,s>>>(d,d.query_xyz,d.query_sdf,d.m,offset,0,trace);
            D4_TRY(cudaPeekAtLastError());
        }
        if(round<k)update_kernel<<<nb,256,0,s>>>(d,round,trace);
        else final_kernel<<<nb,256,0,s>>>(d);
        D4_TRY(cudaPeekAtLastError());
    }
    return 0;
}
bool d4_valid(D4Handle* h,int k,int trace) {
    int device=-1;
    return h && h->ready && cudaGetDevice(&device)==cudaSuccess && device==h->device && valid_run(h->base,k,trace);
}
}
extern "C" int d4_solver_create(FieldView field,int n,int m,const int* off,
    const double* centers,const double* endpoints,int max_k,void** output) {
    if(!output)return cudaErrorInvalidValue;*output=nullptr;
    if(field.kind!=4 || !field.ip || max_k<0 || max_k>6)return cudaErrorInvalidValue;
    auto* h=new(std::nothrow) D4Handle;if(!h)return cudaErrorMemoryAllocation;*output=h;
    D4_TRY(cudaGetDevice(&h->device));
    int params[2];D4_TRY(cudaMemcpy(params,field.ip,sizeof(params),cudaMemcpyDeviceToHost));
    if(params[1]<1 || params[1]>256)return cudaErrorInvalidValue;
    void* b=nullptr;
    int code=d4_base_create(field,n,m,off,centers,endpoints,max_k,&b);
    h->base=static_cast<SolverHandle*>(b);D4_TRY(code);
    D4_TRY(plan_allocation(h->base,3*size_t(n),sizeof(float)));
    if(h->base->requested_bytes>h->base->memory_budget)return cudaErrorMemoryAllocation;
    h->pipeline.external_budget=h->base->memory_budget-h->base->requested_bytes;
    D4_TRY(d4::init(h->pipeline,field,std::max(n,m),params[1],max_k+2,h->base->stream));
    // Additional center coordinate buffer; original endpoint xyz kernel unchanged.
    D4_TRY(allocate(h->base,&h->center_xyz,3*size_t(n)));
    D4_TRY(cudaStreamSynchronize(h->base->stream));
    h->ready=true;return 0;
}
extern "C" int d4_solver_run(void* opaque,int mode,int k,int trace,int diagnostic) {
    auto* h=static_cast<D4Handle*>(opaque);
    if(!d4_valid(h,k,trace) || mode<1 || mode>3 || (diagnostic!=0 && diagnostic!=1))return cudaErrorInvalidValue;
    auto* b=h->base;b->last_k=-1;++b->normal_runs;
#if BM_SOLVER_DEBUG_GUARDS
    D4_TRY(debug_begin(b));
#endif
    D4_TRY(d4_enqueue(h,mode,k,trace,diagnostic));
    D4_TRY(cudaStreamSynchronize(b->stream));
    b->last_k=k;b->last_trace=trace;++b->completed_solves;h->last_mode=mode;h->last_diagnostic=diagnostic;
    return 0;
}
extern "C" int d4_solver_readback(void* opaque,float* position,int* witness,int* valid,
    double* left,double* right,float* aux,float* sdf,float* xyz,int* sign,double* tl,double* tr) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h)return cudaErrorInvalidValue;
    return d4_base_readback(h->base,position,witness,valid,left,right,aux,sdf,xyz,sign,tl,tr);
}
extern "C" int d4_solver_get_info(void* opaque,int k,uint64_t* out,size_t capacity) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h)return cudaErrorInvalidValue;
    return d4_base_get_info(h->base,k,out,capacity);
}
extern "C" int d4_solver_read_counts(void* opaque,uint64_t* out,size_t capacity) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h || !out)return cudaErrorInvalidValue;
    std::vector<d4::Stats> stats;
    D4_TRY(d4::read_stats(h->pipeline,stats));
    if(capacity<stats.size()*20)return cudaErrorInvalidValue;
    static_assert(sizeof(d4::Stats)==20*sizeof(uint64_t),"stats ABI");
    std::memcpy(out,stats.data(),stats.size()*sizeof(d4::Stats));return 0;
}
extern "C" int d4_solver_read_validation(void* opaque,uint64_t* out,size_t capacity) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h || !out || capacity<12)return cudaErrorInvalidValue;
    std::fill(out,out+12,0);return d4::check(h->pipeline,out);
}
extern "C" int d4_solver_read_memory(void* opaque,uint64_t* out,size_t capacity) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h || !out || capacity<16)return cudaErrorInvalidValue;
    std::fill(out,out+16,0);out[0]=1;
    auto& p=h->pipeline;
    out[1]=p.request_capacity;out[2]=p.max_queries;out[3]=p.lattices;out[4]=p.max_calls;
    out[5]=p.bytes;out[6]=p.sort_bytes;out[7]=p.scan_bytes;out[8]=p.free_before;out[9]=p.memory_budget;
    out[10]=h->base->requested_bytes;out[11]=size_t(h->base->d.n)*3*sizeof(float)+2*kGuardBytes;
    return 0;
}
extern "C" int d4_solver_debug_check(void* opaque,uint64_t* out,size_t capacity) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h)return cudaErrorInvalidValue;
    D4_TRY(d4_base_debug_check(h->base,out,capacity));
    uint64_t errors[12]={};D4_TRY(d4::check(h->pipeline,errors));
    out[2]+=errors[0];return out[2]?cudaErrorAssert:0;
}
extern "C" int d4_solver_timing_prepare(void* opaque,int allowed,double* ms) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h)return cudaErrorInvalidValue;
    return d4_base_timing_prepare(h->base,allowed,ms);
}
extern "C" int d4_solver_measure_resident(void* opaque,int mode,int k,int repeats,int allowed,double* event_ms,double* wall_ms) {
#if BM_SOLVER_DEBUG_GUARDS
    return cudaErrorNotSupported;
#else
    auto* h=static_cast<D4Handle*>(opaque);
    if(!d4_valid(h,k,0) || mode<1 || mode>3 || repeats<1 || !event_ms || !wall_ms)return cudaErrorInvalidValue;
    if(allowed!=1)return cudaErrorNotPermitted;
    auto* b=h->base;if(!b->timing_ready)return cudaErrorInvalidValue;
    b->last_k=-1;
    auto begin=std::chrono::steady_clock::now();
    D4_TRY(cudaEventRecord(b->timing_begin,b->stream));++b->event_records;
    for(int i=0;i<repeats;i++)D4_TRY(d4_enqueue(h,mode,k,0,0));
    D4_TRY(cudaEventRecord(b->timing_end,b->stream));++b->event_records;
    D4_TRY(cudaEventSynchronize(b->timing_end));
    auto end=std::chrono::steady_clock::now();float ms=0;
    D4_TRY(cudaEventElapsedTime(&ms,b->timing_begin,b->timing_end));
    *event_ms=ms;*wall_ms=std::chrono::duration<double,std::milli>(end-begin).count();
    b->last_k=k;b->last_trace=0;b->completed_solves+=repeats;h->last_mode=mode;h->last_diagnostic=0;
    return 0;
#endif
}
extern "C" int d4_solver_destroy(void* opaque) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h)return 0;
    h->ready=false;
    if(h->base && h->base->stream)D4_TRY(cudaStreamSynchronize(h->base->stream));
    D4_TRY(d4::destroy(h->pipeline));D4_TRY(d4_base_destroy(h->base));delete h;return 0;
}

// Diagnostic export only. Inactive slots remain explicitly outside the records
// identified by the per-call counts. These arrays are never candidate inputs.
extern "C" int d4_solver_read_diagnostics(void* opaque,uint32_t* rk,uint32_t* ck,
    uint32_t* rd,uint32_t* cd,float* xyz,float* sdf,float* aux,size_t requests,size_t points) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h || !h->last_diagnostic)return cudaErrorInvalidValue;
    auto& p=h->pipeline;auto& d=p.d;size_t nr=size_t(p.request_capacity)*p.max_calls,nq=size_t(p.max_queries)*p.max_calls;
    if(requests<nr || points<nq)return cudaErrorInvalidValue;
    D4_TRY(download(rk,reinterpret_cast<uint32_t*>(d.reference_keys),nr*5,p.stream));
    D4_TRY(download(ck,reinterpret_cast<uint32_t*>(d.candidate_keys),nr*5,p.stream));
    D4_TRY(download(rd,reinterpret_cast<uint32_t*>(d.reference_descriptors),nr*6,p.stream));
    D4_TRY(download(cd,reinterpret_cast<uint32_t*>(d.candidate_descriptors),nr*6,p.stream));
    D4_TRY(download(xyz,d.diagnostic_xyz,nq*3,p.stream));
    D4_TRY(download(sdf,d.reference_sdf,nq,p.stream));
    D4_TRY(download(aux,d.reference_aux,nq*3,p.stream));
    return cudaStreamSynchronize(p.stream);
}
__global__ void d4_fixture_prepare(d4::Device d) {
    if(blockIdx.x || threadIdx.x)return;
    d4::Key a=d4::classify(d.field,bm_upstream::float3_nonbuiltin(1,2,0),0),b=a,c=a;
    b.x=__float_as_uint(__uint_as_float(a.x)+100.0f);c.lattice=1;
    d.keys[0]=a;d.keys[1]=b;d.keys[2]=c;
    // Explicitly permute request/lattice order before the actual GPU sort.
    d.sorted[0]=c;d.sorted[1]=a;d.sorted[2]=b;
    d.request_ids[0]=2;d.request_ids[1]=0;d.request_ids[2]=1;
}
__global__ void d4_fixture_check(d4::Device d) {
    if(blockIdx.x || threadIdx.x)return;
    d.error[0]=d.unique_counts[0];d.error[1]=0;d.error[2]=0;
    for(int j=0;j<3;j++) {
        uint64_t id=d.mapping[j];
        if(id>=uint64_t(d.capacity)*d.calls){d.error[1]++;continue;}
        if(!d4::equal(d.keys[j],d.run_keys[id]))d.error[1]++;
        d4::Descriptor independent=d4::construct(d.field,d.keys[j]);
        if(!d4::equal_descriptor(independent,d.descriptors[id]))d.error[2]++;
    }
}
extern "C" int d4_solver_key_fixture(void* opaque,uint64_t* result,size_t capacity) {
    auto* h=static_cast<D4Handle*>(opaque);if(!h || !result || capacity<8)return cudaErrorInvalidValue;
    auto& p=h->pipeline;if(p.request_capacity<3 || p.lattices<2)return cudaErrorInvalidValue;
    D4_TRY(d4::key_fixture(p,result+3));
    D4_TRY(d4::reset(p));
    d4_fixture_prepare<<<1,128,0,p.stream>>>(p.d);D4_TRY(cudaPeekAtLastError());
    D4_TRY(d4::group_requests(p,3,0,0));
    d4::lookup_kernel<<<1,128,0,p.stream>>>(p.d,3,0,2,0);D4_TRY(cudaPeekAtLastError());
    D4_TRY(cub::DeviceScan::InclusiveSum(p.scan_temp,p.scan_bytes,p.d.miss_flags,p.d.miss_scan,3,p.stream));
    d4::publish_keys_kernel<<<1,128,0,p.stream>>>(p.d,3,0,0);
    d4::construct_kernel<<<1,128,0,p.stream>>>(p.d,3,0,2,0);
    d4::restore_kernel<<<1,128,0,p.stream>>>(p.d,3,0,0);
    d4_fixture_check<<<1,128,0,p.stream>>>(p.d);D4_TRY(cudaPeekAtLastError());
    D4_TRY(download(result,p.error,3,p.stream));
    return cudaStreamSynchronize(p.stream);
}