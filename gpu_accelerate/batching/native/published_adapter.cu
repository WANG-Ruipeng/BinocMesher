// Host lifecycle adaptation of the measured D1_PUBLISHED wrapper. Numerical
// implementation comes from the repository's unchanged published translation unit.
#include "published_api.h"
#include "memory_guard.h"
#include <cuda.h>
#include <unordered_map>
#include <memory>
#include <cstdio>
#include <cstring>
#include <climits>
#include "validated_domain.h"
#define cudaMalloc mb_cuda_malloc
#define cudaFree mb_cuda_free
#include "d4_solver.cu"
#undef cudaMalloc
#undef cudaFree

namespace mb {
constexpr int invalid=int(cudaErrorInvalidValue),nomem=int(cudaErrorMemoryAllocation);
struct Budget {int device=-1;CUcontext context=nullptr;unsigned long long context_id=0;
 size_t initial_free=0,total=0,limit=0,used=0,requested_limit=size_t(2147483648ULL);bool poisoned=false;} budget;
std::unordered_map<void*,size_t> allocations;
int context_ready(){
 CUcontext current=nullptr;unsigned long long identity=0;
 if(budget.context){
  if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||current!=budget.context||
     cuCtxGetId(current,&identity)!=CUDA_SUCCESS||identity!=budget.context_id)return invalid;
  int device=-1;auto e=cudaGetDevice(&device);if(e)return int(e);
  return device==budget.device?0:invalid;
 }
 int device=-1;auto e=cudaGetDevice(&device);if(e)return int(e);
 size_t f=0,t=0;e=cudaMemGetInfo(&f,&t);if(e)return int(e);
 if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||!current||cuCtxGetId(current,&identity)!=CUDA_SUCCESS)return invalid;
 budget.device=device;budget.context=current;budget.context_id=identity;
 budget.initial_free=f;budget.total=t;budget.limit=std::min(budget.requested_limit,f/2);return 0;
}
struct Field{void*raw=nullptr;FieldView view{};bool active=false;};
struct Solver{D4Handle*raw=nullptr;uint64_t field=0,generation=1,attempt=0,success=0,runs=0;
 bool ran=false;uint64_t errors[12]{};};
std::unordered_map<uint64_t,std::unique_ptr<Field>> fields;
std::unordered_map<uint64_t,std::unique_ptr<Solver>> solvers;
uint64_t next_token=1;
template<class T>T*find(std::unordered_map<uint64_t,std::unique_ptr<T>>&reg,uint64_t id){
 auto i=reg.find(id);return i==reg.end()?nullptr:i->second.get();}
bool active(Solver*s){auto*f=s?find(fields,s->field):nullptr;return f&&f->active;}
void clear(Solver*s){if(!s)return;s->ran=false;++s->attempt;s->success=0;
 if(s->raw&&s->raw->base){s->raw->base->last_k=-1;s->raw->last_diagnostic=0;}}
bool last(Solver*s){return s&&active(s)&&s->ran&&s->success==s->attempt&&!budget.poisoned;}
bool config(Solver*s,int k,int trace,int audit){return s&&active(s)&&s->raw&&s->raw->ready&&
 k>=0&&k<=s->raw->base->max_k&&(trace==0||trace==1)&&(audit==0||audit==1)&&!budget.poisoned;}
int failed(Solver*s,int code){if(code){clear(s);budget.poisoned=true;}return code;}
}
cudaError_t mb_cuda_malloc(void**out,size_t bytes){
 if(!out)return cudaErrorInvalidValue;*out=nullptr;int c=mb::context_ready();if(c)return cudaError_t(c);
 auto&b=mb::budget;if(b.used>b.limit||bytes>b.limit-b.used)return cudaErrorMemoryAllocation;
 void*p=nullptr;auto e=cudaMalloc(&p,bytes);if(e)return e;
 try{mb::allocations.emplace(p,bytes);}catch(...){
  auto cleanup=cudaFree(p);
  if(cleanup!=cudaSuccess){mb::budget.poisoned=true;
   std::fprintf(stderr,"MB_ALLOCATION_REGISTRY_FAILURE cleanup_cuda_status=%d\n",int(cleanup));
   // Allocation registry failure followed by failed cleanup is never a safe
   // pre-submit OOM fallback, even if cudaFree itself returned status 2.
   return cudaErrorUnknown;
  }
  return cudaErrorMemoryAllocation;
 }
 b.used+=bytes;*out=p;return cudaSuccess;
}
cudaError_t mb_cuda_free(void*p){
 if(!p)return cudaSuccess;auto i=mb::allocations.find(p);if(i==mb::allocations.end())return cudaErrorInvalidValue;
 auto e=cudaFree(p);if(!e){mb::budget.used-=i->second;mb::allocations.erase(i);}return e;
}
extern "C" int mb_abi(uint64_t*out,size_t cap){
 if(!out||cap<8)return mb::invalid;uint64_t v[8]={1,sizeof(uint64_t),sizeof(size_t),sizeof(double),1,128,6,0};
 std::copy(v,v+8,out);return 0;
}
extern "C" int mb_configure(size_t limit){
 if(!limit||limit>size_t(2147483648ULL))return mb::invalid;
 if(limit<mb::budget.used)return mb::nomem;
 mb::budget.requested_limit=std::min(mb::budget.requested_limit,limit);
 if(mb::budget.context)mb::budget.limit=std::min(mb::budget.limit,limit);return 0;
}
extern "C" int mb_context(uint64_t*out,size_t cap,char*name,size_t nc,char*uuid,size_t uc){
 if(!out||cap<16||!name||nc<256||!uuid||uc<41)return mb::invalid;
 int c=mb::context_ready();if(c)return c;cudaDeviceProp p{};
 auto e=cudaGetDeviceProperties(&p,mb::budget.device);if(e)return int(e);
 size_t f=0,t=0;e=cudaMemGetInfo(&f,&t);if(e)return int(e);
 int driver=0,runtime=0;e=cudaDriverGetVersion(&driver);if(e)return int(e);
 e=cudaRuntimeGetVersion(&runtime);if(e)return int(e);
 uint64_t v[16]={1,uint64_t(mb::budget.device),uint64_t(p.major),uint64_t(p.minor),uint64_t(p.warpSize),
 uint64_t(driver),uint64_t(runtime),f,t,mb::budget.limit,mb::budget.used,mb::budget.initial_free,
 mb::budget.context_id,0,mb::fields.size(),mb::solvers.size()};
 auto*b=reinterpret_cast<unsigned char*>(p.uuid.bytes);char u[41];
 std::snprintf(u,sizeof(u),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
 b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
 std::copy(v,v+16,out);std::strncpy(name,p.name,nc);name[nc-1]=0;std::memcpy(uuid,u,41);return 0;
}
extern "C" int mb_field_create(int meta,size_t ni,const int*ip,size_t nf,const float*fp,
 size_t ni2,const int*ip2,size_t nf2,const float*fp2,uint64_t*out){
 if(!out)return mb::invalid;*out=0;
 if(!ip||!fp||!ip2||!fp2||ni<2||nf<12||ni2<8||nf2<18||
 ni>SIZE_MAX/4||nf>SIZE_MAX/4||ni2>SIZE_MAX/4||nf2>SIZE_MAX/4||ip[1]<1||ip[1]>256||mb::budget.poisoned)return mb::invalid;
 if(!mb_admission::field(meta,ni,ip,nf,fp,ni2,ip2,nf2,fp2))return int(cudaErrorNotSupported);
 int c=mb::context_ready();if(c)return c;
 if(mb::next_token==UINT64_MAX)return mb::invalid;
 const uint64_t token=mb::next_token++;
 try{mb::fields.emplace(token,std::unique_ptr<mb::Field>(new mb::Field));}
 catch(...){return mb::nomem;}
 *out=token;auto*f=mb::find(mb::fields,token);
 c=bm_field_create(4,meta,ni,ip,nf,fp,ni2,ip2,nf2,fp2,&f->raw);
 if(!c)c=bm_field_get_view(f->raw,&f->view);if(!c)f->active=true;return c;
}
extern "C" int mb_field_invalidate(uint64_t id){
 auto*f=mb::find(mb::fields,id);if(!f)return mb::invalid;f->active=false;
 for(auto&i:mb::solvers)if(i.second->field==id){mb::clear(i.second.get());++i.second->generation;}return 0;
}
extern "C" int mb_field_destroy(uint64_t id){
 auto*f=mb::find(mb::fields,id);if(!f)return mb::invalid;
 int c=mb::context_ready();if(c)return c;mb_field_invalidate(id);
 for(auto i=mb::solvers.begin();i!=mb::solvers.end();){
  if(i->second->field!=id){++i;continue;}uint64_t sid=i->first;++i;c=mb_solver_destroy(sid);if(c)return c;}
 c=bm_field_destroy(f->raw);if(c)return c;mb::fields.erase(id);return 0;
}
extern "C" int mb_solver_create(uint64_t field,int n,int m,const int*off,const double*centers,
 const double*endpoints,int maxk,uint64_t*out){
 if(!out)return mb::invalid;*out=0;auto*f=mb::find(mb::fields,field);
 if(!f||!f->active||n<0||m<0||!off||(n==0&&m!=0)||(n&&!centers)||(m&&!endpoints)||maxk<0||maxk>6||mb::budget.poisoned)return mb::invalid;
 if(!mb_admission::geometry(n,m,centers,endpoints))return int(cudaErrorNotSupported);
 int c=mb::context_ready();if(c)return c;if(mb::next_token==UINT64_MAX)return mb::invalid;
 const uint64_t token=mb::next_token++;
 try{mb::solvers.emplace(token,std::unique_ptr<mb::Solver>(new mb::Solver));}catch(...){return mb::nomem;}
 *out=token;auto*s=mb::find(mb::solvers,token);s->field=field;void*raw=nullptr;
 try{c=d4_solver_create(f->view,n,m,off,centers,endpoints,maxk,&raw);}
 catch(...){c=int(cudaErrorUnknown);mb::budget.poisoned=true;}
 s->raw=static_cast<D4Handle*>(raw);return c;
}
extern "C" int mb_solver_prepare(uint64_t id,int k,int trace,int audit){
 auto*s=mb::find(mb::solvers,id);if(!s)return mb::invalid;mb::clear(s);++s->generation;
 if(!mb::config(s,k,trace,audit))return mb::invalid;return mb::context_ready();
}
extern "C" int mb_solver_run(uint64_t id,int k,int trace,int audit){
 auto*s=mb::find(mb::solvers,id);mb::clear(s);if(!mb::config(s,k,trace,audit))return mb::invalid;
 int c=mb::context_ready();if(c)return c;
 c=d4_solver_run(s->raw,1,k,trace,audit);if(c)return mb::failed(s,c);
 uint64_t errors[12]{};c=d4_solver_read_validation(s->raw,errors,12);if(c)return mb::failed(s,c);
 std::copy(errors,errors+12,s->errors);if(errors[0])return mb::failed(s,int(cudaErrorAssert));
 ++s->runs;s->success=s->attempt;s->ran=true;return 0;
}
extern "C" int mb_solver_readback(uint64_t id,float*p,size_t pc,int*w,size_t wc,int*v,size_t vc,double*l,size_t lc,double*r,size_t rc){
 auto*s=mb::find(mb::solvers,id);if(!mb::last(s))return mb::invalid;size_t n=s->raw->base->d.n;
 if(pc<3*n||wc<n||vc<n||lc<n||rc<n||(n&&(!p||!w||!v||!l||!r)))return mb::invalid;
 int c=mb::context_ready();if(c)return c;return mb::failed(s,d4_solver_readback(s->raw,p,w,v,l,r,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr));
}
extern "C" int mb_solver_trace(uint64_t id,float*a,float*f,float*x,int*sign,double*l,double*r,size_t qc,size_t bc){
 auto*s=mb::find(mb::solvers,id);if(!mb::last(s))return mb::invalid;auto*b=s->raw->base;
 size_t q=size_t(b->d.n)+(size_t(b->last_k)+1)*b->d.m,nb=(size_t(b->last_k)+1)*b->d.n;
 if((!a&&!f&&!x&&!sign&&!l&&!r)||((a||f||x||sign)&&qc<q)||((l||r)&&bc<nb)||(!b->last_trace&&(f||x||sign||l||r)))return mb::invalid;
 int c=mb::context_ready();if(c)return c;return mb::failed(s,d4_solver_readback(s->raw,nullptr,nullptr,nullptr,nullptr,nullptr,a,f,x,sign,l,r));
}
extern "C" int mb_solver_counts(uint64_t id,uint64_t*out,size_t cap){
 auto*s=mb::find(mb::solvers,id);if(!mb::last(s)||!out||!s->raw->last_diagnostic||cap<size_t(s->raw->base->max_k+2)*20)return mb::invalid;
 int c=mb::context_ready();if(c)return c;std::fill(out,out+cap,0);return mb::failed(s,d4_solver_read_counts(s->raw,out,cap));
}
extern "C" int mb_solver_validation(uint64_t id,uint64_t*out,size_t cap){
 auto*s=mb::find(mb::solvers,id);if(!mb::last(s)||!out||cap<12)return mb::invalid;
 int c=mb::context_ready();if(c)return c;std::copy(s->errors,s->errors+12,out);return 0;
}
extern "C" int mb_solver_info(uint64_t id,uint64_t*out,size_t cap){
 auto*s=mb::find(mb::solvers,id);if(!s||!mb::active(s)||!s->raw||!s->raw->ready||!out||cap<24)return mb::invalid;
 int c=mb::context_ready();if(c)return c;auto*b=s->raw->base;
 uint64_t v[24]={1,0,uint64_t(b->d.n),uint64_t(b->d.m),uint64_t(b->max_k),s->field,id,s->generation,s->success,
 b->last_k<0?UINT64_MAX:uint64_t(b->last_k),uint64_t(b->last_trace),uint64_t(s->raw->last_diagnostic),s->generation,0,0,
 b->requested_bytes,s->raw->pipeline.bytes,mb::budget.used,mb::budget.limit,0,s->runs,s->attempt,BM_SOLVER_DEBUG_GUARDS,0};
 std::copy(v,v+24,out);return 0;
}
extern "C" int mb_solver_destroy(uint64_t id){
 auto*s=mb::find(mb::solvers,id);if(!s)return mb::invalid;int c=mb::context_ready();if(c)return c;
 mb::clear(s);c=d4_solver_destroy(s->raw);if(c)return c;mb::solvers.erase(id);return 0;
}
