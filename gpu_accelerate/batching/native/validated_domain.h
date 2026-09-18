#ifndef BM_BATCHING_VALIDATED_DOMAIN_H
#define BM_BATCHING_VALIDATED_DOMAIN_H
#include <cstdint>
#include <cstring>
#include <cmath>
namespace mb_admission {
static constexpr uint32_t tree_i[]={0x40e9916cu,0x3cu};
static constexpr uint32_t tree_f[]={0x3f000000u,0x3dcccccdu,0x3fc00000u,0x3f000000u,0x3f000000u,0x3f000000u,0x41100000u,0x3dcccccdu,0x41a00000u,0x41300000u,0x3d4ccccdu,0xbe4ccccdu};
static constexpr uint32_t land_i[]={0x6f82c7b1u,0x1u,0x1u,0x0u,0x0u,0x1u,0x800u,0x0u};
static constexpr uint32_t land_f[]={0x0u,0x3b23065eu,0x3f800000u,0x4e6e6b28u,0x0u,0x44160000u,0x0u,0x0u,0x0u,0x0u,0x3f800000u,0x40000000u,0xce6e6b28u,0x0u,0x0u,0x0u,0x0u,0x0u};
inline bool same(const void*raw,const uint32_t*expected,size_t n){
 for(size_t i=0;i<n;++i){uint32_t bits;std::memcpy(&bits,static_cast<const unsigned char*>(raw)+4*i,4);if(bits!=expected[i])return false;}return true;
}
inline bool field(int meta,size_t ni,const int*ip,size_t nf,const float*fp,size_t ni2,const int*ip2,size_t nf2,const float*fp2){
 // L may vary coherently for tail tests. Height/cover values may vary, including
 // ordinary NaNs, without altering coordinate-based indexing or buffer extents.
 return meta==0&&ni==2&&nf==12&&ni2==8&&nf2==8388627&&ip&&fp&&ip2&&fp2&&
 uint32_t(ip[0])==tree_i[0]&&ip[1]>=1&&ip[1]<=256&&same(fp,tree_f,12)&&
 same(ip2,land_i,8)&&same(fp2,land_f,18);
}
inline bool geometry(int n,int m,const double*c,const double*e){
 for(int kind=0;kind<2;++kind){const double*p=kind?e:c;size_t count=kind?size_t(m):size_t(n);
  for(size_t i=0;i<count;++i)for(int axis=0;axis<3;++axis){double x=p[3*i+axis];
   double lo=axis==2?-0.6234582841396334:-6.600000000000001;
   double hi=axis==2?4.739041715860368:6.600000000000001;
   if(!std::isfinite(x)||x<lo||x>hi)return false;
  }
 }return true;
}
}
#endif
