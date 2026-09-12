// CPU-only original native bisection replay under ASan/UBSan. No timings.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
int load_parameters(double*,double,double,int,double*,double,double,double,double,double,int,const char*);
void load_tree_size();
int bm_checkpoint_read(const char*);
void bisection_init(int);
void bisection_init_t(int);
int bisection_hypermesh_verts(int,int*,int*);
void bisection_hypermesh_verts_output_center(double*);
void bisection_hypermesh_verts_output(double*,int);
void bisection_hypermesh_verts_iter(float*,float*);
void bisection_hypermesh_verts_finishing(int,float*,float*);
void bisection_clean_up();
int bm_output_sizes(int64_t*);
int bm_copy_outputs(float*,int32_t*,int8_t*,int32_t*);
}
struct Input {
 std::ifstream f;
 explicit Input(const char*p):f(p,std::ios::binary){if(!f)throw std::runtime_error("open input");}
 template<class T> std::vector<T> read(size_t n){std::vector<T>v(n);if(n&&!f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T)))throw std::runtime_error("short input");return v;}
 int32_t integer(){return read<int32_t>(1)[0];}
};
void require(bool x,const char*s){if(!x)throw std::runtime_error(s);}
template<class T>void equal(const std::vector<T>&a,const std::vector<T>&b,const char*s){require(a.size()==b.size(),s);require(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0,s);}
int main(int argc,char**argv){try{
 require(argc==2,"one input argument required");Input in(argv[1]);require(in.integer()==0x424d4731,"bad version");
 int path_size=in.integer();require(path_size>0&&path_size<4096,"path length");auto chars=in.read<char>(path_size);std::string path(chars.begin(),chars.end());
 auto center=in.read<double>(3);auto sizes=in.read<double>(2);int nc=in.integer();require(nc>0&&nc<1000,"camera count");auto cameras=in.read<double>(27*nc);auto params=in.read<double>(5);
 int ne=in.integer(),group_size=in.integer(),K=in.integer(),batches=in.integer();require(ne>0&&ne<100&&K>=1&&K<=64&&batches>0,"configuration");
 load_parameters(center.data(),sizes[0],sizes[1],nc,cameras.data(),params[0],params[1],params[2],params[3],params[4],ne,path.c_str());require(bm_checkpoint_read((path+"/native_state.bin").c_str())==0,"native immutable checkpoint import");load_tree_size();bisection_init(group_size);
 int group=-1;
 for(int b=0;b<batches;b++){
  int g=in.integer(),batch=in.integer();
  if(g!=group){require(group==-1,"this bounded CPU oracle supports one captured group only");bisection_init_t(g);group=g;}
  auto ec=in.read<int32_t>(ne),en=in.read<int32_t>(ne);std::vector<int32_t> counts(ne),nodes(ne);
  require(bisection_hypermesh_verts(g,counts.data(),nodes.data())!=0,"missing expected batch");equal(ec,counts,"endpoint counts");equal(en,nodes,"node counts");
  size_t N=0,M=0;for(int e=0;e<ne;e++){require(nodes[e]>=0&&counts[e]>=0,"negative counts");N+=nodes[e];M+=counts[e];}
  auto expected_center=in.read<double>(3*N);auto center_sdf=in.read<float>(N);std::vector<double> actual_center(3*N);
  bisection_hypermesh_verts_output_center(actual_center.data());equal(actual_center,expected_center,"center coordinates");
  for(int step=0;step<=K;step++){
   auto expected_xyz=in.read<double>(3*M);auto sdf=in.read<float>(M);std::vector<double> actual_xyz(3*M);
   bisection_hypermesh_verts_output(actual_xyz.data(),step==K);equal(actual_xyz,expected_xyz,"round coordinates");
   if(step==K)bisection_hypermesh_verts_finishing(g,sdf.data(),center_sdf.data());else bisection_hypermesh_verts_iter(sdf.data(),center_sdf.data());
  }
  int nv=in.integer(),nm=in.integer();require(nv>=0&&nm>=0,"output sizes");int64_t actual_sizes[2];require(bm_output_sizes(actual_sizes)==0,"native sizes API");require(actual_sizes[0]==nv&&actual_sizes[1]==nm,"native output sizes");
  auto ex=in.read<float>(3*size_t(nv));auto et=in.read<int32_t>(2*size_t(nv));auto eg=in.read<int8_t>(nv);auto em=in.read<int32_t>(nm);
  std::vector<float>x(ex.size());std::vector<int32_t>t(et.size()),map(em.size());std::vector<int8_t>tag(eg.size());
  require(bm_copy_outputs(x.data(),t.data(),tag.data(),map.data())==0,"copy outputs");equal(x,ex,"native positions");equal(t,et,"native times");equal(tag,eg,"native tags");equal(map,em,"native vertex_map");
  std::cout<<"{\"group\":"<<g<<",\"batch\":"<<batch<<",\"N\":"<<N<<",\"M\":"<<M<<",\"status\":\"PASS\",\"GPU_executed\":false}"<<std::endl;
 }
 std::vector<int32_t> c(ne),n(ne);require(!bisection_hypermesh_verts(group,c.data(),n.data()),"unconsumed natural batch");
 require(in.f.peek()==std::ifstream::traits_type::eof(),"trailing input");bisection_clean_up();
 std::cout<<"{\"status\":\"PASS_ON_CAPTURED_NATIVE_CHECKPOINT\",\"performance_test\":false}"<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<"CORRECTNESS FAILURE: "<<e.what()<<std::endl;return 1;}}
