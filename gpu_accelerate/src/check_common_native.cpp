// CPU-only memory/assembly check for the new native common bridge.
// Fixed three complete checkpoint reloads in one process; no GPU or timings.
// Consumes the existing check_native_sanitized.cpp input.bin schema unchanged.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

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
const char* bm_common_last_error();
int bm_common_reset();
int bm_common_plan_batch();
int bm_common_field_sizes(int,int64_t*);
int bm_common_copy_field(int,int64_t,int64_t,int32_t*,double*,double*,int64_t*,int32_t*);
int bm_common_commit(int64_t,const float*,const int32_t*,const int32_t*,const double*,const double*);
int bm_output_sizes(int64_t*);
int bm_copy_outputs(float*,int32_t*,int8_t*,int32_t*);
}
namespace {
void require(bool ok,const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void common_check(int code,const char* operation) {
    if (code) throw std::runtime_error(std::string(operation)+": "+bm_common_last_error());
}
struct Input {
    std::ifstream file;
    explicit Input(const char* path):file(path,std::ios::binary) {
        require(file.good(),"cannot open CPU common input");
    }
    template<class V> std::vector<V> read(size_t n) {
        require(n<=100000000/sizeof(V),"CPU input allocation exceeds private limit");
        std::vector<V> result(n);
        if (n) require(bool(file.read(reinterpret_cast<char*>(result.data()),n*sizeof(V))),
                       "short CPU common input");
        return result;
    }
    int32_t integer() { return read<int32_t>(1)[0]; }
};
template<class V> void equal(const std::vector<V>& a,const std::vector<V>& b,const char* what) {
    require(a.size()==b.size(),what);
    require(a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(V))==0,what);
}
struct Field {
    int64_t n=0,m=0,center_base=0,query_base=0;
    std::vector<int32_t> offsets,meta;
    std::vector<int64_t> heads;
    std::vector<double> centers,endpoints;
};
void run_once(const char* input_path,int repeat) {
    Input input(input_path);
    require(input.integer()==0x424d4731,"unsupported CPU input schema");
    const int path_size=input.integer();
    require(path_size>0 && path_size<4096,"invalid checkpoint path length");
    const auto path_chars=input.read<char>(path_size);
    const std::string path(path_chars.begin(),path_chars.end());
    auto center=input.read<double>(3);
    auto scene_sizes=input.read<double>(2);
    const int ncam=input.integer();
    require(ncam>0 && ncam<1000,"invalid camera count");
    auto cameras=input.read<double>(27*static_cast<size_t>(ncam));
    auto parameters=input.read<double>(5);
    const int ne=input.integer(),batch_group=input.integer(),K=input.integer(),nb=input.integer();
    require(ne>0 && ne<100 && batch_group>0 && K>=1 && K<=64 && nb>0 && nb<10000,
            "invalid CPU input configuration");

    // A real full reset: no process fork and no snapshot restore shortcuts.
    common_check(bm_common_reset(),"reset before checkpoint import");
    const int groups=load_parameters(center.data(),scene_sizes[0],scene_sizes[1],
                                    ncam,cameras.data(),parameters[0],parameters[1],
                                    parameters[2],parameters[3],parameters[4],ne,path.c_str());
    require(groups==1,"CPU common check supports max_tL=0 only");
    require(bm_checkpoint_read((path+"/native_state.bin").c_str())==0,"native checkpoint import");
    load_tree_size();
    bisection_init(batch_group);
    bisection_init_t(0);
    int previous_vertices=0;
    std::vector<int32_t> previous_map;
    for (int b=0;b<nb;++b) {
        const int group=input.integer(),batch=input.integer();
        require(group==0,"CPU common check supports only group 0");
        const auto expected_counts=input.read<int32_t>(ne);
        const auto expected_nodes=input.read<int32_t>(ne);
        std::vector<int32_t> counts(ne),nodes(ne);
        require(bisection_hypermesh_verts(0,counts.data(),nodes.data())!=0,
                "missing captured batch");
        equal(counts,expected_counts,"endpoint counts differ");
        equal(nodes,expected_nodes,"node counts differ");
        int64_t n=0,m=0;
        for (int e=0;e<ne;++e) {
            require(counts[e]>=0 && nodes[e]>=0,"negative native batch count");
            n+=nodes[e];m+=counts[e];
        }
        require(n>0 && n<1000000 && m>=n && m<10000000,"invalid bounded natural batch size");
        const auto expected_center=input.read<double>(3*static_cast<size_t>(n));
        auto center_sdf=input.read<float>(n);
        std::vector<double> native_center(3*static_cast<size_t>(n));
        bisection_hypermesh_verts_output_center(native_center.data());
        equal(native_center,expected_center,"original native center differs");

        common_check(bm_common_plan_batch(),"plan batch");
        std::vector<Field> fields(ne);
        int64_t next_center=0,next_query=0;
        std::vector<double> left(n,0),right(n,1);
        std::vector<int32_t> valid(n,1),witness(n,-1);
        std::vector<uint8_t> covered(n,0);
        for (int e=0;e<ne;++e) {
            auto& f=fields[e];
            int64_t dimensions[4]={};
            common_check(bm_common_field_sizes(e,dimensions),"field sizes");
            f.n=dimensions[0];f.m=dimensions[1];
            f.center_base=dimensions[2];f.query_base=dimensions[3];
            require(f.n==nodes[e] && f.m==counts[e] &&
                    f.center_base==next_center && f.query_base==next_query,
                    "field dimensions or original bases differ");
            f.offsets.resize(f.n+1);f.meta.resize(6*f.n);f.heads.resize(f.n);
            f.centers.resize(3*f.n);f.endpoints.resize(3*f.m);
            common_check(bm_common_copy_field(e,f.n,f.m,f.offsets.data(),
                f.centers.data(),f.endpoints.data(),f.heads.data(),f.meta.data()),"copy field");
            require(f.offsets.front()==0 && f.offsets.back()==f.m,"CSR endpoints differ");
            for (int64_t local=0;local<f.n;++local) {
                const int* meta=f.meta.data()+6*local;
                const int ci=meta[1],start=meta[2];
                require(ci==f.center_base+local && start==f.query_base+f.offsets[local],
                        "field metadata center/query order differs");
                require(ci>=0 && ci<n && !covered[ci],"duplicate center metadata");
                covered[ci]=1;
                require(f.heads[local]>=0 && f.heads[local]%ne==e &&
                        f.heads[local]/ne==meta[0],"node-element metadata differs");
                require(local==0 || f.heads[local]>f.heads[local-1],"native heads are not ordered");
                require(f.offsets[local]>=0 && f.offsets[local+1]>f.offsets[local],
                        "natural CSR node has no endpoints");
                require(std::memcmp(f.centers.data()+3*local,
                                    expected_center.data()+3*ci,3*sizeof(double))==0,
                        "CSR center bits differ from original capture");
            }
            next_center+=f.n;next_query+=f.m;
        }
        require(next_center==n && next_query==m,"CSR aggregate dimensions differ");
        for (auto present:covered) require(present!=0,"missing center metadata");
        for (float value:center_sdf) require(std::isfinite(value),"nonfinite captured center SDF");

        // These are frozen actual captured GPU values. Reading them is not a
        // new field evaluation and cannot validate the current GPU program.
        for (int step=0;step<=K;++step) {
            const auto expected_xyz=input.read<double>(3*static_cast<size_t>(m));
            auto sdf=input.read<float>(m);
            for (float value:sdf) require(std::isfinite(value),"nonfinite captured endpoint SDF");
            std::vector<double> native_xyz(3*static_cast<size_t>(m)),csr_xyz(3*static_cast<size_t>(m));
            bisection_hypermesh_verts_output(native_xyz.data(),step==K);
            equal(native_xyz,expected_xyz,"original native query bits differ");
            for (const auto& f:fields) {
                for (int64_t local=0;local<f.n;++local) {
                    const int ci=f.meta[6*local+1];
                    const double t=step==K?right[ci]:(left[ci]+right[ci])/2;
                    for (int p=f.offsets[local];p<f.offsets[local+1];++p)
                        for (int d=0;d<3;++d)
                            csr_xyz[3*(f.query_base+p)+d]=
                                f.centers[3*local+d]*(1-t)+f.endpoints[3*int64_t(p)+d]*t;
                    bool any=false;
                    for (int p=f.offsets[local];p<f.offsets[local+1];++p) {
                        const bool diff=(sdf[f.query_base+p]>=0)!=(center_sdf[ci]>=0);
                        any|=diff;
                        if (step==K && diff && witness[ci]<0)
                            witness[ci]=p-f.offsets[local];
                    }
                    if (step<K) {
                        const double mid=(left[ci]+right[ci])/2;
                        if (any) right[ci]=mid;else left[ci]=mid;
                    } else require(witness[ci]>=0,"captured final query has no witness");
                }
            }
            equal(csr_xyz,expected_xyz,"CSR query bits/order differ from original capture");
            if (step<K) bisection_hypermesh_verts_iter(sdf.data(),center_sdf.data());
            // Finishing is replaced only by common_commit below.
        }

        const int nv=input.integer(),nm=input.integer();
        require(nv>=0 && nm>=0 && nv<1000000 && nm<10000000 &&
                nv==previous_vertices+n,"invalid cumulative captured output dimensions");
        const auto expected_xyz=input.read<float>(3*static_cast<size_t>(nv));
        const auto expected_times=input.read<int32_t>(2*static_cast<size_t>(nv));
        const auto expected_tags=input.read<int8_t>(nv);
        const auto expected_map=input.read<int32_t>(nm);
        std::vector<float> supplied(expected_xyz.begin()+3*previous_vertices,expected_xyz.end());

        // The supplied positions are explicitly captured CPU-test inputs.
        // The helper generates only metadata/map/native storage, from the
        // loaded checkpoint plan. Production callers supply GPU results.
        common_check(bm_common_commit(n,supplied.data(),valid.data(),witness.data(),
                                    left.data(),right.data()),"commit captured positions");
        int64_t output_sizes[2]={};
        require(bm_output_sizes(output_sizes)==0 &&
                output_sizes[0]==nv && output_sizes[1]==nm,"native output dimensions differ");
        std::vector<float> xyz(3*static_cast<size_t>(nv));
        std::vector<int32_t> times(2*static_cast<size_t>(nv)),mapping(nm);
        std::vector<int8_t> tags(nv);
        require(bm_copy_outputs(xyz.data(),times.data(),tags.data(),mapping.data())==0,
                "native output copy failed");
        equal(xyz,expected_xyz,"committed position bits differ");
        equal(times,expected_times,"committed native times differ");
        equal(tags,expected_tags,"committed native tags differ");
        equal(mapping,expected_map,"committed native vertex_map differs");
        if (b==0) require(previous_map.empty(),"unexpected prior map in repeat");
        previous_map=std::move(mapping);
        previous_vertices=nv;
        std::cout<<"{\"repeat\":"<<repeat<<",\"group\":0,\"batch\":"<<batch
                 <<",\"N\":"<<n<<",\"M\":"<<m
                 <<",\"status\":\"PASS\",\"GPU_executed\":false,"
                   "\"purpose\":\"CPU_CSR_AND_COMMIT_WITH_FROZEN_CAPTURE\"}"<<std::endl;
    }
    std::vector<int32_t> counts(ne),nodes(ne);
    require(!bisection_hypermesh_verts(0,counts.data(),nodes.data()),"unconsumed natural batch");
    require(input.file.peek()==std::ifstream::traits_type::eof(),"trailing CPU input data");
    common_check(bm_common_reset(),"reset after complete output");
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"usage: check_common_native INPUT.bin");
        std::cout<<"{\"purpose\":\"CPU input/assembly and same-process full checkpoint reset\","
                   "\"repetitions\":3,\"warmup\":false,\"GPU_executed\":false,"
                   "\"performance_test\":false}"<<std::endl;
        for (int repeat=0;repeat<3;++repeat) run_once(argv[1],repeat);
        std::cout<<"{\"status\":\"PASS_CPU_COMMON_CHECKPOINT_RESET_AND_ASSEMBLY\","
                   "\"repetitions\":3,\"GPU_executed\":false,"
                   "\"performance_test\":false}"<<std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"CORRECTNESS FAILURE: "<<e.what()<<std::endl;
        return 1;
    }
}
