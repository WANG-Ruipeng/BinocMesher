// Private semantic checkpoint for the exact single-group native bisection gate.
// Link into the same core.so/native executable as original source definitions.
// No GPU calls. Original algorithms remain unchanged. This is NOT a complete
// mesher snapshot and requires the matching graph/fine-node files on disk.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <type_traits>
#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"
#include "bisection.h"

namespace bm_checkpoint {
constexpr uint64_t MAGIC = UINT64_C(0x3154504b434d4243);
constexpr uint32_t VERSION = 1;
constexpr uint64_t MAX_BYTES = UINT64_C(536870912);
constexpr uint32_t MAX_COUNT = 10000000;

void check(bool ok,const char* text) {if(!ok)throw std::runtime_error(text);}
struct IO {
    std::fstream file;
    uint64_t transferred=0;
    explicit IO(const char* path,bool writing) {
        check(path && *path,"empty checkpoint path");
        if(writing) {
            check(!std::filesystem::exists(path),"checkpoint output already exists");
            file.open(path,std::ios::binary|std::ios::out);
        } else {
            check(std::filesystem::file_size(path)<=MAX_BYTES,"checkpoint exceeds private size limit");
            file.open(path,std::ios::binary|std::ios::in);
        }
        check(file.good(),"cannot open checkpoint");
    }
    template<class V> void put(V value) {
        static_assert(std::is_arithmetic<V>::value,"POD scalar only");
        check(transferred+sizeof(V)<=MAX_BYTES,"checkpoint size limit");
        file.write(reinterpret_cast<const char*>(&value),sizeof(V));transferred+=sizeof(V);
        check(file.good(),"checkpoint write failed");
    }
    template<class V> V get() {
        static_assert(std::is_arithmetic<V>::value,"POD scalar only");
        check(transferred+sizeof(V)<=MAX_BYTES,"checkpoint size limit");
        V value{};
        file.read(reinterpret_cast<char*>(&value),sizeof(V));transferred+=sizeof(V);
        check(file.good(),"short checkpoint");return value;
    }
    void count(uint32_t n) {check(n<=MAX_COUNT,"vec too large");put<uint32_t>(n);}
    uint32_t count() {auto n=get<uint32_t>();check(n<=MAX_COUNT,"vec too large");return n;}
};
bool exact(double a,double b) {return std::memcmp(&a,&b,sizeof(a))==0;}
void boundary() {
    check(params::center && params::n_elements>0 && params::max_tL==0,
          "checkpoint is limited to one initialized group: max_tL=0");
    check(dual_contouring::cache_map.empty() && dual_contouring::cache_map2.empty() &&
          dual_contouring::priority.empty() && dual_contouring::priority2.empty(),
          "graph/edge cache must already be flushed by original clean_cache");
    check(dual_contouring::inview_tag.empty(),"inview_tag must be empty before init_t");
    for(int i=0;i<CACHECNT;++i)
        check(!dual_contouring::occupied[i] && !dual_contouring::occupied2[i] &&
              dual_contouring::edges_fp_cache[i]==nullptr,
              "cache slot/open edge stream remains live");
}
void header(IO& io,bool write) {
    const uint32_t machine_sizes[]={sizeof(int),sizeof(double),sizeof(timeT),sizeof(node),
                                    sizeof(medium_cube),CACHECNT,N_THREAD};
    if(write) {
        io.put<uint64_t>(MAGIC);io.put<uint32_t>(VERSION);
        for(auto value:machine_sizes)io.put<uint32_t>(value);
        io.put<int32_t>(params::n_elements);io.put<int32_t>(params::max_tL);
        for(int d=0;d<3;++d)io.put<double>(params::center[d]);
        io.put<double>(params::size);io.put<double>(params::tsize);
    } else {
        check(io.get<uint64_t>()==MAGIC && io.get<uint32_t>()==VERSION,"private checkpoint schema mismatch");
        for(auto value:machine_sizes)check(io.get<uint32_t>()==value,"native ABI size/config mismatch");
        check(io.get<int32_t>()==params::n_elements && io.get<int32_t>()==params::max_tL,
              "load_parameters must precede checkpoint import with same scene");
        for(int d=0;d<3;++d)check(exact(io.get<double>(),params::center[d]),"scene center mismatch");
        check(exact(io.get<double>(),params::size) && exact(io.get<double>(),params::tsize),"scene extent mismatch");
    }
}
void put_cube(IO& io,const hypercube& c) {
    for(int d=0;d<3;++d)io.put<int32_t>(c.coords[d]);
    io.put<int32_t>(c.tcoord);io.put<int32_t>(c.L);io.put<int32_t>(c.tL);
}
hypercube get_cube(IO& io) {
    hypercube c{};
    for(int d=0;d<3;++d)c.coords[d]=io.get<int32_t>();
    int tc=io.get<int32_t>(),L=io.get<int32_t>(),tL=io.get<int32_t>();
    check(tc>=0 && tc<=127 && L>=0 && L<31 && tL>=0 && tL<7,"invalid cube metadata");
    c.tcoord=static_cast<timeT>(tc);c.L=static_cast<lT>(L);c.tL=static_cast<lT>(tL);return c;
}
void put_ints(IO& io,vec<int,int>& v) {
    io.count(static_cast<uint32_t>(v.size()));
    for(int i=0;i<v.size();++i)io.put<int32_t>(v[i]);
}
std::vector<int> get_ints(IO& io) {
    std::vector<int> v(io.count());
    for(auto& x:v)x=io.get<int32_t>();
    return v;
}
void put_nodes(IO& io,vec<int,node>& v,bool rearranged) {
    io.count(static_cast<uint32_t>(v.size()));
    for(int i=0;i<v.size();++i) {
        auto& n=v[i];put_cube(io,n.c);io.put<int32_t>(n.nxts[0]);
        if(!rearranged)continue;
        if(leaf_node(n))io.put<int32_t>(n.nxts[1]);
        else {
            const int count=is_tsplit(n)?2:8;io.put<int32_t>(count);
            for(int j=0;j<count;++j)io.put<int32_t>(n.nxts[j]);
        }
    }
}
std::vector<node> get_nodes(IO& io,bool rearranged) {
    std::vector<node> v(io.count());  // initialize unused fields; never read original indeterminate bytes
    for(auto& n:v) {
        n=node{};n.c=get_cube(io);n.nxts[0]=io.get<int32_t>();
        if(!rearranged)continue;
        if(leaf_node(n))n.nxts[1]=io.get<int32_t>();
        else {
            check(n.nxts[0]>=0,"invalid rearranged node tag");
            int count=io.get<int32_t>();check(count==2 || count==8,"invalid child count");
            const int first=n.nxts[0];
            for(int j=0;j<count;++j)n.nxts[j]=io.get<int32_t>();
            check(n.nxts[0]==first,"first child/tag mismatch");
            if(count==2)n.nxts[2]=-1;
        }
    }
    return v;
}
void put_medium(IO& io,vec<int,medium_cube>& v) {
    io.count(static_cast<uint32_t>(v.size()));
    for(int i=0;i<v.size();++i) {
        io.put<int32_t>(v[i].first);
        for(int d=0;d<3;++d)io.put<int32_t>(v[i].second[d]);
    }
}
std::vector<medium_cube> get_medium(IO& io) {
    std::vector<medium_cube> v(io.count());
    for(auto& x:v) {
        x.first=io.get<int32_t>();
        for(int d=0;d<3;++d) {
            int q=io.get<int32_t>();check(q>=-128 && q<=127,"medium coordinate overflow");
            x.second[d]=static_cast<int8_t>(q);
        }
    }
    return v;
}
struct State {
    std::vector<node> coarse_nodes,rearranged_nodes;
    std::vector<int> coarse_leaves,fine_sizes,group_start,group_end,tree_sizes;
    std::vector<medium_cube> visible;
};
void validate(State& s) {
    check(!s.coarse_nodes.empty() && !s.rearranged_nodes.empty(),"missing coarse/rearranged nodes");
    check(s.group_start.size()==s.rearranged_nodes.size() &&
          s.group_end.size()==s.rearranged_nodes.size(),"group range size mismatch");
    check(s.fine_sizes.size()>=s.visible.size() && s.tree_sizes.size()==1 && s.tree_sizes[0]>0,
          "fine/tree sizes missing");
    for(int i:s.coarse_leaves)
        check(i>=0 && i<static_cast<int>(s.coarse_nodes.size()) && leaf_node(s.coarse_nodes[i]),"coarse leaf index invalid");
    for(const auto& mc:s.visible)
        check(mc.first>=0 && mc.first<static_cast<int>(s.coarse_leaves.size()),"medium coarse-leaf index invalid");
    for(int size:s.fine_sizes)check(size>=1,"invalid fine subtree size");
    for(int i=0;i<static_cast<int>(s.rearranged_nodes.size());++i) {
        auto& n=s.rearranged_nodes[i];
        check(s.group_start[i]>=-1 && s.group_end[i]>=-1 &&
              s.group_start[i]<=s.group_end[i] && s.group_end[i]<=0,"unsupported group range");
        if(leaf_node(n)) {
            const int label=get_node_label(n);
            check(label>=-1 && label<static_cast<int>(s.visible.size()),"rearranged leaf label invalid");
        } else {
            const int count=is_tsplit(n)?2:8;
            for(int j=0;j<count;++j)
                check(n.nxts[j]>=0 && n.nxts[j]<static_cast<int>(s.rearranged_nodes.size()),"child index invalid");
        }
    }
    // Original recursive loader assumes a tree; reject cycles and duplicate
    // child ownership before permitting native traversal of the private file.
    std::vector<uint8_t> seen(s.rearranged_nodes.size(),0);
    std::vector<int> pending{0};
    while(!pending.empty()) {
        int i=pending.back();pending.pop_back();
        check(!seen[i],"rearranged graph is not a tree");seen[i]=1;
        auto& n=s.rearranged_nodes[i];
        if(!leaf_node(n))for(int j=0;j<(is_tsplit(n)?2:8);++j)pending.push_back(n.nxts[j]);
    }
    for(auto x:seen)check(x!=0,"unreachable rearranged node");
}
template<class V,class S>void commit(V& target,S& source) {
    target.a.swap(source);target._size=static_cast<int>(target.a.size());
}
} // namespace bm_checkpoint

extern "C" int bm_checkpoint_write(const char* path) {
    using namespace bm_checkpoint;
    try {
        boundary();
        check(!coarse::nodes.empty() && !rearrange::nodes.empty(),"required in-memory state already cleaned");
        IO io(path,true);header(io,true);
        // Exact read-only dependency closure of load_group(t,0,1) for group 0.
        put_nodes(io,coarse::nodes,false);
        put_ints(io,coarse::leaf_nodes_vector);
        put_medium(io,medium::visible_cubes);
        put_ints(io,fine::visible_cubes_nodes_size);
        put_nodes(io,rearrange::nodes,true);
        put_ints(io,rearrange::group_id_start);put_ints(io,rearrange::group_id_end);
        put_ints(io,rearrange::tree_sizes);
        io.file.flush();check(io.file.good(),"checkpoint flush failed");
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"bm_checkpoint_write: "<<e.what()<<std::endl;return 1;
    }
}
extern "C" int bm_checkpoint_read(const char* path) {
    using namespace bm_checkpoint;
    try {
        boundary();
        check(coarse::nodes.empty() && rearrange::nodes.empty() &&
              medium::visible_cubes.empty() && fine::visible_cubes_nodes_size.empty(),
              "checkpoint import requires a fresh native process");
        IO io(path,false);header(io,false);State s;
        s.coarse_nodes=get_nodes(io,false);
        s.coarse_leaves=get_ints(io);
        s.visible=get_medium(io);
        s.fine_sizes=get_ints(io);
        s.rearranged_nodes=get_nodes(io,true);
        s.group_start=get_ints(io);s.group_end=get_ints(io);s.tree_sizes=get_ints(io);
        check(io.file.peek()==std::fstream::traits_type::eof(),"trailing checkpoint data");
        validate(s);
        // No global mutation until complete bounded parse and semantic checks.
        commit(coarse::nodes,s.coarse_nodes);
        commit(coarse::leaf_nodes_vector,s.coarse_leaves);
        commit(medium::visible_cubes,s.visible);
        commit(fine::visible_cubes_nodes_size,s.fine_sizes);
        commit(rearrange::nodes,s.rearranged_nodes);
        commit(rearrange::group_id_start,s.group_start);
        commit(rearrange::group_id_end,s.group_end);
        commit(rearrange::tree_sizes,s.tree_sizes);
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"bm_checkpoint_read: "<<e.what()<<std::endl;return 1;
    }
}
