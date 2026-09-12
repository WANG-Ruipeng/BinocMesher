// Common checkpoint-to-native-output bridge for Task 01.
// This translation unit adds no field evaluations, timers, or upstream edits.
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"
#include "bisection.h"

extern "C" int bm_checkpoint_read(const char*);
extern "C" void bm_replay_invalidate();
gridID edge_key_fn(queriedEdge&, int);

namespace bm_common {
static thread_local std::string error;
static double owned_center[3] = {0, 0, 0};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Fn> int protect(Fn fn) {
    error.clear();
    try { fn(); return 0; }
    catch (const std::exception& e) { error=e.what(); return 1; }
    catch (...) { error="unknown native common exception"; return 1; }
}
template<class V> void release_vec(V& v) {
    // vec::clear alone retains both storage and previously written elements.
    // Release the backing store so inactive vertex_map entries start at zero,
    // exactly as they do on the first resize in a fresh native process.
    decltype(v.a) empty;
    v.a.swap(empty);
    v._size=0;
}
template<class V> void release_container(V& v) {
    V empty;
    v.swap(empty);
}
struct Field {
    int first_center=0, first_query=0;
    std::vector<int32_t> offsets{0}, metadata;
    std::vector<int64_t> heads;
    std::vector<double> centers, endpoints;
};
struct NodeRef {
    ll head=0;
    int center=0, field=0, local=0;
};
struct Plan {
    bool valid=false;
    ll first=0, last=-1;
    int previous_vertices=0, node_count=0;
    int64_t query_count=0;
    std::vector<Field> fields;
    std::vector<NodeRef> nodes;
};
static Plan plan;

void discard_state() {
    // This API is for the single-group bisection checkpoint closure, not a
    // live mesh construction pipeline. Refuse append streams: fclose could
    // flush pending writes into the supposedly immutable checkpoint.
    for (int i=0;i<CACHECNT;++i)
        require(dual_contouring::edges_fp_cache[i]==nullptr,
                "reset refuses a live edge output stream");
    plan=Plan{};
    bm_replay_invalidate();
    release_vec(coarse::nodes);
    release_vec(coarse::leaf_nodes_vector);
    release_vec(coarse::nodemap);
    release_vec(medium::visible_cubes);
    release_vec(fine::visible_cubes_nodes_size);
    release_vec(fine::active_nodes);
    release_vec(fine::mask);
    release_vec(rearrange::nodes);
    release_vec(rearrange::group_id_start);
    release_vec(rearrange::group_id_end);
    release_vec(rearrange::tree_sizes);
    release_vec(rearrange::framing_cubes);
    release_vec(rearrange::visited_subtree);
    release_vec(rearrange::flatten_nodes);
    release_vec(rearrange::info_offset_rearranged_vec);
    using namespace dual_contouring;
    release_container(cache_map);release_container(cache_map2);
    release_container(priority);release_container(priority2);
    for (int i=0;i<CACHECNT;++i) {
        release_vec(head_cache[i]);release_vec(nxt_cache[i]);
        release_vec(edge_pointer_cache[i]);release_vec(edges_cache[i]);
        occupied[i]=occupied2[i]=0;
        edges_size_cache[i]=0;
        edges_fp_cache[i]=nullptr;
    }
    head_=nullptr;nxt_=nullptr;edge_pointer_=nullptr;edges_=nullptr;
    edges_fp_=nullptr;edges_size_=nullptr;
    release_vec(inview_tag);
    release_vec(restricted_set);
    release_vec(node_labels);release_vec(subtree_st);release_vec(subtree_ed);
    release_vec(info_offset_rearranged);release_vec(pointers);
    release_container(output_vertices);release_container(output_vertices_index);
    release_container(unfinalized_edges);release_container(finalized_edges);
    release_container(new_unfinalized_edges);release_container(pending_edges);
    release_container(propagated_edges);release_container(absorbed);
    release_container(backref);release_container(pending_edges_results);
    for (int g=0;g<N_THREAD;++g) {
        release_vec(vertices[g]);release_vec(nodes_queue[g]);
        release_container(output_vertices_buffer[g]);
        release_container(output_vertices_index_buffer[g]);
        release_container(pending_edges_buffer[g]);
    }
    unfinalized_edges_sorted_cnt=0;t_group=-1;
    release_vec(bisection::query_cnt);
    release_vec(bisection::lefts);release_vec(bisection::rights);
    release_vec(bisection::starts);release_vec(bisection::center_indices);
    release_vec(bisection::vertex_map);release_vec(bisection::computed_vertices);
    release_vec(bisection::hyperpolys);release_vec(bisection::hypervertices_vec);
    release_container(bisection::hypervertices);
    for (int g=0;g<N_THREAD;++g) release_container(bisection::hypervertices_g[g]);
    bisection::last_head=0;bisection::current_head=-1;bisection::bisection_group=0;
    release_vec(params::cams);release_vec(params::cam_lookup);
    // The external-loader path calls original load_parameters after reset.
    // Never free its previous Python-owned center; merely detach the pointer.
    params::center=nullptr;params::n_elements=0;params::max_tL=0;
    params::output_path.clear();params::log_path.clear();
}
void loaded_group() {
    require(params::center && params::n_elements>0 && params::max_tL==0,
            "only initialized max_tL=0 single-group scenes are supported");
    require(dual_contouring::t_group==0 && dual_contouring::head_ &&
            dual_contouring::nxt_ && dual_contouring::edge_pointer_ &&
            dual_contouring::edges_,"original bisection_init_t(0) must precede this call");
    require(dual_contouring::head_->size()==
            static_cast<ll>(fine::active_nodes.size())*params::n_elements,
            "native graph/node dimensions differ");
    require(bisection::query_cnt.size()==dual_contouring::head_->size() &&
            bisection::vertex_map.size()==dual_contouring::head_->size(),
            "original bisection group arrays are not initialized");
}
void live_plan() {
    require(plan.valid,"no prepared common batch plan");
    loaded_group();
    require(plan.first==bisection::last_head && plan.last==bisection::current_head &&
            plan.previous_vertices==bisection::computed_vertices.size() &&
            plan.node_count==bisection::lefts.size() &&
            plan.node_count==bisection::rights.size(),
            "native batch changed since plan creation");
}
void build_plan() {
    plan=Plan{};
    loaded_group();
    Plan next;
    next.first=bisection::last_head;
    next.last=bisection::current_head;
    next.previous_vertices=bisection::computed_vertices.size();
    next.node_count=bisection::lefts.size();
    auto& heads=*dual_contouring::head_;
    auto& nexts=*dual_contouring::nxt_;
    auto& edge_ptrs=*dual_contouring::edge_pointer_;
    auto& edges=*dual_contouring::edges_;
    const int ne=params::n_elements;
    require(next.node_count>0 && next.node_count==bisection::rights.size(),
            "plan_batch requires a successful original bisection_hypermesh_verts");
    require(next.first>=0 && next.first<=next.last && next.last<heads.size() &&
            next.first%ne==0 && (next.last+1)%ne==0,
            "invalid or unaligned native batch head interval");
    require(next.last-next.first+1<=INT_MAX &&
            bisection::starts.size()==next.last-next.first+1 &&
            bisection::center_indices.size()==next.last-next.first+1,
            "native batch index arrays differ from head interval");
    require(nexts.size()==edge_ptrs.size(),"nxt/edge_pointer sizes differ");
    next.fields.resize(ne);
    int center_count=0;
    int64_t global_query=0;
    for (int ele=0;ele<ne;++ele) {
        Field& f=next.fields[ele];
        f.first_center=center_count;
        require(global_query<=INT_MAX,"native query index exceeds int32");
        f.first_query=static_cast<int>(global_query);
        for (ll h=next.first+ele;h<=next.last;h+=ne) {
            if (heads[h]<0) continue; // Never read inactive starts/center_indices.
            const int relative=static_cast<int>(h-next.first);
            const int ci=bisection::center_indices[relative];
            const int start=bisection::starts[relative];
            const int count=bisection::query_cnt[h];
            require(ci==center_count && start==global_query && count>0,
                    "native center/query order or count is inconsistent");
            require(ci<next.node_count && bisection::lefts[ci]==0.0 &&
                    bisection::rights[ci]==1.0,
                    "plan must be built before the first native solve round");
            const int ni=static_cast<int>(h/ne);
            require(ni>=0 && ni<fine::active_nodes.size() &&
                    ni<dual_contouring::inview_tag.size(),"node metadata index out of range");
            auto& cube=fine::active_nodes[ni].c;
            require(cube.L>=0 && cube.L<31 && cube.tL==0 && cube.tcoord>=0,
                    "unsupported cube precision/time metadata");
            double center[3];
            compute_center(center,cube);
            for (double x:center) {
                require(std::isfinite(x),"nonfinite native center");
                f.centers.push_back(x);
            }
            const int local=static_cast<int>(f.heads.size());
            f.heads.push_back(h);
            const int time_half=1<<(params::max_tL-cube.tL);
            const int time_mid=(cube.tcoord<<(1+params::max_tL-cube.tL))+time_half;
            require(time_half<=127 && time_mid<=127,"native time metadata would narrow");
            const int meta[6]={ni,ci,start,time_mid,time_half,
                              dual_contouring::inview_tag[ni]};
            f.metadata.insert(f.metadata.end(),meta,meta+6);
            set<gridID> seen;
            int links=0;
            const int before=f.offsets.back();
            for (int link=heads[h];link!=-1;link=nexts[link]) {
                require(link>=0 && link<nexts.size() && ++links<=nexts.size(),
                        "invalid or cyclic native adjacency");
                const int ep=edge_ptrs[link];
                require(ep>=0 && ep<edges.size(),"native edge index out of range");
                auto& ed=edges[ep];
                require(ed.e.dir>=0 && ed.e.dir<3 && ed.e.L>=0 && ed.e.L<31,
                        "invalid edge direction or level");
                require(ed.e.coords[ed.e.dir]<INT_MAX,"edge endpoint int overflow");
                for (int side=0;side<2;++side) {
                    // Reuse original canonical key, but retain encounter order,
                    // not std::set's sorted order, for the field-local CSR.
                    const auto key=edge_key_fn(ed,side);
                    if (!seen.insert(key).second) continue;
                    int raw[3]={ed.e.coords[0],ed.e.coords[1],ed.e.coords[2]};
                    raw[ed.e.dir]+=side;
                    double endpoint[3];
                    compute_coords(endpoint,raw,ed.e.L);
                    for (double x:endpoint) {
                        require(std::isfinite(x),"nonfinite native endpoint");
                        f.endpoints.push_back(x);
                    }
                }
            }
            require(seen.size()==static_cast<size_t>(count),
                    "CSR endpoint count differs from original query_cnt");
            require(static_cast<int64_t>(before)+count<=INT_MAX,
                    "field CSR offsets exceed int32");
            f.offsets.push_back(before+count);
            next.nodes.push_back(NodeRef{h,ci,ele,local});
            ++center_count;global_query+=count;
        }
    }
    require(center_count==next.node_count && global_query<=INT_MAX,
            "prepared native batch totals differ");
    next.query_count=global_query;
    next.valid=true;
    plan=std::move(next);
}
uint32_t float_bits(float x) {
    uint32_t bits;
    std::memcpy(&bits,&x,sizeof(bits));
    return bits;
}
} // namespace bm_common

extern "C" const char* bm_common_last_error() {
    return bm_common::error.c_str();
}
extern "C" int bm_common_reset() {
    return bm_common::protect([] { bm_common::discard_state(); });
}
// Optional convenience loader. The external Python loader is equally valid:
// reset -> original load_parameters -> checkpoint_read -> load_tree_size
// -> bisection_init -> bisection_init_t(0).
extern "C" int bm_common_begin(
    const char* memory_file,const char* data_dir,const double* center3,
    double size,double tsize,int ncam,const double* cameras27,double fading,
    double ppc,double ppc_coarse,double ppc_out,double min_dist,
    int n_elements,int batch_group) {
    using namespace bm_common;
    return protect([&] {
        require(memory_file && *memory_file && data_dir && *data_dir,
                "checkpoint paths are empty");
        require(center3 && cameras27 && ncam>0 && ncam<=1000000,
                "invalid center/camera arguments");
        require(n_elements>0 && n_elements<=127 && batch_group>0,
                "invalid element or batch count");
        require(std::isfinite(size) && size>0 && std::isfinite(tsize) && tsize>0 &&
                std::isfinite(fading) && fading>0 && std::isfinite(ppc) && ppc>0 &&
                std::isfinite(ppc_coarse) && ppc_coarse>0 &&
                std::isfinite(ppc_out) && ppc_out>0 &&
                std::isfinite(min_dist) && min_dist>0,"invalid scene parameters");
        const std::string cp(memory_file),dir(data_dir);
        double center_copy[3]={center3[0],center3[1],center3[2]};
        for (double x:center_copy) require(std::isfinite(x),"nonfinite scene center");
        std::vector<double> cams(cameras27,cameras27+static_cast<int64_t>(ncam)*27);
        std::vector<double> times;
        for (int i=0;i<ncam;++i) {
            const int64_t base=static_cast<int64_t>(i)*27;
            for (int j=0;j<27;++j)
                require(std::isfinite(cams[base+j]),"nonfinite camera parameter");
            require(cams[base+21]>=1 && cams[base+21]<=INT_MAX &&
                    cams[base+22]>=1 && cams[base+22]<=INT_MAX,"invalid camera extent");
            times.push_back(cams[base+23]);
        }
        sort(times.begin(),times.end());
        double frame=std::numeric_limits<double>::infinity();
        for (int i=1;i<ncam;++i)
            if (times[i]!=times[i-1]) frame=smin(frame,times[i]-times[i-1]);
        require(smax(fading,frame)*2>=tsize,"common loader supports max_tL=0 only");
        require(std::filesystem::is_regular_file(cp),"memory checkpoint is missing");
        const char* files[]={"rearranged_fine_nodes/0.bin","rearranged_nodemap/0.bin",
                             "graphs/0.bin","bip_edges/0.bin","tree_sizes.bin"};
        for (const char* file:files) {
            const auto p=std::filesystem::path(dir)/file;
            require(std::filesystem::is_regular_file(p),"required group checkpoint file is missing");
            require(std::filesystem::file_size(p)<=UINT64_C(536870912),
                    "group file exceeds private size limit");
        }
        require(std::filesystem::file_size(std::filesystem::path(dir)/"tree_sizes.bin")==sizeof(int),
                "single-group tree_sizes.bin must contain exactly one int");
        discard_state();
        for (int d=0;d<3;++d) owned_center[d]=center_copy[d];
        std::vector<char> path(dir.begin(),dir.end());path.push_back('\0');
        const int groups=load_parameters(owned_center,size,tsize,ncam,cams.data(),
                                         fading,ppc,ppc_coarse,ppc_out,min_dist,
                                         n_elements,path.data());
        require(groups==1 && params::max_tL==0,"original parameters produced multiple groups");
        require(bm_checkpoint_read(cp.c_str())==0,"bm_checkpoint_read failed; inspect its stderr");
        const int expected_tree=rearrange::tree_sizes[0];
        load_tree_size();
        require(rearrange::tree_sizes[0]==expected_tree,
                "memory checkpoint/tree_sizes.bin mismatch");
        bisection_init(batch_group);
        bisection_init_t(0);
        loaded_group();
    });
}
extern "C" int bm_common_plan_batch() {
    return bm_common::protect([] { bm_common::build_plan(); });
}
// sizes4 = [N, M, original global center base, original global query base].
extern "C" int bm_common_field_sizes(int field,int64_t* sizes4) {
    using namespace bm_common;
    return protect([&] {
        live_plan();
        require(sizes4 && field>=0 && field<static_cast<int>(plan.fields.size()),
                "invalid field or sizes pointer");
        const auto& f=plan.fields[field];
        sizes4[0]=static_cast<int64_t>(f.heads.size());
        sizes4[1]=f.offsets.back();
        sizes4[2]=f.first_center;sizes4[3]=f.first_query;
    });
}
extern "C" int bm_common_copy_field(
    int field,int64_t n,int64_t m,int32_t* offsets,double* centers,double* endpoints,
    int64_t* head_ids,int32_t* metadata6) {
    using namespace bm_common;
    return protect([&] {
        live_plan();
        require(field>=0 && field<static_cast<int>(plan.fields.size()),"invalid field");
        const auto& f=plan.fields[field];
        require(n==static_cast<int64_t>(f.heads.size()) && m==f.offsets.back(),
                "field output capacities differ from reported dimensions");
        require(offsets && (!n || (centers && head_ids && metadata6)) &&
                (!m || endpoints),"null field output array");
        std::copy(f.offsets.begin(),f.offsets.end(),offsets);
        if (n) {
            std::copy(f.centers.begin(),f.centers.end(),centers);
            std::copy(f.heads.begin(),f.heads.end(),head_ids);
            std::copy(f.metadata.begin(),f.metadata.end(),metadata6);
        }
        if (m) std::copy(f.endpoints.begin(),f.endpoints.end(),endpoints);
    });
}
extern "C" int bm_common_commit(
    int64_t n,const float* positions,const int32_t* valid,const int32_t* witness,
    const double* left,const double* right) {
    using namespace bm_common;
    return protect([&] {
        live_plan();
        require(n==plan.node_count && positions && valid && witness && left && right,
                "invalid candidate commit dimensions/pointers");
        require(n+plan.previous_vertices<=INT_MAX,"native output count exceeds int32");
        std::vector<HV> values(static_cast<size_t>(n));
        for (const auto& ref:plan.nodes) {
            const int ci=ref.center;
            const auto& f=plan.fields[ref.field];
            const int local=ref.local;
            const int first=f.offsets[local],count=f.offsets[local+1]-first;
            require(valid[ci]==1 && witness[ci]>=0 && witness[ci]<count,
                    "candidate has no valid node-local witness");
            require(std::isfinite(left[ci]) && std::isfinite(right[ci]) &&
                    left[ci]>=0 && left[ci]<=right[ci] && right[ci]<=1,
                    "candidate has invalid bounds");
            auto& value=values[ci];
            for (int d=0;d<3;++d) {
                const float supplied=positions[3*static_cast<int64_t>(ci)+d];
                require(std::isfinite(supplied),"candidate position is nonfinite");
                const double c=f.centers[3*static_cast<int64_t>(local)+d];
                const double endpoint=f.endpoints[3*static_cast<int64_t>(first+witness[ci])+d];
                const float expected=static_cast<float>(c*(1-right[ci])+endpoint*right[ci]);
                require(float_bits(supplied)==float_bits(expected),
                        "candidate position differs from original finishing arithmetic");
                value.first.first[d]=supplied;
            }
            const int* meta=&f.metadata[6*static_cast<int64_t>(local)];
            value.first.second[0]=static_cast<timeT>(meta[3]);
            value.first.second[1]=static_cast<timeT>(meta[4]);
            value.second=static_cast<int8_t>(meta[5]);
        }
        // All semantic checks and temporary allocations precede native output
        // mutation. Native HV order is previous_vertices + original center.
        bisection::computed_vertices.resize(plan.previous_vertices+plan.node_count);
        for (const auto& ref:plan.nodes) {
            const int ci=ref.center,index=plan.previous_vertices+ci;
            bisection::computed_vertices[index]=values[ci];
            bisection::vertex_map[ref.head]=index;
            bisection::lefts[ci]=left[ci];bisection::rights[ci]=right[ci];
        }
        bisection::last_head=plan.last+1;
        plan.valid=false;
    });
}
