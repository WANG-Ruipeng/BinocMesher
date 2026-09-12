// Observations and bounded replay snapshots of the fixed native bisection state.
// Compile with the original source directory on the include path. This unit
// deliberately owns no mesher state and performs no field evaluations.
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <limits>
#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"
#include "bisection.h"

#ifdef _WIN32
#define CAPTURE_API extern "C" __declspec(dllexport)
#else
#define CAPTURE_API extern "C"
#endif

namespace native_capture {
struct Key {
    int xyz[3], level;
    bool operator<(const Key& other) const {
        for (int d = 0; d < 3; ++d) {
            if (xyz[d] != other.xyz[d]) return xyz[d] < other.xyz[d];
        }
        return level < other.level;
    }
};
struct Endpoint {
    Key key;
    int raw_xyz[3], level, link_rank, link_index, edge_index, side;
    double xyz[3];
};
struct Link {
    int index, next, edge_index;
    queriedEdge edge_data;
};
struct Node {
    long long head_index;
    int node_index, element, center_index, start, original_count;
    int expected_vertex_map, time_mid, time_halfwidth, inview;
    double center[3], previous_left, previous_right;
    std::vector<Endpoint> endpoints;
};
struct Snapshot {
    bool valid = false;
    int group = -1, previous_vertices = 0;
    long long first = 0, last = -1;
    long long endpoints = 0;
    std::vector<Node> nodes;
};
static Snapshot captured;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void quoted(std::ostream& out, const char* s) {
    out << '"';
    for (; *s; ++s) {
        const unsigned char c = static_cast<unsigned char>(*s);
        if (c == '"' || c == '\\') out << '\\' << *s;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << '?';
        else out << *s;
    }
    out << '"';
}
void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}
void xyz(std::ostream& out, const double* values) {
    out << '[';
    for (int d = 0; d < 3; ++d) {
        if (d) out << ',';
        number(out, values[d]);
    }
    out << ']';
}
void ints3(std::ostream& out, const int* values) {
    out << '[' << values[0] << ',' << values[1] << ',' << values[2] << ']';
}
uint32_t bits32(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
uint64_t bits64(double value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
std::ofstream output(const char* path) {
    require(path != nullptr && *path != '\0', "capture output path is empty");
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    require(out.good(), "cannot open capture output path");
    out << std::scientific << std::setprecision(17);
    return out;
}
int failure(const char* path, const char* function, const char* message) {
    if (path != nullptr && *path != '\0') {
        std::ofstream out(path, std::ios::out | std::ios::app);
        out << "{\"type\":\"error\",\"function\":";
        quoted(out, function);
        out << ",\"message\":";
        quoted(out, message);
        out << "}\n";
    }
    return 1;
}
Key independently_normalized_key(const edge& ed, int side) {
    Key key{{ed.coords[0], ed.coords[1], ed.coords[2]}, ed.L};
    require(ed.dir >= 0 && ed.dir < 3, "invalid edge direction");
    require(side == 0 || key.xyz[ed.dir] != INT_MAX, "edge endpoint overflows int");
    key.xyz[ed.dir] += side;
    while (key.level > 0 && key.xyz[0] % 2 == 0 &&
           key.xyz[1] % 2 == 0 && key.xyz[2] % 2 == 0) {
        for (int d = 0; d < 3; ++d) key.xyz[d] /= 2;
        --key.level;
    }
    return key;
}
void identity(std::ostream& out, const Node& node) {
    out << "\"key\":[" << captured.group << ',' << node.head_index
        << "],\"node_index\":" << node.node_index
        << ",\"element\":" << node.element
        << ",\"center_index\":" << node.center_index
        << ",\"start\":" << node.start;
}
void field_values(std::ostream& out, const Node& node,
                  const float* sdfs, const float* center_sdfs) {
    out << ",\"center_sdf\":";
    number(out, center_sdfs[node.center_index]);
    out << ",\"center_sdf_bits\":" << bits32(center_sdfs[node.center_index]);
    out << ",\"endpoint_sdfs\":[";
    for (size_t r = 0; r < node.endpoints.size(); ++r) {
        if (r) out << ',';
        number(out, sdfs[node.start + r]);
    }
    out << "],\"endpoint_sdf_bits\":[";
    for (size_t r = 0; r < node.endpoints.size(); ++r) {
        if (r) out << ',';
        out << bits32(sdfs[node.start + r]);
    }
    out << ']';
}
} // namespace native_capture

// Call after a successful bisection_hypermesh_verts, before any native round.
CAPTURE_API int bm_capture_input(const char* path, int t) {
    using namespace native_capture;
    try {
        captured = Snapshot{};
        captured.group = t;
        captured.first = bisection::last_head;
        captured.last = bisection::current_head;
        captured.previous_vertices = bisection::computed_vertices.size();
        require(dual_contouring::head_ && dual_contouring::nxt_ &&
                dual_contouring::edge_pointer_ && dual_contouring::edges_,
                "native adjacency is not loaded");
        auto& heads = *dual_contouring::head_;
        auto& nexts = *dual_contouring::nxt_;
        auto& pointers = *dual_contouring::edge_pointer_;
        auto& edges = *dual_contouring::edges_;
        require(params::n_elements > 0, "invalid element count");
        require(captured.first >= 0 && captured.first <= heads.size(), "invalid first head");
        require(captured.last < heads.size(), "invalid last head");
        auto out = output(path);
        out << "{\"type\":\"header\",\"schema\":\"binocmesher.native_capture.v2\","
               "\"kind\":\"input\",\"group\":" << t
            << ",\"first_head\":" << captured.first << ",\"last_head\":" << captured.last
            << ",\"head_size\":" << heads.size() << ",\"nxt_size\":" << nexts.size()
            << ",\"edge_pointer_size\":" << pointers.size() << ",\"edges_size\":" << edges.size()
            << ",\"n_elements\":" << params::n_elements << ",\"max_tL\":" << params::max_tL
            << ",\"previous_vertices\":" << captured.previous_vertices
            << ",\"bisection_group\":" << bisection::bisection_group << ",\"scene_center\":";
        xyz(out, params::center);
        out << ",\"scene_size\":"; number(out, params::size);
        out << ",\"scene_tsize\":"; number(out, params::tsize);
        out << "}\n";
        out << "{\"type\":\"graph_arrays\",\"head\":[";
        for (ll i=0; i<heads.size(); ++i) { if(i) out << ','; out << heads[i]; }
        out << "],\"nxt\":[";
        for (int i=0; i<nexts.size(); ++i) { if(i) out << ','; out << nexts[i]; }
        out << "],\"edge_pointer\":[";
        for (int i=0; i<pointers.size(); ++i) { if(i) out << ','; out << pointers[i]; }
        out << "]}\n";
        for (int i=0; i<edges.size(); ++i) {
            auto& ed=edges[i];
            out << "{\"type\":\"graph_edge\",\"index\":" << i << ",\"coords\":";
            ints3(out,ed.e.coords);
            out << ",\"L\":" << int(ed.e.L) << ",\"tL\":" << int(ed.e.tL)
                << ",\"element\":" << int(ed.e.ele) << ",\"tcoord\":" << int(ed.e.tcoord)
                << ",\"dir\":" << int(ed.e.dir) << ",\"vertices_nid\":[";
            for(int k=0;k<8;++k) {if(k) out<<',';out<<ed.vertices_nid[k];}
            out << "],\"vertices_gid\":[";
            for(int k=0;k<8;++k) {if(k) out<<',';out<<int(ed.vertices_gid[k]);}
            out << "]}\n";
        }
        int count_mismatches = 0;
        for (int element = 0; element < params::n_elements; ++element) {
            for (long long i = captured.first + element; i <= captured.last; i += params::n_elements) {
                const int original_head = heads[i];
                if (original_head < 0) {
                    out << "{\"type\":\"inactive_head\",\"key\":[" << t << ',' << i
                        << "],\"head\":" << original_head << "}\n";
                    continue;
                }
                require(i < bisection::query_cnt.size(), "query count index out of range");
                const int local = static_cast<int>(i - captured.first);
                require(local >= 0 && local < bisection::starts.size() &&
                        local < bisection::center_indices.size(), "native batch index out of range");
                Node node{};
                node.head_index = i;
                node.node_index = static_cast<int>(i / params::n_elements);
                node.element = static_cast<int>(i % params::n_elements);
                node.center_index = bisection::center_indices[local];
                node.start = bisection::starts[local];
                node.original_count = bisection::query_cnt[i];
                node.expected_vertex_map = captured.previous_vertices + node.center_index;
                require(node.center_index >= 0 && node.center_index < bisection::lefts.size() &&
                        node.center_index < bisection::rights.size(), "center index out of range");
                require(node.node_index >= 0 && node.node_index < fine::active_nodes.size() &&
                        node.node_index < dual_contouring::inview_tag.size(), "native node index out of range");
                auto& cube = fine::active_nodes[node.node_index].c;
                compute_center(node.center, cube);
                node.previous_left = bisection::lefts[node.center_index];
                node.previous_right = bisection::rights[node.center_index];
                const int shift = params::max_tL - cube.tL;
                require(shift >= 0 && shift < 30, "unsupported native time shift");
                node.time_halfwidth = static_cast<timeT>(1 << shift);
                node.time_mid = static_cast<timeT>((cube.tcoord << (1 + shift)) + node.time_halfwidth);
                node.inview = dual_contouring::inview_tag[node.node_index];
                std::vector<Link> links;
                set<Key> seen;
                for (int current = original_head; current != -1; current = nexts[current]) {
                    require(current >= 0 && current < nexts.size() && current < pointers.size(),
                            "native adjacency link index out of range");
                    require(links.size() <= static_cast<size_t>(nexts.size()), "cycle in native adjacency");
                    const int edge_index = pointers[current];
                    require(edge_index >= 0 && edge_index < edges.size(), "native edge pointer out of range");
                    const auto& ed = edges[edge_index];
                    const int link_rank = static_cast<int>(links.size());
                    links.push_back(Link{current, nexts[current], edge_index, ed});
                    for (int side = 0; side < 2; ++side) {
                        Key key = independently_normalized_key(ed.e, side);
                        if (!seen.insert(key).second) continue;
                        Endpoint point{};
                        point.key = key; point.level = ed.e.L;
                        point.link_rank = link_rank; point.link_index = current;
                        point.edge_index = edge_index; point.side = side;
                        for (int d = 0; d < 3; ++d) point.raw_xyz[d] = ed.e.coords[d];
                        point.raw_xyz[ed.e.dir] += side;
                        compute_coords(point.xyz, point.raw_xyz, point.level);
                        node.endpoints.push_back(point);
                    }
                }
                const bool count_match = node.endpoints.size() == static_cast<size_t>(node.original_count);
                if (!count_match) ++count_mismatches;
                captured.endpoints += node.endpoints.size();
                out << "{\"type\":\"node\","; identity(out, node);
                out << ",\"head\":" << original_head << ",\"query_cnt_native\":" << node.original_count
                    << ",\"query_cnt_independent\":" << node.endpoints.size()
                    << ",\"count_match\":" << (count_match ? "true" : "false")
                    << ",\"cube_coords\":"; ints3(out, cube.coords);
                out << ",\"cube_L\":" << int(cube.L) << ",\"cube_tL\":" << int(cube.tL)
                    << ",\"cube_tcoord\":" << int(cube.tcoord) << ",\"center\":"; xyz(out, node.center);
                out << ",\"initial_left\":"; number(out, node.previous_left);
                out << ",\"initial_right\":"; number(out, node.previous_right);
                out << ",\"expected_vertex_map\":" << node.expected_vertex_map
                    << ",\"time_mid\":" << node.time_mid << ",\"time_halfwidth\":" << node.time_halfwidth
                    << ",\"inview\":" << node.inview << ",\"links\":[";
                for (size_t j = 0; j < links.size(); ++j) {
                    if (j) out << ',';
                    const auto& link = links[j]; const auto& ed = link.edge_data;
                    out << "{\"rank\":" << j << ",\"index\":" << link.index << ",\"next\":" << link.next
                        << ",\"edge_pointer\":" << link.edge_index << ",\"coords\":"; ints3(out, ed.e.coords);
                    out << ",\"L\":" << int(ed.e.L) << ",\"tL\":" << int(ed.e.tL)
                        << ",\"element\":" << int(ed.e.ele) << ",\"tcoord\":" << int(ed.e.tcoord)
                        << ",\"dir\":" << int(ed.e.dir) << ",\"vertices_nid\":[";
                    for (int k = 0; k < 8; ++k) { if (k) out << ','; out << ed.vertices_nid[k]; }
                    out << "],\"vertices_gid\":[";
                    for (int k = 0; k < 8; ++k) { if (k) out << ','; out << int(ed.vertices_gid[k]); }
                    out << "]}";
                }
                out << "],\"endpoints\":[";
                for (size_t rank = 0; rank < node.endpoints.size(); ++rank) {
                    if (rank) out << ',';
                    const auto& point = node.endpoints[rank];
                    out << "{\"rank\":" << rank << ",\"native_query_index\":" << node.start + rank
                        << ",\"first_link_rank\":" << point.link_rank << ",\"first_link_index\":" << point.link_index
                        << ",\"edge_pointer\":" << point.edge_index << ",\"edge_endpoint\":" << point.side
                        << ",\"key_coords\":"; ints3(out, point.key.xyz);
                    out << ",\"key_L\":" << point.key.level << ",\"raw_coords\":"; ints3(out, point.raw_xyz);
                    out << ",\"raw_L\":" << point.level << ",\"xyz\":"; xyz(out, point.xyz);
                    out << '}';
                }
                out << "]}\n";
                captured.nodes.push_back(std::move(node));
            }
        }
        out << "{\"type\":\"validation\",\"N\":" << captured.nodes.size()
            << ",\"M\":" << captured.endpoints << ",\"count_mismatches\":" << count_mismatches << "}\n";
        out.flush(); require(out.good(), "capture input write failed");
        captured.valid = true;
        return count_mismatches == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        captured.valid = false;
        return failure(path, "bm_capture_input", error.what());
    }
}

// Call immediately after bisection_hypermesh_verts_iter with the same arrays.
CAPTURE_API int bm_capture_round_after(const char* path, int round, const float* sdfs, const float* center_sdfs) {
    using namespace native_capture;
    try {
        require(captured.valid, "bm_capture_input must precede bm_capture_round_after");
        require(captured.nodes.empty() || (sdfs && center_sdfs), "null native field arrays");
        auto out = output(path);
        out << "{\"type\":\"header\",\"schema\":\"binocmesher.native_capture.v2\","
               "\"kind\":\"round\",\"group\":" << captured.group << ",\"round\":" << round << "}\n";
        int mismatches = 0;
        for (auto& node : captured.nodes) {
            bool diff = false;
            for (size_t rank = 0; rank < node.endpoints.size(); ++rank)
                diff |= (sdfs[node.start + rank] >= 0) != (center_sdfs[node.center_index] >= 0);
            const double middle = (node.previous_left + node.previous_right) / 2;
            const double expected_left = diff ? node.previous_left : middle;
            const double expected_right = diff ? middle : node.previous_right;
            const double left = bisection::lefts[node.center_index];
            const double right = bisection::rights[node.center_index];
            const bool match = bits64(left) == bits64(expected_left) && bits64(right) == bits64(expected_right);
            if (!match) ++mismatches;
            out << "{\"type\":\"round_node\","; identity(out, node);
            out << ",\"round\":" << round << ",\"left_before\":"; number(out, node.previous_left);
            out << ",\"right_before\":"; number(out, node.previous_right);
            out << ",\"left\":"; number(out, left);
            out << ",\"right\":"; number(out, right);
            out << ",\"expected_left\":"; number(out, expected_left);
            out << ",\"expected_right\":"; number(out, expected_right);
            out << ",\"left_before_bits\":" << bits64(node.previous_left)
                << ",\"right_before_bits\":" << bits64(node.previous_right)
                << ",\"left_bits\":" << bits64(left) << ",\"right_bits\":" << bits64(right);
            out << ",\"sign_or\":" << (diff ? "true" : "false")
                << ",\"bounds_match\":" << (match ? "true" : "false");
            field_values(out, node, sdfs, center_sdfs); out << "}\n";
            node.previous_left = left; node.previous_right = right;
        }
        out << "{\"type\":\"validation\",\"bounds_mismatches\":" << mismatches << "}\n";
        out.flush(); require(out.good(), "capture round write failed");
        return mismatches == 0 ? 0 : 2;
    } catch (const std::exception& error) { return failure(path, "bm_capture_round_after", error.what()); }
}

// Call after finishing and before write_final_hypermesh; last_head has advanced.
CAPTURE_API int bm_capture_final(const char* path, int t, const float* sdfs, const float* center_sdfs) {
    using namespace native_capture;
    try {
        require(captured.valid && captured.group == t, "bm_capture_final has no matching input group");
        require(captured.nodes.empty() || (sdfs && center_sdfs), "null native field arrays");
        auto out = output(path);
        out << "{\"type\":\"header\",\"schema\":\"binocmesher.native_capture.v2\","
               "\"kind\":\"final\",\"group\":" << t
            << ",\"first_head_captured\":" << captured.first << ",\"last_head_captured\":" << captured.last
            << ",\"native_last_head_after_finishing\":" << bisection::last_head << "}\n";
        int mismatches = 0, missing_witnesses = 0;
        for (const auto& node : captured.nodes) {
            int witness = -1;
            for (size_t rank = 0; rank < node.endpoints.size(); ++rank) {
                if ((sdfs[node.start + rank] >= 0) != (center_sdfs[node.center_index] >= 0)) {
                    witness = static_cast<int>(rank); break;
                }
            }
            if (witness < 0) ++missing_witnesses;
            require(node.head_index < bisection::vertex_map.size(), "native vertex map index out of range");
            const int mapped = bisection::vertex_map[node.head_index];
            require(mapped >= 0 && mapped < bisection::computed_vertices.size(), "native output index out of range");
            const auto& vertex = bisection::computed_vertices[mapped];
            const double right = bisection::rights[node.center_index];
            double expected[3] = {0, 0, 0};
            bool position_match = witness >= 0;
            if (witness >= 0) for (int d = 0; d < 3; ++d) {
                expected[d] = node.center[d] * (1 - right) + node.endpoints[witness].xyz[d] * right;
                position_match &= bits32(static_cast<float>(expected[d])) == bits32(vertex.first.first[d]);
            }
            const bool map_match = mapped == node.expected_vertex_map;
            const bool metadata_match = int(vertex.first.second[0]) == node.time_mid &&
                int(vertex.first.second[1]) == node.time_halfwidth && int(vertex.second) == node.inview;
            if (!(position_match && map_match && metadata_match)) ++mismatches;
            out << "{\"type\":\"final_node\","; identity(out, node);
            out << ",\"vertex_map\":" << mapped << ",\"expected_vertex_map\":" << node.expected_vertex_map
                << ",\"witness_rank\":" << witness << ",\"left\":";
            number(out, bisection::lefts[node.center_index]);
            out << ",\"right\":"; number(out, right);
            out << ",\"reconstructed_position_fp64\":";
            if (witness >= 0) xyz(out, expected); else out << "null";
            out << ",\"native_position_fp32\":[";
            for (int d = 0; d < 3; ++d) { if (d) out << ','; number(out, vertex.first.first[d]); }
            out << "],\"native_position_fp32_bits\":[";
            for (int d = 0; d < 3; ++d) { if (d) out << ','; out << bits32(vertex.first.first[d]); }
            out << "],\"time_mid\":" << int(vertex.first.second[0])
                << ",\"time_halfwidth\":" << int(vertex.first.second[1]) << ",\"inview\":" << int(vertex.second)
                << ",\"map_match\":" << (map_match ? "true" : "false")
                << ",\"metadata_match\":" << (metadata_match ? "true" : "false")
                << ",\"position_bits_match\":" << (position_match ? "true" : "false");
            field_values(out, node, sdfs, center_sdfs); out << "}\n";
        }
        out << "{\"type\":\"validation\",\"output_mismatches\":" << mismatches
            << ",\"missing_witnesses\":" << missing_witnesses << "}\n";
        out.flush(); require(out.good(), "capture final write failed");
        return mismatches == 0 && missing_witnesses == 0 ? 0 : 2;
    } catch (const std::exception& error) { return failure(path, "bm_capture_final", error.what()); }
}


// Check before native iter/finishing can assert. Always flush every node and the
// validation record before returning 2. No native algorithm is called here.
CAPTURE_API int bm_capture_round_before(const char* path, int round,
                                        const float* sdfs, const float* center_sdfs) {
    using namespace native_capture;
    try {
        require(captured.valid, "bm_capture_input must precede round");
        require(captured.nodes.empty() || (sdfs && center_sdfs), "null field arrays");
        auto out=output(path);
        out << "{\"type\":\"header\",\"schema\":\"binocmesher.native_capture.v2\",\"kind\":\"round_before\",\"group\":"
            << captured.group << ",\"round\":" << round << "}\n";
        int invalid=0, bounds_mismatch=0;
        for (const auto& node: captured.nodes) {
            bool any=false;
            bool finite=std::isfinite(center_sdfs[node.center_index]);
            for(size_t rank=0;rank<node.endpoints.size();++rank) {
                const float value=sdfs[node.start+rank];
                finite &= std::isfinite(value);
                any |= (value>=0)!=(center_sdfs[node.center_index]>=0);
            }
            const double left=bisection::lefts[node.center_index], right=bisection::rights[node.center_index];
            const bool match=bits64(left)==bits64(node.previous_left) && bits64(right)==bits64(node.previous_right);
            invalid += !finite; bounds_mismatch += !match;
            out << "{\"type\":\"round_before_node\","; identity(out,node);
            out << ",\"round\":" << round << ",\"left\":";number(out,left);
            out << ",\"right\":";number(out,right);
            out << ",\"left_bits\":" << bits64(left) << ",\"right_bits\":" << bits64(right)
                << ",\"sign_or\":" << (any?"true":"false") << ",\"finite\":" << (finite?"true":"false")
                << ",\"previous_bounds_match\":" << (match?"true":"false");
            field_values(out,node,sdfs,center_sdfs);out << "}\n";
        }
        out << "{\"type\":\"validation\",\"nonfinite_nodes\":" << invalid
            << ",\"previous_bounds_mismatches\":" << bounds_mismatch << "}\n";
        out.flush();require(out.good(),"round-before write failed");
        return invalid==0 && bounds_mismatch==0 ? 0 : 2;
    } catch(const std::exception& e) {return failure(path,"bm_capture_round_before",e.what());}
}
CAPTURE_API int bm_capture_prefinish(const char* path, int t,
                                     const float* sdfs, const float* center_sdfs) {
    using namespace native_capture;
    try {
        require(captured.valid && captured.group==t,"prefinish group mismatch");
        require(captured.nodes.empty() || (sdfs && center_sdfs),"null field arrays");
        auto out=output(path);
        out << "{\"type\":\"header\",\"schema\":\"binocmesher.native_capture.v2\",\"kind\":\"prefinish\",\"group\":"
            << t << ",\"native_last_head\":" << bisection::last_head << "}\n";
        int missing=0, invalid=0;
        for(const auto& node:captured.nodes) {
            int witness=-1;bool finite=std::isfinite(center_sdfs[node.center_index]);
            for(size_t rank=0;rank<node.endpoints.size();++rank) {
                const float value=sdfs[node.start+rank];
                finite &= std::isfinite(value);
                if(witness<0 && (value>=0)!=(center_sdfs[node.center_index]>=0)) witness=int(rank);
            }
            missing += witness<0;invalid += !finite;
            const double left=bisection::lefts[node.center_index], right=bisection::rights[node.center_index];
            out << "{\"type\":\"prefinish_node\",";identity(out,node);
            out << ",\"witness_rank\":" << witness << ",\"left\":";number(out,left);
            out << ",\"right\":";number(out,right);
            out << ",\"left_bits\":" << bits64(left) << ",\"right_bits\":" << bits64(right)
                << ",\"finite\":" << (finite?"true":"false");
            field_values(out,node,sdfs,center_sdfs);out << "}\n";
        }
        out << "{\"type\":\"validation\",\"missing_witnesses\":" << missing << ",\"nonfinite_nodes\":" << invalid << "}\n";
        out.flush();require(out.good(),"prefinish write failed");
        return missing==0 && invalid==0 ? 0 : 2;
    } catch(const std::exception& e) {return failure(path,"bm_capture_prefinish",e.what());}
}

namespace bm_replay {
template<class V> struct SavedVec {
    using storage_type=decltype(V{}.a);
    using size_type=decltype(V{}._size);
    storage_type a; size_type logical=0;
    void save(V& v) {a=v.a;logical=v._size;}
    void restore(V& v) {v.a=a;v._size=logical;}
};
struct State {
    bool valid=false;int group=-1, bisection_group=0;
    ll last_head=0,current_head=0,head_size=0;
    int node_size=0,edge_size=0;
    void* head=nullptr;void* edges=nullptr;void* nodes=nullptr;
    SavedVec<decltype(bisection::query_cnt)> query_cnt;
    SavedVec<decltype(bisection::lefts)> lefts,rights;
    SavedVec<decltype(bisection::starts)> starts,center_indices;
    SavedVec<decltype(bisection::vertex_map)> vertex_map;
    SavedVec<decltype(bisection::computed_vertices)> computed_vertices;
    native_capture::Snapshot observation;
};
static State states[2];
}
CAPTURE_API int bm_replay_snapshot(int slot, int t) {
    using namespace native_capture;
    try {
        require(slot==0||slot==1,"snapshot slot must be 0(group) or 1(batch)");
        require(dual_contouring::head_ && dual_contouring::edges_,"no graph loaded");
        auto& s=bm_replay::states[slot];s.valid=false;s.group=t;
        require(dual_contouring::t_group==t,"snapshot group differs from loaded graph");
        s.bisection_group=bisection::bisection_group;
        s.last_head=bisection::last_head;s.current_head=bisection::current_head;
        s.head=dual_contouring::head_;s.edges=dual_contouring::edges_;
        s.head_size=dual_contouring::head_->size();s.edge_size=dual_contouring::edges_->size();
        s.node_size=fine::active_nodes.size();s.nodes=fine::active_nodes.a.data();
        s.query_cnt.save(bisection::query_cnt);s.lefts.save(bisection::lefts);
        s.rights.save(bisection::rights);s.starts.save(bisection::starts);
        s.center_indices.save(bisection::center_indices);s.vertex_map.save(bisection::vertex_map);
        s.computed_vertices.save(bisection::computed_vertices);
        s.observation=captured;s.valid=true;return 0;
    } catch(const std::exception&) {return 1;}
}
CAPTURE_API int bm_replay_restore(int slot) {
    using namespace native_capture;
    try {
        require(slot==0||slot==1,"invalid snapshot slot");
        auto& s=bm_replay::states[slot];require(s.valid,"snapshot not initialized");
        require(dual_contouring::t_group==s.group && dual_contouring::head_==s.head &&
                dual_contouring::edges_==s.edges,"graph changed since snapshot");
        require(dual_contouring::head_->size()==s.head_size && dual_contouring::edges_->size()==s.edge_size &&
                fine::active_nodes.size()==s.node_size && fine::active_nodes.a.data()==s.nodes,
                "graph/nodes invalidated since snapshot; restore before write_final_hypermesh only");
        bisection::bisection_group=s.bisection_group;
        bisection::last_head=s.last_head;bisection::current_head=s.current_head;
        s.query_cnt.restore(bisection::query_cnt);s.lefts.restore(bisection::lefts);
        s.rights.restore(bisection::rights);s.starts.restore(bisection::starts);
        s.center_indices.restore(bisection::center_indices);s.vertex_map.restore(bisection::vertex_map);
        s.computed_vertices.restore(bisection::computed_vertices);
        captured=s.observation;return 0;
    } catch(const std::exception&) {return 1;}
}
CAPTURE_API void bm_replay_invalidate() {
    bm_replay::states[0].valid=false;bm_replay::states[1].valid=false;
}
CAPTURE_API int bm_output_sizes(int64_t* sizes) {
    if(!sizes)return 1;
    sizes[0]=bisection::computed_vertices.size();sizes[1]=bisection::vertex_map.size();
    return 0;
}
CAPTURE_API int bm_copy_outputs(float* xyz, int32_t* times, int8_t* tags, int32_t* mapping) {
    if((bisection::computed_vertices.size() && (!xyz||!times||!tags)) ||
       (bisection::vertex_map.size() && !mapping))return 1;
    for(int i=0;i<bisection::computed_vertices.size();++i) {
        auto& v=bisection::computed_vertices[i];
        for(int d=0;d<3;++d)xyz[3*i+d]=v.first.first[d];
        for(int d=0;d<2;++d)times[2*i+d]=v.first.second[d];
        tags[i]=v.second;
    }
    for(ll i=0;i<bisection::vertex_map.size();++i)mapping[i]=bisection::vertex_map[i];
    return 0;
}

CAPTURE_API int bm_num_elements() { return params::n_elements; }
