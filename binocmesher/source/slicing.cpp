#include <numeric>
#include <optional>

#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"
#include "bisection.h"
#include "slicing.h"
#include "event_registry.h"
#include "hyperpoly_provenance.h"

namespace {

thread_local std::string slicing_last_error_message;

struct ExactSliceTime {
    std::int64_t numerator = 0;
    std::int64_t denominator = 1;
};

thread_local std::optional<ExactSliceTime> exact_slice_time;

int checked_max_temporal_level() {
    // This translation unit evaluates both 1 << level and 2 << level.
    // Validate before either expression can shift by a negative count, the
    // type width, or into the sign bit.
    if (params::max_tL < 0 ||
        params::max_tL >= std::numeric_limits<int>::digits - 1) {
        throw std::runtime_error("temporal cache level cannot be represented");
    }
    const std::int64_t maximum_discrete_time =
        static_cast<std::int64_t>(2) << params::max_tL;
    if (maximum_discrete_time >
        static_cast<std::int64_t>(std::numeric_limits<timeT>::max())) {
        throw std::runtime_error(
            "temporal cache time exceeds serialized timeT range");
    }
    return params::max_tL;
}

class ScopedFile {
public:
    explicit ScopedFile(FILE *file) noexcept : file_(file) {}
    ~ScopedFile() {
        if (file_ != nullptr) fclose(file_);
    }
    ScopedFile(const ScopedFile&) = delete;
    ScopedFile& operator=(const ScopedFile&) = delete;
    FILE *get() const noexcept { return file_; }

private:
    FILE *file_;
};

int slicing_cache_group(T physical_time) {
    if (!std::isfinite(physical_time) || !std::isfinite(params::tsize) ||
        params::tsize <= 0) {
        throw std::runtime_error(
            "slicing time domain is not finite and positive");
    }
    if (physical_time < 0 || physical_time > params::tsize) {
        throw std::runtime_error("slicing time is outside the production cache");
    }
    const int group_count = 1 << checked_max_temporal_level();
    // The time domain is closed. Terminal hypervertices are serialized in the
    // final source group, not in floor(1 * group_count) == group_count.
    if (physical_time == params::tsize) return group_count - 1;
    const long double scaled =
        static_cast<long double>(physical_time) /
        static_cast<long double>(params::tsize) *
        static_cast<long double>(group_count);
    const int group = static_cast<int>(std::floor(scaled));
    if (group < 0 || group >= group_count) {
        throw std::runtime_error(
            "slicing time maps outside temporal cache groups");
    }
    return group;
}

event_registry::Rational64 current_exact_rational() {
    if (!exact_slice_time.has_value()) {
        throw std::runtime_error("exact slicing time is not active");
    }
    return event_registry::Rational64{
        exact_slice_time->numerator, exact_slice_time->denominator};
}

int compare_slice_time_to_integer(std::int64_t value, T physical_time) {
    if (exact_slice_time.has_value()) {
        return event_registry::compare_exact_rational(
            current_exact_rational(),
            event_registry::Rational64{value, 1});
    }
    const T discrete_time = physical_time / params::deltaT;
    return (discrete_time > static_cast<T>(value)) -
           (discrete_time < static_cast<T>(value));
}

int slicing_cache_group_exact(
    int group_count,
    std::int64_t maximum_discrete_time
) {
    if (!exact_slice_time.has_value() || group_count <= 0 ||
        maximum_discrete_time <= 0) {
        throw std::runtime_error("invalid exact temporal cache mapping state");
    }
    const event_registry::Rational64 time = current_exact_rational();
    const int maximum_relation = event_registry::compare_exact_rational(
        time, event_registry::Rational64{maximum_discrete_time, 1});
    if (maximum_relation > 0 ||
        event_registry::compare_exact_rational(
            time, event_registry::Rational64{0, 1}) < 0) {
        throw std::runtime_error(
            "exact slicing time is outside the production cache");
    }
    if (maximum_relation == 0) return group_count - 1;

    // Binary-search exact boundaries instead of forming
    // numerator * group_count or maximum_time * denominator.
    int lower = 0;
    int upper = group_count;
    while (lower + 1 < upper) {
        const int middle = lower + (upper - lower) / 2;
        const std::int64_t boundary_numerator =
            static_cast<std::int64_t>(middle) * maximum_discrete_time;
        const event_registry::Rational64 boundary{
            boundary_numerator, group_count};
        if (event_registry::compare_exact_rational(time, boundary) >= 0) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return lower;
}

void clear_partial_slicing_state() {
    using namespace slicing;
    for (int element = 0; element < N_ELE; ++element) {
        unmerged_vertices[element].clear();
        faces[element].clear();
        vertices[element].clear();
        vertices_inview[element].clear();
    }
    merging_map.clear();
    vertices_tmp.clear();
    head.clear();
    nxt.clear();
    ind.clear();
    bisection::hypervertices.clear();
    bisection::hypervertices_vec.clear();
    bisection::hyperpolys.clear();
}

}  // namespace


// given an edge connecting two 4D vertices and a time slicing plane, compute the sliced vertex with ID and coordinates
pair<VUnmerged, pair<bool, array<spaceT, 3> > > compute_slice(HVTable &hypervertices, T t0, VID &new_v) {
    array<spaceT, 3> xyz;
    bool inview = 1;
    auto p1 = hypervertices[new_v[0]];
    auto p2 = hypervertices[new_v[1]];
    timeT t1 = p1.first.second[0], t2 = p2.first.second[0];
    // make the first vertex have smaller time
    if (t1 > t2) {
        swap(p1, p2);
        sswap(t1, t2);
    }
    if (t0 < t1) t0 = t1;
    else if (t0 > t2) t0 = t2;
    // A technique to prevent coarse polygons got into views too early - we do piecewise interpolation
    // To interpolate between an out-of-view vertex to a in-view vertex, we interpolate between t1+dt1 to t2 or t1 to t2-dt2 (effective interpolation range),
    // where 2dt1 and 2dt2 are the time span of the nodes
    T dt1 = p1.first.second[1], dt2 = p2.first.second[1];
    auto inview1 = p1.second, inview2 = p2.second;
    T effective_t0 = t0;
    if (inview1 != inview2 && t1 + dt1 <= t2 - dt2) {
        if (inview2) {
            t2 -= dt2;
        }
        else {
            t1 += dt1;
        }
        effective_t0 = smax((T)t1, smin(t0, (T)t2));
    }
    // vertex_id indicates whether the interpolated vertex is still on the middle of the edge or on endpoint
    int vertex_id = 0;
    if (t1 != t2) {
        // in a regular case, we interpolate in the effective interpolation range
        if (effective_t0 == t1) vertex_id = 1;
        else if (effective_t0 == t2) vertex_id = 2;
        for (int i = 0; i < 3; i++) {
            xyz[i] = (p1.first.first[i] * (t2 - effective_t0) + p2.first.first[i] * (effective_t0 - t1)) / (t2 - t1);
        }
    }
    else {
        // in this edge case, we must have a sudden change
        if (t0 < t1) {
            for (int i = 0; i < 3; i++) {
                xyz[i] = p1.first.first[i];
            }
            vertex_id = 1;
        }
        else {
            for (int i = 0; i < 3; i++) {
                xyz[i] = p2.first.first[i];
            }
            vertex_id = 2;
        }
    }
    // mark when the interpolation still belong to out of view part
    if ((!inview1 && t0 < t2) || (!inview2 && t0 > t1) || (!inview1 && !inview2)) {
        inview = 0;
    }
    VUnmerged unmerged_vert;
    unmerged_vert.first = new_v;
    // adjust unmerged_vert ID based on previous vertex_id flag
    if (vertex_id == 1) {
        unmerged_vert.first[1] = unmerged_vert.first[0];
    }
    else if (vertex_id == 2) {
        unmerged_vert.first[0] = unmerged_vert.first[1];
    }
    return mp(unmerged_vert, mp(inview, xyz));
}

// Given a 4D polyhedra and a time slicing plane, this function precompute the sliced polygon faces
std::vector<vec<int, VID> > preprocess_hyperpoly(HVTable &hypervertices, HP &hyperpoly, timeT disc_t0) {
    std::vector<vec<int, VID> > ret;
    int c[8] = {0};
    for (int i = 0; i < 8; i++) {
        timeT disc_t = hypervertices[hyperpoly.first[i]].first.second[0];
        c[i] = disc_t > disc_t0;
    }
    std::vector<II> edges;
    for (int e = 0; e < 12; e++) {
        int coords[3];
        int e0 = e / 4;
        int e1 = (e0+1) % 3;
        int e2 = (e1+1) % 3;
        coords[e0] = 0;
        coords[e1] = e&1;
        coords[e2] = (e>>1) & 1;
        int index1 = cube_index(coords[0], coords[1], coords[2], 2);
        coords[e0] = 1;
        int index2 = cube_index(coords[0], coords[1], coords[2], 2);
        if (c[index1] != c[index2]) {
            II edge;
            if (!c[index1]) edge = mp(index1, index2);
            else edge = mp(index2, index1);
            edges.push_back(edge);
        }
    }
    std::vector<int> nxt_edge;
    nxt_edge.resize(edges.size());
    // It first find all intersection point on all edges
    // candidates store the potential "next" intersection point for each intersection point
    // the final next pointer is "nxt_edge"
    for (int i = 0; i < edges.size(); i++) {
        std::vector<int> candidates;
        for (int j = 0; j < edges.size(); j++) {
            if (j != i) {
                int mins[3];
                int maxs[3] = {0};
                mins[0] = mins[1] = mins[2] = 1;
                for (int k = 0; k < 3; k++) {
                    mins[k] = smin(mins[k], (edges[i].first >> k) & 1);
                    mins[k] = smin(mins[k], (edges[i].second >> k) & 1);
                    mins[k] = smin(mins[k], (edges[j].first >> k) & 1);
                    mins[k] = smin(mins[k], (edges[j].second >> k) & 1);
                    maxs[k] = smax(maxs[k], (edges[i].first >> k) & 1);
                    maxs[k] = smax(maxs[k], (edges[i].second >> k) & 1);
                    maxs[k] = smax(maxs[k], (edges[j].first >> k) & 1);
                    maxs[k] = smax(maxs[k], (edges[j].second >> k) & 1);
                }
                if (mins[0] == maxs[0] || mins[1] == maxs[1] || mins[2] == maxs[2]) {
                    int a[3][3];
                    for (int k = 0; k < 3; k++) a[0][k] = ((edges[i].first>>k)&1) * 2 - 1;
                    for (int k = 0; k < 3; k++) a[1][k] = ((edges[i].second>>k)&1) * 2 - 1;
                    for (int k = 0; k < 3; k++) a[2][k] = ((edges[j].first>>k)&1) * 2 - 1;
                    if (non_neg(a)) {
                        for (int k = 0; k < 3; k++) a[2][k] = ((edges[j].second>>k)&1) * 2 - 1;
                        if (non_neg(a)) candidates.push_back(j);
                    }
                }
            }
        }
        if (candidates.size() > 1) {
            bool flag = 0;
            for (int j: candidates) {
                if (edges[i].first == edges[j].first) {
                    nxt_edge[i] = j;
                    flag = 1;
                    break;
                }
            }
            assert(flag);
        }
        else {
            nxt_edge[i] = candidates[0];
        }
    }
    // connect intersection points to faces according to the pointer
    std::vector<int> visited;
    vec<int, VID> face;
    visited.resize(edges.size(), 0);
    for (int i = 0; i < edges.size(); i++) {
        if (!visited[i]) {
            face.clear();
            visited[i] = 1;
            VID last_v;
            last_v[0] = hyperpoly.first[edges[i].first];
            last_v[1] = hyperpoly.first[edges[i].second];
            face.push_back(last_v);
            for (int j = nxt_edge[i]; j != i; j = nxt_edge[j]) {
                visited[j] = 1;
                VID new_v;
                new_v[0] = hyperpoly.first[edges[j].first];
                new_v[1] = hyperpoly.first[edges[j].second];
                if (new_v != last_v) face.push_back(new_v);
                last_v = new_v;
            }
            while (face.size() > 1 && face[(int)face.size() - 1] == face[0]) {
                face.erase(face.size() - 1, face.size());
            }
            if (face.size() > 2) {
                ret.push_back(face);
            }
        }
    }
    face.clear();
    ret.push_back(face);
    return ret;
}

std::vector<hyperpoly_provenance::SourceRecord> load_hyperpoly_provenance(
    int t_group,
    int expected_records
) {
    using namespace hyperpoly_provenance;
    std::stringstream filename;
    filename << params::output_path << "/hyperpoly_meta/" << t_group
             << ".bin";
    ScopedFile infile_owner(fopen(filename.str().c_str(), "rb"));
    FILE *infile = infile_owner.get();
    if (infile == nullptr) {
        throw std::runtime_error("missing hyperpoly provenance stream");
    }
    SourceHeader header{};
    if (read_scalar_checked(infile, header) != CheckedReadStatus::record) {
        throw std::runtime_error(
            "truncated hyperpoly provenance header");
    }
    if (header.magic != kSourceMagic || header.version != kVersion ||
        header.record_size != sizeof(SourceRecord) ||
        header.layout_version != kLayoutVersion || expected_records < 0 ||
        header.record_count !=
            static_cast<std::uint64_t>(expected_records)) {
        throw std::runtime_error(
            "hyperpoly provenance header/count mismatch");
    }
    std::vector<SourceRecord> records(
        static_cast<size_t>(expected_records));
    for (int index = 0; index < expected_records; ++index) {
        if (read_scalar_checked(
                infile, records[static_cast<size_t>(index)]) !=
            CheckedReadStatus::record) {
            throw std::runtime_error(
                "truncated hyperpoly provenance payload");
        }
    }
    SourceRecord trailing{};
    if (read_scalar_checked(infile, trailing) !=
        CheckedReadStatus::clean_eof) {
        throw std::runtime_error(
            "trailing bytes or records in hyperpoly provenance stream");
    }
    for (int index = 0; index < expected_records; ++index) {
        const SourceRecord& record =
            records[static_cast<size_t>(index)];
        if (record.source_t_group != t_group ||
            record.source_record_index != index) {
            throw std::runtime_error(
                "hyperpoly provenance record order mismatch");
        }
    }
    return records;
}

void write_processed_provenance(
    FILE *outfile,
    int t_group,
    int t_start,
    int sorted_record_index,
    const hyperpoly_provenance::SourceRecord& source
) {
    using namespace hyperpoly_provenance;
    if (outfile == nullptr) {
        throw std::runtime_error(
            "null processed provenance output");
    }
    ProcessedRecord record{};
    record.magic = kProcessedMagic;
    record.version = kVersion;
    record.record_size = sizeof(ProcessedRecord);
    record.layout_version = kLayoutVersion;
    record.t_group = t_group;
    record.t_start = t_start;
    record.sorted_record_index = sorted_record_index;
    record.source = source;
    if (fwrite(&record, sizeof(record), 1, outfile) != 1) {
        throw std::runtime_error(
            "failed to write processed provenance record");
    }
}

CheckedReadStatus read_processed_provenance(
    FILE *infile,
    hyperpoly_provenance::ProcessedRecord& record
) {
    const CheckedReadStatus status = read_scalar_checked(infile, record);
    if (status != CheckedReadStatus::record) return status;
    if (record.magic != hyperpoly_provenance::kProcessedMagic ||
        record.version != hyperpoly_provenance::kVersion ||
        record.record_size != sizeof(record) ||
        record.layout_version != hyperpoly_provenance::kLayoutVersion) {
        throw std::runtime_error(
            "invalid processed provenance record header");
    }
    return CheckedReadStatus::record;
}

// These are functions exposed to Python
extern "C" {
    // The preprocessing function that prebuild a lookup table for which edge to cut given a timestamp
    int slicing_preprocess() noexcept {
        FILE *log = nullptr;
        try {
        const int max_temporal_level = checked_max_temporal_level();
        const int temporal_group_count = 1 << max_temporal_level;
        const int maximum_discrete_time = 2 << max_temporal_level;
        log = fopen(params::log_path.c_str(), "a");
        if (log == nullptr) {
            throw std::runtime_error("failed to open slicing log");
        }
        using namespace bisection;
        const bool provenance_enabled = hyperpoly_provenance::enabled();
        event_registry::begin(params::output_path);
        // Sort polyhedra according the time span
        // First group together polyhedra with the same starting time, then sort by ending time within each group
        vec<int, pair<array<timeT, 2>, int> > hyperpolys_timed;
        for (int t_group = 0; t_group < temporal_group_count; t_group++) {
            fprintf(log, "processing %d\n", t_group);
            MEASURE_TIME("load hyperpolys", 1, {
                load_hyperpolys(t_group);
                load_vertices(t_group);
            });
            std::vector<hyperpoly_provenance::SourceRecord>
                source_provenance;
            if (provenance_enabled) {
                source_provenance = load_hyperpoly_provenance(
                    t_group, hyperpolys.size());
            }
            MEASURE_TIME("get timed hyperpolys", 1, {
                hyperpolys_timed.resize(hyperpolys.size());
                OMP_PRAGMA(omp parallel for schedule(dynamic))
                for (int index = 0; index < hyperpolys.size(); index++) {
                    auto hyperpoly = hyperpolys[index];
                    timeT t_min = maximum_discrete_time;
                    timeT t_max = 0;
                    for (int i = 0; i < 8; i++) {
                        timeT disc_t = hypervertices[hyperpoly.first[i]].first.second[0];
                        t_min = smin(t_min, disc_t);
                        t_max = smax(t_max, disc_t);
                    }
                    hyperpolys_timed[index].first[0] = t_min;
                    hyperpolys_timed[index].first[1] = -t_max;
                    hyperpolys_timed[index].second = index;
                }
            });
            vec<int, int> starting_index;
            MEASURE_TIME("sort hyperpolys", 1, {
                _sort(hyperpolys_timed);
                starting_index.resize(maximum_discrete_time + 1);
                starting_index.fill(-1);
                starting_index[maximum_discrete_time] = hyperpolys.size();
                for (int index = 0; index < hyperpolys.size(); index++) {
                    if (index == 0 || hyperpolys_timed[index].first[0] != hyperpolys_timed[index-1].first[0]) {
                        starting_index[(int)hyperpolys_timed[index].first[0]] = index;
                    }
                }
                for (int i = maximum_discrete_time - 1; i >= 0; i--) {
                    if (starting_index[i] == -1) starting_index[i] = starting_index[i+1];
                }
            });
            typedef vec<int, timeT> vec_timeT;
            typedef set<VID> set_VID;
            typedef set<set_VID> set_set_VID;
            // parallel_faces keeps track of faces that are parallel to the spatial hyperplane (perpendicular to the time axis)
            map<set_VID, pair<int, pair<int, int> > > parallel_faces;
            vec<int, pair<int, pair<int, int> > > discontinuity_list;
            // Slice the polyhedra in order
            MEASURE_TIME("process hyperpolys", 1, {
                for (int t_start = 0; t_start < maximum_discrete_time; t_start++) {
                    set_VID vertices_set;
                    set_set_VID parallel_faces_to_remove;
                    std::stringstream filename;
                    filename << params::output_path << "/processed_hyperpolys/" << t_group << "_" << t_start << ".bin";
                    ScopedFile outfile_owner(fopen(filename.str().c_str(), "ab"));
                    FILE *outfile = outfile_owner.get();
                    if (outfile == nullptr) {
                        throw std::runtime_error(
                            "failed to open processed hyperpoly output");
                    }
                    std::stringstream metadata_filename;
                    metadata_filename
                        << params::output_path << "/processed_hyperpolys/"
                        << t_group << "_" << t_start << "_hpmeta.bin";
                    ScopedFile metadata_outfile_owner(
                        provenance_enabled
                            ? fopen(metadata_filename.str().c_str(), "ab")
                            : nullptr);
                    FILE *metadata_outfile =
                        metadata_outfile_owner.get();
                    if (provenance_enabled && metadata_outfile == nullptr) {
                        throw std::runtime_error(
                            "failed to open processed hyperpoly metadata output");
                    }
                    for (int index = starting_index[t_start]; index < starting_index[t_start+1]; index++) {
                        const int source_index =
                            hyperpolys_timed[index].second;
                        auto hyperpoly = hyperpolys[source_index];
                        if (provenance_enabled) {
                            const auto &source = source_provenance[
                                static_cast<size_t>(source_index)];
                            write_processed_provenance(
                                metadata_outfile, t_group, t_start, index,
                                source);
                            event_registry::observe_hyperpoly(
                                hypervertices, hyperpoly, source,
                                t_group, t_start, index);
                        }
                        // Given a polyhedra, find the list of critital time point
                        // For each time segment, precompute which edge to slice and what faces to generate
                        vec_timeT times;
                        for (int i = 0; i < 8; i++) {
                            timeT disc_t = hypervertices[hyperpoly.first[i]].first.second[0];
                            times.push_back(disc_t);
                        }
                        make_unique(times);
                        _sort(times);
                        if (fwrite(&hyperpoly.second, sizeof(eleT), 1, outfile) != 1) {
                            throw std::runtime_error(
                                "failed to write processed hyperpoly element");
                        }
                        write_vec_checked(outfile, times);
                        for (int i_times = 0; i_times < times.size() - 1; i_times++) {
                            auto disc_t0 = times[i_times];
                            auto res = preprocess_hyperpoly(hypervertices, hyperpoly, disc_t0);
                            for (auto &face: res) {
                                write_vec_checked(outfile, face);
                            }
                            if (i_times == 0) {
                                T t0 = times[0];
                                bool t0_critical = times[0] - lowbit(times[0]) == 2 * t_group;
                                if (t0_critical) {
                                    for (int i_face = 0; i_face < res.size(); i_face++) {
                                        auto &face = res[i_face];
                                        vertices_set.clear();
                                        for (int i_verts = 0; i_verts < face.size(); i_verts++) {
                                            auto vert = face[i_verts];
                                            auto res = compute_slice(hypervertices, t0, vert);
                                            VUnmerged unmerged_vert = res.first;
                                            vertices_set.insert(unmerged_vert.first);
                                        }
                                        if (vertices_set.size() > 1) {
                                            int cnt = parallel_faces.count(vertices_set);
                                            if (cnt == 0) {
                                                if (times[0] != 0) {
                                                    discontinuity_list.push_back(mp(index, mp(0, i_face)));
                                                }
                                            }
                                            else {
                                                parallel_faces_to_remove.insert(vertices_set);
                                            }
                                        }
                                    }
                                }
                            }
                            if (i_times == times.size() - 2) {
                                T t0 = times[times.size() - 1];
                                bool t0_critical = times[times.size() - 1] - lowbit(times[times.size() - 1]) == 2 * t_group;
                                if ((times[times.size() - 1] != maximum_discrete_time) && t0_critical) {
                                    for (int i_face = 0; i_face < res.size(); i_face++) {
                                        auto &face = res[i_face];
                                        vertices_set.clear();
                                        for (int i_verts = 0; i_verts < face.size(); i_verts++) {
                                            auto vert = face[i_verts];
                                            auto res = compute_slice(hypervertices, t0, vert);
                                            VUnmerged unmerged_vert = res.first;
                                            vertices_set.insert(unmerged_vert.first);
                                        }
                                        if (vertices_set.size() > 1) {
                                            parallel_faces.emplace(
                                                vertices_set,
                                                mp(index, mp(1, i_face))
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (ferror(outfile)) {
                        throw std::runtime_error(
                            "I/O error in processed hyperpoly output");
                    }
                    if (provenance_enabled && ferror(metadata_outfile)) {
                        throw std::runtime_error(
                            "I/O error in processed hyperpoly metadata output");
                    }
                    for (auto &vertices_set: parallel_faces_to_remove) {
                        parallel_faces.erase(vertices_set);
                    }
                }
            });
            // discontinuity_list are those remaining faces that are parallel to the spatial hyperplane that caused the discontinuity
            for (auto &kv : parallel_faces) {
                discontinuity_list.push_back(kv.second);
            }
            _sort(discontinuity_list);
            // write these discontinuity faces to disk
            vec<int, pair<int, int> > buffer;
            MEASURE_TIME("process discontinuity", 1, {
                int discontinuity_list_i = 0;
                for (int t_start = 0; t_start < maximum_discrete_time; t_start++) {
                    std::stringstream filename;
                    filename << params::output_path << "/processed_hyperpolys/" << t_group << "_" << t_start << "_discon.bin";
                    ScopedFile outfile_owner(fopen(filename.str().c_str(), "ab"));
                    FILE *outfile = outfile_owner.get();
                    if (outfile == nullptr) {
                        throw std::runtime_error(
                            "failed to open discontinuity output");
                    }
                    for (int index = starting_index[t_start]; index < starting_index[t_start+1]; index++) {
                        buffer.clear();
                        while (discontinuity_list_i < discontinuity_list.size() && discontinuity_list[discontinuity_list_i].first == index) {
                            buffer.push_back(discontinuity_list[discontinuity_list_i].second);
                            discontinuity_list_i++;
                        }
                        write_vec_checked(outfile, buffer);
                    }
                    if (ferror(outfile)) {
                        throw std::runtime_error(
                            "I/O error in discontinuity output");
                    }

                }
            });
        }
        event_registry::finish();
        if (fclose(log) != 0) {
            log = nullptr;
            throw std::runtime_error("failed to close slicing log");
        }
        log = nullptr;
        slicing_last_error_message.clear();
        return 0;
        } catch (const std::exception& error) {
            slicing_last_error_message = error.what();
            if (log != nullptr) fclose(log);
            clear_partial_slicing_state();
            return -1;
        } catch (...) {
            slicing_last_error_message = "unknown slicing_preprocess failure";
            if (log != nullptr) fclose(log);
            clear_partial_slicing_state();
            return -2;
        }
    }
}

// Functions for sorting and deduplicating faces
bool compare_face(const array<int, 3> &a, const array<int, 3> &b) {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] < b[i]) return 1;
        if (a[i] > b[i]) return 0;
    }
    return 0;
}

bool equal_face(const array<int, 3> &a, const array<int, 3> &b) {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

// A bucket sort implementation for sorting unmerged vertices
void bucket_sort(int ele, int index) {
    using namespace slicing;
    auto &a = unmerged_vertices[ele];
    int max_n = 0, min_n = INT_MAX;
    for (int i = 0; i < a.size(); i++) {
        max_n = smax(max_n, index%2? a[i].first[index/2].first: a[i].first[index/2].second);
        min_n = smin(min_n, index%2? a[i].first[index/2].first: a[i].first[index/2].second);
    }
    for (int i = 0; i < a.size(); i++) {
        if (index%2) {
            a[i].first[index/2].first -= min_n;
        }
        else {
            a[i].first[index/2].second -= min_n;
        }
    }
    max_n -= min_n;
    head.clear();
    head.resizefill(max_n+1, -1);
    ind.clear();
    nxt.clear();
    for (int i = 0; i < a.size(); i++) {
        int v = index%2? a[i].first[index/2].first: a[i].first[index/2].second;
        ind.push_back(i);
        nxt.push_back(head[v]);
        head[v] = nxt.size() - 1;
    }
    vertices_tmp.clear();
    for (int v = head.size() - 1; v >= 0; v--) {
        for (int i = head[v]; i != -1; i = nxt[i]) {
            vertices_tmp.push_back(a[ind[i]]);
        }
    }
    for (int i = 0; i * 2 < vertices_tmp.size(); i++) {
        swap(vertices_tmp[i], vertices_tmp[vertices_tmp.size() - 1 - i]);
    }
    vertices_tmp.a.swap(a.a);
    vertices_tmp.clear();
}

// These are functions exposed to Python
extern "C" {
    // the main slicing function that returns the number of vertices and faces per element
    // extra_smooth indicates whether "Extension to Ameliorate Popping Artifacts" in Sec. 7 is turned On
    int run_slicing(T t0, int *v_cnts, int *f_cnts, bool extra_smooth) noexcept {
        FILE *log = nullptr;
        try {
        if (v_cnts == nullptr || f_cnts == nullptr) {
            throw std::runtime_error("invalid run_slicing output pointers");
        }
        if (params::n_elements < 0 || params::n_elements > N_ELE) {
            throw std::runtime_error("invalid slicing element count");
        }
        const int max_temporal_level = checked_max_temporal_level();
        const int temporal_group_count = 1 << max_temporal_level;
        const int maximum_discrete_time = 2 << max_temporal_level;
        if (!std::isfinite(params::deltaT) || params::deltaT <= 0) {
            throw std::runtime_error("invalid slicing discrete-time scale");
        }
        log = fopen(params::log_path.c_str(), "a");
        if (log == nullptr) {
            throw std::runtime_error("failed to open slicing log");
        }
        using namespace bisection;
        using namespace slicing;
        const bool provenance_enabled = hyperpoly_provenance::enabled();
        const int disc_t = exact_slice_time.has_value()
            ? slicing_cache_group_exact(
                temporal_group_count, maximum_discrete_time)
            : slicing_cache_group(t0);
        int discon_cnt = 0;
        typedef set<VID> set_VID;
        map<set_VID, int> discon_map;
        set_VID vertices_set;
        for (int t_group = disc_t; ; t_group -= lowbit(t_group)) {
            fprintf(log, "processing %d\n", t_group);
            MEASURE_TIME("load_vertices", 1, {
                load_vertices(t_group);
            });
            typedef vec<int, timeT> vec_timeT;
            typedef vec<int, VID> VID_vec;
            typedef array<int, 3> array_int_3;
            typedef array<spaceT, 3> array_spaceT_3;
            vec<int, pair<int, int> > buffer;
            set<int> start_discon_face, end_discon_face;
            std::vector<array_spaceT_3> full_vertices;
            MEASURE_TIME("getting unmerged mesh", 1, {
                for (int t_start = 0; t_start < maximum_discrete_time; t_start++) {
                    // read the preprocessed polyhedra in order until the starting time exceed t0
                    if (compare_slice_time_to_integer(t_start - 1, t0) < 0) break;
                    vec_timeT times;
                    VID_vec unmerged_face;
                    array_int_3 face;
                    std::stringstream filename;
                    filename << params::output_path << "/processed_hyperpolys/" << t_group << "_" << t_start << ".bin";
                    ScopedFile infile_owner(fopen(filename.str().c_str(), "rb"));
                    FILE *infile = infile_owner.get();
                    std::stringstream filename2;
                    filename2 << params::output_path << "/processed_hyperpolys/" << t_group << "_" << t_start << "_discon.bin";
                    ScopedFile infile_discon_owner(
                        fopen(filename2.str().c_str(), "rb"));
                    FILE *infile_discon = infile_discon_owner.get();
                    std::stringstream filename3;
                    filename3
                        << params::output_path << "/processed_hyperpolys/"
                        << t_group << "_" << t_start << "_hpmeta.bin";
                    ScopedFile infile_metadata_owner(
                        provenance_enabled
                            ? fopen(filename3.str().c_str(), "rb")
                            : nullptr);
                    FILE *infile_metadata = infile_metadata_owner.get();
                    if (infile == nullptr || infile_discon == nullptr ||
                        (provenance_enabled && infile_metadata == nullptr)) {
                        throw std::runtime_error(
                            "missing synchronized processed hyperpoly stream");
                    }
                    int hyperpolys_cnt = 0;
                    int previous_sorted_record_index = -1;
                    for (;;) {
                        start_discon_face.clear();
                        end_discon_face.clear();

                        // The primary stream owns record lifetime. On its clean
                        // EOF the companion must also be exactly at a record
                        // boundary with no trailing record.
                        eleT ele{};
                        const CheckedReadStatus primary_status =
                            read_scalar_checked(infile, ele);
                        if (primary_status == CheckedReadStatus::clean_eof) {
                            const CheckedReadStatus discon_status =
                                read_vec_checked(
                                    infile_discon, buffer,
                                    static_cast<int>(64));
                            CheckedReadStatus metadata_status =
                                CheckedReadStatus::clean_eof;
                            if (provenance_enabled) {
                                hyperpoly_provenance::ProcessedRecord
                                    trailing_metadata{};
                                metadata_status = read_processed_provenance(
                                    infile_metadata, trailing_metadata);
                            }
                            if (discon_status !=
                                    CheckedReadStatus::clean_eof ||
                                metadata_status !=
                                    CheckedReadStatus::clean_eof) {
                                throw std::runtime_error(
                                    "synchronized processed hyperpoly stream has trailing records");
                            }
                            break;
                        }
                        if (ele < 0 || ele >= params::n_elements) {
                            throw std::runtime_error(
                                "processed hyperpoly element is out of range");
                        }
                        if (read_vec_checked(
                                infile_discon, buffer,
                                static_cast<int>(64)) !=
                            CheckedReadStatus::record) {
                            throw std::runtime_error(
                                "processed_hyperpolys discontinuity stream ended before primary stream");
                        }
                        if (provenance_enabled) {
                            hyperpoly_provenance::ProcessedRecord metadata{};
                            if (read_processed_provenance(
                                    infile_metadata, metadata) !=
                                CheckedReadStatus::record) {
                                throw std::runtime_error(
                                    "processed provenance stream ended before primary stream");
                            }
                            if (metadata.t_group != t_group ||
                                metadata.t_start != t_start ||
                                metadata.source.element != ele ||
                                metadata.sorted_record_index < 0 ||
                                (previous_sorted_record_index >= 0 &&
                                 metadata.sorted_record_index !=
                                     previous_sorted_record_index + 1)) {
                                throw std::runtime_error(
                                    "processed provenance stream alignment mismatch");
                            }
                            previous_sorted_record_index =
                                metadata.sorted_record_index;
                        }
                        if (extra_smooth) {
                            for (auto &p: buffer) {
                                if (p.second < 0 || p.second >= 16) {
                                    throw std::runtime_error(
                                        "discontinuity face index is out of range");
                                }
                                if (p.first == 0) {
                                    start_discon_face.insert(p.second);
                                } else if (p.first == 1) {
                                    end_discon_face.insert(p.second);
                                } else {
                                    throw std::runtime_error(
                                        "invalid discontinuity endpoint side");
                                }
                            }
                        }
                        if (read_vec_checked(
                                infile, times, static_cast<int>(8)) !=
                            CheckedReadStatus::record) {
                            throw std::runtime_error(
                                "processed hyperpoly times ended before primary record");
                        }
                        if (times.size() < 2) {
                            throw std::runtime_error(
                                "processed hyperpoly must contain at least two distinct times");
                        }
                        for (int time_index = 1;
                             time_index < times.size(); ++time_index) {
                            if (times[time_index - 1] >= times[time_index]) {
                                throw std::runtime_error(
                                    "processed hyperpoly times are not strictly increasing");
                            }
                        }
                        if (times[0] == std::numeric_limits<timeT>::min() ||
                            times[times.size() - 1] ==
                                std::numeric_limits<timeT>::max()) {
                            throw std::runtime_error(
                                "processed hyperpoly endpoint expansion exceeds timeT");
                        }
                        times[0] -= 1;
                        times[times.size() - 1] += 1;
                        int slice_group;
                        for (slice_group = 0;
                             slice_group < times.size() &&
                             compare_slice_time_to_integer(
                                 times[slice_group], t0) >= 0;
                             slice_group++);
                        slice_group--;
                        // Consume every serialized interval even when the time
                        // plane does not intersect this record. An early break
                        // would hide a corrupt synchronized suffix.
                        const bool record_relevant = slice_group >= 0 &&
                            slice_group < times.size() - 1;
                        for (int s = 0; s < times.size() - 1; s++) {
                            int unmerged_face_cnt = 0;
                            for (;;) {
                                if (read_vec_checked(
                                        infile, unmerged_face,
                                        static_cast<int>(12)) !=
                                    CheckedReadStatus::record) {
                                    throw std::runtime_error(
                                        "processed hyperpoly face stream ended inside a record");
                                }
                                if (unmerged_face.size() == 0) break;
                                if (unmerged_face_cnt >= 16) {
                                    throw std::runtime_error(
                                        "processed hyperpoly interval exceeds face-count bound");
                                }
                                if (record_relevant && s == slice_group) {
                                    if (start_discon_face.count(unmerged_face_cnt) == 0 &&
                                        s == 0 &&
                                        compare_slice_time_to_integer(
                                            times[0] + 1, t0) < 0) {
                                        unmerged_face_cnt++;
                                        continue;
                                    }
                                    if (end_discon_face.count(unmerged_face_cnt) == 0 &&
                                        s == times.size() - 2 &&
                                        compare_slice_time_to_integer(
                                            times[times.size() - 1] - 1,
                                            t0) > 0) {
                                        unmerged_face_cnt++;
                                        continue;
                                    }
                                    int base = unmerged_vertices[ele].size();
                                    array_spaceT_3 center;
                                    center[0] = center[1] = center[2] = 0;
                                    vertices_set.clear();
                                    full_vertices.clear();
                                    for (int i_verts = 0; i_verts < unmerged_face.size(); i_verts++) {
                                        auto vert = unmerged_face[i_verts];
                                        auto res = compute_slice(
                                            hypervertices,
                                            t0 / params::deltaT,
                                            vert);
                                        VUnmerged unmerged_vert = res.first;
                                        unmerged_vert.second = mp(unmerged_vertices[ele].size(), res.second);
                                        unmerged_vertices[ele].push_back(unmerged_vert);
                                        for (int p = 0; p < 3; p++) center[p] += res.second.second[p];
                                        vertices_set.insert(unmerged_vert.first);
                                        full_vertices.push_back(res.second.second);
                                    }
                                    for (int p = 0; p < 3; p++) center[p] /= unmerged_face.size();
                                    // We extrude these time-perpendicular faces and do the special interpolation within the time span of one node
                                    bool needs_resolve_discon =
                                        start_discon_face.count(unmerged_face_cnt) != 0 &&
                                        s == 0 &&
                                        compare_slice_time_to_integer(
                                            times[0] + 1, t0) < 0;
                                    needs_resolve_discon |=
                                        end_discon_face.count(unmerged_face_cnt) != 0 &&
                                        s == times.size() - 2 &&
                                        compare_slice_time_to_integer(
                                            times[times.size() - 1] - 1,
                                            t0) > 0;
                                    if (needs_resolve_discon) {
                                        T weight;
                                        if (s == 0 &&
                                            compare_slice_time_to_integer(
                                                times[0] + 1, t0) < 0) {
                                            weight = t0 / params::deltaT - times[0];
                                        }
                                        else weight = times[times.size() - 1] - t0 / params::deltaT;
                                        for (int i_verts = 0; i_verts < unmerged_face.size(); i_verts++) {
                                            // We interpolate from the center of the face to the actual face so it looks like a point grows into a triangle face
                                            for (int p = 0; p < 3; p++) {
                                                auto &full = unmerged_vertices[ele][base + i_verts].second.second.second[p];
                                                full = full * weight + center[p] * (1 - weight);
                                            }
                                            // The vertices as a result of the extra discontinuity removal technique have special IDs
                                            unmerged_vertices[ele][base + i_verts].first[0].first = --discon_cnt;
                                        }
                                    }
                                    for (int j = 0; j < unmerged_face.size() - 2; j++) {
                                        face[0] = base;
                                        face[1] = base + (j + 1) % unmerged_face.size();
                                        face[2] = base + (j + 2) % unmerged_face.size();
                                        faces[ele].push_back(face);
                                    }
                                }
                                unmerged_face_cnt++;
                            }
                        }
                        hyperpolys_cnt++;
                    }
                }
            });
            if (t_group == 0) break;
        }
        // For each scene element, run bucket sort to merge vertices
        MEASURE_TIME("merging vertices", 1, {
            for (int ele = 0; ele < params::n_elements; ele++) {
                for (int i = 0; i < 4; i++) bucket_sort(ele, i);
                int cnt = 0;
                int start_i = -1;
                merging_map.resize(unmerged_vertices[ele].size());
                for (int i = 0; i < unmerged_vertices[ele].size(); i++) {
                    if (i == 0 || unmerged_vertices[ele][i].first != unmerged_vertices[ele][i-1].first) {
                        vertices[ele].push_back(unmerged_vertices[ele][i].second.second.second);
                        vertices_inview[ele].push_back(unmerged_vertices[ele][i].second.second.first);
                        if (start_i != -1) {
                            for (int j = start_i; j < i; j++) merging_map[unmerged_vertices[ele][j].second.first] = cnt - 1;
                        }
                        start_i = i;
                        cnt++;
                    }
                }
                if (start_i != -1) {
                    for (int j = start_i; j < unmerged_vertices[ele].size(); j++) merging_map[unmerged_vertices[ele][j].second.first] = cnt - 1;
                }
                for (int i = 0; i < faces[ele].size(); i++) {
                    for (int j = 0; j < faces[ele][i].size(); j++) {
                        faces[ele][i][j] = merging_map[faces[ele][i][j]];
                    }
                }
                CLS(unmerged_vertices[ele]);
                v_cnts[ele] = vertices[ele].size();
            }
        });
        CLS(head);
        CLS(nxt);
        CLS(ind);
        CLS(merging_map);
        // Deduplicate faces according to unique vertex IDs
        MEASURE_TIME("cleaning up faces", 1, {
            for (int ele = 0; ele < params::n_elements; ele++) {
                auto &facesele = faces[ele];
                OMP_PRAGMA(omp parallel for schedule(dynamic))
                for (int i = 0; i < facesele.size(); i++) {
                    auto face = facesele[i];
                    int min_ind = 0;
                    for (int j = 1; j < face.size(); j++) {
                        if (face[j] < face[min_ind]) {
                            min_ind = j;
                        }
                    }
                    facesele[i][0] = face[min_ind];
                    facesele[i][1] = face[(min_ind+1)%3];
                    facesele[i][2] = face[(min_ind+2)%3];
                }
                fprintf(log, "before merging faces, face count: %d\n", (int)facesele.size());
                make_unique(facesele, compare_face, equal_face);
                fprintf(log, "after merging faces, face count: %d\n", (int)facesele.size());
                f_cnts[ele] = facesele.size();
            }
        });
        if (fclose(log) != 0) {
            log = nullptr;
            throw std::runtime_error("failed to close slicing log");
        }
        log = nullptr;
        slicing_last_error_message.clear();
        return 0;
        } catch (const std::exception& error) {
            slicing_last_error_message = error.what();
            if (log != nullptr) fclose(log);
            if (v_cnts != nullptr && f_cnts != nullptr &&
                params::n_elements >= 0 && params::n_elements <= N_ELE) {
                for (int element = 0; element < params::n_elements; ++element) {
                    v_cnts[element] = 0;
                    f_cnts[element] = 0;
                }
            }
            clear_partial_slicing_state();
            return -1;
        } catch (...) {
            slicing_last_error_message = "unknown run_slicing failure";
            if (log != nullptr) fclose(log);
            if (v_cnts != nullptr && f_cnts != nullptr &&
                params::n_elements >= 0 && params::n_elements <= N_ELE) {
                for (int element = 0; element < params::n_elements; ++element) {
                    v_cnts[element] = 0;
                    f_cnts[element] = 0;
                }
            }
            clear_partial_slicing_state();
            return -2;
        }
    }

    int run_slicing_rational(
        std::int64_t time_numerator,
        std::int64_t time_denominator,
        int *v_cnts,
        int *f_cnts,
        bool extra_smooth
    ) noexcept {
        try {
            if (v_cnts == nullptr || f_cnts == nullptr) {
                throw std::runtime_error("invalid exact slicing output pointers");
            }
            if (time_numerator < 0 || time_denominator <= 0) {
                throw std::runtime_error(
                    "exact slicing time must have a nonnegative numerator and positive denominator");
            }
            if (exact_slice_time.has_value()) {
                throw std::runtime_error("nested exact slicing is not allowed");
            }
            const int max_temporal_level = checked_max_temporal_level();
            const std::int64_t maximum_discrete_time =
                static_cast<std::int64_t>(2) << max_temporal_level;
            const std::int64_t divisor =
                std::gcd(time_numerator, time_denominator);
            time_numerator /= divisor;
            time_denominator /= divisor;
            const event_registry::Rational64 time{
                time_numerator, time_denominator};
            const int maximum_relation = event_registry::compare_exact_rational(
                time,
                event_registry::Rational64{maximum_discrete_time, 1});
            if (maximum_relation > 0) {
                throw std::runtime_error(
                    "exact slicing time is outside the production cache");
            }
            const long double physical_time = maximum_relation == 0
                ? static_cast<long double>(params::tsize)
                : static_cast<long double>(time_numerator) *
                    static_cast<long double>(params::deltaT) /
                    static_cast<long double>(time_denominator);
            if (!std::isfinite(physical_time) || physical_time < 0 ||
                physical_time >
                    static_cast<long double>(std::numeric_limits<T>::max())) {
                throw std::runtime_error(
                    "exact slicing time cannot be represented physically");
            }
            exact_slice_time = ExactSliceTime{
                time_numerator, time_denominator};
            const int status = run_slicing(
                static_cast<T>(physical_time),
                v_cnts,
                f_cnts,
                extra_smooth);
            exact_slice_time.reset();
            return status;
        } catch (const std::exception& error) {
            exact_slice_time.reset();
            slicing_last_error_message = error.what();
            if (v_cnts != nullptr && f_cnts != nullptr &&
                params::n_elements >= 0 && params::n_elements <= N_ELE) {
                for (int element = 0; element < params::n_elements; ++element) {
                    v_cnts[element] = 0;
                    f_cnts[element] = 0;
                }
            }
            clear_partial_slicing_state();
            return -1;
        } catch (...) {
            exact_slice_time.reset();
            slicing_last_error_message = "unknown exact slicing failure";
            if (v_cnts != nullptr && f_cnts != nullptr &&
                params::n_elements >= 0 && params::n_elements <= N_ELE) {
                for (int element = 0; element < params::n_elements; ++element) {
                    v_cnts[element] = 0;
                    f_cnts[element] = 0;
                }
            }
            clear_partial_slicing_state();
            return -2;
        }
    }

    const char *slicing_last_error() noexcept {
        return slicing_last_error_message.c_str();
    }
    
    // actually output the mesh data
    void slicing_output(int ele, T *output_verts, int *output_faces, int *output_inview) {
        using namespace slicing;
        int cnt = 0;
        for (int i = 0; i < vertices[ele].size(); i++) {
            auto vert = vertices[ele][i];
            for (int j = 0; j < 3; j++) output_verts[cnt * 3 + j] = vert[j];
            output_inview[cnt] = vertices_inview[ele][i];
            cnt++;
        }
        cnt = 0;
        for (int i = 0; i < faces[ele].size(); i++) {
            auto &face = faces[ele][i];
            for (int j = 0; j < face.size(); j++) {
                output_faces[cnt++] = face[j];
            }
        }
        CLS(faces[ele]);
        CLS(vertices[ele]);
        CLS(vertices_inview[ele]);
    }

    // clean up slicing data structures
    void slicing_clean_up() {
        using namespace slicing;
        CL(bisection::hypervertices);
    }
}
