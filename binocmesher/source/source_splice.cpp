#include "source_splice.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "event_registry.h"

namespace source_splice {
namespace {

enum class VertexKind {
    source,
    internal,
};

struct PlanVertex {
    std::int32_t local_id = -1;
    std::int32_t element = -1;
    VertexKind kind = VertexKind::source;
    SourceVID source_key{};
    array<spaceT, 3> position{};
    std::int32_t in_view = 1;
};

struct PlanFace {
    std::int32_t element = -1;
    array<std::int32_t, 3> local_vertices{};
};

struct BoundaryReuse {
    std::int32_t element = -1;
    std::int32_t local_id = -1;
    std::int32_t global_id = -1;
};

struct State {
    bool enabled = false;
    std::string id;
    event_registry::Rational64 exact_time{0, 1};
    set<TriangleRef> suppressions;
    set<TriangleRef> observed_suppressions;
    map<std::int32_t, PlanVertex> plan_vertices;
    std::vector<PlanFace> plan_faces;
    map<std::int32_t, map<SourceVID, std::int32_t>> ordinary_vertices;
    set<std::int32_t> emitted_elements;
    std::vector<BoundaryReuse> boundary_reuses;
    std::int32_t expected_boundary_vertices = 0;
    std::int32_t expected_internal_vertices = 0;
    std::int32_t expected_faces = 0;
    std::int32_t boundary_vertices_reused = 0;
    std::int32_t internal_vertices_added = 0;
    std::int32_t replacement_faces_added = 0;
    std::filesystem::path audit_path;
};

thread_local State state;

std::int32_t checked_int32(long long value, const char* field) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(
            std::string("source-splice ") + field + " is outside int32 range");
    }
    return static_cast<std::int32_t>(value);
}

std::int8_t checked_int8(long long value, const char* field) {
    if (value < std::numeric_limits<std::int8_t>::min() ||
        value > std::numeric_limits<std::int8_t>::max()) {
        throw std::runtime_error(
            std::string("source-splice ") + field + " is outside int8 range");
    }
    return static_cast<std::int8_t>(value);
}

event_registry::Rational64 normalize(
    std::int64_t numerator,
    std::int64_t denominator
) {
    if (numerator < 0 || denominator <= 0) {
        throw std::runtime_error("source-splice exact time is invalid");
    }
    const std::int64_t divisor = std::gcd(numerator, denominator);
    return event_registry::Rational64{
        numerator / divisor, denominator / divisor};
}

SourceVID canonical_vid(HVID first, HVID second) {
    SourceVID result;
    if (second < first) sswap(first, second);
    result[0] = first;
    result[1] = second;
    return result;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(ch)
                           << std::dec;
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

void require_token(std::istream& input, const char* expected) {
    std::string token;
    if (!(input >> token) || token != expected) {
        throw std::runtime_error(
            std::string("source-splice plan expected token ") + expected);
    }
}

// SSP1 describes the complete replacement support for an element.  Therefore
// every plan-boundary edge must be resolvable through the ordinary SourceVID
// map; a new internal vertex on that boundary would be an undeclared
// T-junction.  Internal edges must form an oriented two-chain and all boundary
// vertices must form cycles before the ordinary whole mesh is modified.
void validate_plan_topology() {
    using EV = pair<std::int32_t, std::int32_t>;
    using Edge = std::tuple<std::int32_t, std::int32_t, std::int32_t>;
    using Face = std::tuple<
        std::int32_t, std::int32_t, std::int32_t, std::int32_t>;
    map<Edge, pair<std::int32_t, std::int32_t>> edges;
    set<Face> faces;
    map<EV, std::int32_t> uses;
    map<EV, std::int32_t> boundary_degree;
    for (const PlanFace& face : state.plan_faces) {
        array<std::int32_t, 3> sorted_face = face.local_vertices;
        sort(sorted_face.begin(), sorted_face.end());
        if (!faces.emplace(
                face.element, sorted_face[0], sorted_face[1],
                sorted_face[2]).second) {
            throw std::runtime_error(__func__);
        }
        for (std::int32_t vertex : face.local_vertices) {
            ++uses[{face.element, vertex}];
        }
        for (int corner = 0; corner < 3; ++corner) {
            const std::int32_t first = face.local_vertices[corner];
            const std::int32_t second = face.local_vertices[(corner + 1) % 3];
            auto& audit = edges[{
                face.element, std::min(first, second),
                std::max(first, second)}];
            ++audit.first;
            audit.second += first < second ? 1 : -1;
        }
    }
    for (const auto& entry : edges) {
        const auto& audit = entry.second;
        if (audit.first > 2 ||
            (audit.first == 2 && audit.second != 0)) {
            throw std::runtime_error(__func__);
        }
        if (audit.first != 1) continue;
        const std::int32_t element = std::get<0>(entry.first);
        const std::int32_t first = std::get<1>(entry.first);
        const std::int32_t second = std::get<2>(entry.first);
        const auto first_vertex = state.plan_vertices.find(first);
        const auto second_vertex = state.plan_vertices.find(second);
        if (first_vertex == state.plan_vertices.end() ||
            second_vertex == state.plan_vertices.end() ||
            first_vertex->second.element != element ||
            second_vertex->second.element != element ||
            first_vertex->second.kind != VertexKind::source ||
            second_vertex->second.kind != VertexKind::source) {
            throw std::runtime_error(__func__);
        }
        ++boundary_degree[{element, first}];
        ++boundary_degree[{element, second}];
    }
    for (const auto& entry : state.plan_vertices) {
        const PlanVertex& vertex = entry.second;
        const EV key{vertex.element, vertex.local_id};
        const std::int32_t degree = boundary_degree[key];
        if (uses.count(key) == 0 ||
            (vertex.kind == VertexKind::source && degree != 2) ||
            (vertex.kind == VertexKind::internal && degree != 0)) {
            throw std::runtime_error(__func__);
        }
    }
}

void load_plan(
    const std::filesystem::path& path,
    const event_registry::Rational64& current_time,
    bool provenance_enabled
) {
    if (!provenance_enabled) {
        throw std::runtime_error(
            "source-splice plan requires BPM2 provenance");
    }
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open source-splice plan");
    }

    std::string magic;
    if (!(input >> magic) || magic != "SSP1") {
        throw std::runtime_error("invalid source-splice plan magic");
    }
    require_token(input, "PLAN");
    if (!(input >> std::quoted(state.id)) || state.id.empty()) {
        throw std::runtime_error("source-splice plan has no plan ID");
    }
    require_token(input, "TIME");
    long long numerator = 0;
    long long denominator = 0;
    if (!(input >> numerator >> denominator)) {
        throw std::runtime_error("truncated source-splice TIME row");
    }
    state.exact_time = normalize(
        static_cast<std::int64_t>(numerator),
        static_cast<std::int64_t>(denominator));
    if (event_registry::compare_exact_rational(
            state.exact_time, current_time) != 0) {
        throw std::runtime_error(
            "source-splice plan exact time does not match run_slicing_rational");
    }

    require_token(input, "EXPECT");
    long long expected_suppressions = 0;
    long long expected_boundary = 0;
    long long expected_internal = 0;
    long long expected_faces = 0;
    if (!(input >> expected_suppressions >> expected_boundary >>
          expected_internal >> expected_faces) ||
        expected_suppressions <= 0 || expected_boundary < 3 ||
        expected_internal < 0 || expected_faces <= 0) {
        throw std::runtime_error("invalid source-splice EXPECT row");
    }
    state.expected_boundary_vertices = checked_int32(
        expected_boundary, "expected boundary count");
    state.expected_internal_vertices = checked_int32(
        expected_internal, "expected internal count");
    state.expected_faces = checked_int32(
        expected_faces, "expected face count");

    bool saw_end = false;
    std::string command;
    while (input >> command) {
        if (command == "SUPPRESS") {
            long long values[7]{};
            for (long long& value : values) {
                if (!(input >> value)) {
                    throw std::runtime_error(
                        "truncated source-splice SUPPRESS row");
                }
            }
            TriangleRef reference{
                checked_int32(values[0], "suppression element"),
                checked_int32(values[1], "suppression t_group"),
                checked_int32(values[2], "suppression t_start"),
                checked_int32(values[3], "suppression sorted index"),
                checked_int32(values[4], "suppression interval"),
                checked_int32(values[5], "suppression face"),
                checked_int32(values[6], "suppression fan"),
            };
            if (reference.element < 0 || reference.t_group < 0 ||
                reference.t_start < 0 || reference.sorted_record_index < 0 ||
                reference.interval_index < 0 || reference.face_index < 0 ||
                reference.fan_index < 0 ||
                !state.suppressions.insert(reference).second) {
                throw std::runtime_error(
                    "duplicate or negative source-splice suppression row");
            }
        } else if (command == "VERTEX_SOURCE") {
            long long local_id = 0;
            long long element = 0;
            long long n0 = 0;
            long long g0 = 0;
            long long n1 = 0;
            long long g1 = 0;
            if (!(input >> local_id >> element >> n0 >> g0 >> n1 >> g1)) {
                throw std::runtime_error(
                    "truncated source-splice VERTEX_SOURCE row");
            }
            PlanVertex vertex;
            vertex.local_id = checked_int32(local_id, "source vertex ID");
            vertex.element = checked_int32(element, "source vertex element");
            vertex.kind = VertexKind::source;
            vertex.source_key = canonical_vid(
                HVID(checked_int32(n0, "source HVID node"),
                     checked_int8(g0, "source HVID group")),
                HVID(checked_int32(n1, "source HVID node"),
                     checked_int8(g1, "source HVID group")));
            if (vertex.local_id < 0 || vertex.element < 0 ||
                !state.plan_vertices.emplace(
                    vertex.local_id, vertex).second) {
                throw std::runtime_error(
                    "duplicate or negative source-splice vertex ID");
            }
        } else if (command == "VERTEX_INTERNAL") {
            long long local_id = 0;
            long long element = 0;
            long long in_view = 0;
            double x = 0;
            double y = 0;
            double z = 0;
            if (!(input >> local_id >> element >> x >> y >> z >> in_view) ||
                !std::isfinite(x) || !std::isfinite(y) ||
                !std::isfinite(z) || (in_view != 0 && in_view != 1)) {
                throw std::runtime_error(
                    "invalid source-splice VERTEX_INTERNAL row");
            }
            PlanVertex vertex;
            vertex.local_id = checked_int32(local_id, "internal vertex ID");
            vertex.element = checked_int32(element, "internal vertex element");
            vertex.kind = VertexKind::internal;
            vertex.position[0] = static_cast<spaceT>(x);
            vertex.position[1] = static_cast<spaceT>(y);
            vertex.position[2] = static_cast<spaceT>(z);
            vertex.in_view = static_cast<std::int32_t>(in_view);
            if (vertex.local_id < 0 || vertex.element < 0 ||
                !state.plan_vertices.emplace(
                    vertex.local_id, vertex).second) {
                throw std::runtime_error(
                    "duplicate or negative source-splice vertex ID");
            }
        } else if (command == "FACE") {
            long long element = 0;
            long long a = 0;
            long long b = 0;
            long long c = 0;
            if (!(input >> element >> a >> b >> c)) {
                throw std::runtime_error(
                    "truncated source-splice FACE row");
            }
            PlanFace face;
            face.element = checked_int32(element, "face element");
            face.local_vertices = {
                checked_int32(a, "face vertex"),
                checked_int32(b, "face vertex"),
                checked_int32(c, "face vertex"),
            };
            if (face.element < 0 || face.local_vertices[0] < 0 ||
                face.local_vertices[1] < 0 || face.local_vertices[2] < 0 ||
                face.local_vertices[0] == face.local_vertices[1] ||
                face.local_vertices[1] == face.local_vertices[2] ||
                face.local_vertices[2] == face.local_vertices[0]) {
                throw std::runtime_error("invalid source-splice FACE row");
            }
            state.plan_faces.push_back(face);
        } else if (command == "END") {
            saw_end = true;
            std::string trailing;
            if (input >> trailing) {
                throw std::runtime_error(
                    "trailing tokens after source-splice END");
            }
            break;
        } else {
            throw std::runtime_error(
                "unknown source-splice plan command: " + command);
        }
    }
    if (!saw_end) {
        throw std::runtime_error("source-splice plan has no END row");
    }

    if (state.suppressions.size() !=
            static_cast<size_t>(expected_suppressions) ||
        state.plan_faces.size() !=
            static_cast<size_t>(state.expected_faces)) {
        throw std::runtime_error(
            "source-splice plan count does not match EXPECT row");
    }
    std::int32_t source_vertices = 0;
    std::int32_t internal_vertices = 0;
    for (const auto& entry : state.plan_vertices) {
        if (entry.second.kind == VertexKind::source) {
            ++source_vertices;
        } else {
            ++internal_vertices;
        }
    }
    if (source_vertices != state.expected_boundary_vertices ||
        internal_vertices != state.expected_internal_vertices) {
        throw std::runtime_error(
            "source-splice vertex counts do not match EXPECT row");
    }
    for (const PlanFace& face : state.plan_faces) {
        for (std::int32_t local_vertex : face.local_vertices) {
            const auto found = state.plan_vertices.find(local_vertex);
            if (found == state.plan_vertices.end() ||
                found->second.element != face.element) {
                throw std::runtime_error(
                    "source-splice FACE references an invalid local vertex");
            }
        }
    }

    const char* audit = std::getenv("BINOC_SOURCE_SPLICE_AUDIT");
    if (audit != nullptr && audit[0] != '\0') {
        state.audit_path = std::filesystem::path(audit);
    }
    validate_plan_topology();
    state.enabled = true;
}

void write_audit() {
    if (state.audit_path.empty()) return;
    const std::filesystem::path parent = state.audit_path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream output(state.audit_path);
    if (!output) {
        throw std::runtime_error("failed to open source-splice audit output");
    }
    output << "{\n";
    output << "  \"schema\": \"binoc-source-splice-audit-v1\",\n";
    output << "  \"plan_id\": \"" << json_escape(state.id) << "\",\n";
    output << "  \"time_numerator\": "
           << state.exact_time.numerator << ",\n";
    output << "  \"time_denominator\": "
           << state.exact_time.denominator << ",\n";
    output << "  \"expected_suppressions\": "
           << state.suppressions.size() << ",\n";
    output << "  \"suppressed_triangles\": "
           << state.observed_suppressions.size() << ",\n";
    output << "  \"expected_boundary_vertices\": "
           << state.expected_boundary_vertices << ",\n";
    output << "  \"resolved_boundary_vertices\": "
           << state.boundary_vertices_reused << ",\n";
    output << "  \"expected_internal_vertices\": "
           << state.expected_internal_vertices << ",\n";
    output << "  \"emitted_internal_vertices\": "
           << state.internal_vertices_added << ",\n";
    output << "  \"expected_replacement_faces\": "
           << state.expected_faces << ",\n";
    output << "  \"emitted_replacement_faces\": "
           << state.replacement_faces_added << ",\n";
    output << "  \"replacement_emissions\": "
           << state.emitted_elements.size() << ",\n";
    output << "  \"boundary_reuses\": [\n";
    for (size_t index = 0; index < state.boundary_reuses.size(); ++index) {
        const BoundaryReuse& reuse = state.boundary_reuses[index];
        output << "    {\"element\": " << reuse.element
               << ", \"local_id\": " << reuse.local_id
               << ", \"global_id\": " << reuse.global_id << "}";
        if (index + 1 != state.boundary_reuses.size()) output << ',';
        output << '\n';
    }
    output << "  ],\n";
    output << "  \"pass\": true\n";
    output << "}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write source-splice audit output");
    }
}

}  // namespace

bool TriangleRef::operator<(const TriangleRef& other) const noexcept {
    return std::tie(
        element, t_group, t_start, sorted_record_index,
        interval_index, face_index, fan_index) <
        std::tie(
            other.element, other.t_group, other.t_start,
            other.sorted_record_index, other.interval_index,
            other.face_index, other.fan_index);
}

bool TriangleRef::operator==(const TriangleRef& other) const noexcept {
    return element == other.element && t_group == other.t_group &&
        t_start == other.t_start &&
        sorted_record_index == other.sorted_record_index &&
        interval_index == other.interval_index &&
        face_index == other.face_index && fan_index == other.fan_index;
}

void begin_exact(
    std::int64_t time_numerator,
    std::int64_t time_denominator,
    bool provenance_enabled
) {
    reset();
    const char* plan = std::getenv("BINOC_SOURCE_SPLICE_PLAN");
    if (plan == nullptr || plan[0] == '\0') return;
    const event_registry::Rational64 current = normalize(
        time_numerator, time_denominator);
    load_plan(std::filesystem::path(plan), current, provenance_enabled);
}

bool active() noexcept {
    return state.enabled;
}

bool should_suppress(const TriangleRef& reference) {
    if (!state.enabled) return false;
    const auto found = state.suppressions.find(reference);
    if (found == state.suppressions.end()) return false;
    if (!state.observed_suppressions.insert(reference).second) {
        throw std::runtime_error(
            "source-splice suppression reference was consumed more than once");
    }
    return true;
}

void register_ordinary_vertex(
    std::int32_t element,
    const SourceVID& source_key,
    std::int32_t final_vertex_index
) {
    if (!state.enabled) return;
    if (element < 0 || final_vertex_index < 0) {
        throw std::runtime_error("invalid ordinary source-splice vertex");
    }
    const SourceVID canonical = canonical_vid(source_key[0], source_key[1]);
    auto& mapping = state.ordinary_vertices[element];
    const auto result = mapping.emplace(canonical, final_vertex_index);
    if (!result.second && result.first->second != final_vertex_index) {
        throw std::runtime_error(
            "one source-edge VID mapped to multiple ordinary vertices");
    }
}

void build_replacement(
    std::int32_t element,
    std::int32_t first_internal_vertex_index,
    std::vector<RuntimeVertex>& internal_vertices,
    std::vector<RuntimeFace>& replacement_faces
) {
    internal_vertices.clear();
    replacement_faces.clear();
    if (!state.enabled) return;
    if (first_internal_vertex_index < 0) {
        throw std::runtime_error(
            "source-splice first internal vertex index is negative");
    }
    if (state.emitted_elements.count(element) != 0) {
        throw std::runtime_error(
            "source-splice replacement was emitted more than once");
    }

    bool has_faces = false;
    for (const PlanFace& face : state.plan_faces) {
        if (face.element == element) {
            has_faces = true;
            break;
        }
    }
    if (!has_faces) return;

    map<std::int32_t, std::int32_t> resolved;
    for (const auto& entry : state.plan_vertices) {
        const PlanVertex& vertex = entry.second;
        if (vertex.element != element) continue;
        if (vertex.kind == VertexKind::source) {
            const auto element_it = state.ordinary_vertices.find(element);
            if (element_it == state.ordinary_vertices.end()) {
                throw std::runtime_error(
                    "source-splice element has no ordinary vertices");
            }
            const auto vertex_it = element_it->second.find(vertex.source_key);
            if (vertex_it == element_it->second.end()) {
                throw std::runtime_error(
                    "source-splice boundary VID was not produced by ordinary mesh");
            }
            resolved.emplace(vertex.local_id, vertex_it->second);
            state.boundary_reuses.push_back(
                BoundaryReuse{element, vertex.local_id, vertex_it->second});
            ++state.boundary_vertices_reused;
        } else {
            RuntimeVertex runtime_vertex;
            runtime_vertex.position = vertex.position;
            runtime_vertex.in_view = vertex.in_view;
            const std::int32_t final_index = first_internal_vertex_index +
                static_cast<std::int32_t>(internal_vertices.size());
            internal_vertices.push_back(runtime_vertex);
            resolved.emplace(vertex.local_id, final_index);
            ++state.internal_vertices_added;
        }
    }

    for (const PlanFace& planned : state.plan_faces) {
        if (planned.element != element) continue;
        RuntimeFace runtime_face;
        for (int corner = 0; corner < 3; ++corner) {
            const auto found = resolved.find(
                planned.local_vertices[static_cast<size_t>(corner)]);
            if (found == resolved.end()) {
                throw std::runtime_error(
                    "source-splice face vertex was not resolved");
            }
            runtime_face.indices[corner] = found->second;
        }
        if (runtime_face.indices[0] == runtime_face.indices[1] ||
            runtime_face.indices[1] == runtime_face.indices[2] ||
            runtime_face.indices[2] == runtime_face.indices[0]) {
            throw std::runtime_error(
                "source-splice replacement produced repeated indices");
        }
        replacement_faces.push_back(runtime_face);
        ++state.replacement_faces_added;
    }
    state.emitted_elements.insert(element);
}

void finish() {
    if (!state.enabled) return;
    if (state.observed_suppressions != state.suppressions) {
        throw std::runtime_error(
            "source-splice source triangle was not suppressed exactly once");
    }
    if (state.boundary_vertices_reused !=
            state.expected_boundary_vertices ||
        state.internal_vertices_added != state.expected_internal_vertices ||
        state.replacement_faces_added != state.expected_faces) {
        throw std::runtime_error(
            "source-splice replacement counts do not match plan contract");
    }
    write_audit();
}

void reset() noexcept {
    state = State{};
}

const std::string& plan_id() noexcept {
    return state.id;
}

}  // namespace source_splice
