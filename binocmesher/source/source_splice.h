#ifndef BINOC_SOURCE_SPLICE_H
#define BINOC_SOURCE_SPLICE_H

#include <cstdint>
#include <string>
#include <vector>

#include "utils.h"
#include "binoctree.h"
#include "bisection.h"

namespace source_splice {

using SourceVID = array<HVID, 2>;

struct TriangleRef {
    std::int32_t element = -1;
    std::int32_t t_group = -1;
    std::int32_t t_start = -1;
    std::int32_t sorted_record_index = -1;
    std::int32_t interval_index = -1;
    std::int32_t face_index = -1;
    std::int32_t fan_index = -1;

    bool operator<(const TriangleRef& other) const noexcept;
    bool operator==(const TriangleRef& other) const noexcept;
};

struct RuntimeVertex {
    array<spaceT, 3> position{};
    std::int32_t in_view = 1;
};

struct RuntimeFace {
    array<std::int32_t, 3> indices{};
};

void begin_exact(
    std::int64_t time_numerator,
    std::int64_t time_denominator,
    bool provenance_enabled
);

bool active() noexcept;
bool should_suppress(const TriangleRef& reference);

void register_ordinary_vertex(
    std::int32_t element,
    const SourceVID& source_key,
    std::int32_t final_vertex_index
);

// Resolve all source-boundary local vertices through the ordinary global-ID
// map and assign internal vertices consecutive IDs starting at
// first_internal_vertex_index. The caller appends returned vertices/faces to
// the upstream containers.
void build_replacement(
    std::int32_t element,
    std::int32_t first_internal_vertex_index,
    std::vector<RuntimeVertex>& internal_vertices,
    std::vector<RuntimeFace>& replacement_faces
);

void finish();
void reset() noexcept;
const std::string& plan_id() noexcept;

}  // namespace source_splice

#endif  // BINOC_SOURCE_SPLICE_H
