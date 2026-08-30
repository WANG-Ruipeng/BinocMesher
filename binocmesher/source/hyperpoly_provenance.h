#ifndef BINOC_HYPERPOLY_PROVENANCE_H
#define BINOC_HYPERPOLY_PROVENANCE_H

#include <cstdint>
#include <cstdlib>

namespace hyperpoly_provenance {

constexpr std::uint32_t kSourceMagic = 0x32504842U;     // "BHP2"
constexpr std::uint32_t kProcessedMagic = 0x324D5042U;  // "BPM2"
constexpr std::uint32_t kVersion = 2U;
constexpr std::uint32_t kLayoutVersion = 1U;

// Provenance is observational and opt-in.  Enabling the event registry also
// enables provenance because the registry needs stable source/processed joins.
// With both flags disabled, the primary cache and mesh output remain identical
// to the pre-provenance branch and no BHP2/BPM2 sidecars are created.
inline bool enabled() noexcept {
    const char* provenance = std::getenv("BINOC_PROVENANCE_V2");
    const char* event_mode = std::getenv("BINOC_EVENT_MODE");
    return (provenance != nullptr && std::atoi(provenance) > 0) ||
           (event_mode != nullptr && std::atoi(event_mode) > 0);
}

#pragma pack(push, 1)
struct SourceHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t record_size;
    std::uint32_t layout_version;
    std::uint64_t record_count;
};

// Stable identity retained from the queriedEdge that produced one HP.  The
// cache group is deliberately separate from the edge key: one logical edge is
// exported into multiple t_group caches by the upstream algorithm.
struct SourceRecord {
    std::int32_t source_t_group;
    std::int32_t source_record_index;
    std::int32_t edge_coords[3];
    std::int32_t edge_L;
    std::int32_t edge_tcoord;
    std::int32_t edge_tL;
    std::int32_t edge_dir;
    std::int32_t element;
    std::int32_t hvid_node[8];
    std::int32_t hvid_group[8];
};

struct ProcessedRecord {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t record_size;
    std::uint32_t layout_version;
    std::int32_t t_group;
    std::int32_t t_start;
    std::int32_t sorted_record_index;
    SourceRecord source;
};
#pragma pack(pop)

static_assert(sizeof(SourceHeader) == 24, "unexpected source metadata ABI");
static_assert(sizeof(SourceRecord) == 104, "unexpected source record ABI");
static_assert(sizeof(ProcessedRecord) == 132, "unexpected processed record ABI");

}  // namespace hyperpoly_provenance

#endif  // BINOC_HYPERPOLY_PROVENANCE_H
