#ifndef BINOC_RUNTIME_IDENTITY_H
#define BINOC_RUNTIME_IDENTITY_H

// Opt-in provenance of ACTUAL raw slicing arrays; never mutates mesh data.
// VID rows preserve the ordered HVID pair used by the upstream merger.
#include <cstring>
#include <vector>

namespace runtime_identity {
enum Status { disabled = 0, ready = 1, unsupported = 2, error = 3 };
constexpr int kElements = 5;
constexpr int kMaximumVertices = 250000;
constexpr int kMaximumOwners = 250000;
constexpr int kMaximumFaces = 250000;
struct VertexRow { int values[4]; };
struct OwnerRow { int values[11]; };
struct State {
    bool enabled = false;
    bool collecting = false;
    int status = disabled;
    int elements = 0;
    int vertex_total = 0;
    int owner_total = 0;
    bool finalized[kElements]{};
    char message[256]{};
    std::vector<VertexRow> vertices[kElements];
    std::vector<OwnerRow> owners[kElements];
};
inline thread_local State state;

inline void message(const char* text) noexcept {
    std::strncpy(state.message, text, sizeof(state.message)-1);
    state.message[sizeof(state.message)-1] = '\0';
}

inline void release_rows() noexcept {
    for (int element = 0; element < kElements; ++element) {
        std::vector<VertexRow>().swap(state.vertices[element]);
        std::vector<OwnerRow>().swap(state.owners[element]);
        state.finalized[element] = false;
    }
    state.vertex_total = state.owner_total = 0;
    state.elements = 0;
    state.collecting = false;
}

inline void invalidate(const char* reason) noexcept {
    release_rows();
    state.status = state.enabled ? unsupported : disabled;
    message(state.enabled ? reason : "Identity observation is disabled.");
}

inline void enable(bool requested) noexcept {
    state.enabled = requested;
    invalidate("No completed raw slicing snapshot is available.");
}

inline void fail(const char* reason) noexcept {
    release_rows();
    state.status = state.enabled ? error : disabled;
    message(state.enabled ? reason : "Identity observation is disabled.");
}

inline void begin(bool raw, bool provenance, bool splice, int elements) noexcept {
    invalidate("Raw identity observation is not ready.");
    if (!state.enabled) return;
    if (!raw || !provenance || splice || elements < 0 || elements > kElements) {
        message("Identity observation requires raw slicing, BPM2 provenance, no SSP1 plan, and at most five elements.");
        return;
    }
    state.elements = elements;
    state.collecting = true;
}

inline bool valid_element(int element) noexcept {
    return element >= 0 && element < state.elements;
}

template<class Owner, class Face>
inline void record_owner(const Owner& owner, const Face& face) noexcept {
    if (!state.collecting) return;
    try {
        if (!valid_element(owner.element) || state.owner_total >= kMaximumOwners) {
            fail("Identity owner ledger exceeds its bounded scope.");
            return;
        }
        OwnerRow row{{owner.element, owner.t_group, owner.t_start,
            owner.sorted_record_index, owner.interval_index, owner.face_index,
            owner.fan_index, -1, face[0], face[1], face[2]}};
        for (int index = 0; index < 7; ++index) {
            if (row.values[index] < 0) {
                fail("An emitted raw owner has invalid provenance.");
                return;
            }
        }
        state.owners[owner.element].push_back(row);
        ++state.owner_total;
    } catch (...) { fail("Identity owner allocation failed; baseline is unchanged."); }
}

template<class VID>
inline void record_vertex(int element, const VID& vid, int final_index) noexcept {
    if (!state.collecting) return;
    try {
        if (!valid_element(element) || state.vertex_total >= kMaximumVertices ||
            final_index != static_cast<int>(state.vertices[element].size())) {
            fail("Identity vertex ledger exceeds its bound or lost final-index alignment.");
            return;
        }
        VertexRow row{{vid[0].first, static_cast<int>(vid[0].second),
                       vid[1].first, static_cast<int>(vid[1].second)}};
        state.vertices[element].push_back(row);
        ++state.vertex_total;
    } catch (...) { fail("Identity vertex allocation failed; baseline is unchanged."); }
}

template<class Mapping>
inline void remap_owners(int element, Mapping& mapping) noexcept {
    if (!state.collecting) return;
    if (!valid_element(element)) { fail("Invalid identity remapping element."); return; }
    for (OwnerRow& row : state.owners[element]) {
        int mapped[3];
        for (int corner = 0; corner < 3; ++corner) {
            const int original = row.values[8+corner];
            if (original < 0 || original >= static_cast<int>(mapping.size())) {
                fail("An owner references an invalid unmerged vertex.");
                return;
            }
            mapped[corner] = mapping[original];
            if (mapped[corner] < 0 ||
                mapped[corner] >= static_cast<int>(state.vertices[element].size())) {
                fail("An owner references an invalid actual merged vertex.");
                return;
            }
        }
        // EXACT upstream first-minimum rule, not the lexicographically least
        // cyclic rotation: those differ for some repeated-index triangles.
        int first = 0;
        for (int corner = 1; corner < 3; ++corner)
            if (mapped[corner] < mapped[first]) first = corner;
        for (int corner = 0; corner < 3; ++corner)
            row.values[8+corner] = mapped[(first+corner)%3];
    }
}

template<class Faces>
inline void finalize_element(int element, Faces& faces, int vertex_count) noexcept {
    if (!state.collecting) return;
    try {
        if (!valid_element(element) || state.finalized[element] ||
            faces.size() > kMaximumFaces ||
            vertex_count != static_cast<int>(state.vertices[element].size())) {
            fail("Final identity arrays exceed their bound or lost alignment.");
            return;
        }
        std::vector<unsigned char> covered(faces.size(), 0);
        for (OwnerRow& row : state.owners[element]) {
            // Existing make_unique already sorted these oriented triples.
            // Read-only binary search preserves the actual final face order.
            int lower = 0, upper = static_cast<int>(faces.size());
            while (lower < upper) {
                const int middle = lower+(upper-lower)/2;
                int comparison = 0;
                for (int corner = 0; corner < 3; ++corner) {
                    const int actual = faces[middle][corner];
                    const int wanted = row.values[8+corner];
                    if (actual != wanted) { comparison = actual < wanted ? -1 : 1; break; }
                }
                if (comparison < 0) lower = middle+1;
                else upper = middle;
            }
            if (lower >= static_cast<int>(faces.size())) {
                fail("An emitted owner has no final oriented face.");
                return;
            }
            for (int corner = 0; corner < 3; ++corner) {
                if (faces[lower][corner] != row.values[8+corner]) {
                    fail("An emitted owner has no final oriented face.");
                    return;
                }
            }
            row.values[7] = lower;
            covered[lower] = 1;
        }
        for (unsigned char value : covered) {
            if (!value) { fail("A final ordinary face has no observed raw owner."); return; }
        }
        state.finalized[element] = true;
    } catch (...) { fail("Identity finalization allocation failed; baseline is unchanged."); }
}

inline void complete() noexcept {
    if (!state.collecting) return;
    for (int element = 0; element < state.elements; ++element) {
        if (!state.finalized[element]) { fail("An identity element was not finalized."); return; }
    }
    state.collecting = false;
    state.status = ready;
    state.message[0] = '\0';
}

inline int count(int element, bool owners) noexcept {
    if (state.status != ready || !valid_element(element)) return -1;
    return static_cast<int>(owners ? state.owners[element].size() : state.vertices[element].size());
}

inline int output_vertices(int element, int* output, int capacity) noexcept {
    const int rows = count(element, false);
    if (rows < 0 || capacity < 0 || capacity < rows*4 || (rows && output == nullptr)) {
        message("Identity vertex output is unavailable or has insufficient int capacity.");
        return -1;
    }
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < 4; ++column)
            output[row*4+column] = state.vertices[element][row].values[column];
    return 0;
}

inline int output_owners(int element, int* output, int capacity) noexcept {
    const int rows = count(element, true);
    if (rows < 0 || capacity < 0 || capacity < rows*11 || (rows && output == nullptr)) {
        message("Identity owner output is unavailable or has insufficient int capacity.");
        return -1;
    }
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < 11; ++column)
            output[row*11+column] = state.owners[element][row].values[column];
    return 0;
}
}  // namespace runtime_identity
#endif
