#ifndef BINOC_EVENT_REGISTRY_H
#define BINOC_EVENT_REGISTRY_H

#include <cstdint>
#include <string>

// The official bisection.h relies on type aliases/macros defined by these
// upstream headers and is not self-contained on its own. Keep this public
// header directly compilable in a fresh official worktree.
#include "utils.h"
#include "binoctree.h"
#include "bisection.h"
#include "hyperpoly_provenance.h"

namespace event_registry {

// Exact normalized rational used for event times. Denominator is always positive.
struct Rational64 {
    std::int64_t numerator = 0;
    std::int64_t denominator = 1;
};

// Exact overflow-safe comparison for normalized rationals.
int compare_exact_rational(const Rational64& first, const Rational64& second);

// Reset read-only instrumentation for one slicing_preprocess() invocation.
// The registry is enabled only when BINOC_EVENT_MODE is a positive integer.
void begin(const std::string& output_path);

// Observe one official HP/HV record without mutating the source cache or output stream.
void observe_hyperpoly(
    const HVTable& hypervertices,
    const HP& hyperpoly,
    const hyperpoly_provenance::SourceRecord& source,
    int t_group,
    int t_start,
    int sorted_record_index
);

// Write the provenance-v2 P1 CSV, summary, and selected-event evidence.
void finish();

}  // namespace event_registry

// Deterministic P2-P4 seam fixtures. They are intentionally tiny and are used only
// to verify that the event code was compiled into the official core.so.
extern "C" {
int binoc_event_fixture_saddle(std::int32_t* numerator, std::int32_t* denominator);
int binoc_event_fixture_endpoint(double* official_per_face_gap, double* shared_gap);
int binoc_event_fixture_replay(
    std::int32_t samples,
    std::int32_t* registry_events,
    std::int32_t* uniform_exact_hits,
    std::int32_t* simultaneous_batch_size
);
}

#endif  // BINOC_EVENT_REGISTRY_H
