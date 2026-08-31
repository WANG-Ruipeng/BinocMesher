#include "hyperpoly_layout.h"
#include "event_registry.h"

// The upstream utils.h intentionally defines short macro aliases such as
// `array` and `map`. Undefine them in this translation unit before including
// additional standard headers so `std::array` cannot become `std::std::array`.
#ifdef pair
#undef pair
#endif
#ifdef array
#undef array
#endif
#ifdef map
#undef map
#endif
#ifdef set
#undef set
#endif
#ifdef size_t
#undef size_t
#endif
#ifdef sort
#undef sort
#endif
#ifdef unique
#undef unique
#endif
#ifdef unordered_map
#undef unordered_map
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "checked_output.h"

namespace event_registry {
namespace {

using FaceKey = std::array<HVID, 4>;

struct SaddleSolution {
    Rational64 time{};
    Rational64 u{};
    Rational64 v{};
    std::int64_t A = 0;
    std::int64_t B = 0;
};

struct CanonicalFace {
    FaceKey key{};
    std::array<std::int64_t, 4> times{};
};

struct SaddleObservation {
    FaceKey key{};
    std::array<std::int64_t, 4> times{};
    SaddleSolution solution{};
    int face_axis = -1;
    int face_side = -1;
    int element = -1;
    int t_group = -1;
    int t_start = -1;
    int sorted_record_index = -1;
    hyperpoly_provenance::SourceRecord source{};
    std::string raw_id;
    std::string logical_incidence_id;
    std::string canonical_event_id;
};

struct EventAggregate {
    FaceKey key{};
    std::array<std::int64_t, 4> times{};
    SaddleSolution solution{};
    int face_axis = -1;
    int element = -1;
    std::int64_t raw_observations = 0;
    std::set<std::string> logical_incidence_ids;
};

struct RegistryState {
    bool enabled = false;
    std::filesystem::path output_path;
    std::vector<SaddleObservation> observations;
    std::set<std::string> raw_observation_ids;
    std::map<std::string, EventAggregate> events;
    std::set<std::string> logical_incidence_ids;
    std::int64_t hyperpolys = 0;
    std::int64_t faces_examined = 0;
    std::int64_t finite_algebraic_roots = 0;
    std::int64_t accepted_saddles = 0;
    std::int64_t a0_b0 = 0;
    std::int64_t a0_b_nonzero = 0;
    std::int64_t reduced_hvid_faces = 0;
    std::int64_t shared_root_mismatches = 0;
    std::optional<std::string> selected_saddle_event_id;
    std::set<std::string> selected_saddle_raw_ids;
    std::set<std::string> selected_saddle_logical_ids;
};

RegistryState state;

std::int64_t abs64(std::int64_t value) {
    if (value == std::numeric_limits<std::int64_t>::min()) {
        throw std::overflow_error("cannot normalize INT64_MIN rational component");
    }
    return value < 0 ? -value : value;
}

Rational64 normalize(std::int64_t numerator, std::int64_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("cannot normalize a rational with zero denominator");
    }
    if (denominator < 0) {
        if (numerator == std::numeric_limits<std::int64_t>::min() ||
            denominator == std::numeric_limits<std::int64_t>::min()) {
            throw std::overflow_error("cannot normalize signed rational minimum");
        }
        numerator = -numerator;
        denominator = -denominator;
    }
    const std::int64_t divisor = std::gcd(abs64(numerator), denominator);
    return Rational64{numerator / divisor, denominator / divisor};
}

bool rational_equal(const Rational64& a, const Rational64& b) {
    return a.numerator == b.numerator && a.denominator == b.denominator;
}

std::uint64_t unsigned_magnitude(std::int64_t value) noexcept {
    if (value >= 0) return static_cast<std::uint64_t>(value);
    return static_cast<std::uint64_t>(-(value + 1)) + UINT64_C(1);
}

int compare_unsigned_rational(
    std::uint64_t first_numerator,
    std::uint64_t first_denominator,
    std::uint64_t second_numerator,
    std::uint64_t second_denominator
) {
    int direction = 1;
    for (;;) {
        const std::uint64_t first_quotient =
            first_numerator / first_denominator;
        const std::uint64_t second_quotient =
            second_numerator / second_denominator;
        if (first_quotient != second_quotient) {
            const int result = first_quotient > second_quotient ? 1 : -1;
            return direction * result;
        }
        const std::uint64_t first_remainder =
            first_numerator % first_denominator;
        const std::uint64_t second_remainder =
            second_numerator % second_denominator;
        if (first_remainder == 0 || second_remainder == 0) {
            if (first_remainder == second_remainder) return 0;
            const int result = first_remainder == 0 ? -1 : 1;
            return direction * result;
        }
        first_numerator = first_denominator;
        first_denominator = first_remainder;
        second_numerator = second_denominator;
        second_denominator = second_remainder;
        direction = -direction;
    }
}

int compare_signed_rational_components(
    std::int64_t first_numerator,
    std::int64_t first_denominator,
    std::int64_t second_numerator,
    std::int64_t second_denominator
) {
    if (first_denominator <= 0 || second_denominator <= 0) {
        throw std::runtime_error(
            "exact rational comparison requires positive denominators");
    }
    const bool first_negative = first_numerator < 0;
    const bool second_negative = second_numerator < 0;
    if (first_negative != second_negative) return first_negative ? -1 : 1;
    const int magnitude_comparison = compare_unsigned_rational(
        unsigned_magnitude(first_numerator),
        static_cast<std::uint64_t>(first_denominator),
        unsigned_magnitude(second_numerator),
        static_cast<std::uint64_t>(second_denominator));
    return first_negative ? -magnitude_comparison : magnitude_comparison;
}

int compare_integer_to_rational(std::int64_t value, const Rational64& rational) {
    return compare_signed_rational_components(
        value, 1, rational.numerator, rational.denominator);
}

int compare_rational(const Rational64& first, const Rational64& second) {
    return compare_signed_rational_components(
        first.numerator, first.denominator,
        second.numerator, second.denominator);
}

CanonicalFace rotate(const CanonicalFace& face, int offset) {
    CanonicalFace result{};
    for (int i = 0; i < 4; ++i) {
        result.key[i] = face.key[(i + offset) & 3];
        result.times[i] = face.times[(i + offset) & 3];
    }
    return result;
}

CanonicalFace reverse_cycle(const CanonicalFace& face) {
    return CanonicalFace{
        FaceKey{face.key[0], face.key[3], face.key[2], face.key[1]},
        std::array<std::int64_t, 4>{
            face.times[0], face.times[3], face.times[2], face.times[1]},
    };
}

CanonicalFace canonical_face(const CanonicalFace& input) {
    CanonicalFace best = rotate(input, 0);
    for (int offset = 1; offset < 4; ++offset) {
        const CanonicalFace candidate = rotate(input, offset);
        if (candidate.key < best.key) best = candidate;
    }
    const CanonicalFace reversed = reverse_cycle(input);
    for (int offset = 0; offset < 4; ++offset) {
        const CanonicalFace candidate = rotate(reversed, offset);
        if (candidate.key < best.key) best = candidate;
    }
    return best;
}

bool all_hvids_distinct(const FaceKey& key) {
    std::set<HVID> unique(key.begin(), key.end());
    return unique.size() == 4;
}

// A face is ordered cyclically as (00, 10, 11, 01).
std::optional<SaddleSolution> admissible_saddle(
    const std::array<std::int64_t, 4>& times
) {
    constexpr std::int64_t minimum_serialized_time =
        std::numeric_limits<std::int8_t>::min();
    constexpr std::int64_t maximum_serialized_time =
        std::numeric_limits<std::int8_t>::max();
    for (const std::int64_t value : times) {
        if (value < minimum_serialized_time ||
            value > maximum_serialized_time) {
            throw std::runtime_error(
                "saddle corner time is outside serialized int8 range");
        }
    }
    const std::int64_t t00 = times[0];
    const std::int64_t t10 = times[1];
    const std::int64_t t11 = times[2];
    const std::int64_t t01 = times[3];

    const std::int64_t A = t00 + t11 - t10 - t01;
    const std::int64_t B = t00 * t11 - t10 * t01;
    if (A == 0) {
        if (B == 0) {
            ++state.a0_b0;
        } else {
            ++state.a0_b_nonzero;
        }
        return std::nullopt;
    }
    ++state.finite_algebraic_roots;
    const Rational64 root = normalize(B, A);

    // Reject a root at or outside the four vertex-time envelope.
    const auto [minimum_it, maximum_it] = std::minmax_element(times.begin(), times.end());
    if (compare_integer_to_rational(*minimum_it, root) >= 0 ||
        compare_integer_to_rational(*maximum_it, root) <= 0) {
        return std::nullopt;
    }
    for (const std::int64_t value : times) {
        if (compare_integer_to_rational(value, root) == 0) {
            return std::nullopt;
        }
    }

    // At an admissible face saddle, opposite corners have the same sign and
    // adjacent corners have opposite signs at the critical value.
    const int s00 = compare_integer_to_rational(t00, root);
    const int s10 = compare_integer_to_rational(t10, root);
    const int s11 = compare_integer_to_rational(t11, root);
    const int s01 = compare_integer_to_rational(t01, root);
    const auto same_nonzero_sign = [](int first, int second) {
        return (first < 0 && second < 0) ||
               (first > 0 && second > 0);
    };
    const auto opposite_sign = [](int first, int second) {
        return (first < 0 && second > 0) ||
               (first > 0 && second < 0);
    };
    if (!(same_nonzero_sign(s00, s11) &&
          same_nonzero_sign(s10, s01) && opposite_sign(s00, s10))) {
        return std::nullopt;
    }

    const Rational64 u = normalize(t00 - t01, A);
    const Rational64 v = normalize(t00 - t10, A);
    if (!(u.numerator > 0 && u.numerator < u.denominator &&
          v.numerator > 0 && v.numerator < v.denominator)) {
        return std::nullopt;
    }
    return SaddleSolution{root, u, v, A, B};
}

std::string hvid_string(const HVID& value) {
    std::ostringstream stream;
    stream << value.first << ':' << static_cast<int>(value.second);
    return stream.str();
}

std::string face_string(const FaceKey& key) {
    std::ostringstream stream;
    for (int corner = 0; corner < 4; ++corner) {
        if (corner != 0) stream << '|';
        stream << hvid_string(key[corner]);
    }
    return stream.str();
}

std::string event_id(int element, int face_axis, const FaceKey& key) {
    std::ostringstream stream;
    stream << "element=" << element
           << ";role=" << hyperpoly_layout::axis_role_name(
               static_cast<hyperpoly_layout::AxisRole>(face_axis))
           << ";face=" << face_string(key);
    return stream.str();
}

std::string logical_incidence_id(
    const std::string& canonical_event_id,
    const hyperpoly_provenance::SourceRecord& source
) {
    std::ostringstream stream;
    stream << canonical_event_id << ";edge="
           << source.edge_coords[0] << ',' << source.edge_coords[1] << ','
           << source.edge_coords[2] << ',' << source.edge_L << ','
           << source.edge_tcoord << ',' << source.edge_tL << ','
           << source.edge_dir << ',' << source.element;
    return stream.str();
}

std::string raw_observation_id(
    int t_group,
    int t_start,
    int sorted_record_index,
    int face_axis,
    int face_side,
    const hyperpoly_provenance::SourceRecord& source
) {
    std::ostringstream stream;
    stream << "cache=" << t_group << ':' << t_start << ':'
           << sorted_record_index << ";source=" << source.source_t_group
           << ':' << source.source_record_index << ";face=" << face_axis
           << ':' << face_side;
    return stream.str();
}

std::string json_escape(const std::string& input) {
    std::ostringstream output;
    for (const unsigned char character : input) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(character)
                           << std::dec << std::setfill(' ');
                } else {
                    output << character;
                }
        }
    }
    return output.str();
}

void validate_source_record(
    const HP& hyperpoly,
    const hyperpoly_provenance::SourceRecord& source
) {
    if (source.element != static_cast<int>(hyperpoly.second)) {
        throw std::runtime_error("hyperpoly provenance element mismatch");
    }
    for (int corner = 0; corner < 8; ++corner) {
        if (source.hvid_node[corner] != hyperpoly.first[corner].first ||
            source.hvid_group[corner] !=
                static_cast<int>(hyperpoly.first[corner].second)) {
            throw std::runtime_error(
                "hyperpoly provenance HVID alignment mismatch");
        }
    }
}

void build_selected_saddle() {
    state.selected_saddle_event_id.reset();
    state.selected_saddle_raw_ids.clear();
    state.selected_saddle_logical_ids.clear();
    const int temporal_axis = static_cast<int>(
        hyperpoly_layout::AxisRole::temporal_neighbour);
    for (const auto& entry : state.events) {
        const EventAggregate& candidate = entry.second;
        if (candidate.face_axis != temporal_axis) continue;
        if (!state.selected_saddle_event_id.has_value()) {
            state.selected_saddle_event_id = entry.first;
            continue;
        }
        const EventAggregate& selected =
            state.events.at(*state.selected_saddle_event_id);
        if (candidate.raw_observations > selected.raw_observations ||
            (candidate.raw_observations == selected.raw_observations &&
             candidate.logical_incidence_ids.size() >
                 selected.logical_incidence_ids.size()) ||
            (candidate.raw_observations == selected.raw_observations &&
             candidate.logical_incidence_ids.size() ==
                 selected.logical_incidence_ids.size() &&
             entry.first < *state.selected_saddle_event_id)) {
            state.selected_saddle_event_id = entry.first;
        }
    }
    if (!state.selected_saddle_event_id.has_value()) return;
    const EventAggregate& selected =
        state.events.at(*state.selected_saddle_event_id);
    state.selected_saddle_logical_ids = selected.logical_incidence_ids;
    for (const SaddleObservation& observation : state.observations) {
        if (observation.canonical_event_id ==
            *state.selected_saddle_event_id) {
            state.selected_saddle_raw_ids.insert(observation.raw_id);
        }
    }
    if (state.selected_saddle_raw_ids.size() !=
            static_cast<std::size_t>(selected.raw_observations) ||
        state.selected_saddle_logical_ids.size() !=
            selected.logical_incidence_ids.size()) {
        throw std::runtime_error(
            "selected saddle provenance cardinality mismatch");
    }
}

void write_summary_json(std::ostream& output) {
    const int temporal_axis = static_cast<int>(
        hyperpoly_layout::AxisRole::temporal_neighbour);
    std::size_t temporal_raw_observations = 0;
    std::set<std::string> temporal_logical_incidences;
    for (const SaddleObservation& observation : state.observations) {
        if (observation.face_axis != temporal_axis) continue;
        ++temporal_raw_observations;
        temporal_logical_incidences.insert(
            observation.logical_incidence_id);
    }
    std::size_t temporal_canonical_events = 0;
    for (const auto& entry : state.events) {
        if (entry.second.face_axis == temporal_axis) {
            ++temporal_canonical_events;
        }
    }
    output << "{\n";
    output << R"(  "all_parameter_faces": {"raw_observations": )"
           << state.observations.size()
           << R"(, "logical_incidences": )"
           << state.logical_incidence_ids.size()
           << R"(, "canonical_events": )" << state.events.size()
           << "},\n";
    output << R"(  "temporal_neighbour_faces": {"raw_observations": )"
           << temporal_raw_observations
           << R"(, "logical_incidences": )"
           << temporal_logical_incidences.size()
           << R"(, "canonical_events": )" << temporal_canonical_events
           << "},\n";
    output << "  \"enabled\": " << (state.enabled ? "true" : "false")
           << ",\n";
    output << "  \"hyperpolys\": " << state.hyperpolys << ",\n";
    output << "  \"faces_examined\": " << state.faces_examined << ",\n";
    output << "  \"finite_algebraic_roots\": "
           << state.finite_algebraic_roots << ",\n";
    output << "  \"accepted_saddle_occurrences\": "
           << state.accepted_saddles << ",\n";
    output << "  \"raw_observations\": " << state.observations.size()
           << ",\n";
    output << "  \"logical_incidences\": "
           << state.logical_incidence_ids.size() << ",\n";
    output << "  \"canonical_events\": " << state.events.size() << ",\n";
    output << "  \"canonical_shared_events\": " << state.events.size()
           << ",\n";
    output << "  \"a0_b0_degeneracies\": " << state.a0_b0 << ",\n";
    output << "  \"a0_b_nonzero_faces\": " << state.a0_b_nonzero << ",\n";
    output << "  \"reduced_hvid_faces\": " << state.reduced_hvid_faces
           << ",\n";
    output << "  \"shared_root_mismatches\": "
           << state.shared_root_mismatches << "\n";
    output << "}\n";
}

void write_selected_event_json(std::ostream& output) {
    if (!state.selected_saddle_event_id.has_value()) {
        output << "{\n  \"selected\": false\n}\n";
        return;
    }
    const std::string& selected_id = *state.selected_saddle_event_id;
    const EventAggregate& event = state.events.at(selected_id);
    output << "{\n";
    output << "  \"selected\": true,\n";
    output << "  \"selection_rule\": \"temporal-neighbour events only; "
              "largest raw count, then logical count, then event_id\",\n";
    output << "  \"event_id\": \"" << json_escape(selected_id) << "\",\n";
    output << "  \"element\": " << event.element << ",\n";
    output << "  \"face_axis\": " << event.face_axis << ",\n";
    output << "  \"face_axis_role\": \""
           << hyperpoly_layout::axis_role_name(
               static_cast<hyperpoly_layout::AxisRole>(event.face_axis))
           << "\",\n";
    output << "  \"temporal_provenance\": {\"layout_version\": "
           << hyperpoly_provenance::kLayoutVersion
           << ", \"producer_mapping\": "
              "\"dual_contouring hp_slot=i+2*j+4*t\", "
              "\"verified_temporal_face\": "
           << (event.face_axis == static_cast<int>(
                   hyperpoly_layout::AxisRole::temporal_neighbour)
                   ? "true" : "false")
           << "},\n";
    output << "  \"producer_temporal_face_slots\": "
              "{\"side_0\": [0,1,3,2], \"side_1\": [4,5,7,6]},\n";
    output << "  \"canonical_hvids\": [";
    for (int corner = 0; corner < 4; ++corner) {
        if (corner != 0) output << ',';
        output << "\"" << hvid_string(event.key[corner]) << "\"";
    }
    output << "],\n  \"corner_times\": [";
    for (int corner = 0; corner < 4; ++corner) {
        if (corner != 0) output << ',';
        output << event.times[corner];
    }
    output << "],\n";
    output << "  \"A\": " << event.solution.A << ",\n";
    output << "  \"B\": " << event.solution.B << ",\n";
    output << "  \"root\": {\"numerator\": "
           << event.solution.time.numerator << ", \"denominator\": "
           << event.solution.time.denominator << "},\n";
    output << "  \"u\": {\"numerator\": " << event.solution.u.numerator
           << ", \"denominator\": " << event.solution.u.denominator
           << "},\n";
    output << "  \"v\": {\"numerator\": " << event.solution.v.numerator
           << ", \"denominator\": " << event.solution.v.denominator
           << "},\n";
    output << "  \"raw_observations\": " << event.raw_observations
           << ",\n";
    output << "  \"logical_incidences\": "
           << event.logical_incidence_ids.size() << ",\n";
    output << "  \"canonical_events\": 1,\n";
    output << "  \"raw_ids\": [";
    std::size_t raw_index = 0;
    for (const std::string& raw_id : state.selected_saddle_raw_ids) {
        if (raw_index++ != 0U) output << ',';
        output << "\"" << json_escape(raw_id) << "\"";
    }
    output << "],\n  \"logical_incidence_ids\": [";
    std::size_t logical_index = 0;
    for (const std::string& logical_id :
         state.selected_saddle_logical_ids) {
        if (logical_index++ != 0U) output << ',';
        output << "\"" << json_escape(logical_id) << "\"";
    }
    output << "]\n}\n";
}

}  // namespace

int compare_exact_rational(
    const Rational64& first,
    const Rational64& second
) {
    return compare_rational(first, second);
}

void begin(const std::string& output_path) {
    state = RegistryState{};
    const char* mode_text = std::getenv("BINOC_EVENT_MODE");
    const int mode = mode_text == nullptr ? 0 : std::atoi(mode_text);
    state.enabled = mode > 0;
    state.output_path = output_path;
}

void observe_hyperpoly(
    const HVTable& hypervertices,
    const HP& hyperpoly,
    const hyperpoly_provenance::SourceRecord& source,
    int t_group,
    int t_start,
    int sorted_record_index
) {
    if (!state.enabled) return;
    ++state.hyperpolys;
    validate_source_record(hyperpoly, source);

    for (int face_axis = 0; face_axis < 3; ++face_axis) {
        for (int face_side = 0; face_side < 2; ++face_side) {
            ++state.faces_examined;
            CanonicalFace face{};
            const auto corners = hyperpoly_layout::face_corners(
                static_cast<hyperpoly_layout::AxisRole>(face_axis),
                face_side);
            bool found = true;
            for (int corner = 0; corner < 4; ++corner) {
                const HVID hvid = hyperpoly.first[corners[corner]];
                face.key[corner] = hvid;
                const auto iterator = hypervertices.find(hvid);
                if (iterator == hypervertices.end()) {
                    found = false;
                    break;
                }
                face.times[corner] =
                    iterator->second.first.second[0];
            }
            if (!found) continue;
            if (!all_hvids_distinct(face.key)) {
                ++state.reduced_hvid_faces;
                continue;
            }
            const CanonicalFace canonical = canonical_face(face);
            const auto solution = admissible_saddle(canonical.times);
            if (!solution.has_value()) continue;

            ++state.accepted_saddles;
            SaddleObservation observation;
            observation.key = canonical.key;
            observation.times = canonical.times;
            observation.solution = *solution;
            observation.face_axis = face_axis;
            observation.face_side = face_side;
            observation.element = static_cast<int>(hyperpoly.second);
            observation.t_group = t_group;
            observation.t_start = t_start;
            observation.sorted_record_index = sorted_record_index;
            observation.source = source;
            observation.canonical_event_id = event_id(
                observation.element, face_axis, canonical.key);
            observation.logical_incidence_id = logical_incidence_id(
                observation.canonical_event_id, source);
            observation.raw_id = raw_observation_id(
                t_group, t_start, sorted_record_index, face_axis,
                face_side, source);
            if (!state.raw_observation_ids.insert(observation.raw_id).second) {
                throw std::runtime_error(
                    "duplicate raw saddle observation identity");
            }

            auto existing = state.events.find(
                observation.canonical_event_id);
            if (existing == state.events.end()) {
                EventAggregate aggregate;
                aggregate.key = canonical.key;
                aggregate.times = canonical.times;
                aggregate.solution = *solution;
                aggregate.face_axis = face_axis;
                aggregate.element = observation.element;
                existing = state.events.emplace(
                    observation.canonical_event_id,
                    aggregate).first;
            } else if (!rational_equal(
                           existing->second.solution.time,
                           solution->time) ||
                       existing->second.times != canonical.times) {
                ++state.shared_root_mismatches;
                throw std::runtime_error(
                    "canonical event has inconsistent root or corner times");
            }
            ++existing->second.raw_observations;
            existing->second.logical_incidence_ids.insert(
                observation.logical_incidence_id);
            state.logical_incidence_ids.insert(
                observation.logical_incidence_id);
            state.observations.push_back(std::move(observation));
        }
    }
}

void finish() {
    if (!state.enabled) return;
    build_selected_saddle();
    std::filesystem::create_directories(state.output_path);
    const std::filesystem::path csv_path =
        state.output_path / "event_registry_p1.csv";
    checked_output::TextFile csv_output(
        csv_path, "event registry CSV output");
    std::ostream& csv = csv_output.stream();
    csv << "raw_id,t_group,t_start,sorted_record_index,source_t_group,"
           "source_record_index,element,edge_x,edge_y,edge_z,edge_L,"
           "edge_tcoord,edge_tL,edge_dir,source_h0,source_h1,source_h2,"
           "source_h3,source_h4,source_h5,source_h6,source_h7,face_axis,"
           "face_axis_role,face_side,h0,h1,h2,h3,t0,t1,t2,t3,A,B,"
           "root_num,root_den,u_num,u_den,v_num,v_den,"
           "logical_incidence_id,canonical_event_id\n";
    for (const SaddleObservation& observation : state.observations) {
        csv << observation.raw_id << ',' << observation.t_group << ','
            << observation.t_start << ','
            << observation.sorted_record_index << ','
            << observation.source.source_t_group << ','
            << observation.source.source_record_index << ','
            << observation.element;
        for (int axis = 0; axis < 3; ++axis) {
            csv << ',' << observation.source.edge_coords[axis];
        }
        csv << ',' << observation.source.edge_L << ','
            << observation.source.edge_tcoord << ','
            << observation.source.edge_tL << ','
            << observation.source.edge_dir;
        for (int corner = 0; corner < 8; ++corner) {
            csv << ',' << observation.source.hvid_node[corner] << ':'
                << observation.source.hvid_group[corner];
        }
        csv << ',' << observation.face_axis << ','
            << hyperpoly_layout::axis_role_name(
                   static_cast<hyperpoly_layout::AxisRole>(
                       observation.face_axis))
            << ',' << observation.face_side;
        for (const HVID& hvid : observation.key) {
            csv << ',' << hvid_string(hvid);
        }
        for (const std::int64_t time : observation.times) {
            csv << ',' << time;
        }
        csv << ',' << observation.solution.A << ','
            << observation.solution.B << ','
            << observation.solution.time.numerator << ','
            << observation.solution.time.denominator << ','
            << observation.solution.u.numerator << ','
            << observation.solution.u.denominator << ','
            << observation.solution.v.numerator << ','
            << observation.solution.v.denominator << ','
            << std::quoted(observation.logical_incidence_id) << ','
            << std::quoted(observation.canonical_event_id) << '\n';
    }
    checked_output::TextFile summary_output(
        state.output_path / "event_registry_p1_summary.json",
        "event registry summary output");
    checked_output::TextFile selected_output(
        state.output_path / "event_registry_selected_event.json",
        "event registry selected-event output");
    write_summary_json(summary_output.stream());
    write_selected_event_json(selected_output.stream());
    csv_output.commit();
    summary_output.commit();
    selected_output.commit();
}

}  // namespace event_registry

namespace {
struct Point3 {
    double x;
    double y;
    double z;
};

Point3 add(const Point3& a, const Point3& b) {
    return Point3{a.x + b.x, a.y + b.y, a.z + b.z};
}
Point3 scale(const Point3& value, double factor) {
    return Point3{value.x * factor, value.y * factor, value.z * factor};
}
Point3 subtract(const Point3& a, const Point3& b) {
    return Point3{a.x - b.x, a.y - b.y, a.z - b.z};
}
double norm(const Point3& value) {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}
Point3 centroid3(const Point3& a, const Point3& b, const Point3& c) {
    return scale(add(add(a, b), c), 1.0 / 3.0);
}
Point3 interpolate_from_carrier(const Point3& carrier, const Point3& value, double alpha) {
    return add(scale(carrier, 1.0 - alpha), scale(value, alpha));
}
}  // namespace

extern "C" {

int binoc_event_fixture_saddle(std::int32_t* numerator, std::int32_t* denominator) {
    if (numerator == nullptr || denominator == nullptr) {
        return -1;
    }
    const std::int64_t t00 = 1;
    const std::int64_t t10 = 4;
    const std::int64_t t11 = 1;
    const std::int64_t t01 = 4;
    std::int64_t n = t00 * t11 - t10 * t01;
    std::int64_t d = t00 + t11 - t10 - t01;
    if (d == 0) {
        return -2;
    }
    if (d < 0) {
        n = -n;
        d = -d;
    }
    const std::int64_t divisor = std::gcd(n < 0 ? -n : n, d);
    *numerator = static_cast<std::int32_t>(n / divisor);
    *denominator = static_cast<std::int32_t>(d / divisor);
    return 0;
}

int binoc_event_fixture_endpoint(double* official_per_face_gap, double* shared_gap) {
    if (official_per_face_gap == nullptr || shared_gap == nullptr) {
        return -1;
    }
    const Point3 a{0.0, 0.0, 0.0};
    const Point3 b{1.0, 0.0, 0.0};
    const Point3 c{0.0, 1.0, 0.0};
    const Point3 d{0.0, -1.0, 0.0};
    const double alpha = 0.5;
    const Point3 centroid_abc = centroid3(a, b, c);
    const Point3 centroid_abd = centroid3(a, b, d);
    const Point3 a_from_abc = interpolate_from_carrier(centroid_abc, a, alpha);
    const Point3 a_from_abd = interpolate_from_carrier(centroid_abd, a, alpha);
    *official_per_face_gap = norm(subtract(a_from_abc, a_from_abd));

    // One shared critical carrier produces one trajectory for the shared HVID.
    const Point3 shared_carrier = scale(add(add(add(a, b), c), d), 0.25);
    const Point3 shared_a_0 = interpolate_from_carrier(shared_carrier, a, alpha);
    const Point3 shared_a_1 = interpolate_from_carrier(shared_carrier, a, alpha);
    *shared_gap = norm(subtract(shared_a_0, shared_a_1));
    return 0;
}

int binoc_event_fixture_replay(
    std::int32_t samples,
    std::int32_t* registry_events,
    std::int32_t* uniform_exact_hits,
    std::int32_t* simultaneous_batch_size
) {
    if (samples < 2 || registry_events == nullptr || uniform_exact_hits == nullptr ||
        simultaneous_batch_size == nullptr) {
        return -1;
    }
    *registry_events = 1;
    *simultaneous_batch_size = 2;  // one endpoint event and one saddle share 5/2.
    *uniform_exact_hits = 0;
    for (std::int32_t i = 0; i < samples; ++i) {
        // Uniform samples over [1,4]: t_i = 1 + 3 i/(samples-1).
        // Compare exactly against 5/2 without floating point.
        const std::int64_t lhs = 2LL * ((samples - 1LL) + 3LL * i);
        const std::int64_t rhs = 5LL * (samples - 1LL);
        if (lhs == rhs) {
            ++(*uniform_exact_hits);
        }
    }
    return 0;
}

}  // extern "C"
