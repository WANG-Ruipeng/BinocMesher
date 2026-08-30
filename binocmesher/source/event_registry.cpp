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
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace event_registry {
namespace {

using FaceKey = std::array<HVID, 4>;

struct SaddleObservation {
    FaceKey key{};
    Rational64 time{};
    int face_axis = -1;
    int face_side = -1;
    int element = -1;
};

struct RegistryState {
    bool enabled = false;
    std::filesystem::path output_path;
    std::vector<SaddleObservation> observations;
    std::map<FaceKey, Rational64> shared_roots;
    std::int64_t hyperpolys = 0;
    std::int64_t faces_examined = 0;
    std::int64_t finite_algebraic_roots = 0;
    std::int64_t accepted_saddles = 0;
    std::int64_t a0_b0 = 0;
    std::int64_t a0_b_nonzero = 0;
    std::int64_t reduced_hvid_faces = 0;
    std::int64_t shared_root_mismatches = 0;
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
    // Avoid negating INT64_MIN in the signed domain.
    return static_cast<std::uint64_t>(-(value + 1)) + UINT64_C(1);
}

int compare_unsigned_rational(
    std::uint64_t first_numerator,
    std::uint64_t first_denominator,
    std::uint64_t second_numerator,
    std::uint64_t second_denominator
) {
    // Compare through the continued-fraction expansion. Division/remainder
    // stays exact when either cross product would overflow uint64_t.
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
    if (first_negative != second_negative) {
        return first_negative ? -1 : 1;
    }

    const int magnitude_comparison = compare_unsigned_rational(
        unsigned_magnitude(first_numerator),
        static_cast<std::uint64_t>(first_denominator),
        unsigned_magnitude(second_numerator),
        static_cast<std::uint64_t>(second_denominator));
    return first_negative ? -magnitude_comparison : magnitude_comparison;
}

int compare_integer_to_rational(
    std::int64_t value,
    const Rational64& rational
) {
    return compare_signed_rational_components(
        value, 1, rational.numerator, rational.denominator);
}

int compare_rational(const Rational64& first, const Rational64& second) {
    return compare_signed_rational_components(
        first.numerator, first.denominator,
        second.numerator, second.denominator);
}

FaceKey rotate(const FaceKey& key, int offset) {
    FaceKey result{};
    for (int i = 0; i < 4; ++i) {
        result[i] = key[(i + offset) & 3];
    }
    return result;
}

FaceKey reverse_cycle(const FaceKey& key) {
    return FaceKey{key[0], key[3], key[2], key[1]};
}

FaceKey canonical_face_key(const FaceKey& input) {
    FaceKey best = rotate(input, 0);
    for (int offset = 1; offset < 4; ++offset) {
        best = std::min(best, rotate(input, offset));
    }
    const FaceKey reversed = reverse_cycle(input);
    for (int offset = 0; offset < 4; ++offset) {
        best = std::min(best, rotate(reversed, offset));
    }
    return best;
}

bool all_hvids_distinct(const FaceKey& key) {
    std::set<HVID> unique(key.begin(), key.end());
    return unique.size() == 4;
}

// A face is ordered cyclically as (00, 10, 11, 01).
std::optional<Rational64> admissible_saddle(const std::array<std::int64_t, 4>& times) {
    constexpr std::int64_t minimum_serialized_time =
        std::numeric_limits<std::int8_t>::min();
    constexpr std::int64_t maximum_serialized_time =
        std::numeric_limits<std::int8_t>::max();
    for (const std::int64_t value : times) {
        // Production corner times use timeT == int8_t. Keep this wider helper
        // fail-closed so the coefficient arithmetic below cannot overflow.
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

    // The bilinear critical point must lie strictly inside the parameter face.
    // f(u,v) = a + b u + c v + d u v.
    const long double b = static_cast<long double>(t10 - t00);
    const long double c = static_cast<long double>(t01 - t00);
    const long double d = static_cast<long double>(t11 - t10 - t01 + t00);
    if (d == 0.0L) {
        return std::nullopt;
    }
    const long double u = -c / d;
    const long double v = -b / d;
    if (!(u > 0.0L && u < 1.0L && v > 0.0L && v < 1.0L)) {
        return std::nullopt;
    }
    return root;
}

std::string hvid_string(const HVID& value) {
    std::ostringstream stream;
    stream << value.first << ':' << static_cast<int>(value.second);
    return stream.str();
}

void write_summary_json(const std::filesystem::path& path) {
    std::ofstream output(path);
    output << "{\n";
    output << "  \"enabled\": " << (state.enabled ? "true" : "false") << ",\n";
    output << "  \"hyperpolys\": " << state.hyperpolys << ",\n";
    output << "  \"faces_examined\": " << state.faces_examined << ",\n";
    output << "  \"finite_algebraic_roots\": " << state.finite_algebraic_roots << ",\n";
    output << "  \"accepted_saddle_occurrences\": " << state.accepted_saddles << ",\n";
    output << "  \"canonical_shared_events\": " << state.shared_roots.size() << ",\n";
    output << "  \"a0_b0_degeneracies\": " << state.a0_b0 << ",\n";
    output << "  \"a0_b_nonzero_faces\": " << state.a0_b_nonzero << ",\n";
    output << "  \"reduced_hvid_faces\": " << state.reduced_hvid_faces << ",\n";
    output << "  \"shared_root_mismatches\": " << state.shared_root_mismatches << "\n";
    output << "}\n";
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

void observe_hyperpoly(const HVTable& hypervertices, const HP& hyperpoly) {
    if (!state.enabled) {
        return;
    }
    ++state.hyperpolys;

    // Cube corner index follows cube_index(x,y,z,2) = 4x + 2y + z.
    static constexpr int faces[6][4] = {
        {0, 1, 3, 2}, {4, 5, 7, 6},
        {0, 1, 5, 4}, {2, 3, 7, 6},
        {0, 2, 6, 4}, {1, 3, 7, 5},
    };

    for (int face_index = 0; face_index < 6; ++face_index) {
        ++state.faces_examined;
        FaceKey key{};
        std::array<std::int64_t, 4> times{};
        bool found = true;
        for (int corner = 0; corner < 4; ++corner) {
            const HVID hvid = hyperpoly.first[faces[face_index][corner]];
            key[corner] = hvid;
            const auto iterator = hypervertices.find(hvid);
            if (iterator == hypervertices.end()) {
                found = false;
                break;
            }
            times[corner] = iterator->second.first.second[0];
        }
        if (!found) {
            continue;
        }
        if (!all_hvids_distinct(key)) {
            ++state.reduced_hvid_faces;
            continue;
        }
        const auto root = admissible_saddle(times);
        if (!root.has_value()) {
            continue;
        }

        ++state.accepted_saddles;
        const FaceKey canonical = canonical_face_key(key);
        const auto existing = state.shared_roots.find(canonical);
        if (existing == state.shared_roots.end()) {
            state.shared_roots.emplace(canonical, *root);
        } else if (!rational_equal(existing->second, *root)) {
            ++state.shared_root_mismatches;
        }

        SaddleObservation observation;
        observation.key = canonical;
        observation.time = *root;
        observation.face_axis = face_index / 2;
        observation.face_side = face_index & 1;
        observation.element = static_cast<int>(hyperpoly.second);
        state.observations.push_back(observation);
    }
}

void finish() {
    if (!state.enabled) {
        return;
    }
    std::filesystem::create_directories(state.output_path);
    const std::filesystem::path csv_path = state.output_path / "event_registry_p1.csv";
    std::ofstream csv(csv_path);
    csv << "element,face_axis,face_side,h0,h1,h2,h3,root_num,root_den\n";
    for (const SaddleObservation& observation : state.observations) {
        csv << observation.element << ',' << observation.face_axis << ',' << observation.face_side;
        for (const HVID& hvid : observation.key) {
            csv << ',' << hvid_string(hvid);
        }
        csv << ',' << observation.time.numerator << ',' << observation.time.denominator << '\n';
    }
    write_summary_json(state.output_path / "event_registry_p1_summary.json");
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
