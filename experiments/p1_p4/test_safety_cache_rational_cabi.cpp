#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>

#include "utils.h"
#include "binoctree.h"
#include "coarse_step.h"
#include "medium_step.h"
#include "fine_step.h"
#include "dual_contouring.h"
#include "bisection.h"
#include "event_registry.h"

extern "C" {
int run_slicing_rational(
    std::int64_t numerator,
    std::int64_t denominator,
    int *vertex_counts,
    int *face_counts,
    bool extra_smooth) noexcept;
const char *slicing_last_error() noexcept;
}

static_assert(noexcept(write_final_hypermesh(0)));
static_assert(std::is_same_v<
              decltype(&write_final_hypermesh), int (*)(int) noexcept>);

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void write_count(const std::filesystem::path& path, int count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    require(static_cast<bool>(output), "failed to write cache count");
}

bool hyperpoly_rejects(const std::filesystem::path& path, int count) {
    write_count(path, count);
    try { load_hyperpolys(0); }
    catch (const std::runtime_error&) { return true; }
    return false;
}

bool hypervertex_rejects(const std::filesystem::path& path, int count) {
    write_count(path, count);
    try { load_vertices(0); }
    catch (const std::runtime_error&) { return true; }
    return false;
}

void test_rational_boundaries() {
    using event_registry::Rational64;
    const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
    const std::int64_t minimum = std::numeric_limits<std::int64_t>::min();
    const auto relation = [](Rational64 first, Rational64 second, int expected) {
        return event_registry::compare_exact_rational(first, second) == expected &&
               event_registry::compare_exact_rational(second, first) == -expected;
    };
    require(relation({maximum, maximum}, {1, 1}, 0),
            "equal full-width rational comparison failed");
    require(relation({maximum, maximum - 1}, {1, 1}, 1),
            "full-width greater rational comparison failed");
    require(relation({minimum, maximum}, {-1, 1}, -1),
            "INT64_MIN rational comparison failed");
    require(relation({maximum - 1, maximum}, {1, 1}, -1),
            "large-denominator rational comparison failed");
    require(relation({32, 3}, {10, 1}, 1) &&
            relation({32, 3}, {11, 1}, -1),
            "target rational ordering failed");
    bool rejected = false;
    try {
        static_cast<void>(event_registry::compare_exact_rational(
            Rational64{1, 0}, Rational64{1, 1}));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "zero rational denominator was accepted");
}

}  // namespace

int main() {
    test_rational_boundaries();

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("binoc-safety-cache-" +
         std::to_string(static_cast<long long>(getpid())));
    std::filesystem::create_directories(root / "hyperpolys");
    std::filesystem::create_directories(root / "hypervertices");
    std::filesystem::create_directories(root / "processed_hyperpolys");
    params::output_path = root.string();
    params::log_path = (root / "log.txt").string();

    const auto hyperpolys = root / "hyperpolys/0.bin";
    require(hyperpoly_rejects(hyperpolys, -1),
            "negative hyperpoly count was accepted");
    require(hyperpoly_rejects(
                hyperpolys, std::numeric_limits<int>::max()),
            "huge hyperpoly count was accepted");
    write_count(hyperpolys, 0);
    load_hyperpolys(0, 0);

    const auto hypervertices = root / "hypervertices/0.bin";
    require(hypervertex_rejects(hypervertices, -1),
            "negative hypervertex count was accepted");
    write_count(hypervertices, 1);
    require(hypervertex_rejects(hypervertices, 1),
            "truncated hypervertex record was accepted");
    write_count(hypervertices, 0);
    load_vertices(0);

    const size_t map_size = bisection::hypervertices.size();
    bool missing_hvid_rejected = false;
    try {
        static_cast<void>(bisection::hypervertices[
            HVID{999, static_cast<std::int8_t>(0)}]);
    } catch (const std::runtime_error&) {
        missing_hvid_rejected = true;
    }
    require(missing_hvid_rejected &&
                bisection::hypervertices.size() == map_size,
            "missing HVID lookup mutated the cache");

    static vec<int, queriedEdge> empty_edges;
    dual_contouring::edges_ = &empty_edges;
    params::output_path = (root / "missing-output-root").string();
    const int write_status = write_final_hypermesh(0);
    require(write_status == -1,
            "write_final_hypermesh exception crossed the C ABI");
    require(std::string(bisection_last_error()) ==
                "failed to open computed-vertex output",
            "write_final_hypermesh lost its diagnostic");

    params::output_path = root.string();
    params::n_elements = 1;
    params::max_tL = 4;
    params::tsize = 32.0;
    params::deltaT = 1.0;
    write_count(root / "hypervertices/15.bin", 0);
    int vertex_counts[1] = {-7};
    int face_counts[1] = {-7};
    const int terminal_status = run_slicing_rational(
        32, 1, vertex_counts, face_counts, true);
    require(terminal_status == -1,
            "terminal exact slice did not fail through status ABI");
    require(std::string(slicing_last_error()) ==
                "missing synchronized processed hyperpoly stream",
            "terminal exact slice did not select final cache group");
    require(vertex_counts[0] == 0 && face_counts[0] == 0,
            "failed exact slice retained output counts");

    require(run_slicing_rational(
                33, 1, vertex_counts, face_counts, true) == -1,
            "out-of-domain exact slice was accepted");
    require(std::string(slicing_last_error()) ==
                "exact slicing time is outside the production cache",
            "out-of-domain exact slice lost its diagnostic");

    const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
    write_count(root / "hypervertices/0.bin", 0);
    require(run_slicing_rational(
                maximum - 1, maximum,
                vertex_counts, face_counts, true) == -1,
            "large-denominator exact slice unexpectedly completed");
    require(std::string(slicing_last_error()) ==
                "missing synchronized processed hyperpoly stream",
            "large denominator was rejected before checked cache consumption");

    std::filesystem::remove_all(root);
    std::cout << "PASS_SAFETY_CACHE_RATIONAL_C_ABI\n";
    return 0;
}
