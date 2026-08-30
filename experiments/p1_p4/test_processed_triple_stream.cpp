// Integration regression for the three synchronized slicing cache streams:
//
//   processed_hyperpolys/<group>_<start>.bin
//   processed_hyperpolys/<group>_<start>_discon.bin
//   processed_hyperpolys/<group>_<start>_hpmeta.bin
//
// The fixture uses the production scalar/vector serialization helpers and the
// packed BPM2 metadata ABI.  It parses a complete processed record, including
// its time vector, polygon-face vectors, and terminal empty-face sentinel.
// Corruptions must therefore fail at the same record boundaries enforced by
// run_slicing(), rather than merely comparing the three files' byte lengths.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "utils.h"
#include "binoctree.h"
#include "bisection.h"
#include "hyperpoly_provenance.h"

namespace {

using SerializedVID = array<HVID, 2>;
using TimeVector = vec<int, timeT>;
using FaceVector = vec<int, SerializedVID>;
using DiscontinuityVector = vec<int, II>;

// The upstream cache intentionally writes these libstdc++ pair/array objects
// as raw bytes.  Pin their observed production ABI here instead of claiming
// std::pair is formally trivially copyable (libstdc++ does not promise that).
static_assert(sizeof(II) == 8, "unexpected discontinuity-pair ABI");
static_assert(sizeof(HVID) == 8, "unexpected HVID ABI");
static_assert(sizeof(SerializedVID) == 16, "unexpected processed VID ABI");
static_assert(sizeof(hyperpoly_provenance::ProcessedRecord) == 132,
              "BPM2 processed metadata ABI changed");

enum class Corruption {
    none,
    metadata_early_eof,
    metadata_trailing_record,
    metadata_truncated_record,
    discontinuity_early_eof,
    discontinuity_trailing_record,
    truncated_times,
    truncated_face,
    missing_face_sentinel,
};

struct FixtureBytes {
    std::vector<std::uint8_t> primary;
    std::vector<std::uint8_t> discontinuity;
    std::vector<std::uint8_t> metadata;
};

template <typename Value>
void append_object(std::vector<std::uint8_t>& output, const Value& value) {
    const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
    output.insert(output.end(), first, first + sizeof(Value));
}

template <typename Value>
void append_vector(
    std::vector<std::uint8_t>& output,
    const std::vector<Value>& values
) {
    if (values.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("fixture vector is too large");
    }
    const int count = static_cast<int>(values.size());
    append_object(output, count);
    for (const Value& value : values) append_object(output, value);
}

hyperpoly_provenance::ProcessedRecord make_metadata(int sorted_index) {
    hyperpoly_provenance::ProcessedRecord record{};
    record.magic = hyperpoly_provenance::kProcessedMagic;
    record.version = hyperpoly_provenance::kVersion;
    record.record_size = sizeof(record);
    record.layout_version = hyperpoly_provenance::kLayoutVersion;
    record.t_group = 0;
    record.t_start = 8;
    record.sorted_record_index = sorted_index;
    record.source.source_t_group = 0;
    record.source.source_record_index = sorted_index;
    record.source.element = 0;
    return record;
}

SerializedVID make_vid(int first_node, int second_node) {
    return SerializedVID{
        HVID{first_node, static_cast<std::int8_t>(0)},
        HVID{second_node, static_cast<std::int8_t>(0)},
    };
}

FixtureBytes make_fixture(Corruption corruption) {
    FixtureBytes fixture;
    const eleT element = 0;
    append_object(fixture.primary, element);

    if (corruption == Corruption::truncated_times) {
        const int declared_count = 2;
        const timeT only_value = 8;
        append_object(fixture.primary, declared_count);
        append_object(fixture.primary, only_value);
    } else {
        append_vector<timeT>(fixture.primary, {8, 12});
        const std::vector<SerializedVID> triangle{
            make_vid(10, 38),
            make_vid(38, 76),
            make_vid(76, 10),
        };
        append_vector(fixture.primary, triangle);
        if (corruption == Corruption::truncated_face) {
            fixture.primary.pop_back();
        } else if (corruption != Corruption::missing_face_sentinel) {
            append_vector<SerializedVID>(fixture.primary, {});
        }
    }

    if (corruption != Corruption::discontinuity_early_eof) {
        append_vector<II>(fixture.discontinuity, {});
    }
    if (corruption == Corruption::discontinuity_trailing_record) {
        append_vector<II>(fixture.discontinuity, {});
    }

    if (corruption != Corruption::metadata_early_eof) {
        append_object(fixture.metadata, make_metadata(0));
    }
    if (corruption == Corruption::metadata_trailing_record) {
        append_object(fixture.metadata, make_metadata(1));
    }
    if (corruption == Corruption::metadata_truncated_record) {
        fixture.metadata.pop_back();
    }
    return fixture;
}

FILE* open_fixture_stream(const std::vector<std::uint8_t>& bytes) {
    FILE* stream = tmpfile();
    if (stream == nullptr) throw std::runtime_error("tmpfile failed");
    if (!bytes.empty() &&
        fwrite(bytes.data(), 1, bytes.size(), stream) != bytes.size()) {
        fclose(stream);
        throw std::runtime_error("fixture write failed");
    }
    rewind(stream);
    return stream;
}

class TripleStreams {
public:
    explicit TripleStreams(const FixtureBytes& fixture)
        : primary_(open_fixture_stream(fixture.primary)),
          discontinuity_(open_fixture_stream(fixture.discontinuity)),
          metadata_(open_fixture_stream(fixture.metadata)) {}

    ~TripleStreams() {
        fclose(primary_);
        fclose(discontinuity_);
        fclose(metadata_);
    }

    TripleStreams(const TripleStreams&) = delete;
    TripleStreams& operator=(const TripleStreams&) = delete;

    FILE* primary() const { return primary_; }
    FILE* discontinuity() const { return discontinuity_; }
    FILE* metadata() const { return metadata_; }

private:
    FILE* primary_;
    FILE* discontinuity_;
    FILE* metadata_;
};

bool read_metadata_checked(
    FILE* stream,
    hyperpoly_provenance::ProcessedRecord& record
) {
    record = hyperpoly_provenance::ProcessedRecord{};
    const size_t bytes = fread(&record, 1, sizeof(record), stream);
    if (bytes == 0) {
        if (ferror(stream)) throw std::runtime_error("metadata I/O error");
        if (feof(stream)) return false;
        throw std::runtime_error("metadata read made no progress");
    }
    if (bytes != sizeof(record)) {
        throw std::runtime_error("truncated BPM2 record");
    }
    if (record.magic != hyperpoly_provenance::kProcessedMagic ||
        record.version != hyperpoly_provenance::kVersion ||
        record.record_size != sizeof(record) ||
        record.layout_version != hyperpoly_provenance::kLayoutVersion) {
        throw std::runtime_error("invalid BPM2 record header");
    }
    return true;
}

// Mirror the complete synchronized-record loop in run_slicing().  The primary
// element owns record lifetime.  Companion and BPM2 EOF are accepted only at
// that same boundary, and every interval must end in an empty face vector.
int consume_triple_stream(TripleStreams& streams) {
    int records = 0;
    int previous_sorted_index = -1;
    DiscontinuityVector discontinuities;
    TimeVector times;
    FaceVector face;

    for (;;) {
        eleT element = 0;
        const CheckedReadStatus primary_status =
            read_scalar_checked(streams.primary(), element);
        if (primary_status == CheckedReadStatus::clean_eof) {
            if (read_vec_checked(streams.discontinuity(), discontinuities, 64) !=
                CheckedReadStatus::clean_eof) {
                throw std::runtime_error("trailing discontinuity record");
            }
            hyperpoly_provenance::ProcessedRecord trailing{};
            if (read_metadata_checked(streams.metadata(), trailing)) {
                throw std::runtime_error("trailing BPM2 record");
            }
            return records;
        }
        if (element < 0 || element >= 5) {
            throw std::runtime_error("element is out of range");
        }
        if (read_vec_checked(streams.discontinuity(), discontinuities, 64) !=
            CheckedReadStatus::record) {
            throw std::runtime_error("discontinuity EOF before primary");
        }

        hyperpoly_provenance::ProcessedRecord metadata{};
        if (!read_metadata_checked(streams.metadata(), metadata)) {
            throw std::runtime_error("BPM2 EOF before primary");
        }
        if (metadata.t_group != 0 || metadata.t_start != 8 ||
            metadata.source.element != element ||
            metadata.sorted_record_index < 0 ||
            (previous_sorted_index >= 0 &&
             metadata.sorted_record_index != previous_sorted_index + 1)) {
            throw std::runtime_error("BPM2 alignment mismatch");
        }
        previous_sorted_index = metadata.sorted_record_index;

        if (read_vec_checked(streams.primary(), times, 8) !=
            CheckedReadStatus::record) {
            throw std::runtime_error("time vector EOF inside primary record");
        }
        if (times.size() < 2) {
            throw std::runtime_error("too few processed times");
        }
        for (int index = 1; index < times.size(); ++index) {
            if (times[index - 1] >= times[index]) {
                throw std::runtime_error("processed times are not increasing");
            }
        }

        for (int interval = 0; interval < times.size() - 1; ++interval) {
            int faces = 0;
            for (;;) {
                if (read_vec_checked(streams.primary(), face, 12) !=
                    CheckedReadStatus::record) {
                    throw std::runtime_error("missing terminal face sentinel");
                }
                if (face.empty()) break;
                if (face.size() < 3) {
                    throw std::runtime_error("processed face has fewer than three vertices");
                }
                if (++faces > 16) {
                    throw std::runtime_error("too many processed faces in interval");
                }
            }
        }
        ++records;
    }
}

bool accepts_valid_fixture() {
    TripleStreams streams(make_fixture(Corruption::none));
    return consume_triple_stream(streams) == 1;
}

bool rejects(Corruption corruption) {
    TripleStreams streams(make_fixture(corruption));
    try {
        static_cast<void>(consume_triple_stream(streams));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    if (!accepts_valid_fixture()) return 10;
    if (!rejects(Corruption::metadata_early_eof)) return 11;
    if (!rejects(Corruption::metadata_trailing_record)) return 12;
    if (!rejects(Corruption::metadata_truncated_record)) return 13;
    if (!rejects(Corruption::discontinuity_early_eof)) return 14;
    if (!rejects(Corruption::discontinuity_trailing_record)) return 15;
    if (!rejects(Corruption::truncated_times)) return 16;
    if (!rejects(Corruption::truncated_face)) return 17;
    if (!rejects(Corruption::missing_face_sentinel)) return 18;

    std::puts("PASS_PROCESSED_PRIMARY_DISCON_BPM2_TRIPLE_STREAM");
    return 0;
}
