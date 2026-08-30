#include "utils.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using Record = vec<int, II>;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void write_scalar(FILE *stream, std::int8_t value) {
    require(fwrite(&value, sizeof(value), 1, stream) == 1,
            "failed to write scalar fixture");
}

void write_record(FILE *stream, Record& record) {
    write_vec_checked(stream, record);
}

int consume_synchronized(FILE *primary, FILE *companion) {
    int records = 0;
    std::int8_t element = 0;
    Record payload;
    for (;;) {
        const CheckedReadStatus primary_status =
            read_scalar_checked(primary, element);
        if (primary_status == CheckedReadStatus::clean_eof) {
            if (read_vec_checked(companion, payload, 8) !=
                CheckedReadStatus::clean_eof) {
                throw std::runtime_error("companion has trailing record");
            }
            return records;
        }
        if (read_vec_checked(companion, payload, 8) !=
            CheckedReadStatus::record) {
            throw std::runtime_error("primary has unmatched record");
        }
        ++records;
    }
}

bool rejects_before_allocation(FILE *stream, int maximum) {
    Record record;
    const size_t capacity = record.a.capacity();
    try {
        static_cast<void>(read_vec_checked(stream, record, maximum));
    } catch (const std::runtime_error&) {
        return record.empty() && record.a.capacity() == capacity;
    }
    return false;
}

void test_matched_and_eof() {
    FILE *primary = tmpfile();
    FILE *companion = tmpfile();
    require(primary != nullptr && companion != nullptr, "tmpfile failed");
    Record first;
    first.push_back(mp(7, 11));
    Record second;
    write_scalar(primary, 1);
    write_scalar(primary, 2);
    write_record(companion, first);
    write_record(companion, second);
    rewind(primary);
    rewind(companion);
    require(consume_synchronized(primary, companion) == 2,
            "matched stream count differs");
    fclose(primary);
    fclose(companion);
}

void test_asymmetric_eof() {
    {
        FILE *primary = tmpfile();
        FILE *companion = tmpfile();
        require(primary != nullptr && companion != nullptr, "tmpfile failed");
        Record record;
        write_scalar(primary, 1);
        write_scalar(primary, 2);
        write_record(companion, record);
        rewind(primary);
        rewind(companion);
        bool stopped = false;
        try { static_cast<void>(consume_synchronized(primary, companion)); }
        catch (const std::runtime_error& error) {
            stopped = std::string(error.what()) ==
                "primary has unmatched record";
        }
        require(stopped, "unmatched primary was accepted");
        fclose(primary);
        fclose(companion);
    }
    {
        FILE *primary = tmpfile();
        FILE *companion = tmpfile();
        require(primary != nullptr && companion != nullptr, "tmpfile failed");
        Record record;
        write_scalar(primary, 1);
        write_record(companion, record);
        write_record(companion, record);
        rewind(primary);
        rewind(companion);
        bool stopped = false;
        try { static_cast<void>(consume_synchronized(primary, companion)); }
        catch (const std::runtime_error& error) {
            stopped = std::string(error.what()) ==
                "companion has trailing record";
        }
        require(stopped, "trailing companion was accepted");
        fclose(primary);
        fclose(companion);
    }
}

void test_bad_lengths_and_truncation() {
    for (const int count : {
             -1, 1025, std::numeric_limits<int>::max()}) {
        FILE *stream = tmpfile();
        require(stream != nullptr, "tmpfile failed");
        require(fwrite(&count, sizeof(count), 1, stream) == 1,
                "failed to write count");
        rewind(stream);
        require(rejects_before_allocation(stream, 1024),
                "bad count allocated or was accepted");
        fclose(stream);
    }
    {
        FILE *stream = tmpfile();
        require(stream != nullptr, "tmpfile failed");
        const unsigned char bytes[sizeof(int) - 1] = {1};
        require(fwrite(bytes, 1, sizeof(bytes), stream) == sizeof(bytes),
                "failed to write partial header");
        rewind(stream);
        require(rejects_before_allocation(stream, 8),
                "partial header was accepted");
        fclose(stream);
    }
    {
        FILE *stream = tmpfile();
        require(stream != nullptr, "tmpfile failed");
        const int count = 2;
        const II one = mp(3, 5);
        require(fwrite(&count, sizeof(count), 1, stream) == 1 &&
                fwrite(&one, sizeof(one), 1, stream) == 1,
                "failed to write partial payload");
        rewind(stream);
        require(rejects_before_allocation(stream, 8),
                "partial payload was accepted");
        fclose(stream);
    }
    {
        FILE *stream = tmpfile();
        require(stream != nullptr, "tmpfile failed");
        const unsigned char byte = 7;
        require(fwrite(&byte, 1, 1, stream) == 1,
                "failed to write partial scalar");
        rewind(stream);
        int value = 0;
        bool stopped = false;
        try { static_cast<void>(read_scalar_checked(stream, value)); }
        catch (const std::runtime_error&) { stopped = true; }
        require(stopped, "partial scalar was accepted");
        fclose(stream);
    }
}

void test_non_regular_stream() {
    int descriptors[2] = {-1, -1};
    require(pipe(descriptors) == 0, "pipe failed");
    const int count = 0;
    require(write(descriptors[1], &count, sizeof(count)) == sizeof(count),
            "pipe write failed");
    close(descriptors[1]);
    FILE *stream = fdopen(descriptors[0], "rb");
    require(stream != nullptr, "fdopen failed");
    Record record;
    bool stopped = false;
    try { static_cast<void>(read_vec_checked(stream, record, 8)); }
    catch (const std::runtime_error& error) {
        stopped = std::string(error.what()) ==
            "serialized vector stream must be a regular file";
    }
    require(stopped, "non-regular vector stream was accepted");
    fclose(stream);
}

}  // namespace

int main() {
    test_matched_and_eof();
    test_asymmetric_eof();
    test_bad_lengths_and_truncation();
    test_non_regular_stream();
    std::puts("PASS_SAFETY_STREAM_READER");
    return 0;
}
