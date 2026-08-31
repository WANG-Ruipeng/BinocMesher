#include "checked_output.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_checked_output_transaction NEW_DIRECTORY\n";
        return 2;
    }
    try {
        const std::filesystem::path root(argv[1]);
        require(!std::filesystem::exists(root),
                "test directory must not already exist");
        std::filesystem::create_directories(root);

        const std::filesystem::path binary_path = root / "cache.bin";
        {
            checked_output::BinaryFile output(
                binary_path, "binary transaction regression");
            const std::string payload = "complete-cache-record";
            require(std::fwrite(payload.data(), 1, payload.size(),
                                output.get()) == payload.size(),
                    "binary write failed");
            output.commit();
        }
        require(read_text(binary_path) == "complete-cache-record",
                "published binary payload differs");
        require(!std::filesystem::exists(binary_path.string() + ".tmp"),
                "committed binary temporary file remains");

        bool overwrite_rejected = false;
        try {
            checked_output::BinaryFile duplicate(
                binary_path, "overwrite regression");
            (void)duplicate;
        } catch (const std::runtime_error&) {
            overwrite_rejected = true;
        }
        require(overwrite_rejected, "existing output was overwritten");

        const std::filesystem::path abandoned_path = root / "abandoned.bin";
        {
            checked_output::BinaryFile abandoned(
                abandoned_path, "abandoned transaction regression");
            const unsigned char byte = 7;
            require(std::fwrite(&byte, 1, 1, abandoned.get()) == 1,
                    "abandoned write failed");
        }
        require(!std::filesystem::exists(abandoned_path),
                "uncommitted output was published");
        require(!std::filesystem::exists(abandoned_path.string() + ".tmp"),
                "uncommitted temporary output was not cleaned");

        const std::filesystem::path text_path = root / "registry.json";
        {
            checked_output::TextFile output(
                text_path, "text transaction regression");
            output.stream() << R"({"complete":true})" << '\n';
            output.commit();
        }
        require(read_text(text_path) ==
                    std::string(R"({"complete":true})") + '\n',
                "published text payload differs");

        const std::filesystem::path full_device("/dev/full");
        if (std::filesystem::exists(full_device)) {
            FILE* full = std::fopen(full_device.string().c_str(), "wb");
            require(full != nullptr, "failed to open /dev/full");
            const unsigned char byte = 1;
            (void)std::fwrite(&byte, 1, 1, full);
            bool close_failure_detected = false;
            try {
                checked_output::close_checked(full, "/dev/full regression");
            } catch (const std::runtime_error&) {
                close_failure_detected = true;
            }
            require(close_failure_detected,
                    "late flush/close failure was not detected");
        }

        std::cout << "PASS_CHECKED_TRANSACTIONAL_OUTPUT\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
