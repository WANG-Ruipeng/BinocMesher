#ifndef BINOC_CHECKED_OUTPUT_H
#define BINOC_CHECKED_OUTPUT_H

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace checked_output {

inline std::runtime_error system_error(
    const std::string& context,
    const std::string& operation,
    int error_number
) {
    return std::runtime_error(
        context + ": " + operation + " failed: " +
        std::strerror(error_number));
}

inline void close_checked(FILE*& file, const std::string& context) {
    if (file == nullptr) return;
    const bool prior_stream_error = std::ferror(file) != 0;
    const int prior_error_number = errno;
    errno = 0;
    const int flush_result = std::fflush(file);
    if (prior_stream_error || flush_result != 0) {
        int error_number = errno;
        if (error_number == 0) error_number = prior_error_number;
        if (error_number == 0) error_number = EIO;
        std::fclose(file);
        file = nullptr;
        throw system_error(context, "flush", error_number);
    }
    FILE* closing = std::exchange(file, nullptr);
    if (std::fclose(closing) != 0) {
        throw system_error(context, "close", errno);
    }
}

inline void remove_noexcept(const std::filesystem::path& path) noexcept {
    std::error_code error;
    std::filesystem::remove(path, error);
}

inline void publish_new_file(
    const std::filesystem::path& temporary,
    const std::filesystem::path& final,
    const std::string& context
) {
    if (std::filesystem::exists(final)) {
        throw std::runtime_error(
            context + ": refusing to overwrite an existing cache file");
    }
    std::error_code error;
    std::filesystem::rename(temporary, final, error);
    if (error) {
        throw std::runtime_error(
            context + ": atomic publish failed: " + error.message());
    }
}

class BinaryFile {
public:
    BinaryFile(
        const std::filesystem::path& final_path,
        std::string context
    ) :
        final_path_(final_path),
        temporary_path_(final_path.string() + ".tmp"),
        context_(std::move(context))
    {
        if (std::filesystem::exists(final_path_)) {
            throw std::runtime_error(
                context_ + ": refusing to overwrite an existing cache file");
        }
        if (std::filesystem::exists(temporary_path_)) {
            throw std::runtime_error(
                context_ + ": stale temporary output exists; clean the "
                "incomplete cache before retrying");
        }
        file_ = std::fopen(temporary_path_.string().c_str(), "wb");
        if (file_ == nullptr) {
            throw system_error(context_, "open temporary output", errno);
        }
    }

    ~BinaryFile() noexcept {
        if (file_ != nullptr) std::fclose(file_);
        if (!committed_) remove_noexcept(temporary_path_);
    }

    BinaryFile(const BinaryFile&) = delete;
    BinaryFile& operator=(const BinaryFile&) = delete;

    FILE* get() const noexcept { return file_; }

    void commit() {
        close_checked(file_, context_);
        publish_new_file(temporary_path_, final_path_, context_);
        committed_ = true;
    }

private:
    std::filesystem::path final_path_;
    std::filesystem::path temporary_path_;
    std::string context_;
    FILE* file_ = nullptr;
    bool committed_ = false;
};

class TextFile {
public:
    TextFile(
        const std::filesystem::path& final_path,
        std::string context
    ) :
        final_path_(final_path),
        temporary_path_(final_path.string() + ".tmp"),
        context_(std::move(context))
    {
        if (std::filesystem::exists(final_path_)) {
            throw std::runtime_error(
                context_ + ": refusing to overwrite an existing output file");
        }
        if (std::filesystem::exists(temporary_path_)) {
            throw std::runtime_error(
                context_ + ": stale temporary output exists; clean the "
                "incomplete cache before retrying");
        }
        stream_.exceptions(std::ios::badbit | std::ios::failbit);
        try {
            stream_.open(
                temporary_path_,
                std::ios::out | std::ios::binary | std::ios::trunc);
        } catch (const std::ios_base::failure& error) {
            throw std::runtime_error(
                context_ + ": open temporary output failed: " + error.what());
        }
    }

    ~TextFile() noexcept {
        if (stream_.is_open()) {
            try {
                stream_.exceptions(std::ios::goodbit);
                stream_.close();
            } catch (...) {
            }
        }
        if (!committed_) remove_noexcept(temporary_path_);
    }

    TextFile(const TextFile&) = delete;
    TextFile& operator=(const TextFile&) = delete;

    std::ostream& stream() noexcept { return stream_; }

    void commit() {
        try {
            stream_.flush();
            stream_.close();
        } catch (const std::ios_base::failure& error) {
            throw std::runtime_error(
                context_ + ": flush/close failed: " + error.what());
        }
        publish_new_file(temporary_path_, final_path_, context_);
        committed_ = true;
    }

private:
    std::filesystem::path final_path_;
    std::filesystem::path temporary_path_;
    std::string context_;
    std::ofstream stream_;
    bool committed_ = false;
};

}  // namespace checked_output

#endif  // BINOC_CHECKED_OUTPUT_H
