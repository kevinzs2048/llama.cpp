#include "flagos-package.h"

#include "flagos-sha256.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Artifacts are single compiled kernels; anything larger is a packaging error.
static constexpr size_t FLAGOS_MAX_ARTIFACT_BYTES = 64u << 20;
static constexpr size_t FLAGOS_MAX_MANIFEST_BYTES = 16u << 20;

static bool flagos_read_file(const fs::path & path, size_t limit, std::vector<uint8_t> & bytes, std::string & error) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) {
        error = "not a regular file: " + path.string();
        return false;
    }
    const auto size = fs::file_size(path, ec);
    if (ec || size > limit) {
        error = "unreadable or oversized file: " + path.string();
        return false;
    }
    std::ifstream stream(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    if (!stream.good() && !stream.eof()) {
        error = "read failed: " + path.string();
        return false;
    }
    return true;
}

// A manifest names files relative to the package; reject anything that could
// escape the package directory.
static bool flagos_is_safe_relative(const std::string & file) {
    if (file.empty() || file.find('\0') != std::string::npos) {
        return false;
    }
    const fs::path path(file);
    if (path.is_absolute()) {
        return false;
    }
    for (const auto & part : path) {
        if (part == "..") {
            return false;
        }
    }
    return true;
}

template <typename T>
static bool flagos_get_int(const json & item, const char * key, T & out, std::string & error) {
    if (!item.contains(key)) {
        return true;
    }
    const json & value = item.at(key);
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        error = std::string("field '") + key + "' is not an integer";
        return false;
    }
    const long long v = value.get<long long>();
    if (v < static_cast<long long>(std::numeric_limits<T>::min()) ||
        (v > 0 && static_cast<unsigned long long>(v) > static_cast<unsigned long long>(std::numeric_limits<T>::max()))) {
        error = std::string("field '") + key + "' is out of range";
        return false;
    }
    out = static_cast<T>(v);
    return true;
}

bool flagos_kernel_package::open(const std::string & dir, std::string & error) {
    *this = flagos_kernel_package();
    std::vector<uint8_t> bytes;
    if (!flagos_read_file(fs::path(dir) / "manifest.json", FLAGOS_MAX_MANIFEST_BYTES, bytes, error)) {
        return false;
    }
    json manifest = json::parse(bytes.begin(), bytes.end(), nullptr, false);
    if (manifest.is_discarded() || !manifest.is_object() || !manifest.contains("kernels") ||
        !manifest.at("kernels").is_array()) {
        error = "manifest.json is not a kernel package manifest";
        return false;
    }
    if (manifest.contains("arch") && manifest.at("arch").is_string()) {
        arch_ = manifest.at("arch").get<std::string>();
    }
    for (const json & item : manifest.at("kernels")) {
        if (!item.is_object() || !item.contains("name") || !item.at("name").is_string() ||
            !item.contains("file") || !item.at("file").is_string()) {
            error = "kernel entry without a name or file";
            return false;
        }
        flagos_kernel_info info;
        info.name = item.at("name").get<std::string>();
        info.file = item.at("file").get<std::string>();
        info.symbol = item.value("symbol", info.name);
        info.sha256 = item.value("sha256", "");
        if (!flagos_is_safe_relative(info.file) || (!info.sha256.empty() && !flagos::is_sha256(info.sha256))) {
            error = "kernel '" + info.name + "' has an invalid file or sha256";
            return false;
        }
        if (!flagos_get_int(item, "size_bytes", info.size_bytes, error) ||
            !flagos_get_int(item, "block_size", info.block_size, error) ||
            !flagos_get_int(item, "num_warps", info.num_warps, error) ||
            !flagos_get_int(item, "warp_size", info.warp_size, error) ||
            !flagos_get_int(item, "shared", info.shared_memory, error) ||
            !flagos_get_int(item, "argument_count", info.argument_count, error) ||
            !flagos_get_int(item, "tile_m", info.tile_m, error) ||
            !flagos_get_int(item, "tile_n", info.tile_n, error) ||
            !flagos_get_int(item, "tile_k", info.tile_k, error) ||
            !flagos_get_int(item, "global_scratch_size", info.global_scratch_size, error) ||
            !flagos_get_int(item, "global_scratch_align", info.global_scratch_align, error) ||
            !flagos_get_int(item, "profile_scratch_size", info.profile_scratch_size, error) ||
            !flagos_get_int(item, "profile_scratch_align", info.profile_scratch_align, error)) {
            error = "kernel '" + info.name + "': " + error;
            return false;
        }
        if (info.block_size < 0 || info.num_warps < 0 || info.warp_size < 0 || info.shared_memory < 0 ||
            info.tile_m < 0 || info.tile_n < 0 || info.tile_k < 0) {
            error = "kernel '" + info.name + "' has negative launch metadata";
            return false;
        }
        if (!index_.emplace(info.name, kernels_.size()).second) {
            error = "duplicate kernel '" + info.name + "'";
            return false;
        }
        kernels_.push_back(std::move(info));
    }
    dir_ = dir;
    return true;
}

const flagos_kernel_info * flagos_kernel_package::find(std::string_view name) const {
    const auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &kernels_[it->second];
}

bool flagos_kernel_package::read_artifact(const flagos_kernel_info & kernel, std::vector<uint8_t> & bytes,
                                          std::string & error) const {
    if (!flagos_read_file(fs::path(dir_) / kernel.file, FLAGOS_MAX_ARTIFACT_BYTES, bytes, error)) {
        return false;
    }
    if (kernel.size_bytes != 0 && bytes.size() != kernel.size_bytes) {
        error = "artifact size mismatch for " + kernel.file;
        return false;
    }
    if (!kernel.sha256.empty() && flagos::sha256_hex(bytes) != kernel.sha256) {
        error = "artifact sha256 mismatch for " + kernel.file;
        return false;
    }
    return true;
}
