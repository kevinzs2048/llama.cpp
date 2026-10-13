#include "flagos-package.h"

#include "flagos-sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
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

// Optional variant fields: "contract", "priority" and
// "when": { "<fact>": { "min": i, "max": i, "multiple_of": i }, ... }.
static bool flagos_get_variant(const json & item, flagos_kernel_info & info, std::string & error) {
    if (item.contains("contract")) {
        if (!item.at("contract").is_string() || item.at("contract").get<std::string>().empty()) {
            error = "field 'contract' is not a kernel name";
            return false;
        }
        info.contract = item.at("contract").get<std::string>();
    }
    if (!flagos_get_int(item, "priority", info.priority, error)) {
        return false;
    }
    if (!item.contains("when")) {
        return true;
    }
    const json & when = item.at("when");
    if (!when.is_object() || info.contract.empty()) {
        error = "field 'when' needs an object and a 'contract'";
        return false;
    }
    for (const auto & [fact, rule] : when.items()) {
        flagos_kernel_constraint c;
        c.fact = fact;
        if (!rule.is_object() || !flagos_get_int(rule, "min", c.min, error) || !flagos_get_int(rule, "max", c.max, error) ||
            !flagos_get_int(rule, "multiple_of", c.multiple_of, error) || c.multiple_of < 1 || c.min > c.max) {
            error = "invalid constraint on '" + fact + "'" + (error.empty() ? "" : ": " + error);
            return false;
        }
        info.when.push_back(std::move(c));
    }
    return true;
}

bool flagos_kernel_package::open(const std::string & dir, std::string & error) {
    // nlohmann::json throws on type mismatches; never let that escape into
    // GGML registry initialization.
    try {
        return open_impl(dir, error);
    } catch (const std::exception & e) {
        *this = flagos_kernel_package();
        error = std::string("malformed manifest: ") + e.what();
        return false;
    }
}

bool flagos_kernel_package::open_impl(const std::string & dir, std::string & error) {
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
        if (!flagos_get_variant(item, info, error)) {
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
    // A variant must name an existing, non-variant kernel and take the same
    // number of arguments; anything else is a packaging error.
    for (size_t i = 0; i < kernels_.size(); ++i) {
        const flagos_kernel_info & v = kernels_[i];
        if (v.contract.empty()) {
            continue;
        }
        const flagos_kernel_info * base = find(v.contract);
        if (base == nullptr || !base->contract.empty() || base == &v ||
            (base->argument_count >= 0 && v.argument_count >= 0 && base->argument_count != v.argument_count)) {
            error = "kernel '" + v.name + "' is not a valid variant of '" + v.contract + "'";
            return false;
        }
        variants_[v.contract].push_back(i);
    }
    for (auto & [contract, list] : variants_) {
        std::stable_sort(list.begin(), list.end(),
                         [this](size_t a, size_t b) { return kernels_[a].priority > kernels_[b].priority; });
    }
    dir_ = dir;
    return true;
}

// GGML_FLAGOS_KERNEL_VARIANTS: unset or "1" selects variants by priority,
// "0" uses only base kernels, and a comma-separated list of kernel names
// allows only those variants (used to validate one variant at a time).
struct flagos_variant_policy {
    bool enabled = true;
    std::vector<std::string> only;
};

static const flagos_variant_policy & flagos_variants() {
    static const flagos_variant_policy policy = [] {
        flagos_variant_policy p;
        const char * value = std::getenv("GGML_FLAGOS_KERNEL_VARIANTS");
        if (value == nullptr || value[0] == '\0' || std::string(value) == "1") {
            return p;
        }
        if (std::string(value) == "0") {
            p.enabled = false;
            return p;
        }
        const std::string list(value);
        size_t start = 0;
        while (start <= list.size()) {
            const size_t end = std::min(list.find(',', start), list.size());
            if (end > start) {
                p.only.push_back(list.substr(start, end - start));
            }
            start = end + 1;
        }
        return p;
    }();
    return policy;
}

static bool flagos_variant_allowed(const flagos_kernel_info & kernel) {
    const flagos_variant_policy & p = flagos_variants();
    return p.only.empty() || std::find(p.only.begin(), p.only.end(), kernel.name) != p.only.end();
}

static bool flagos_constraints_hold(const flagos_kernel_info & kernel, std::initializer_list<flagos_kernel_fact> facts) {
    for (const flagos_kernel_constraint & c : kernel.when) {
        const flagos_kernel_fact * fact = nullptr;
        for (const flagos_kernel_fact & f : facts) {
            if (c.fact == f.name) {
                fact = &f;
                break;
            }
        }
        if (fact == nullptr || fact->value < c.min || fact->value > c.max ||
            (c.multiple_of > 1 && fact->value % c.multiple_of != 0)) {
            return false;
        }
    }
    return true;
}

const flagos_kernel_info * flagos_kernel_package::select(std::string_view contract,
                                                         std::initializer_list<flagos_kernel_fact> facts) const {
    const flagos_kernel_info * base = find(contract);
    if (base == nullptr || !flagos_variants().enabled) {
        return base;
    }
    const auto it = variants_.find(std::string(contract));
    if (it != variants_.end()) {
        for (size_t i : it->second) {
            const flagos_kernel_info & v = kernels_[i];
            if (v.priority >= base->priority && flagos_variant_allowed(v) && flagos_constraints_hold(v, facts)) {
                return &v;
            }
        }
    }
    return base;
}

void flagos_kernel_package::retain_variants(const std::function<bool(const flagos_kernel_info &)> & keep) {
    for (auto & [contract, list] : variants_) {
        list.erase(std::remove_if(list.begin(), list.end(), [&](size_t i) { return !keep(kernels_[i]); }), list.end());
    }
}

size_t flagos_kernel_package::variant_count() const {
    size_t n = 0;
    for (const auto & [contract, list] : variants_) {
        n += list.size();
    }
    return n;
}

const flagos_kernel_info * flagos_kernel_package::find(std::string_view name) const {
    const auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &kernels_[it->second];
}

bool flagos_kernel_package::read_artifact(const flagos_kernel_info & kernel, std::vector<uint8_t> & bytes,
                                          std::string & error) const {
    // Resolve symlinks and require the artifact to stay inside the package.
    std::error_code ec;
    const fs::path root = fs::canonical(fs::path(dir_), ec);
    const fs::path file = ec ? fs::path() : fs::canonical(fs::path(dir_) / kernel.file, ec);
    if (ec) {
        error = "artifact not found: " + kernel.file;
        return false;
    }
    const auto rel = file.lexically_relative(root);
    if (rel.empty() || rel.is_absolute() || *rel.begin() == "..") {
        error = "artifact outside the package: " + kernel.file;
        return false;
    }
    if (!flagos_read_file(file, FLAGOS_MAX_ARTIFACT_BYTES, bytes, error)) {
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
