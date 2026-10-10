#pragma once

// Target-neutral reader for a FlagOS kernel package: a directory holding
// manifest.json plus the compiled artifacts it lists. The manifest format is
// the one the AMD AOT exporter already writes; fields this reader does not
// know are ignored, so provider-specific metadata can coexist.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct flagos_kernel_info {
    std::string name;          // name the core asks for, e.g. "flagos_add_f32"
    std::string symbol;        // symbol inside the artifact
    std::string file;          // artifact file, relative to the package
    std::string sha256;        // empty when the manifest gives none
    size_t size_bytes = 0;     // 0 when unknown
    int block_size = 0;        // logical elements per program (Triton BLOCK)
    int num_warps = 0;
    int warp_size = 0;
    int shared_memory = 0;
    int argument_count = -1;   // -1 when the manifest does not declare it
    int tile_m = 0;            // optional multi-dimensional tile, 0 when unused
    int tile_n = 0;
    int tile_k = 0;
    size_t global_scratch_size = 0;
    size_t global_scratch_align = 1;
    size_t profile_scratch_size = 0;
    size_t profile_scratch_align = 1;
};

class flagos_kernel_package {
public:
    // Parse <dir>/manifest.json. Returns false with `error` set if the
    // manifest is missing or malformed.
    bool open(const std::string & dir, std::string & error);

    bool is_open() const { return !dir_.empty(); }
    const std::string & dir() const { return dir_; }
    const std::string & arch() const { return arch_; }
    size_t size() const { return kernels_.size(); }

    const flagos_kernel_info * find(std::string_view name) const;

    // Read an artifact and check its size and SHA-256 against the manifest.
    bool read_artifact(const flagos_kernel_info & kernel, std::vector<uint8_t> & bytes, std::string & error) const;

private:
    std::string dir_;
    std::string arch_;
    std::vector<flagos_kernel_info> kernels_;
    std::unordered_map<std::string, size_t> index_;
};
