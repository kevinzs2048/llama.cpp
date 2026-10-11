#pragma once

// Target-neutral reader for a FlagOS kernel package: a directory holding
// manifest.json plus the compiled artifacts it lists. The manifest format is
// the one the AMD AOT exporter already writes; fields this reader does not
// know are ignored, so provider-specific metadata can coexist.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// One condition a kernel variant puts on a named fact of the call.
struct flagos_kernel_constraint {
    std::string fact;
    int64_t min = std::numeric_limits<int64_t>::min();
    int64_t max = std::numeric_limits<int64_t>::max();
    int64_t multiple_of = 1;
};

// A named integer the op adapter knows about a call (row length, number of
// rows, pointer alignment, ...). Which facts exist is part of each contract.
struct flagos_kernel_fact {
    const char * name;
    int64_t value;
};

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
    // Variants: a kernel with a non-empty `contract` implements the same call
    // (arguments and grid rule) as the kernel of that name, tuned for the
    // calls where every constraint in `when` holds. Higher priority wins.
    std::string contract;
    int priority = 0;
    std::vector<flagos_kernel_constraint> when;
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

    // The kernel to launch for `contract` given what the caller knows about
    // the call: the highest-priority variant whose constraints all hold
    // (a constraint on a fact not given fails), else the kernel named
    // `contract`. NULL when there is no such kernel. Variant selection is
    // off with GGML_FLAGOS_KERNEL_VARIANTS=0.
    const flagos_kernel_info * select(std::string_view contract, std::initializer_list<flagos_kernel_fact> facts) const;

    // Keep only the variants `keep` accepts (e.g. the ones a device can
    // launch); base kernels are never dropped.
    void retain_variants(const std::function<bool(const flagos_kernel_info &)> & keep);

    size_t variant_count() const;
    const std::vector<flagos_kernel_info> & kernels() const { return kernels_; }

    // Read an artifact and check its size and SHA-256 against the manifest.
    bool read_artifact(const flagos_kernel_info & kernel, std::vector<uint8_t> & bytes, std::string & error) const;

private:
    bool open_impl(const std::string & dir, std::string & error);

    std::string dir_;
    std::string arch_;
    std::vector<flagos_kernel_info> kernels_;
    std::unordered_map<std::string, size_t> index_;
    // contract name -> variant indices, highest priority first
    std::unordered_map<std::string, std::vector<size_t>> variants_;
};
