// hostsim: a reference RUNTIME-kind provider that runs on the host CPU.
//
// It exists to test the core's runtime device end to end without hardware:
// GGML objects, buffers, streams, kernel-package loading and the L2 op
// adapter all run exactly as they would for a GPU provider. Its "modules"
// are text artifacts listing kernel symbols and their BLOCK size; launching
// a kernel runs a C++ reference of the FlagGems kernel once per program id,
// so grid and argument bugs in the core show up as wrong results.
//
// The plugin does not link GGML.

#include "flagos-runtime.h"
#include "reference_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct hs_module {
    std::unordered_map<std::string, uint32_t> blocks;  // symbol -> BLOCK baked into the "binary"
    std::unordered_map<std::string, uint32_t> tiles;   // symbol -> tile_n, 0 if untiled
    std::unordered_map<std::string, uint32_t> tiles_m; // symbol -> tile_m, 0 if untiled
};

struct hs_kernel {
    std::string name;
    uint32_t block;
    uint32_t tile_n;
    uint32_t tile_m;
    uint32_t n_args;
};

struct hs_buffer {
    void * data;
};

std::string g_package_dir;

flagos_rt_status device_count(uint32_t * count) {
    *count = 1;
    return FLAGOS_RT_OK;
}

flagos_rt_status device_open(uint32_t index, flagos_rt_device_t * device) {
    if (index != 0) return FLAGOS_RT_NOT_FOUND;
    *device = reinterpret_cast<flagos_rt_device_t>(uintptr_t(1));
    return FLAGOS_RT_OK;
}

void device_close(flagos_rt_device_t) {}

flagos_rt_status device_props(flagos_rt_device_t, flagos_rt_device_props * out) {
    flagos_rt_device_props local {};
    flagos_rt_device_props * props = &local;
    props->struct_size = sizeof(*props);
    props->type = FLAGOS_RT_DEVICE_OTHER;
    props->name = "host";
    props->description = "FlagOS host reference runtime";
    props->target = "hostsim-v1";
    props->package_format = "hostsim";
    props->uuid_hi = 0x686f737473696dull;
    props->uuid_lo = 1;
    props->memory_total = 1ull << 30;
    props->memory_flags = FLAGOS_RT_MEM_UNIFIED | FLAGOS_RT_MEM_HOST_VISIBLE;
    props->alignment = 64;
    props->warp_size = 1;
    props->compute_units = 1;
    props->package_dir = g_package_dir.empty() ? nullptr : g_package_dir.c_str();
    const uint32_t size = std::min<uint32_t>(out->struct_size, sizeof(local));   // caller may be older
    std::memcpy(out, &local, size);
    out->struct_size = size;
    return FLAGOS_RT_OK;
}

flagos_rt_status device_memory(flagos_rt_device_t, uint64_t * free, uint64_t * total) {
    *free = *total = 1ull << 30;
    return FLAGOS_RT_OK;
}

flagos_rt_status buffer_alloc(flagos_rt_device_t, size_t size, flagos_rt_buffer_t * buffer) {
    const size_t rounded = (size + 63) / 64 * 64;
    void * data = std::aligned_alloc(64, rounded);
    if (data == nullptr) return FLAGOS_RT_OUT_OF_MEMORY;
    *buffer = reinterpret_cast<flagos_rt_buffer_t>(new hs_buffer { data });
    return FLAGOS_RT_OK;
}

void buffer_free(flagos_rt_buffer_t buffer) {
    auto * b = reinterpret_cast<hs_buffer *>(buffer);
    std::free(b->data);
    delete b;
}

void * buffer_base(flagos_rt_buffer_t buffer) {
    return reinterpret_cast<hs_buffer *>(buffer)->data;
}

// A stream records launches while capturing; a graph replays them. This is
// the host analogue of HIP/CUDA stream capture and keeps the same rule:
// recorded work does not run until the graph is launched.
struct hs_record {
    hs_kernel * kernel;
    uint32_t grid[3];
    std::vector<flagos_rt_arg> args;   // copied by value at record time
};

struct hs_graph {
    std::vector<hs_record> records;
};

struct hs_stream {
    hs_graph * recording = nullptr;
};

flagos_rt_status stream_create(flagos_rt_device_t, flagos_rt_stream_t * stream) {
    *stream = reinterpret_cast<flagos_rt_stream_t>(new hs_stream);
    return FLAGOS_RT_OK;
}

void stream_destroy(flagos_rt_stream_t stream) {
    auto * s = reinterpret_cast<hs_stream *>(stream);
    delete s->recording;
    delete s;
}

flagos_rt_status stream_synchronize(flagos_rt_stream_t) { return FLAGOS_RT_OK; }

flagos_rt_status rt_memcpy(flagos_rt_stream_t, void * dst, const void * src, size_t size, flagos_rt_copy_kind) {
    if (size != 0) std::memmove(dst, src, size);
    return FLAGOS_RT_OK;
}

flagos_rt_status rt_memset(flagos_rt_stream_t, void * dst, int value, size_t size) {
    std::memset(dst, value, size);
    return FLAGOS_RT_OK;
}

// Artifact format: first line "hostsim-module v1", then "<symbol> <BLOCK> <tile_n> <tile_m>".
flagos_rt_status module_load(flagos_rt_device_t, const void * image, size_t size, flagos_rt_module_t * module) {
    std::istringstream in(std::string(static_cast<const char *>(image), size));
    std::string header;
    std::getline(in, header);
    if (header != "hostsim-module v1") return FLAGOS_RT_INVALID_ARGUMENT;
    auto * m = new hs_module;
    // Each line: "<symbol> <BLOCK> <tile_n> <tile_m>".
    std::string symbol;
    uint32_t block = 0;
    uint32_t tile = 0;
    uint32_t tile_m = 0;
    while (in >> symbol >> block >> tile >> tile_m) {
        m->blocks[symbol] = block;
        m->tiles[symbol] = tile;
        m->tiles_m[symbol] = tile_m;
    }
    *module = reinterpret_cast<flagos_rt_module_t>(m);
    return FLAGOS_RT_OK;
}

void module_unload(flagos_rt_module_t module) {
    delete reinterpret_cast<hs_module *>(module);
}

flagos_rt_status kernel_get(flagos_rt_module_t module, const flagos_rt_kernel_desc * desc, flagos_rt_kernel_t * kernel) {
    auto * m = reinterpret_cast<hs_module *>(module);
    const auto in_module = m->blocks.find(desc->symbol);
    if (in_module == m->blocks.end()) return FLAGOS_RT_NOT_FOUND;
    // The manifest and the artifact must agree on BLOCK, as they would for a
    // compiled Triton kernel.
    if (desc->block_size != 0 && desc->block_size != in_module->second) return FLAGOS_RT_INVALID_ARGUMENT;
    *kernel = reinterpret_cast<flagos_rt_kernel_t>(
        new hs_kernel { desc->symbol, in_module->second, m->tiles[desc->symbol], m->tiles_m[desc->symbol], desc->n_args });
    return FLAGOS_RT_OK;
}

flagos_rt_status run_kernel(hs_kernel * k, const uint32_t grid[3], const flagos_rt_arg * args, uint32_t n_args);

// Test hook for the core's GGML_FLAGOS_GUARD: FLAGOS_HOSTSIM_OVERRUN=
// <kernel>:<arg>:<offset> makes every launch of <kernel> also write four
// bytes at pointer argument <arg> + <offset>, like a kernel that writes
// past its output.
void inject_overrun(const std::string & kernel, const flagos_rt_arg * args, uint32_t n_args) {
    struct overrun { std::string kernel; uint32_t arg = 0; size_t offset = 0; bool set = false; };
    static const overrun o = [] {
        overrun r;
        const char * value = std::getenv("FLAGOS_HOSTSIM_OVERRUN");
        if (value == nullptr) return r;
        std::string text(value);
        const size_t a = text.find(':'), b = a == std::string::npos ? a : text.find(':', a + 1);
        if (b == std::string::npos) return r;
        r.kernel = text.substr(0, a);
        r.arg = static_cast<uint32_t>(std::strtoul(text.c_str() + a + 1, nullptr, 10));
        r.offset = std::strtoull(text.c_str() + b + 1, nullptr, 10);
        r.set = true;
        return r;
    }();
    if (!o.set || kernel != o.kernel || o.arg >= n_args || args[o.arg].kind != FLAGOS_RT_ARG_PTR) return;
    const uint32_t junk = 0x0BADF00Du;
    std::memcpy(static_cast<char *>(args[o.arg].value.ptr) + o.offset, &junk, sizeof(junk));
}

flagos_rt_status launch(flagos_rt_stream_t stream, flagos_rt_kernel_t kernel, const flagos_rt_launch * launch) {
    auto * k = reinterpret_cast<hs_kernel *>(kernel);
    if (launch->n_args != k->n_args || launch->n_args > 32) return FLAGOS_RT_INVALID_ARGUMENT;
    auto * s = reinterpret_cast<hs_stream *>(stream);
    if (s != nullptr && s->recording != nullptr) {
        s->recording->records.push_back({ k, { launch->grid[0], launch->grid[1], launch->grid[2] },
                                          std::vector<flagos_rt_arg>(launch->args, launch->args + launch->n_args) });
        return FLAGOS_RT_OK;
    }
    return run_kernel(k, launch->grid, launch->args, launch->n_args);
}

flagos_rt_status run_kernel(hs_kernel * k, const uint32_t grid[3], const flagos_rt_arg * args, uint32_t n_args) {
    struct { uint32_t grid[3]; uint32_t n_args; const flagos_rt_arg * args; } l = { { grid[0], grid[1], grid[2] }, n_args, args };
    const auto * launch = &l;
    // Same argument passing as hipModuleLaunchKernel: one pointer per value.
    void * values[32];
    for (uint32_t i = 0; i < launch->n_args; ++i) {
        values[i] = const_cast<void *>(static_cast<const void *>(&launch->args[i].value));
    }
    for (uint32_t z = 0; z < launch->grid[2]; ++z) {
        for (uint32_t y = 0; y < launch->grid[1]; ++y) {
            for (uint32_t x = 0; x < launch->grid[0]; ++x) {
                if (!flagos_ref::run(k->name, { x, y, z, k->block, k->tile_n, k->tile_m, values })) return FLAGOS_RT_NOT_FOUND;
            }
        }
    }
    inject_overrun(k->name, args, n_args);
    return FLAGOS_RT_OK;
}

// hostsim accepts every pattern; the core still requires the fused kernel
// in the package, so the package decides what is fused.
int fusion_supported(flagos_rt_device_t, uint32_t) {
    return 1;
}

flagos_rt_status graph_capture_begin(flagos_rt_stream_t stream) {
    auto * s = reinterpret_cast<hs_stream *>(stream);
    if (s->recording != nullptr) return FLAGOS_RT_INVALID_ARGUMENT;
    s->recording = new hs_graph;
    return FLAGOS_RT_OK;
}

flagos_rt_status graph_capture_end(flagos_rt_stream_t stream, flagos_rt_graph_t * graph) {
    auto * s = reinterpret_cast<hs_stream *>(stream);
    if (s->recording == nullptr) return FLAGOS_RT_INVALID_ARGUMENT;
    *graph = reinterpret_cast<flagos_rt_graph_t>(s->recording);
    s->recording = nullptr;
    return FLAGOS_RT_OK;
}

// Replace the recorded launches of `existing` with the new recording; the
// host graph has no fixed topology, so the update always succeeds.
flagos_rt_status graph_capture_end_update(flagos_rt_stream_t stream, flagos_rt_graph_t existing, flagos_rt_graph_t * graph) {
    auto * s = reinterpret_cast<hs_stream *>(stream);
    if (s->recording == nullptr) return FLAGOS_RT_INVALID_ARGUMENT;
    reinterpret_cast<hs_graph *>(existing)->records = std::move(s->recording->records);
    delete s->recording;
    s->recording = nullptr;
    *graph = existing;
    return FLAGOS_RT_OK;
}

flagos_rt_status graph_launch(flagos_rt_stream_t, flagos_rt_graph_t graph) {
    for (const hs_record & r : reinterpret_cast<hs_graph *>(graph)->records) {
        const flagos_rt_status status = run_kernel(r.kernel, r.grid, r.args.data(), uint32_t(r.args.size()));
        if (status != FLAGOS_RT_OK) return status;
    }
    return FLAGOS_RT_OK;
}

void graph_destroy(flagos_rt_graph_t graph) {
    delete reinterpret_cast<hs_graph *>(graph);
}

flagos_runtime_v1 make_runtime() {
    flagos_runtime_v1 rt {};
    rt.struct_size = sizeof(rt);
    rt.abi_version = FLAGOS_PROVIDER_ABI_VERSION;
    rt.name = "hostsim";
    rt.version = 1;
    rt.device_count = device_count;
    rt.device_open = device_open;
    rt.device_close = device_close;
    rt.device_props = device_props;
    rt.device_memory = device_memory;
    rt.buffer_alloc = buffer_alloc;
    rt.buffer_free = buffer_free;
    rt.buffer_base = buffer_base;
    rt.stream_create = stream_create;
    rt.stream_destroy = stream_destroy;
    rt.stream_synchronize = stream_synchronize;
    rt.memcpy = rt_memcpy;
    rt.memset = rt_memset;
    rt.module_load = module_load;
    rt.module_unload = module_unload;
    rt.kernel_get = kernel_get;
    rt.launch = launch;
    rt.fusion_supported = fusion_supported;
    rt.graph_capture_begin = graph_capture_begin;
    rt.graph_capture_end = graph_capture_end;
    rt.graph_launch = graph_launch;
    rt.graph_destroy = graph_destroy;
    rt.graph_capture_end_update = graph_capture_end_update;
    rt.device_synchronize = [](flagos_rt_device_t) -> flagos_rt_status { return FLAGOS_RT_OK; };
    return rt;
}

} // namespace

extern "C" FLAGOS_PROVIDER_EXPORT const flagos_provider_entry_v1 * flagos_provider_get_entry_v1(const flagos_host_v1 * host) {
    static const flagos_runtime_v1 runtime = make_runtime();
    static const flagos_provider_entry_v1 entry = {
        sizeof(flagos_provider_entry_v1), FLAGOS_PROVIDER_ABI_VERSION, FLAGOS_PROVIDER_ENTRY_RUNTIME,
        "hostsim", nullptr, &runtime, nullptr,
    };
    if (host != nullptr && host->provider_dir != nullptr) {
        g_package_dir = std::string(host->provider_dir) + "/hostsim-kernels";
    }
    return &entry;
}
