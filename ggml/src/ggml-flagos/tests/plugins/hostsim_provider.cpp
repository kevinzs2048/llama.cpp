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
};

struct hs_kernel {
    std::string name;
    uint32_t block;
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

flagos_rt_status device_props(flagos_rt_device_t, flagos_rt_device_props * props) {
    std::memset(props, 0, sizeof(*props));
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

flagos_rt_status stream_create(flagos_rt_device_t, flagos_rt_stream_t * stream) {
    *stream = reinterpret_cast<flagos_rt_stream_t>(uintptr_t(1));
    return FLAGOS_RT_OK;
}

void stream_destroy(flagos_rt_stream_t) {}
flagos_rt_status stream_synchronize(flagos_rt_stream_t) { return FLAGOS_RT_OK; }

flagos_rt_status rt_memcpy(flagos_rt_stream_t, void * dst, const void * src, size_t size, flagos_rt_copy_kind) {
    if (size != 0) std::memmove(dst, src, size);
    return FLAGOS_RT_OK;
}

flagos_rt_status rt_memset(flagos_rt_stream_t, void * dst, int value, size_t size) {
    std::memset(dst, value, size);
    return FLAGOS_RT_OK;
}

// Artifact format: first line "hostsim-module v1", then "<symbol> <BLOCK>".
flagos_rt_status module_load(flagos_rt_device_t, const void * image, size_t size, flagos_rt_module_t * module) {
    std::istringstream in(std::string(static_cast<const char *>(image), size));
    std::string header;
    std::getline(in, header);
    if (header != "hostsim-module v1") return FLAGOS_RT_INVALID_ARGUMENT;
    auto * m = new hs_module;
    std::string symbol;
    uint32_t block = 0;
    while (in >> symbol >> block) {
        m->blocks[symbol] = block;
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
    *kernel = reinterpret_cast<flagos_rt_kernel_t>(new hs_kernel { desc->symbol, in_module->second, desc->n_args });
    return FLAGOS_RT_OK;
}

flagos_rt_status launch(flagos_rt_stream_t, flagos_rt_kernel_t kernel, const flagos_rt_launch * launch) {
    auto * k = reinterpret_cast<hs_kernel *>(kernel);
    if (launch->n_args != k->n_args || launch->n_args > 32) return FLAGOS_RT_INVALID_ARGUMENT;
    // Same argument passing as hipModuleLaunchKernel: one pointer per value.
    void * values[32];
    for (uint32_t i = 0; i < launch->n_args; ++i) {
        values[i] = const_cast<void *>(static_cast<const void *>(&launch->args[i].value));
    }
    for (uint32_t z = 0; z < launch->grid[2]; ++z) {
        for (uint32_t y = 0; y < launch->grid[1]; ++y) {
            for (uint32_t x = 0; x < launch->grid[0]; ++x) {
                if (!flagos_ref::run(k->name, { x, y, k->block, values })) return FLAGOS_RT_NOT_FOUND;
            }
        }
    }
    return FLAGOS_RT_OK;
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
