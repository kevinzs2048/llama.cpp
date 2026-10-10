#include "flagos-rt-device.h"

#include "flagos-ops.h"
#include "flagos-package.h"

#include "../ggml-backend-impl.h"
#include "../ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct rt_device {
    const flagos_runtime_v1 * rt = nullptr;
    std::string provider;
    size_t local_index = 0;
    flagos_rt_device_t handle = nullptr;
    flagos_rt_device_props props {};
    std::string name;
    std::string description;
    std::string target;
    std::string package_format;
    uint64_t uuid_hi = 0;
    uint64_t uuid_lo = 0;
    ggml_backend_device dev {};
    ggml_backend_buffer_type buft {};
    flagos_kernel_package package;

    std::mutex kernel_mutex;
    std::unordered_map<std::string, flagos_rt_module_t> modules;  // by artifact file
    std::unordered_map<std::string, flagos_rt_kernel_t> kernels;  // by kernel name
};

std::vector<std::unique_ptr<rt_device>> g_devices;

rt_device * device_of(ggml_backend_dev_t dev) {
    return static_cast<rt_device *>(dev->context);
}

bool rt_check(flagos_rt_status status, const rt_device * device, const char * what) {
    if (status == FLAGOS_RT_OK) {
        return true;
    }
    GGML_LOG_ERROR("FlagOS %s: %s failed with status %d\n", device->name.c_str(), what, status);
    return false;
}

uint64_t fnv1a(const std::string & text, uint64_t seed = 0xcbf29ce484222325ull) {
    uint64_t hash = seed;
    for (unsigned char c : text) {
        hash = (hash ^ c) * 0x100000001b3ull;
    }
    return hash;
}

// Load (once) the module that holds a kernel and resolve the kernel handle.
flagos_rt_kernel_t resolve_kernel(rt_device * device, const flagos_kernel_call & call) {
    const flagos_kernel_info & info = *call.kernel;
    std::lock_guard<std::mutex> lock(device->kernel_mutex);
    const auto found = device->kernels.find(info.name);
    if (found != device->kernels.end()) {
        return found->second;
    }
    flagos_rt_module_t module = nullptr;
    const auto loaded = device->modules.find(info.file);
    if (loaded != device->modules.end()) {
        module = loaded->second;
    } else {
        std::vector<uint8_t> image;
        std::string error;
        if (!device->package.read_artifact(info, image, error)) {
            GGML_LOG_ERROR("FlagOS %s: %s\n", device->name.c_str(), error.c_str());
            return nullptr;
        }
        if (!rt_check(device->rt->module_load(device->handle, image.data(), image.size(), &module), device,
                      "module_load")) {
            return nullptr;
        }
        device->modules.emplace(info.file, module);
    }
    flagos_rt_kernel_desc desc {};
    desc.struct_size = sizeof(desc);
    desc.symbol = info.symbol.c_str();
    desc.threads = static_cast<uint32_t>(info.num_warps * info.warp_size);
    desc.block_size = static_cast<uint32_t>(info.block_size);
    desc.shared_memory = static_cast<uint32_t>(info.shared_memory);
    desc.n_args = call.n_args;
    desc.global_scratch_size = info.global_scratch_size;
    desc.global_scratch_align = static_cast<uint32_t>(info.global_scratch_align);
    desc.profile_scratch_size = info.profile_scratch_size;
    desc.profile_scratch_align = static_cast<uint32_t>(info.profile_scratch_align);
    flagos_rt_kernel_t kernel = nullptr;
    if (!rt_check(device->rt->kernel_get(module, &desc, &kernel), device, "kernel_get")) {
        return nullptr;
    }
    device->kernels.emplace(info.name, kernel);
    return kernel;
}

// ---------------------------------------------------------------------------
// Buffer

struct rt_buffer {
    rt_device * device;
    flagos_rt_buffer_t buffer;
    void * base;
};

rt_buffer * buffer_of(ggml_backend_buffer_t buffer) {
    return static_cast<rt_buffer *>(buffer->context);
}

void buffer_free(ggml_backend_buffer_t buffer) {
    rt_buffer * ctx = buffer_of(buffer);
    ctx->device->rt->buffer_free(ctx->buffer);
    delete ctx;
}

void * buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer_of(buffer)->base;
}

void buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    rt_buffer * ctx = buffer_of(buffer);
    rt_check(ctx->device->rt->memset(nullptr, static_cast<char *>(tensor->data) + offset, value, size), ctx->device,
             "memset");
}

void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    rt_buffer * ctx = buffer_of(buffer);
    rt_check(ctx->device->rt->memcpy(nullptr, static_cast<char *>(tensor->data) + offset, data, size,
                                     FLAGOS_RT_COPY_HOST_TO_DEVICE), ctx->device, "memcpy h2d");
}

void buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    rt_buffer * ctx = buffer_of(buffer);
    rt_check(ctx->device->rt->memcpy(nullptr, data, static_cast<const char *>(tensor->data) + offset, size,
                                     FLAGOS_RT_COPY_DEVICE_TO_HOST), ctx->device, "memcpy d2h");
}

bool buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    rt_buffer * ctx = buffer_of(buffer);
    if (src->buffer == nullptr) {
        return false;
    }
    flagos_rt_copy_kind kind;
    if (src->buffer->buft == buffer->buft) {
        kind = FLAGOS_RT_COPY_DEVICE_TO_DEVICE;
    } else if (ggml_backend_buffer_is_host(src->buffer)) {
        kind = FLAGOS_RT_COPY_HOST_TO_DEVICE;
    } else {
        return false;
    }
    return rt_check(ctx->device->rt->memcpy(nullptr, dst->data, src->data, ggml_nbytes(src), kind), ctx->device,
                    "memcpy tensor");
}

void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    rt_buffer * ctx = buffer_of(buffer);
    rt_check(ctx->device->rt->memset(nullptr, ctx->base, value, buffer->size), ctx->device, "memset buffer");
}

const ggml_backend_buffer_i buffer_interface = {
    /* .free_buffer   = */ buffer_free,
    /* .get_base      = */ buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ buffer_memset_tensor,
    /* .set_tensor    = */ buffer_set_tensor,
    /* .get_tensor    = */ buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ buffer_cpy_tensor,
    /* .clear         = */ buffer_clear,
    /* .reset         = */ nullptr,
};

const char * buft_get_name(ggml_backend_buffer_type_t buft) {
    return static_cast<rt_device *>(buft->context)->name.c_str();
}

ggml_backend_buffer_t buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    rt_device * device = static_cast<rt_device *>(buft->context);
    flagos_rt_buffer_t handle = nullptr;
    if (!rt_check(device->rt->buffer_alloc(device->handle, size == 0 ? 1 : size, &handle), device, "buffer_alloc")) {
        return nullptr;
    }
    auto * ctx = new rt_buffer { device, handle, device->rt->buffer_base(handle) };
    return ggml_backend_buffer_init(buft, buffer_interface, ctx, size);
}

size_t buft_get_alignment(ggml_backend_buffer_type_t buft) {
    const uint32_t alignment = static_cast<rt_device *>(buft->context)->props.alignment;
    return alignment >= 32 ? alignment : 128;
}

const ggml_backend_buffer_type_i buft_interface = {
    /* .get_name       = */ buft_get_name,
    /* .alloc_buffer   = */ buft_alloc_buffer,
    /* .get_alignment  = */ buft_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

// ---------------------------------------------------------------------------
// Backend

struct rt_backend {
    rt_device * device;
    flagos_rt_stream_t stream;
};

rt_backend * backend_of(ggml_backend_t backend) {
    return static_cast<rt_backend *>(backend->context);
}

const char * backend_get_name(ggml_backend_t backend) {
    return backend_of(backend)->device->name.c_str();
}

void backend_free(ggml_backend_t backend) {
    rt_backend * ctx = backend_of(backend);
    ctx->device->rt->stream_synchronize(ctx->stream);
    ctx->device->rt->stream_destroy(ctx->stream);
    delete ctx;
    delete backend;
}

void backend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    rt_backend * ctx = backend_of(backend);
    rt_check(ctx->device->rt->memcpy(ctx->stream, static_cast<char *>(tensor->data) + offset, data, size,
                                     FLAGOS_RT_COPY_HOST_TO_DEVICE), ctx->device, "async memcpy h2d");
}

void backend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    rt_backend * ctx = backend_of(backend);
    rt_check(ctx->device->rt->memcpy(ctx->stream, data, static_cast<const char *>(tensor->data) + offset, size,
                                     FLAGOS_RT_COPY_DEVICE_TO_HOST), ctx->device, "async memcpy d2h");
}

void backend_synchronize(ggml_backend_t backend) {
    rt_backend * ctx = backend_of(backend);
    rt_check(ctx->device->rt->stream_synchronize(ctx->stream), ctx->device, "stream_synchronize");
}

enum ggml_status backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    rt_backend * ctx = backend_of(backend);
    rt_device * device = ctx->device;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || flagos_op_is_noop(node)) {
            continue;
        }
        flagos_kernel_call call;
        if (!flagos_op_plan(device->package, node, &call)) {
            // supports_op promised this node; refusing here means the graph
            // changed after placement.
            GGML_LOG_ERROR("FlagOS %s: no kernel for %s (%s)\n", device->name.c_str(), ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        flagos_rt_kernel_t kernel = resolve_kernel(device, call);
        if (kernel == nullptr) {
            return GGML_STATUS_FAILED;
        }
        flagos_rt_launch launch {};
        launch.struct_size = sizeof(launch);
        std::memcpy(launch.grid, call.grid, sizeof(launch.grid));
        launch.n_args = call.n_args;
        launch.args = call.args;
        if (!rt_check(device->rt->launch(ctx->stream, kernel, &launch), device, call.kernel->name.c_str())) {
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

void backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    rt_backend * ctx = backend_of(backend);
    rt_check(ctx->device->rt->event_record(static_cast<flagos_rt_event_t>(event->context), ctx->stream), ctx->device,
             "event_record");
}

void backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    rt_backend * ctx = backend_of(backend);
    rt_check(ctx->device->rt->event_wait(ctx->stream, static_cast<flagos_rt_event_t>(event->context)), ctx->device,
             "event_wait");
}

ggml_backend_i backend_interface(const rt_device * device) {
    const bool events = device->rt->event_create != nullptr;
    return {
        /* .get_name            = */ backend_get_name,
        /* .free                = */ backend_free,
        /* .set_tensor_async    = */ backend_set_tensor_async,
        /* .get_tensor_async    = */ backend_get_tensor_async,
        /* .set_tensor_2d_async = */ nullptr,
        /* .get_tensor_2d_async = */ nullptr,
        /* .cpy_tensor_async    = */ nullptr,
        /* .synchronize         = */ backend_synchronize,
        /* .graph_plan_create   = */ nullptr,
        /* .graph_plan_free     = */ nullptr,
        /* .graph_plan_update   = */ nullptr,
        /* .graph_plan_compute  = */ nullptr,
        /* .graph_compute       = */ backend_graph_compute,
        /* .event_record        = */ events ? backend_event_record : nullptr,
        /* .event_wait          = */ events ? backend_event_wait : nullptr,
        /* .graph_optimize      = */ nullptr,
    };
}

ggml_guid_t backend_guid() {
    static ggml_guid guid = { 0x46, 0x4c, 0x41, 0x47, 0x4f, 0x53, 0x2d, 0x52, 0x54, 0x2d, 0x76, 0x31, 0x00, 0x00, 0x00, 0x01 };
    return &guid;
}

// ---------------------------------------------------------------------------
// Device

const char * dev_get_name(ggml_backend_dev_t dev) {
    return device_of(dev)->name.c_str();
}

const char * dev_get_description(ggml_backend_dev_t dev) {
    return device_of(dev)->description.c_str();
}

void dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    rt_device * device = device_of(dev);
    uint64_t f = 0;
    uint64_t t = 0;
    if (device->rt->device_memory(device->handle, &f, &t) != FLAGOS_RT_OK) {
        f = 0;
        t = device->props.memory_total;
    }
    *free = static_cast<size_t>(f);
    *total = static_cast<size_t>(t);
}

enum ggml_backend_dev_type dev_get_type(ggml_backend_dev_t dev) {
    switch (device_of(dev)->props.type) {
        case FLAGOS_RT_DEVICE_GPU:  return GGML_BACKEND_DEVICE_TYPE_GPU;
        case FLAGOS_RT_DEVICE_IGPU: return GGML_BACKEND_DEVICE_TYPE_IGPU;
        default:                    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    }
}

void dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    rt_device * device = device_of(dev);
    props->name = device->name.c_str();
    props->description = device->description.c_str();
    dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->type = dev_get_type(dev);
    props->device_id = nullptr;
    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ device->rt->event_create != nullptr,
        /* .mmap_support         = */ false,
    };
}

ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char *) {
    rt_device * device = device_of(dev);
    flagos_rt_stream_t stream = nullptr;
    if (!rt_check(device->rt->stream_create(device->handle, &stream), device, "stream_create")) {
        return nullptr;
    }
    return new ggml_backend {
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_interface(device),
        /* .device  = */ dev,
        /* .context = */ new rt_backend { device, stream },
    };
}

ggml_backend_buffer_type_t dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &device_of(dev)->buft;
}

bool dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return flagos_op_is_noop(op) || flagos_op_plan(device_of(dev)->package, op, nullptr);
}

bool dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == &device_of(dev)->buft;
}

ggml_backend_event_t dev_event_new(ggml_backend_dev_t dev) {
    rt_device * device = device_of(dev);
    flagos_rt_event_t event = nullptr;
    if (device->rt->event_create == nullptr ||
        !rt_check(device->rt->event_create(device->handle, &event), device, "event_create")) {
        return nullptr;
    }
    return new ggml_backend_event { dev, event };
}

void dev_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    device_of(dev)->rt->event_destroy(static_cast<flagos_rt_event_t>(event->context));
    delete event;
}

void dev_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    rt_device * device = device_of(dev);
    rt_check(device->rt->event_synchronize(static_cast<flagos_rt_event_t>(event->context)), device, "event_synchronize");
}

ggml_backend_device_i device_interface(const rt_device * device) {
    const bool events = device->rt->event_create != nullptr;
    return {
        /* .get_name             = */ dev_get_name,
        /* .get_description      = */ dev_get_description,
        /* .get_memory           = */ dev_get_memory,
        /* .get_type             = */ dev_get_type,
        /* .get_props            = */ dev_get_props,
        /* .init_backend         = */ dev_init_backend,
        /* .get_buffer_type      = */ dev_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ dev_supports_op,
        /* .supports_buft        = */ dev_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ events ? dev_event_new : nullptr,
        /* .event_free           = */ events ? dev_event_free : nullptr,
        /* .event_synchronize    = */ events ? dev_event_synchronize : nullptr,
    };
}

// Kernel package lookup: GGML_FLAGOS_KERNEL_DIR (as the package itself or as
// a root holding one directory per target), then the runtime's default.
std::string find_package_dir(const rt_device & device, const char * plugin_dir) {
    std::error_code ec;
    if (const char * root = std::getenv("GGML_FLAGOS_KERNEL_DIR")) {
        const fs::path base(root);
        if (fs::is_regular_file(base / device.target / "manifest.json", ec)) {
            return (base / device.target).string();
        }
        if (fs::is_regular_file(base / "manifest.json", ec)) {
            return base.string();
        }
    }
    if (device.props.package_dir != nullptr && device.props.package_dir[0] != '\0') {
        fs::path dir(device.props.package_dir);
        if (dir.is_relative() && plugin_dir != nullptr) {
            dir = fs::path(plugin_dir) / dir;
        }
        return dir.string();
    }
    return {};
}

// ---------------------------------------------------------------------------
// Adapter descriptor exposing every runtime device to the registry.

bool adapter_probe(ggml_backend_reg_t) {
    return !g_devices.empty();
}

size_t adapter_device_count() {
    return g_devices.size();
}

ggml_backend_dev_t adapter_device_get(size_t index) {
    return index < g_devices.size() ? &g_devices[index]->dev : nullptr;
}

bool adapter_device_identity(size_t index, flagos_device_identity * identity) {
    if (index >= g_devices.size() || identity == nullptr) {
        return false;
    }
    const rt_device & device = *g_devices[index];
    const bool unified = (device.props.memory_flags & FLAGOS_RT_MEM_UNIFIED) != 0;
    *identity = {
        FLAGOS_RUNTIME_ADAPTER_ID,
        device.uuid_hi,
        device.uuid_lo,
        unified ? fnv1a(device.provider + ":unified") : (device.uuid_hi ^ device.uuid_lo),
        static_cast<uint32_t>(index),
    };
    return true;
}

bool adapter_device_caps(size_t index, flagos_device_caps * caps) {
    if (index >= g_devices.size() || caps == nullptr) {
        return false;
    }
    const rt_device & device = *g_devices[index];
    flagos_provider_kind kind = flagos_provider_kind::other;
    switch (device.props.type) {
        case FLAGOS_RT_DEVICE_GPU:
        case FLAGOS_RT_DEVICE_IGPU:      kind = flagos_provider_kind::gpu; break;
        case FLAGOS_RT_DEVICE_NPU:       kind = flagos_provider_kind::ai_accelerator; break;
        case FLAGOS_RT_DEVICE_CPU_ACCEL: kind = flagos_provider_kind::cpu_accelerator; break;
        default:                         kind = flagos_provider_kind::other; break;
    }
    uint64_t memory = FLAGOS_MEMORY_DEVICE_LOCAL;
    if (device.props.memory_flags & FLAGOS_RT_MEM_UNIFIED) {
        memory |= FLAGOS_MEMORY_UNIFIED_COHERENT;
    }
    if (device.props.memory_flags & FLAGOS_RT_MEM_HOST_VISIBLE) {
        memory |= FLAGOS_MEMORY_HOST_VISIBLE;
    }
    uint64_t execution = FLAGOS_EXECUTION_ASYNC_QUEUE | FLAGOS_EXECUTION_AOT_MODULE;
    if (device.rt->event_create != nullptr) {
        execution |= FLAGOS_EXECUTION_EVENTS;
    }
    if (device.rt->graph_capture_begin != nullptr) {
        execution |= FLAGOS_EXECUTION_NATIVE_GRAPH;
    }
    *caps = { kind, memory, execution, device.package_format.c_str() };
    return true;
}

bool adapter_is_backend(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == backend_get_name;
}

int adapter_score() {
    return g_devices.empty() ? 0 : 1;
}

bool adapter_device_profile(size_t index, flagos_device_profile * profile) {
    if (index >= g_devices.size() || profile == nullptr) {
        return false;
    }
    const rt_device & device = *g_devices[index];
    *profile = {};
    profile->struct_size = sizeof(*profile);
    switch (device.props.type) {
        case FLAGOS_RT_DEVICE_GPU:
        case FLAGOS_RT_DEVICE_IGPU:      profile->engine = flagos_engine_kind::gpu; break;
        case FLAGOS_RT_DEVICE_NPU:       profile->engine = flagos_engine_kind::npu; break;
        case FLAGOS_RT_DEVICE_CPU_ACCEL: profile->engine = flagos_engine_kind::cpu; break;
        default:                         profile->engine = flagos_engine_kind::other; break;
    }
    profile->lane_count = device.props.warp_size;
    profile->concurrency = device.props.compute_units;
    profile->vendor = device.provider.c_str();
    profile->architecture = device.props.name;
    profile->target = device.target.c_str();
    profile->runtime = device.provider.c_str();
    profile->aot_format = device.package_format.c_str();
    return true;
}

} // namespace

const flagos_provider_v1 * flagos_runtime_adapter(const std::vector<flagos_runtime_source> & runtimes) {
    static const flagos_provider_v1 adapter = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { FLAGOS_RUNTIME_ADAPTER_ID, "flagos-runtime", 1 },
        adapter_probe,
        adapter_device_count,
        adapter_device_get,
        adapter_device_identity,
        adapter_device_caps,
        adapter_is_backend,
        nullptr,
        nullptr,
        adapter_score,
        adapter_device_profile,
    };
    static std::once_flag once;
    std::call_once(once, [&]() {
        for (const flagos_runtime_source & source : runtimes) {
            const flagos_runtime_v1 * rt = source.runtime;
            uint32_t count = 0;
            if (rt->device_count(&count) != FLAGOS_RT_OK) {
                GGML_LOG_WARN("FlagOS: runtime '%s' failed to count devices\n", rt->name);
                continue;
            }
            for (uint32_t index = 0; index < count; ++index) {
                auto device = std::make_unique<rt_device>();
                device->rt = rt;
                device->provider = rt->name;
                device->local_index = g_devices.size();
                device->name = std::string("FlagOS-") + rt->name + std::to_string(index);
                if (rt->device_open(index, &device->handle) != FLAGOS_RT_OK) {
                    GGML_LOG_WARN("FlagOS: runtime '%s' failed to open device %u\n", rt->name, index);
                    continue;
                }
                device->props.struct_size = sizeof(device->props);
                if (rt->device_props(device->handle, &device->props) != FLAGOS_RT_OK ||
                    device->props.struct_size < offsetof(flagos_rt_device_props, package_dir)) {
                    GGML_LOG_WARN("FlagOS: runtime '%s' device %u has no valid properties\n", rt->name, index);
                    rt->device_close(device->handle);
                    continue;
                }
                const flagos_rt_device_props & p = device->props;
                device->target = p.target ? p.target : "";
                device->package_format = p.package_format ? p.package_format : "";
                device->description = std::string(p.description ? p.description : rt->name) + " (" + device->target + ")";
                device->uuid_hi = p.uuid_hi;
                device->uuid_lo = p.uuid_lo;
                if (device->uuid_hi == 0 && device->uuid_lo == 0) {
                    device->uuid_hi = fnv1a(device->provider);
                    device->uuid_lo = fnv1a(device->target, index + 1);
                }
                const std::string package_dir = find_package_dir(*device, source.plugin_dir);
                std::string error;
                if (package_dir.empty()) {
                    GGML_LOG_WARN("FlagOS %s: no kernel package; only view operations are supported\n",
                                  device->name.c_str());
                } else if (!device->package.open(package_dir, error)) {
                    GGML_LOG_WARN("FlagOS %s: kernel package %s rejected: %s\n", device->name.c_str(),
                                  package_dir.c_str(), error.c_str());
                } else {
                    GGML_LOG_DEBUG("FlagOS %s: kernel package %s (%zu kernels)\n", device->name.c_str(),
                                   package_dir.c_str(), device->package.size());
                }
                device->buft = { buft_interface, &device->dev, device.get() };
                device->dev = { device_interface(device.get()), nullptr, device.get() };
                g_devices.push_back(std::move(device));
            }
        }
    });
    return g_devices.empty() ? nullptr : &adapter;
}
