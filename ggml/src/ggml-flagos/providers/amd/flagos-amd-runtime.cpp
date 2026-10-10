// AMD implementation of flagos_runtime_v1 (RUNTIME-kind provider "amd-rt").
//
// This is the target form of the AMD provider: HIP device, memory, stream,
// event, module and launch only. GGML objects, planning and the
// GGML-to-kernel binding live in the FlagOS core, and the kernels come from
// the same AMD package (manifest.json + HSACO) the legacy provider uses.
//
// During migration it is a separate plugin, libflagos-provider-amd-rt.so,
// next to the DEVICE-kind AMD provider. Install only one of the two: both
// would expose the same GPU.

#include "../../flagos-runtime.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

const flagos_host_v1 * g_host = nullptr;
std::string g_default_package_dir;

void log_error(const char * what, hipError_t error) {
    if (g_host != nullptr && g_host->log != nullptr) {
        const std::string message = std::string(what) + " failed: HIP error " + std::to_string(static_cast<int>(error));
        g_host->log(FLAGOS_LOG_ERROR, "amd-rt", message.c_str());
    }
}

flagos_rt_status check(hipError_t error, const char * what) {
    if (error == hipSuccess) {
        return FLAGOS_RT_OK;
    }
    log_error(what, error);
    return error == hipErrorOutOfMemory ? FLAGOS_RT_OUT_OF_MEMORY : FLAGOS_RT_ERROR;
}

struct amd_device {
    int ordinal = 0;
    hipDeviceProp_t prop {};
    std::string arch;          // base gfx name, also the package target
    std::string description;
};

struct amd_buffer {
    amd_device * device;
    void * data;
};

struct amd_stream {
    amd_device * device;
    hipStream_t stream;
};

struct amd_event {
    amd_device * device;
    hipEvent_t event;
};

struct amd_module {
    amd_device * device;
    hipModule_t module;
};

struct amd_kernel {
    amd_device * device;
    hipFunction_t function;
    uint32_t threads;
    uint32_t shared_memory;
    uint32_t n_args;
};

std::string base_arch(const char * name) {
    std::string arch(name);
    return arch.substr(0, arch.find(':'));
}

flagos_rt_status device_count(uint32_t * count) {
    int n = 0;
    const hipError_t error = hipGetDeviceCount(&n);
    if (error == hipErrorNoDevice) {
        *count = 0;
        return FLAGOS_RT_OK;
    }
    if (check(error, "hipGetDeviceCount") != FLAGOS_RT_OK) {
        return FLAGOS_RT_ERROR;
    }
    *count = static_cast<uint32_t>(n > 0 ? n : 0);
    return FLAGOS_RT_OK;
}

flagos_rt_status device_open(uint32_t index, flagos_rt_device_t * out) {
    auto * device = new amd_device;
    device->ordinal = static_cast<int>(index);
    if (check(hipGetDeviceProperties(&device->prop, device->ordinal), "hipGetDeviceProperties") != FLAGOS_RT_OK) {
        delete device;
        return FLAGOS_RT_ERROR;
    }
    device->arch = base_arch(device->prop.gcnArchName);
    device->description = std::string(device->prop.name[0] != '\0' ? device->prop.name : "AMD GPU") + " " + device->arch;
    *out = reinterpret_cast<flagos_rt_device_t>(device);
    return FLAGOS_RT_OK;
}

void device_close(flagos_rt_device_t device) {
    delete reinterpret_cast<amd_device *>(device);
}

flagos_rt_status device_props(flagos_rt_device_t handle, flagos_rt_device_props * out) {
    const amd_device * device = reinterpret_cast<amd_device *>(handle);
    const hipDeviceProp_t & p = device->prop;
    if (out == nullptr || out->struct_size < offsetof(flagos_rt_device_props, struct_size) + sizeof(uint32_t)) {
        return FLAGOS_RT_INVALID_ARGUMENT;
    }
    // Write only what the caller's struct can hold (it may be older).
    flagos_rt_device_props local {};
    flagos_rt_device_props * props = &local;
    props->struct_size = sizeof(*props);
    props->type = p.integrated ? FLAGOS_RT_DEVICE_IGPU : FLAGOS_RT_DEVICE_GPU;
    props->name = device->arch.c_str();
    props->description = device->description.c_str();
    props->target = device->arch.c_str();
    props->package_format = "hsaco";
    std::memcpy(&props->uuid_hi, p.uuid.bytes, sizeof(uint64_t));
    std::memcpy(&props->uuid_lo, p.uuid.bytes + sizeof(uint64_t), sizeof(uint64_t));
    props->memory_total = p.totalGlobalMem;
    // An APU's device allocations live in system memory but are not host
    // dereferenceable through this runtime, so HOST_VISIBLE is not claimed.
    props->memory_flags = FLAGOS_RT_MEM_DEVICE_LOCAL | (p.integrated ? FLAGOS_RT_MEM_UNIFIED : 0u);
    props->alignment = 256;
    props->max_threads_per_block = static_cast<uint32_t>(p.maxThreadsPerBlock);
    props->warp_size = static_cast<uint32_t>(p.warpSize);
    props->compute_units = static_cast<uint32_t>(p.multiProcessorCount);
    props->shared_memory_per_block = static_cast<uint32_t>(p.sharedMemPerBlock);
    props->package_dir = g_default_package_dir.empty() ? nullptr : g_default_package_dir.c_str();
    const uint32_t size = std::min<uint32_t>(out->struct_size, sizeof(local));
    std::memcpy(out, &local, size);
    out->struct_size = size;
    return FLAGOS_RT_OK;
}

flagos_rt_status device_memory(flagos_rt_device_t handle, uint64_t * free, uint64_t * total) {
    const amd_device * device = reinterpret_cast<amd_device *>(handle);
    size_t f = 0;
    size_t t = 0;
    if (check(hipSetDevice(device->ordinal), "hipSetDevice") != FLAGOS_RT_OK ||
        check(hipMemGetInfo(&f, &t), "hipMemGetInfo") != FLAGOS_RT_OK) {
        return FLAGOS_RT_ERROR;
    }
    *free = f;
    *total = t;
    return FLAGOS_RT_OK;
}

flagos_rt_status buffer_alloc(flagos_rt_device_t handle, size_t size, flagos_rt_buffer_t * out) {
    auto * device = reinterpret_cast<amd_device *>(handle);
    void * data = nullptr;
    if (check(hipSetDevice(device->ordinal), "hipSetDevice") != FLAGOS_RT_OK) {
        return FLAGOS_RT_ERROR;
    }
    const flagos_rt_status status = check(hipMalloc(&data, size), "hipMalloc");
    if (status != FLAGOS_RT_OK) {
        return status;
    }
    *out = reinterpret_cast<flagos_rt_buffer_t>(new amd_buffer { device, data });
    return FLAGOS_RT_OK;
}

void buffer_free(flagos_rt_buffer_t handle) {
    auto * buffer = reinterpret_cast<amd_buffer *>(handle);
    hipSetDevice(buffer->device->ordinal);
    check(hipFree(buffer->data), "hipFree");
    delete buffer;
}

void * buffer_base(flagos_rt_buffer_t handle) {
    return reinterpret_cast<amd_buffer *>(handle)->data;
}

flagos_rt_status stream_create(flagos_rt_device_t handle, flagos_rt_stream_t * out) {
    auto * device = reinterpret_cast<amd_device *>(handle);
    hipStream_t stream = nullptr;
    if (check(hipSetDevice(device->ordinal), "hipSetDevice") != FLAGOS_RT_OK ||
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags") != FLAGOS_RT_OK) {
        return FLAGOS_RT_ERROR;
    }
    *out = reinterpret_cast<flagos_rt_stream_t>(new amd_stream { device, stream });
    return FLAGOS_RT_OK;
}

void stream_destroy(flagos_rt_stream_t handle) {
    auto * stream = reinterpret_cast<amd_stream *>(handle);
    check(hipStreamDestroy(stream->stream), "hipStreamDestroy");
    delete stream;
}

flagos_rt_status stream_synchronize(flagos_rt_stream_t handle) {
    return check(hipStreamSynchronize(reinterpret_cast<amd_stream *>(handle)->stream), "hipStreamSynchronize");
}

hipMemcpyKind copy_kind(flagos_rt_copy_kind kind) {
    switch (kind) {
        case FLAGOS_RT_COPY_HOST_TO_DEVICE: return hipMemcpyHostToDevice;
        case FLAGOS_RT_COPY_DEVICE_TO_HOST: return hipMemcpyDeviceToHost;
        default:                            return hipMemcpyDeviceToDevice;
    }
}

flagos_rt_status rt_memcpy(flagos_rt_stream_t handle, void * dst, const void * src, size_t size, flagos_rt_copy_kind kind) {
    if (size == 0) {
        return FLAGOS_RT_OK;
    }
    if (handle == nullptr) {
        // hipMemcpy resolves the device from the pointers (unified addressing).
        return check(hipMemcpy(dst, src, size, copy_kind(kind)), "hipMemcpy");
    }
    hipSetDevice(reinterpret_cast<amd_stream *>(handle)->device->ordinal);
    return check(hipMemcpyAsync(dst, src, size, copy_kind(kind), reinterpret_cast<amd_stream *>(handle)->stream),
                 "hipMemcpyAsync");
}

flagos_rt_status rt_memset(flagos_rt_stream_t handle, void * dst, int value, size_t size) {
    if (size == 0) {
        return FLAGOS_RT_OK;
    }
    if (handle == nullptr) {
        // hipMemset is asynchronous for device memory; complete it here so a
        // NULL-stream call returns finished, as the ABI requires.
        return check(hipMemset(dst, value, size), "hipMemset") == FLAGOS_RT_OK
            ? check(hipDeviceSynchronize(), "hipDeviceSynchronize") : FLAGOS_RT_ERROR;
    }
    hipSetDevice(reinterpret_cast<amd_stream *>(handle)->device->ordinal);
    return check(hipMemsetAsync(dst, value, size, reinterpret_cast<amd_stream *>(handle)->stream), "hipMemsetAsync");
}

flagos_rt_status module_load(flagos_rt_device_t handle, const void * image, size_t size, flagos_rt_module_t * out) {
    auto * device = reinterpret_cast<amd_device *>(handle);
    if (image == nullptr || size == 0) {
        return FLAGOS_RT_INVALID_ARGUMENT;
    }
    hipModule_t module = nullptr;
    if (check(hipSetDevice(device->ordinal), "hipSetDevice") != FLAGOS_RT_OK ||
        check(hipModuleLoadData(&module, image), "hipModuleLoadData") != FLAGOS_RT_OK) {
        return FLAGOS_RT_ERROR;
    }
    *out = reinterpret_cast<flagos_rt_module_t>(new amd_module { device, module });
    return FLAGOS_RT_OK;
}

void module_unload(flagos_rt_module_t handle) {
    auto * module = reinterpret_cast<amd_module *>(handle);
    check(hipModuleUnload(module->module), "hipModuleUnload");
    delete module;
}

flagos_rt_status kernel_get(flagos_rt_module_t handle, const flagos_rt_kernel_desc * desc, flagos_rt_kernel_t * out) {
    auto * module = reinterpret_cast<amd_module *>(handle);
    if (desc == nullptr || desc->struct_size < sizeof(flagos_rt_kernel_desc)) {
        return FLAGOS_RT_INVALID_ARGUMENT;   // every field of the v1 descriptor is required
    }
    const hipDeviceProp_t & p = module->device->prop;
    // No scratch allocator yet: the Triton scratch slots are passed as null,
    // which is only valid for kernels that request none.
    if (desc->global_scratch_size != 0 || desc->profile_scratch_size != 0) {
        return FLAGOS_RT_UNSUPPORTED;
    }
    if (desc->threads == 0 || desc->threads > static_cast<uint32_t>(p.maxThreadsPerBlock) ||
        desc->shared_memory > p.sharedMemPerBlock || desc->n_args > 30) {
        return FLAGOS_RT_INVALID_ARGUMENT;
    }
    hipFunction_t function = nullptr;
    if (check(hipModuleGetFunction(&function, module->module, desc->symbol), "hipModuleGetFunction") != FLAGOS_RT_OK) {
        return FLAGOS_RT_NOT_FOUND;
    }
    *out = reinterpret_cast<flagos_rt_kernel_t>(
        new amd_kernel { module->device, function, desc->threads, desc->shared_memory, desc->n_args });
    return FLAGOS_RT_OK;
}

flagos_rt_status launch(flagos_rt_stream_t handle, flagos_rt_kernel_t kernel_handle, const flagos_rt_launch * launch) {
    auto * stream = reinterpret_cast<amd_stream *>(handle);
    auto * kernel = reinterpret_cast<amd_kernel *>(kernel_handle);
    if (launch == nullptr || launch->struct_size < sizeof(flagos_rt_launch)) {
        return FLAGOS_RT_INVALID_ARGUMENT;
    }
    const hipDeviceProp_t & p = kernel->device->prop;
    if (launch->n_args != kernel->n_args || launch->grid[0] == 0 || launch->grid[1] == 0 || launch->grid[2] == 0 ||
        launch->grid[0] > static_cast<uint32_t>(p.maxGridSize[0]) ||
        launch->grid[1] > static_cast<uint32_t>(p.maxGridSize[1]) ||
        launch->grid[2] > static_cast<uint32_t>(p.maxGridSize[2])) {
        return FLAGOS_RT_INVALID_ARGUMENT;
    }
    // hipModuleLaunchKernel takes one pointer per argument value. Triton's
    // AMD launcher appends global and profile scratch pointers after the
    // explicit signature; both are null for zero-scratch kernels.
    std::array<void *, 32> values {};
    void * global_scratch = nullptr;
    void * profile_scratch = nullptr;
    for (uint32_t i = 0; i < launch->n_args; ++i) {
        values[i] = const_cast<void *>(static_cast<const void *>(&launch->args[i].value));
    }
    values[launch->n_args] = &global_scratch;
    values[launch->n_args + 1] = &profile_scratch;
    return check(hipModuleLaunchKernel(kernel->function, launch->grid[0], launch->grid[1], launch->grid[2],
                                       kernel->threads, 1, 1, kernel->shared_memory, stream->stream,
                                       values.data(), nullptr),
                 "hipModuleLaunchKernel");
}

flagos_rt_status event_create(flagos_rt_device_t handle, flagos_rt_event_t * out) {
    auto * device = reinterpret_cast<amd_device *>(handle);
    hipEvent_t event = nullptr;
    if (check(hipSetDevice(device->ordinal), "hipSetDevice") != FLAGOS_RT_OK ||
        check(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreateWithFlags") != FLAGOS_RT_OK) {
        return FLAGOS_RT_ERROR;
    }
    *out = reinterpret_cast<flagos_rt_event_t>(new amd_event { device, event });
    return FLAGOS_RT_OK;
}

void event_destroy(flagos_rt_event_t handle) {
    auto * event = reinterpret_cast<amd_event *>(handle);
    check(hipEventDestroy(event->event), "hipEventDestroy");
    delete event;
}

flagos_rt_status event_record(flagos_rt_event_t event, flagos_rt_stream_t stream) {
    return check(hipEventRecord(reinterpret_cast<amd_event *>(event)->event, reinterpret_cast<amd_stream *>(stream)->stream),
                 "hipEventRecord");
}

flagos_rt_status event_wait(flagos_rt_stream_t stream, flagos_rt_event_t event) {
    return check(hipStreamWaitEvent(reinterpret_cast<amd_stream *>(stream)->stream,
                                    reinterpret_cast<amd_event *>(event)->event, 0),
                 "hipStreamWaitEvent");
}

flagos_rt_status event_synchronize(flagos_rt_event_t event) {
    return check(hipEventSynchronize(reinterpret_cast<amd_event *>(event)->event), "hipEventSynchronize");
}

// Graph capture with HIP stream capture. Recorded work does not run until
// the executable graph is launched.
flagos_rt_status graph_capture_begin(flagos_rt_stream_t handle) {
    return check(hipStreamBeginCapture(reinterpret_cast<amd_stream *>(handle)->stream, hipStreamCaptureModeRelaxed),
                 "hipStreamBeginCapture");
}

flagos_rt_status graph_capture_end(flagos_rt_stream_t handle, flagos_rt_graph_t * out) {
    hipGraph_t graph = nullptr;
    if (check(hipStreamEndCapture(reinterpret_cast<amd_stream *>(handle)->stream, &graph), "hipStreamEndCapture") !=
            FLAGOS_RT_OK || graph == nullptr) {
        return FLAGOS_RT_ERROR;
    }
    hipGraphExec_t exec = nullptr;
    const flagos_rt_status status = check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "hipGraphInstantiate");
    check(hipGraphDestroy(graph), "hipGraphDestroy");
    if (status != FLAGOS_RT_OK) {
        return status;
    }
    *out = reinterpret_cast<flagos_rt_graph_t>(exec);
    return FLAGOS_RT_OK;
}

flagos_rt_status graph_capture_end_update(flagos_rt_stream_t handle, flagos_rt_graph_t existing, flagos_rt_graph_t * out) {
    hipGraph_t graph = nullptr;
    if (check(hipStreamEndCapture(reinterpret_cast<amd_stream *>(handle)->stream, &graph), "hipStreamEndCapture") !=
            FLAGOS_RT_OK || graph == nullptr) {
        return FLAGOS_RT_ERROR;
    }
    hipGraphExec_t exec = reinterpret_cast<hipGraphExec_t>(existing);
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    if (hipGraphExecUpdate(exec, graph, &error_node, &result) == hipSuccess && result == hipGraphExecUpdateSuccess) {
        check(hipGraphDestroy(graph), "hipGraphDestroy");
        *out = existing;
        return FLAGOS_RT_OK;
    }
    // Topology changed: instantiate a new executable; `existing` stays valid.
    hipGraphExec_t fresh = nullptr;
    const flagos_rt_status status = check(hipGraphInstantiate(&fresh, graph, nullptr, nullptr, 0), "hipGraphInstantiate");
    check(hipGraphDestroy(graph), "hipGraphDestroy");
    if (status != FLAGOS_RT_OK) {
        return status;
    }
    *out = reinterpret_cast<flagos_rt_graph_t>(fresh);
    return FLAGOS_RT_OK;
}

flagos_rt_status graph_launch(flagos_rt_stream_t handle, flagos_rt_graph_t graph) {
    return check(hipGraphLaunch(reinterpret_cast<hipGraphExec_t>(graph), reinterpret_cast<amd_stream *>(handle)->stream),
                 "hipGraphLaunch");
}

void graph_destroy(flagos_rt_graph_t graph) {
    check(hipGraphExecDestroy(reinterpret_cast<hipGraphExec_t>(graph)), "hipGraphExecDestroy");
}

flagos_rt_status device_synchronize(flagos_rt_device_t handle) {
    hipSetDevice(reinterpret_cast<amd_device *>(handle)->ordinal);
    return check(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

void kernel_release(flagos_rt_kernel_t kernel) {
    delete reinterpret_cast<amd_kernel *>(kernel);
}

// Patterns (flagos_pattern_id) this runtime opts into. The core also needs
// the fused kernel in the package and a fused contract in its op adapter.
int fusion_supported(flagos_rt_device_t, uint32_t pattern_id) {
    switch (pattern_id) {
        case 1:  // rms_norm_mul
        case 2:  // add_rms_norm_mul
            return 1;
        default:
            return 0;
    }
}

flagos_runtime_v1 make_runtime() {
    flagos_runtime_v1 rt {};
    rt.struct_size = sizeof(rt);
    rt.abi_version = FLAGOS_PROVIDER_ABI_VERSION;
    rt.name = "amd-rt";
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
    rt.event_create = event_create;
    rt.event_destroy = event_destroy;
    rt.event_record = event_record;
    rt.event_wait = event_wait;
    rt.event_synchronize = event_synchronize;
    rt.fusion_supported = fusion_supported;
    rt.graph_capture_begin = graph_capture_begin;
    rt.graph_capture_end = graph_capture_end;
    rt.graph_launch = graph_launch;
    rt.graph_destroy = graph_destroy;
    rt.graph_capture_end_update = graph_capture_end_update;
    rt.device_synchronize = device_synchronize;
    rt.kernel_release = kernel_release;
    return rt;
}

} // namespace

const flagos_runtime_v1 * flagos_amd_runtime(const flagos_host_v1 * host) {
    static const flagos_runtime_v1 runtime = make_runtime();
    static std::once_flag once;
    std::call_once(once, [host]() {
        g_host = host;
#ifdef FLAGOS_AMD_CONFIGURED_KERNEL_DIR
        g_default_package_dir = FLAGOS_AMD_CONFIGURED_KERNEL_DIR;
#endif
    });
    return &runtime;
}

#ifdef GGML_FLAGOS_PROVIDER_MODULE
extern "C" FLAGOS_PROVIDER_EXPORT const flagos_provider_entry_v1 * flagos_provider_get_entry_v1(const flagos_host_v1 * host) {
    static const flagos_provider_entry_v1 entry = {
        sizeof(flagos_provider_entry_v1), FLAGOS_PROVIDER_ABI_VERSION, FLAGOS_PROVIDER_ENTRY_RUNTIME,
        "amd-rt", nullptr, flagos_amd_runtime(host), nullptr,
    };
    return &entry;
}
#endif
