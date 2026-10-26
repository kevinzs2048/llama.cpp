// Minimal RUNTIME-kind provider written in C. It proves flagos-runtime.h is
// consumable from C and gives the loader test a plugin with no GGML link.
#include "flagos-runtime.h"

#include <stdlib.h>
#include <string.h>

static flagos_rt_status mock_device_count(uint32_t * count) { *count = 1; return FLAGOS_RT_OK; }
static flagos_rt_status mock_device_open(uint32_t index, flagos_rt_device_t * device) {
    if (index != 0) return FLAGOS_RT_NOT_FOUND;
    *device = (flagos_rt_device_t) (uintptr_t) 1;
    return FLAGOS_RT_OK;
}
static void mock_device_close(flagos_rt_device_t device) { (void) device; }
static flagos_rt_status mock_device_props(flagos_rt_device_t device, flagos_rt_device_props * props) {
    (void) device;
    memset(props, 0, sizeof(*props));
    props->struct_size = sizeof(*props);
    props->type = FLAGOS_RT_DEVICE_OTHER;
    props->name = "mock0";
    props->description = "mock runtime device";
    props->target = "mock-target";
    props->package_format = "none";
    props->memory_flags = FLAGOS_RT_MEM_UNIFIED | FLAGOS_RT_MEM_HOST_VISIBLE;
    props->alignment = 64;
    return FLAGOS_RT_OK;
}
static flagos_rt_status mock_device_memory(flagos_rt_device_t device, uint64_t * free_bytes, uint64_t * total) {
    (void) device; *free_bytes = *total = 1u << 20; return FLAGOS_RT_OK;
}
static flagos_rt_status mock_alloc(flagos_rt_device_t device, size_t size, flagos_rt_buffer_t * buffer) {
    (void) device; *buffer = (flagos_rt_buffer_t) aligned_alloc(64, (size + 63) / 64 * 64); return *buffer ? FLAGOS_RT_OK : FLAGOS_RT_OUT_OF_MEMORY;
}
static void mock_free(flagos_rt_buffer_t buffer) { free(buffer); }
static void * mock_base(flagos_rt_buffer_t buffer) { return buffer; }
static flagos_rt_status mock_stream_create(flagos_rt_device_t device, flagos_rt_stream_t * stream) {
    (void) device; *stream = (flagos_rt_stream_t) (uintptr_t) 1; return FLAGOS_RT_OK;
}
static void mock_stream_destroy(flagos_rt_stream_t stream) { (void) stream; }
static flagos_rt_status mock_stream_sync(flagos_rt_stream_t stream) { (void) stream; return FLAGOS_RT_OK; }
static flagos_rt_status mock_memcpy(flagos_rt_stream_t stream, void * dst, const void * src, size_t size, flagos_rt_copy_kind kind) {
    (void) stream; (void) kind; memcpy(dst, src, size); return FLAGOS_RT_OK;
}
static flagos_rt_status mock_memset(flagos_rt_stream_t stream, void * dst, int value, size_t size) {
    (void) stream; memset(dst, value, size); return FLAGOS_RT_OK;
}
static flagos_rt_status mock_module_load(flagos_rt_device_t device, const void * image, size_t size, flagos_rt_module_t * module) {
    (void) device; (void) image; (void) size; (void) module; return FLAGOS_RT_UNSUPPORTED;
}
static void mock_module_unload(flagos_rt_module_t module) { (void) module; }
static flagos_rt_status mock_kernel_get(flagos_rt_module_t module, const flagos_rt_kernel_desc * desc, flagos_rt_kernel_t * kernel) {
    (void) module; (void) desc; (void) kernel; return FLAGOS_RT_NOT_FOUND;
}
static flagos_rt_status mock_launch(flagos_rt_stream_t stream, flagos_rt_kernel_t kernel, const flagos_rt_launch * launch) {
    (void) stream; (void) kernel; (void) launch; return FLAGOS_RT_UNSUPPORTED;
}

static const flagos_runtime_v1 mock_runtime = {
    .struct_size = sizeof(flagos_runtime_v1),
    .abi_version = FLAGOS_PROVIDER_ABI_VERSION,
    .name = "mock-runtime",
    .version = 1,
    .device_count = mock_device_count,
    .device_open = mock_device_open,
    .device_close = mock_device_close,
    .device_props = mock_device_props,
    .device_memory = mock_device_memory,
    .buffer_alloc = mock_alloc,
    .buffer_free = mock_free,
    .buffer_base = mock_base,
    .stream_create = mock_stream_create,
    .stream_destroy = mock_stream_destroy,
    .stream_synchronize = mock_stream_sync,
    .memcpy = mock_memcpy,
    .memset = mock_memset,
    .module_load = mock_module_load,
    .module_unload = mock_module_unload,
    .kernel_get = mock_kernel_get,
    .launch = mock_launch,
};

static const flagos_provider_entry_v1 mock_entry = {
    .struct_size = sizeof(flagos_provider_entry_v1),
    .abi_version = FLAGOS_PROVIDER_ABI_VERSION,
    .kind = FLAGOS_PROVIDER_ENTRY_RUNTIME,
    .name = "mock-runtime",
    .ggml_abi_tag = NULL,
    .runtime = &mock_runtime,
    .device = NULL,
};

#ifdef FLAGOS_MOCK_BAD_ABI
FLAGOS_PROVIDER_EXPORT const flagos_provider_entry_v1 * flagos_provider_get_entry_v1(const flagos_host_v1 * host) {
    static flagos_provider_entry_v1 bad;
    (void) host;
    bad = mock_entry;
    bad.abi_version = 99;
    bad.name = "mock-bad-abi";
    return &bad;
}
#else
FLAGOS_PROVIDER_EXPORT const flagos_provider_entry_v1 * flagos_provider_get_entry_v1(const flagos_host_v1 * host) {
    if (host == NULL || host->struct_size < sizeof(flagos_host_v1) || host->abi_version != FLAGOS_PROVIDER_ABI_VERSION) {
        return NULL;
    }
    if (host->log) host->log(FLAGOS_LOG_DEBUG, "mock-runtime", "entry called");
    return &mock_entry;
}
#endif
