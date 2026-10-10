#pragma once

// FlagOS provider plugin ABI, version 1.
//
// This header is plain C. A provider plugin is a shared object named
// libflagos-provider-<name>.so that exports flagos_provider_get_entry_v1().
// The FlagOS core loads plugins when its GGML registry is first initialized.
//
// Two provider kinds share the entry point:
//   - FLAGOS_PROVIDER_ENTRY_RUNTIME: the provider implements flagos_runtime_v1
//     and knows nothing about GGML. The core owns the GGML device, buffer and
//     backend objects, graph planning, and the GGML-to-kernel binding. This is
//     the target form for new GPUs and NPUs.
//   - FLAGOS_PROVIDER_ENTRY_DEVICE: the provider returns complete GGML devices
//     through the C++ flagos_provider_v1 descriptor (flagos-provider.h). It is
//     coupled to the GGML build it was compiled against, so the entry carries
//     a GGML ABI tag that must match the core. This kind exists for vendor
//     native backends and for providers that have not migrated yet.
//
// Every struct starts with struct_size so fields can only be appended. All
// functions return a flagos_rt_status; handles are opaque.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLAGOS_PROVIDER_ABI_VERSION 1u
#define FLAGOS_PROVIDER_ENTRY_SYMBOL "flagos_provider_get_entry_v1"

#if defined(_WIN32)
#    define FLAGOS_PROVIDER_EXPORT __declspec(dllexport)
#else
#    define FLAGOS_PROVIDER_EXPORT __attribute__((visibility("default")))
#endif

typedef int32_t flagos_rt_status;
enum {
    FLAGOS_RT_OK = 0,
    FLAGOS_RT_ERROR = 1,
    FLAGOS_RT_INVALID_ARGUMENT = 2,
    FLAGOS_RT_OUT_OF_MEMORY = 3,
    FLAGOS_RT_NOT_FOUND = 4,
    FLAGOS_RT_UNSUPPORTED = 5,
    FLAGOS_RT_DEVICE_LOST = 6,
};

// ---------------------------------------------------------------------------
// Host services handed to a provider when it is loaded.

typedef enum {
    FLAGOS_LOG_DEBUG = 0,
    FLAGOS_LOG_INFO = 1,
    FLAGOS_LOG_WARN = 2,
    FLAGOS_LOG_ERROR = 3,
} flagos_log_level;

typedef struct flagos_host_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    // Directory that contained the provider plugin, or NULL for a built-in
    // provider. Providers resolve relative kernel-package paths against it.
    const char * provider_dir;
    void (*log)(flagos_log_level level, const char * provider, const char * message);
} flagos_host_v1;

// ---------------------------------------------------------------------------
// Runtime provider interface.

typedef struct flagos_rt_device * flagos_rt_device_t;
typedef struct flagos_rt_buffer * flagos_rt_buffer_t;
typedef struct flagos_rt_stream * flagos_rt_stream_t;
typedef struct flagos_rt_event * flagos_rt_event_t;
typedef struct flagos_rt_module * flagos_rt_module_t;
typedef struct flagos_rt_kernel * flagos_rt_kernel_t;
typedef struct flagos_rt_graph * flagos_rt_graph_t;

typedef enum {
    FLAGOS_RT_DEVICE_GPU = 0,
    FLAGOS_RT_DEVICE_IGPU = 1,
    FLAGOS_RT_DEVICE_NPU = 2,
    FLAGOS_RT_DEVICE_CPU_ACCEL = 3,
    FLAGOS_RT_DEVICE_OTHER = 4,
} flagos_rt_device_type;

// Memory properties of the device's default allocations.
enum {
    FLAGOS_RT_MEM_DEVICE_LOCAL = 1u << 0,
    // Device memory and host memory are the same physical pool (APU/SoC).
    FLAGOS_RT_MEM_UNIFIED = 1u << 1,
    // map_buffer() returns a host pointer that aliases the allocation.
    FLAGOS_RT_MEM_HOST_VISIBLE = 1u << 2,
    // register_host_ptr() can wrap caller memory (zero-copy weight loading).
    FLAGOS_RT_MEM_HOST_REGISTER = 1u << 3,
};

typedef struct flagos_rt_device_props {
    uint32_t struct_size;
    flagos_rt_device_type type;
    const char * name;          // short, stable, e.g. "gfx1150"
    const char * description;   // human readable
    const char * target;        // kernel-package target id, e.g. "amdgcn-gfx1150"
    const char * package_format; // e.g. "hsaco", "elf-rv64"
    uint64_t uuid_hi;
    uint64_t uuid_lo;
    uint64_t memory_total;
    uint64_t memory_flags;      // FLAGOS_RT_MEM_*
    uint32_t alignment;         // minimum buffer alignment in bytes
    uint32_t max_threads_per_block;
    uint32_t warp_size;
    uint32_t compute_units;
    uint32_t shared_memory_per_block;
    // Default kernel-package directory for this device, or NULL. The core
    // prefers GGML_FLAGOS_KERNEL_DIR when it is set.
    const char * package_dir;
} flagos_rt_device_props;

typedef enum {
    FLAGOS_RT_ARG_PTR = 0,  // device address, value.ptr
    FLAGOS_RT_ARG_I32 = 1,
    FLAGOS_RT_ARG_U32 = 2,
    FLAGOS_RT_ARG_I64 = 3,
    FLAGOS_RT_ARG_F32 = 4,
    FLAGOS_RT_ARG_F64 = 5,
} flagos_rt_arg_kind;

typedef struct flagos_rt_arg {
    flagos_rt_arg_kind kind;
    union {
        void * ptr;
        int32_t i32;
        uint32_t u32;
        int64_t i64;
        float f32;
        double f64;
    } value;
} flagos_rt_arg;

// Launch properties of one kernel, taken from the package manifest. The
// runtime keeps what it needs when the kernel is looked up.
typedef struct flagos_rt_kernel_desc {
    uint32_t struct_size;
    const char * symbol;
    uint32_t threads;           // threads per program (workgroup size)
    uint32_t block_size;        // logical elements per program (Triton BLOCK)
    uint32_t shared_memory;     // dynamic shared memory per launch, bytes
    uint32_t n_args;            // arguments the core passes to launch()
    // Compiler-managed scratch that the runtime allocates and appends after
    // the core's arguments (Triton global/profile scratch). Zero if unused.
    uint64_t global_scratch_size;
    uint32_t global_scratch_align;
    uint64_t profile_scratch_size;
    uint32_t profile_scratch_align;
} flagos_rt_kernel_desc;

typedef struct flagos_rt_launch {
    uint32_t struct_size;
    uint32_t grid[3];           // number of programs per dimension
    uint32_t n_args;
    const flagos_rt_arg * args;
} flagos_rt_launch;

typedef enum {
    FLAGOS_RT_COPY_HOST_TO_DEVICE = 0,
    FLAGOS_RT_COPY_DEVICE_TO_HOST = 1,
    FLAGOS_RT_COPY_DEVICE_TO_DEVICE = 2,
} flagos_rt_copy_kind;

typedef struct flagos_runtime_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char * name;      // provider name, e.g. "amd"
    uint32_t version;

    // Tier 1: required.
    flagos_rt_status (*device_count)(uint32_t * count);
    flagos_rt_status (*device_open)(uint32_t index, flagos_rt_device_t * device);
    void (*device_close)(flagos_rt_device_t device);
    flagos_rt_status (*device_props)(flagos_rt_device_t device, flagos_rt_device_props * props);
    flagos_rt_status (*device_memory)(flagos_rt_device_t device, uint64_t * free, uint64_t * total);

    flagos_rt_status (*buffer_alloc)(flagos_rt_device_t device, size_t size, flagos_rt_buffer_t * buffer);
    void (*buffer_free)(flagos_rt_buffer_t buffer);
    // Device address of the allocation's first byte; used for pointer arguments
    // and for GGML tensor->data. Must be aligned to props.alignment.
    void * (*buffer_base)(flagos_rt_buffer_t buffer);

    flagos_rt_status (*stream_create)(flagos_rt_device_t device, flagos_rt_stream_t * stream);
    void (*stream_destroy)(flagos_rt_stream_t stream);
    flagos_rt_status (*stream_synchronize)(flagos_rt_stream_t stream);

    // Copies are ordered on the stream. A NULL stream means a synchronous copy.
    flagos_rt_status (*memcpy)(flagos_rt_stream_t stream, void * dst, const void * src, size_t size,
                               flagos_rt_copy_kind kind);
    flagos_rt_status (*memset)(flagos_rt_stream_t stream, void * dst, int value, size_t size);

    // Load one artifact from the device's kernel package. `image` is the raw
    // artifact bytes (HSACO, ELF, ...); the core reads and verifies them.
    flagos_rt_status (*module_load)(flagos_rt_device_t device, const void * image, size_t size,
                                    flagos_rt_module_t * module);
    void (*module_unload)(flagos_rt_module_t module);
    flagos_rt_status (*kernel_get)(flagos_rt_module_t module, const flagos_rt_kernel_desc * desc,
                                   flagos_rt_kernel_t * kernel);
    flagos_rt_status (*launch)(flagos_rt_stream_t stream, flagos_rt_kernel_t kernel, const flagos_rt_launch * launch);

    // Tier 1, optional when the device reports no event support.
    flagos_rt_status (*event_create)(flagos_rt_device_t device, flagos_rt_event_t * event);
    void (*event_destroy)(flagos_rt_event_t event);
    flagos_rt_status (*event_record)(flagos_rt_event_t event, flagos_rt_stream_t stream);
    flagos_rt_status (*event_wait)(flagos_rt_stream_t stream, flagos_rt_event_t event);
    flagos_rt_status (*event_synchronize)(flagos_rt_event_t event);

    // Tier 2: each may be NULL.
    // Host pointer aliasing the allocation (FLAGOS_RT_MEM_HOST_VISIBLE).
    void * (*buffer_map)(flagos_rt_buffer_t buffer);
    // Wrap caller memory as a device buffer (FLAGOS_RT_MEM_HOST_REGISTER).
    flagos_rt_status (*register_host_ptr)(flagos_rt_device_t device, void * ptr, size_t size,
                                          flagos_rt_buffer_t * buffer);
    // Stream capture and replay (HIP/CUDA graphs, NPU graph offload).
    flagos_rt_status (*graph_capture_begin)(flagos_rt_stream_t stream);
    flagos_rt_status (*graph_capture_end)(flagos_rt_stream_t stream, flagos_rt_graph_t * graph);
    flagos_rt_status (*graph_launch)(flagos_rt_stream_t stream, flagos_rt_graph_t graph);
    void (*graph_destroy)(flagos_rt_graph_t graph);
    // End a capture by updating an existing recorded graph in place with the
    // newly captured work (same structure, new addresses or scalars), as
    // cudaGraphExecUpdate / hipGraphExecUpdate do. On success *graph is the
    // updated `existing`; if the update is impossible the provider returns a
    // fresh graph in *graph and leaves `existing` untouched. May be NULL.
    flagos_rt_status (*graph_capture_end_update)(flagos_rt_stream_t stream, flagos_rt_graph_t existing,
                                                 flagos_rt_graph_t * graph);
    // Whether the provider accepts a Common fusion pattern (flagos_pattern_id)
    // on this device. The core additionally requires the fused kernel in the
    // device's package.
    int (*fusion_supported)(flagos_rt_device_t device, uint32_t pattern_id);
} flagos_runtime_v1;

// ---------------------------------------------------------------------------
// Plugin entry.

typedef enum {
    FLAGOS_PROVIDER_ENTRY_RUNTIME = 0,
    FLAGOS_PROVIDER_ENTRY_DEVICE = 1,
} flagos_provider_entry_kind;

struct flagos_provider_v1; // C++ descriptor, see flagos-provider.h

typedef struct flagos_provider_entry_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    flagos_provider_entry_kind kind;
    const char * name;
    // DEVICE kind only: flagos_ggml_abi_tag() of the GGML build the provider
    // was compiled against. The core refuses a mismatching provider.
    const char * ggml_abi_tag;
    const flagos_runtime_v1 * runtime;              // RUNTIME kind
    const struct flagos_provider_v1 * device;       // DEVICE kind
} flagos_provider_entry_v1;

typedef const flagos_provider_entry_v1 * (*flagos_provider_get_entry_v1_fn)(const flagos_host_v1 * host);

#ifdef __cplusplus
}
#endif
