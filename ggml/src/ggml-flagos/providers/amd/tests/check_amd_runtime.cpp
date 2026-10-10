// End-to-end host check of the AMD runtime provider under the core runtime
// device: real AMD package (manifest + HSACO, SHA-256 verified), core L2 op
// adapter, flagos_runtime_v1 calls into HIP. HIP itself is replaced by the
// functions below, which never touch the driver. hipModuleLaunchKernel
// checks the launch ABI (grid, threads, argument slots, Triton scratch
// slots) and then evaluates the kernel on the host, so the test compares
// real numbers against a CPU reference.
//
// The test links only ggml-base and ggml-cpu, so no real HIP provider is
// ever initialized; keep it that way (strace shows no /dev/kfd access).
//
// Usage: flagos-check-amd-runtime <amd-package-dir>

#include "flagos-package.h"
#include "flagos-rt-device.h"
#include "flagos-runtime.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

const flagos_runtime_v1 * flagos_amd_runtime(const flagos_host_v1 * host);

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

static flagos_kernel_package g_package;
static std::map<uintptr_t, std::string> g_functions;
static int g_modules = 0;
static int g_launches = 0;
static int g_allocations = 0;
static int g_frees = 0;

extern "C" {
hipError_t hipGetDeviceCount(int * count) { *count = 1; return hipSuccess; }
hipError_t hipSetDevice(int) { return hipSuccess; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t * p, int) {
    *p = {};
    std::strcpy(p->name, "Mock Radeon");
    std::strcpy(p->gcnArchName, "gfx1150:sramecc-:xnack-");
    p->integrated = 1;
    p->totalGlobalMem = size_t(16) << 30;
    p->warpSize = 32;
    p->maxThreadsPerBlock = 1024;
    p->sharedMemPerBlock = 65536;
    p->multiProcessorCount = 16;
    p->maxGridSize[0] = INT32_MAX;
    p->maxGridSize[1] = p->maxGridSize[2] = 65535;
    return hipSuccess;
}
hipError_t hipMemGetInfo(size_t * free, size_t * total) { *free = *total = size_t(16) << 30; return hipSuccess; }
hipError_t hipMalloc(void ** ptr, size_t size) { *ptr = std::aligned_alloc(256, (size + 255) / 256 * 256); ++g_allocations; return *ptr ? hipSuccess : hipErrorOutOfMemory; }
hipError_t hipFree(void * ptr) { std::free(ptr); ++g_frees; return hipSuccess; }
hipError_t hipMemcpy(void * dst, const void * src, size_t size, hipMemcpyKind) { std::memmove(dst, src, size); return hipSuccess; }
hipError_t hipMemcpyAsync(void * dst, const void * src, size_t size, hipMemcpyKind, hipStream_t) { std::memmove(dst, src, size); return hipSuccess; }
hipError_t hipMemset(void * dst, int value, size_t size) { std::memset(dst, value, size); return hipSuccess; }
hipError_t hipMemsetAsync(void * dst, int value, size_t size, hipStream_t) { std::memset(dst, value, size); return hipSuccess; }
hipError_t hipStreamCreateWithFlags(hipStream_t * stream, unsigned int) { *stream = reinterpret_cast<hipStream_t>(uintptr_t(0x51)); return hipSuccess; }
hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
hipError_t hipEventCreateWithFlags(hipEvent_t * event, unsigned int) { *event = reinterpret_cast<hipEvent_t>(uintptr_t(0xe1)); return hipSuccess; }
hipError_t hipEventDestroy(hipEvent_t) { return hipSuccess; }
hipError_t hipEventRecord(hipEvent_t, hipStream_t) { return hipSuccess; }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int) { return hipSuccess; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }
hipError_t hipModuleLoadData(hipModule_t * module, const void * image) {
    // A real HSACO is an ELF object.
    CHECK(std::memcmp(image, "\x7f" "ELF", 4) == 0);
    *module = reinterpret_cast<hipModule_t>(uintptr_t(++g_modules));
    return hipSuccess;
}
hipError_t hipModuleUnload(hipModule_t) { return hipSuccess; }
hipError_t hipModuleGetFunction(hipFunction_t * function, hipModule_t, const char * name) {
    const uintptr_t id = g_functions.size() + 1;
    g_functions[id] = name;
    *function = reinterpret_cast<hipFunction_t>(id);
    return hipSuccess;
}
}

// Host evaluation of the FlagGems kernels, one Triton program at a time.
static void run_program(const std::string & name, uint32_t pid, uint32_t block, void ** a) {
    auto P = [&](int i) { return *static_cast<float **>(a[i]); };
    auto I = [&](int i) { return *static_cast<int32_t *>(a[i]); };
    auto F = [&](int i) { return *static_cast<float *>(a[i]); };
    auto span = [&](int32_t n, int64_t & b, int64_t & e) { b = int64_t(pid) * block; e = std::min<int64_t>(b + block, n); };
    int64_t b, e;
    if (name == "flagos_add_f32") { span(I(3), b, e); for (int64_t i = b; i < e; ++i) P(2)[i] = P(0)[i] + P(1)[i]; }
    else if (name == "flagos_add_repeat_f32") { span(I(3), b, e); for (int64_t i = b; i < e; ++i) P(2)[i] = P(0)[i] + P(1)[i % I(4)]; }
    else if (name == "flagos_mul_f32") { span(I(3), b, e); for (int64_t i = b; i < e; ++i) P(2)[i] = P(0)[i] * P(1)[i % I(4)]; }
    else if (name == "flagos_scale_f32") { span(I(4), b, e); for (int64_t i = b; i < e; ++i) P(1)[i] = P(0)[i] * F(2) + F(3); }
    else if (name == "flagos_silu_f32") { span(I(2), b, e); for (int64_t i = b; i < e; ++i) P(1)[i] = P(0)[i] / (1.0f + std::exp(-P(0)[i])); }
    else if (name == "flagos_rms_norm_f32") {
        const int32_t n = I(2);
        const float * x = P(1) + int64_t(pid) * n;
        float * y = P(0) + int64_t(pid) * n;
        double sum = 0;
        for (int32_t i = 0; i < n; ++i) sum += double(x[i]) * x[i];
        const float s = 1.0f / std::sqrt(float(sum / n) + F(3));
        for (int32_t i = 0; i < n; ++i) y[i] = x[i] * s;
    } else {
        std::fprintf(stderr, "unexpected kernel %s\n", name.c_str());
        std::exit(1);
    }
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
                                            unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                            hipStream_t stream, void ** args, void ** extra) {
    const std::string & name = g_functions.at(reinterpret_cast<uintptr_t>(function));
    const flagos_kernel_info * info = g_package.find(name);
    CHECK(info != nullptr);
    CHECK(stream == reinterpret_cast<hipStream_t>(uintptr_t(0x51)));
    CHECK(extra == nullptr && gy == 1 && gz == 1 && by == 1 && bz == 1);
    CHECK(bx == uint32_t(info->num_warps * info->warp_size));
    CHECK(shared == uint32_t(info->shared_memory));
    // Triton scratch slots follow the declared arguments and are null.
    CHECK(info->argument_count > 0);
    CHECK(*static_cast<void **>(args[info->argument_count]) == nullptr);
    CHECK(*static_cast<void **>(args[info->argument_count + 1]) == nullptr);
    for (uint32_t pid = 0; pid < gx; ++pid) {
        run_program(name, pid, uint32_t(info->block_size), args);
    }
    ++g_launches;
    return hipSuccess;
}

static void host_log(flagos_log_level, const char * provider, const char * message) {
    std::fprintf(stderr, "[%s] %s\n", provider, message);
}

// Run `build` on the AMD runtime device and on the CPU, compare outputs.
template <typename Build>
static void compare(ggml_backend_t device_backend, ggml_backend_dev_t device, const char * label, Build build,
                    bool expect_supported = true) {
    // Use the CPU backend directly: going through the backend registry would
    // initialize every registered backend, including real GPU providers.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu != nullptr);
    std::vector<float> results[2];
    for (int pass = 0; pass < 2; ++pass) {
        ggml_backend_t backend = pass == 0 ? device_backend : cpu;
        ggml_init_params params = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        std::vector<ggml_tensor *> inputs;
        ggml_tensor * out = build(ctx, inputs);
        if (pass == 0) {
            const bool supported = ggml_backend_dev_supports_op(device, out);
            CHECK(supported == expect_supported);
            if (!supported) {
                ggml_free(ctx);
                ggml_backend_free(cpu);
                std::printf("  %-28s rejected as expected\n", label);
                return;
            }
        }
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buffer != nullptr);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (ggml_tensor * input : inputs) {
            std::vector<float> data(ggml_nelements(input));
            for (float & v : data) v = dist(rng);
            ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        }
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        results[pass].resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, results[pass].data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    double err = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        err = std::max(err, double(std::fabs(results[0][i] - results[1][i])));
    }
    std::printf("  %-28s max |diff| %.3g\n", label, err);
    CHECK(err < 1e-4);
    ggml_backend_free(cpu);
}

int main(int argc, char ** argv) {
    CHECK(argc == 2);
    std::string error;
    CHECK(g_package.open(argv[1], error));
    CHECK(setenv("GGML_FLAGOS_KERNEL_DIR", argv[1], 1) == 0);

    const flagos_host_v1 host = { sizeof(flagos_host_v1), FLAGOS_PROVIDER_ABI_VERSION, nullptr, host_log };
    const flagos_runtime_v1 * runtime = flagos_amd_runtime(&host);
    CHECK(runtime != nullptr && std::strcmp(runtime->name, "amd-rt") == 0);
    const flagos_provider_v1 * adapter = flagos_runtime_adapter({ { runtime, nullptr } });
    CHECK(adapter != nullptr && adapter->device_count() == 1);
    ggml_backend_dev_t device = adapter->device_get(0);
    CHECK(std::strcmp(ggml_backend_dev_name(device), "FlagOS-amd-rt0") == 0);
    CHECK(ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU);
    flagos_device_profile profile {};
    CHECK(adapter->get_device_profile(0, &profile) && std::strcmp(profile.target, "gfx1150") == 0);
    flagos_device_caps caps {};
    CHECK(adapter->device_caps(0, &caps) && (caps.memory & FLAGOS_MEMORY_UNIFIED_COHERENT) != 0);
    CHECK((caps.execution & FLAGOS_EXECUTION_EVENTS) != 0);

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);

    std::puts("amd-rt on core runtime device vs CPU:");
    compare(backend, device, "ADD same shape", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 7), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 7) };
        return ggml_add(ctx, in[0], in[1]);
    });
    compare(backend, device, "ADD row repeat", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 513, 9), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 513) };
        return ggml_add(ctx, in[0], in[1]);
    });
    compare(backend, device, "MUL row repeat", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 5), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048) };
        return ggml_mul(ctx, in[0], in[1]);
    });
    compare(backend, device, "SCALE with bias", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4099) };
        return ggml_scale_bias(ctx, in[0], 1.25f, -0.5f);
    });
    compare(backend, device, "SILU", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 300, 11) };
        return ggml_silu(ctx, in[0]);
    });
    compare(backend, device, "RMS_NORM hidden 2048", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 6) };
        return ggml_rms_norm(ctx, in[0], 1e-6f);
    });
    compare(backend, device, "norm*w+x chain", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 4), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024) };
        return ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, in[0], 1e-5f), in[1]), in[0]);
    });
    // Contract boundaries the package cannot satisfy fall back to the CPU.
    const flagos_kernel_info * norm = g_package.find("flagos_rms_norm_f32");
    CHECK(norm != nullptr);
    const int64_t too_wide = (norm->block_size / 4 + 1) * 4;
    compare(backend, device, "RMS_NORM wider than BLOCK", [too_wide](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, too_wide, 2) };
        return ggml_rms_norm(ctx, in[0], 1e-6f);
    }, false);
    compare(backend, device, "ADD column broadcast", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 8) };
        return ggml_add(ctx, in[0], in[1]);
    }, false);

    ggml_backend_free(backend);
    CHECK(g_launches > 0 && g_modules > 0);
    CHECK(g_allocations == g_frees);
    std::printf("amd-rt checks passed: %d launches, %d modules loaded\n", g_launches, g_modules);
    return 0;
}
