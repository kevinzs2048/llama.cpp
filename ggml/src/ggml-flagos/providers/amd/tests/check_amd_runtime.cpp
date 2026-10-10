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

#include "reference_kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <random>
#include <string>
#include <tuple>
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
static int g_fused_rms_norm_mul = 0;
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
// Modules are identified by matching the image to a package artifact, since
// an artifact's symbol may differ from the manifest kernel name.
static std::map<uintptr_t, std::string> g_module_names;
hipError_t hipModuleLoadData(hipModule_t * module, const void * image) {
    // A real HSACO is an ELF object.
    CHECK(std::memcmp(image, "\x7f" "ELF", 4) == 0);
    std::string name;
    for (const flagos_kernel_info & info : g_package.kernels()) {
        std::vector<uint8_t> bytes;
        std::string error;
        if (g_package.read_artifact(info, bytes, error) && std::memcmp(bytes.data(), image, bytes.size()) == 0) {
            name = info.name;
            break;
        }
    }
    CHECK(!name.empty());
    *module = reinterpret_cast<hipModule_t>(uintptr_t(++g_modules));
    g_module_names[uintptr_t(g_modules)] = name;
    return hipSuccess;
}
hipError_t hipModuleUnload(hipModule_t) { return hipSuccess; }
hipError_t hipModuleGetFunction(hipFunction_t * function, hipModule_t module, const char *) {
    const uintptr_t id = g_functions.size() + 1;
    g_functions[id] = g_module_names.at(reinterpret_cast<uintptr_t>(module));
    *function = reinterpret_cast<hipFunction_t>(id);
    return hipSuccess;
}
}

// HIP graph capture: launches on a capturing stream are recorded with their
// argument values copied (hipModuleLaunchKernel copies kernel parameters at
// capture), and replayed by hipGraphLaunch.
struct mock_launch {
    hipFunction_t function;
    unsigned gx, gy, gz, bx, by, bz, shared;
    std::vector<std::array<unsigned char, 8>> values;
};
struct mock_graph {
    std::vector<mock_launch> launches;
};
static mock_graph * g_capturing = nullptr;
static int g_graph_launches = 0;
static int g_graph_instantiations = 0;
static hipError_t execute_launch(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
                                 unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                 hipStream_t stream, void ** args, void ** extra);
extern "C" hipError_t hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode) {
    CHECK(g_capturing == nullptr);
    g_capturing = new mock_graph;
    return hipSuccess;
}
extern "C" hipError_t hipStreamEndCapture(hipStream_t, hipGraph_t * graph) {
    CHECK(g_capturing != nullptr);
    *graph = reinterpret_cast<hipGraph_t>(g_capturing);
    g_capturing = nullptr;
    return hipSuccess;
}
extern "C" hipError_t hipGraphInstantiate(hipGraphExec_t * exec, hipGraph_t graph, hipGraphNode_t *, char *, size_t) {
    // The executable owns a copy, as in HIP; the graph itself is destroyed.
    *exec = reinterpret_cast<hipGraphExec_t>(new mock_graph(*reinterpret_cast<mock_graph *>(graph)));
    ++g_graph_instantiations;
    return hipSuccess;
}
extern "C" hipError_t hipGraphDestroy(hipGraph_t graph) { delete reinterpret_cast<mock_graph *>(graph); return hipSuccess; }
static int g_graph_updates = 0;
extern "C" hipError_t hipGraphExecUpdate(hipGraphExec_t exec, hipGraph_t graph, hipGraphNode_t * error_node,
                                         hipGraphExecUpdateResult * result) {
    auto * e = reinterpret_cast<mock_graph *>(exec);
    auto * g = reinterpret_cast<mock_graph *>(graph);
    if (error_node) *error_node = nullptr;
    bool same = e->launches.size() == g->launches.size();
    for (size_t i = 0; same && i < e->launches.size(); ++i) same = e->launches[i].function == g->launches[i].function;
    if (!same) { *result = hipGraphExecUpdateErrorTopologyChanged; return hipErrorGraphExecUpdateFailure; }
    e->launches = g->launches;
    *result = hipGraphExecUpdateSuccess;
    ++g_graph_updates;
    return hipSuccess;
}
extern "C" hipError_t hipGraphExecDestroy(hipGraphExec_t exec) { delete reinterpret_cast<mock_graph *>(exec); return hipSuccess; }
extern "C" hipError_t hipGraphLaunch(hipGraphExec_t exec, hipStream_t stream) {
    ++g_graph_launches;
    for (mock_launch & l : reinterpret_cast<mock_graph *>(exec)->launches) {
        std::vector<void *> args;
        for (auto & v : l.values) args.push_back(v.data());
        CHECK(execute_launch(l.function, l.gx, l.gy, l.gz, l.bx, l.by, l.bz, l.shared, stream, args.data(), nullptr) == hipSuccess);
    }
    return hipSuccess;
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
                                            unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                            hipStream_t stream, void ** args, void ** extra) {
    if (g_capturing != nullptr) {
        const flagos_kernel_info * info = g_package.find(g_functions.at(reinterpret_cast<uintptr_t>(function)));
        CHECK(info != nullptr && info->argument_count >= 0);
        mock_launch l { function, gx, gy, gz, bx, by, bz, shared, {} };
        for (int i = 0; i < info->argument_count + 2; ++i) {
            std::array<unsigned char, 8> v {};
            std::memcpy(v.data(), args[i], v.size());
            l.values.push_back(v);
        }
        g_capturing->launches.push_back(l);
        return hipSuccess;
    }
    return execute_launch(function, gx, gy, gz, bx, by, bz, shared, stream, args, extra);
}

static hipError_t execute_launch(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
                                            unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                            hipStream_t stream, void ** args, void ** extra) {
    const std::string & name = g_functions.at(reinterpret_cast<uintptr_t>(function));
    const flagos_kernel_info * info = g_package.find(name);
    CHECK(info != nullptr);
    CHECK(stream == reinterpret_cast<hipStream_t>(uintptr_t(0x51)));
    CHECK(extra == nullptr && by == 1 && bz == 1);
    CHECK(bx == uint32_t(info->num_warps * info->warp_size));
    CHECK(shared == uint32_t(info->shared_memory));
    // Triton scratch slots follow the declared arguments and are null.
    CHECK(info->argument_count > 0);
    CHECK(*static_cast<void **>(args[info->argument_count]) == nullptr);
    CHECK(*static_cast<void **>(args[info->argument_count + 1]) == nullptr);
    for (uint32_t z = 0; z < gz; ++z) {
        for (uint32_t y = 0; y < gy; ++y) {
            for (uint32_t x = 0; x < gx; ++x) {
                CHECK(flagos_ref::run(name, { x, y, z, uint32_t(info->block_size), uint32_t(info->tile_n), args }));
            }
        }
    }
    ++g_launches;
    g_fused_rms_norm_mul += name == "flagos_rms_norm_mul_f32";
    return hipSuccess;
}

static void host_log(flagos_log_level, const char * provider, const char * message) {
    std::fprintf(stderr, "[%s] %s\n", provider, message);
}

// Run `build` on the AMD runtime device and on the CPU, compare outputs.
template <typename Build>
static void compare(ggml_backend_t device_backend, ggml_backend_dev_t device, const char * label, Build build,
                    bool expect_supported = true, double max_nmse = 1e-10) {
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
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (std::strcmp(ggml_get_name(t), "ids") == 0) {
                std::vector<int32_t> ids(ggml_nelements(t));
                for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t((i * 5 + 3) % 8);
                ggml_backend_tensor_set(t, ids.data(), 0, ggml_nbytes(t));
            }
        }
        for (ggml_tensor * input : inputs) {
            std::vector<float> data(ggml_nelements(input));
            for (float & v : data) v = dist(rng);
            if (input->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
            } else {
                std::vector<uint8_t> q(ggml_nbytes(input));
                ggml_quantize_chunk(input->type, data.data(), q.data(), 0, ggml_nrows(input), input->ne[0], nullptr);
                ggml_backend_tensor_set(input, q.data(), 0, q.size());
            }
        }
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        results[pass].resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, results[pass].data(), 0, ggml_nbytes(out));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    // Normalized mean squared error, as test-backend-ops uses. The CPU
    // backend quantizes activations to Q8 for quantized dot products, so
    // those cases get a looser bound than exact float operations.
    double num = 0;
    double den = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        const double d = double(results[0][i]) - results[1][i];
        num += d * d;
        den += double(results[1][i]) * results[1][i];
    }
    const double nmse = den > 0 ? num / den : num;
    std::printf("  %-28s nmse %.3g\n", label, nmse);
    CHECK(nmse <= max_nmse);
    ggml_backend_free(cpu);
}

int main(int argc, char ** argv) {
    CHECK(argc == 2);
    std::string error;
    CHECK(g_package.open(argv[1], error));
    CHECK(setenv("GGML_FLAGOS_KERNEL_DIR", argv[1], 1) == 0);
    CHECK(setenv("GGML_FLAGOS_GRAPH_CAPTURE", "1", 1) == 0);

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
    // The norm*w+x chain went through the Common planner and the fused
    // RMS_NORM+MUL kernel the AMD package ships.
    CHECK(g_fused_rms_norm_mul == 1);
    // Quantized MUL_MAT: decode GEMV (1 column) and batched prefill.
    const ggml_type quant_types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0,
                                      GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    for (ggml_type type : quant_types) {
        for (int64_t columns : { int64_t(1), int64_t(5), int64_t(37) }) {
            const std::string label = std::string("MUL_MAT ") + ggml_type_name(type) + " cols=" + std::to_string(columns);
            compare(backend, device, label.c_str(), [type, columns](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
                in = { ggml_new_tensor_2d(ctx, type, 512, 96), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, columns) };
                return ggml_mul_mat(ctx, in[0], in[1]);
            }, true, 5e-4);
        }
    }
    // Graph capture: the same graph with the same buffers runs three times
    // with new input values each time. Run 1 executes directly, run 2 is
    // recorded and launched, run 3 replays; all must match the CPU.
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_init_params params = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * dctx = ggml_init(params);
        ggml_context * cctx = ggml_init(params);
        auto build = [](ggml_context * ctx, ggml_tensor *& x, ggml_tensor *& w, ggml_tensor *& q) {
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 3);
            w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
            q = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 512, 64);
            ggml_tensor * h = ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w), x);
            return ggml_mul_mat(ctx, q, ggml_silu(ctx, h));
        };
        ggml_tensor *dx, *dw, *dq, *cx, *cw, *cq;
        ggml_tensor * dout = build(dctx, dx, dw, dq);
        ggml_tensor * cout = build(cctx, cx, cw, cq);
        ggml_cgraph * dg = ggml_new_graph(dctx);
        ggml_cgraph * cg = ggml_new_graph(cctx);
        ggml_build_forward_expand(dg, dout);
        ggml_build_forward_expand(cg, cout);
        ggml_backend_buffer_t db = ggml_backend_alloc_ctx_tensors(dctx, backend);
        ggml_backend_buffer_t cb = ggml_backend_alloc_ctx_tensors(cctx, cpu);
        CHECK(db != nullptr && cb != nullptr);
        for (int run = 0; run < 3; ++run) {
            std::mt19937 rng(100 + run);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> xv(512 * 3), wv(512), qv(512 * 64);
            for (float & v : xv) v = dist(rng);
            for (float & v : wv) v = dist(rng);
            for (float & v : qv) v = dist(rng);
            std::vector<uint8_t> qb(ggml_nbytes(dq));
            ggml_quantize_chunk(GGML_TYPE_Q4_K, qv.data(), qb.data(), 0, 64, 512, nullptr);
            for (auto [d, c, data, bytes] : { std::tuple{ dx, cx, (const void *) xv.data(), ggml_nbytes(dx) },
                                              std::tuple{ dw, cw, (const void *) wv.data(), ggml_nbytes(dw) },
                                              std::tuple{ dq, cq, (const void *) qb.data(), qb.size() } }) {
                ggml_backend_tensor_set(d, data, 0, bytes);
                ggml_backend_tensor_set(c, data, 0, bytes);
            }
            CHECK(ggml_backend_graph_compute(backend, dg) == GGML_STATUS_SUCCESS);
            CHECK(ggml_backend_graph_compute(cpu, cg) == GGML_STATUS_SUCCESS);
            std::vector<float> a(ggml_nelements(dout)), b(ggml_nelements(cout));
            ggml_backend_tensor_get(dout, a.data(), 0, ggml_nbytes(dout));
            ggml_backend_tensor_get(cout, b.data(), 0, ggml_nbytes(cout));
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * double(a[i] - b[i]); den += double(b[i]) * b[i]; }
            std::printf("  %-28s nmse %.3g\n", ("captured graph run " + std::to_string(run + 1)).c_str(), num / den);
            CHECK(num / den < 5e-4);
        }
        CHECK(g_graph_instantiations == 1);
        CHECK(g_graph_launches == 2);
        // Same structure in new buffers (as decode moves KV views every
        // token): the recorded graph is updated in place, not re-instantiated.
        ggml_context * d2 = ggml_init(params);
        ggml_tensor *ex, *ew, *eq;
        ggml_tensor * eout = build(d2, ex, ew, eq);
        ggml_cgraph * eg = ggml_new_graph(d2);
        ggml_build_forward_expand(eg, eout);
        ggml_backend_buffer_t eb = ggml_backend_alloc_ctx_tensors(d2, backend);
        CHECK(eb != nullptr);
        for (auto [d, c] : { std::pair{ ex, cx }, std::pair{ ew, cw }, std::pair{ eq, cq } }) {
            std::vector<uint8_t> bytes(ggml_nbytes(c));
            ggml_backend_tensor_get(c, bytes.data(), 0, bytes.size());
            ggml_backend_tensor_set(d, bytes.data(), 0, bytes.size());
        }
        CHECK(ggml_backend_graph_compute(backend, eg) == GGML_STATUS_SUCCESS);
        std::vector<float> a(ggml_nelements(eout)), b(ggml_nelements(cout));
        ggml_backend_tensor_get(eout, a.data(), 0, ggml_nbytes(eout));
        ggml_backend_tensor_get(cout, b.data(), 0, ggml_nbytes(cout));
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * double(a[i] - b[i]); den += double(b[i]) * b[i]; }
        std::printf("  %-28s nmse %.3g\n", "captured graph, new buffers", num / den);
        CHECK(num / den < 5e-4);
        CHECK(g_graph_updates == 1 && g_graph_instantiations == 1 && g_graph_launches == 3);
        ggml_backend_buffer_free(eb);
        ggml_free(d2);
        ggml_backend_buffer_free(db);
        ggml_backend_buffer_free(cb);
        ggml_free(dctx);
        ggml_free(cctx);
        ggml_backend_free(cpu);
    }
    // Contract boundaries the package cannot satisfy fall back to the CPU.
    const flagos_kernel_info * norm = g_package.find("flagos_rms_norm_f32");
    CHECK(norm != nullptr);
    const int64_t too_wide = (norm->block_size / 4 + 1) * 4;
    compare(backend, device, "RMS_NORM wider than BLOCK", [too_wide](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, too_wide, 2) };
        return ggml_rms_norm(ctx, in[0], 1e-6f);
    }, false);
    // Per-dimension broadcast runs through the strided repeat kernel.
    compare(backend, device, "ADD column broadcast", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 8) };
        return ggml_add(ctx, in[0], in[1]);
    });
    // MoE experts: Q4_K is in the AMD package; Q6_K (the Qwen3.6-35B-A3B
    // Q4_K_M ffn_down_exps type) is not, so it must fall back to the CPU.
    compare(backend, device, "MUL_MAT_ID q4_K 8 experts", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, 256, 64, 8), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 3) };
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
        ggml_set_name(ids, "ids");
        return ggml_mul_mat_id(ctx, in[0], in[1], ids);
    }, true, 5e-4);
    compare(backend, device, "MUL_MAT_ID q6_K (no kernel)", [](ggml_context * ctx, std::vector<ggml_tensor *> & in) {
        in = { ggml_new_tensor_3d(ctx, GGML_TYPE_Q6_K, 256, 64, 8), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 3) };
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
        return ggml_mul_mat_id(ctx, in[0], in[1], ids);
    }, false);

    ggml_backend_free(backend);
    CHECK(g_launches > 0 && g_modules > 0);
    CHECK(g_allocations == g_frees);
    std::printf("amd-rt checks passed: %d launches, %d modules loaded\n", g_launches, g_modules);
    return 0;
}
