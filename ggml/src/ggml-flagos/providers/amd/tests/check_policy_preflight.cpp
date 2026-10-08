// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// CPU fixture: HIP execution functions are replaced below. No device compute.
#ifndef FLAGOS_PROVIDER_SOURCE
#define FLAGOS_PROVIDER_SOURCE "../flagos-amd.cpp"
#endif
#include FLAGOS_PROVIDER_SOURCE

#include <cstdio>
#include <map>
static int launches = 0, copy_writes = 0, matmul_writes = 0, module_loads = 0, module_unloads = 0;
static std::map<uintptr_t, std::string> functions;

extern "C" hipError_t hipSetDevice(int) {
    return hipSuccess;
}

extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t * p, int) {
    *p                    = {};
    p->warpSize           = 32;
    p->maxGridSize[0]     = INT32_MAX;
    p->maxGridSize[1]     = 65535;
    p->maxGridSize[2]     = 65535;
    p->maxThreadsPerBlock = 1024;
    p->sharedMemPerBlock  = 65536;
    std::strcpy(p->gcnArchName, "gfx1150");
    return hipSuccess;
}

extern "C" hipError_t hipModuleLoadData(hipModule_t * m, const void *) {
    *m = reinterpret_cast<hipModule_t>(uintptr_t(++module_loads));
    return hipSuccess;
}

extern "C" hipError_t hipModuleUnload(hipModule_t) {
    ++module_unloads;
    return hipSuccess;
}

extern "C" hipError_t hipModuleGetFunction(hipFunction_t * f, hipModule_t, const char * n) {
    uintptr_t id  = functions.size() + 1;
    functions[id] = n;
    *f            = reinterpret_cast<hipFunction_t>(id);
    return hipSuccess;
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t) {
    return hipSuccess;
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                            unsigned,
                                            unsigned,
                                            unsigned,
                                            unsigned,
                                            unsigned,
                                            unsigned,
                                            unsigned,
                                            hipStream_t,
                                            void ** args,
                                            void **) {
    ++launches;
    auto name = functions.at(reinterpret_cast<uintptr_t>(f));
    if (name.find("flagos_copy_strided_f16_bits") == 0) {
        const auto * x   = *static_cast<const uint16_t **>(args[0]);
        auto *       out = *static_cast<uint16_t **>(args[1]);
        int          n   = *static_cast<int *>(args[18]);
        std::copy(x, x + n, out);
        ++copy_writes;
    } else if (name.find("flagos_mul_mat_f16_f32_strided_batched") == 0) {
        auto * out  = *static_cast<float **>(args[2]);
        int    rows = *static_cast<int *>(args[4]), cols = *static_cast<int *>(args[5]);
        std::fill(out, out + rows * cols, 42.f);
        ++matmul_writes;
    } else {
        std::abort();
    }
    return hipSuccess;
}

static void check(bool v, const char * s) {
    if (!v) {
        std::fprintf(stderr, "CHECK_FAILED %s\n", s);
        std::exit(2);
    }
}

int main(int argc, char ** argv) {
    check(argc == 2, "package");
    setenv("FLAGOS_AMD_F16_COPY", "1", 1);
    setenv("FLAGOS_AMD_STRIDED_BATCHED_F16_GEMM", "1", 1);
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    setenv("FLAGOS_AMD_KERNEL_PROFILE", "0", 1);
    setenv("FLAGOS_AMD_SPLITKV_DECODE", "0", 1);
    amd_device_context device;
    device.arch                 = "gfx1150";
    device.ordinal              = 0;
    device.device_iface.context = &device;
    device.aot                  = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "fake modules");
    device.props.warpSize   = 32;
    device.props.integrated = 1;
    amd_fill_device_profile(device, &device.profile);
    amd_backend_context context(&device, nullptr);
    ggml_backend        backend{};
    backend.context             = &context;
    auto *                ctx   = ggml_init({ 1024 * 1024, nullptr, true });
    auto *                x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 17);
    auto *                state = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 17);
    auto *                copy  = ggml_cpy(ctx, x, state);
    auto *                b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 3);
    auto *                out   = ggml_mul_mat(ctx, copy, b);
    std::vector<uint16_t> source(32 * 17, 0x3c00), persistent(32 * 17, 0x7c00);
    std::vector<float>    activation(32 * 3, 1.f), output(17 * 3, -99.f);
    x->data     = source.data();
    state->data = copy->data = persistent.data();
    b->data                  = activation.data();
    out->data                = output.data();
    ggml_tensor * nodes[]    = { copy, out };
    ggml_cgraph   graph{};
    graph.nodes   = nodes;
    graph.n_nodes = 2;
    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS, "initial valid graph");
    check(copy_writes == 1 && matmul_writes == 1, "valid initial dispatches");
    std::fill(persistent.begin(), persistent.end(), 0x7c00);
    std::fill(output.begin(), output.end(), -99.f);
    int prior = launches;
    setenv("FLAGOS_AMD_STRIDED_BATCHED_F16_GEMM", "0", 1);
    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED, "disabled later node fails");
    check(launches - prior == 0, "policy preflight launch count");
    check(std::all_of(output.begin(), output.end(), [](float v) { return v == -99.f; }), "later output untouched");
    check(std::all_of(persistent.begin(), persistent.end(), [=](auto v) { return v == 0x7c00; }),
          "persistent first node state");
    std::printf("POLICY_DISABLE_REPRO API=%d phase=%s earlier_state_writes=%d actual_GPU_calls=0\n",
                GGML_BACKEND_API_VERSION, "after", launches - prior);
    // A new context must reject a graph containing an unsupported later node too.
    std::fill(persistent.begin(), persistent.end(), 0x7c00);
    amd_backend_context cold(&device, nullptr);
    backend.context = &cold;
    ggml_cgraph empty{};
    check(amd_backend_graph_compute(&backend, &empty) == GGML_STATUS_SUCCESS, "empty graph");
    check(cold.execution_policy_key == 0, "empty graph leaves policy cold");
    prior = launches;
    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED, "invalid initial placement contract");
    check(launches - prior == 0, "cold graph preflight");
    backend.context = &context;
    setenv("FLAGOS_AMD_STRIDED_BATCHED_F16_GEMM", "1", 1);
    prior = launches;
    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS, "reenable original policy");
    check(launches - prior == 2, "restored valid launches");
    setenv("FLAGOS_AMD_FUSIONS", "none", 1);
    prior = launches;
    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS, "valid policy change");
    check(launches - prior == 2, "valid policy retains computation");
    context.graph_plans.clear();
    cold.graph_plans.clear();
    device.aot.reset();
    check(module_loads == module_unloads, "balanced module lifecycle");
    ggml_free(ctx);
    std::printf("POLICY_PREFLIGHT_AUDIT_PASS API=%d phase=%s actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, "after");
}
