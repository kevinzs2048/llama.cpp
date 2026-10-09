// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Host bindings only. HIP functions below never call the driver.
#ifndef FLAGOS_PROVIDER_SOURCE
#define FLAGOS_PROVIDER_SOURCE "../flagos-amd.cpp"
#endif
#include FLAGOS_PROVIDER_SOURCE
#include <cstdio>
#include <functional>
#include <map>

static int launches = 0, loaded = 0, unloaded = 0;
static std::map<uintptr_t, std::string> symbols;
static ggml_tensor * expected_node = nullptr;
static amd_quantized_gemv_launch expected;
static void check(bool value, const char * name) {
    if (!value) { std::fprintf(stderr, "CHECK_FAILED %s\n", name); std::exit(2); }
}
extern "C" hipError_t hipSetDevice(int) { return hipSuccess; }
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t * p, int) {
    *p = {}; p->warpSize = 32; p->maxThreadsPerBlock = 1024; p->sharedMemPerBlock = 65536;
    p->maxGridSize[0] = INT32_MAX; p->maxGridSize[1] = p->maxGridSize[2] = 65535;
    std::strcpy(p->gcnArchName, "gfx1150"); return hipSuccess;
}
extern "C" hipError_t hipModuleLoadData(hipModule_t * m, const void *) {
    *m = reinterpret_cast<hipModule_t>(uintptr_t(++loaded)); return hipSuccess;
}
extern "C" hipError_t hipModuleUnload(hipModule_t) { ++unloaded; return hipSuccess; }
extern "C" hipError_t hipModuleGetFunction(hipFunction_t * f, hipModule_t, const char * name) {
    const auto id = symbols.size() + 1; symbols[id] = name;
    *f = reinterpret_cast<hipFunction_t>(id); return hipSuccess;
}
extern "C" hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_node && gx == expected.grid_x && gy == 1 && gz == 1, "selected grid");
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected.kernel) == 0 ||
          (std::strstr(expected.kernel, "_model_aligned") &&
           symbols.at(reinterpret_cast<uintptr_t>(function)) == amd_quantized_gemv_kernel(amd_quantized_kind(expected_node->src[0]->type))), "selected symbol");
    check(*static_cast<void **>(args[0]) == expected_node->src[0]->data &&
          *static_cast<void **>(args[1]) == expected_node->src[0]->data &&
          *static_cast<void **>(args[2]) == expected_node->src[1]->data &&
          *static_cast<void **>(args[3]) == expected_node->data &&
          *static_cast<int *>(args[4]) == expected_node->src[0]->ne[0] &&
          *static_cast<int *>(args[5]) == expected_node->ne[0], "six-slot ABI bindings");
    std::fill_n(static_cast<float *>(expected_node->data), ggml_nelements(expected_node), -123.0f);
    ++launches; return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2, "package"); setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    setenv("FLAGOS_AMD_FUSIONS", "none", 1); setenv("FLAGOS_AMD_PREFILL_F16_GEMM", "0", 1);
    amd_device_context device; device.arch = "gfx1150"; device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock modules"); int checks = 0;
    using mutation = std::function<void(ggml_tensor *)>;
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        const auto kind = amd_quantized_kind(type); const char * name = amd_quantized_gemv_kernel(kind);
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name));
        check(metadata != nullptr, "baseline metadata"); const auto saved = *metadata;
        auto test = [&](const char * label, bool accepted, const mutation & modify, bool execute = false) {
            for (bool bound : {false, true}) {
                auto * ctx = ggml_init({1024 * 1024, nullptr, true});
                auto * weights = ggml_new_tensor_2d(ctx, type, 256, 65);
                auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
                auto * out = ggml_mul_mat(ctx, weights, input);
                std::vector<uint16_t> packed((ggml_nbytes(weights) + 1) / 2 + 32, 0);
                std::vector<float> x(ggml_nelements(input) + 32, 3.0f), result(ggml_nelements(out) + 32, 37.0f);
                if (bound) { weights->data = packed.data() + 8; input->data = x.data() + 8; out->data = result.data() + 8; }
                *metadata = saved; hipGetDeviceProperties(&device.props, 0);
                for (const char * flag : {"FLAGOS_AMD_Q4_GEMV_NARROW", "FLAGOS_AMD_Q5_GEMV_NARROW", "FLAGOS_AMD_Q40_GEMV_NARROW"}) setenv(flag, "0", 1);
                modify(out); check(amd_device_supports_op(&device.device_iface, out) == accepted, label);
                if (execute && bound) {
                    check(amd_describe_quantized_gemv(&device, out, kind, 256, 65, &expected), "selected description");
                    expected_node = out; amd_backend_context context(&device, nullptr); ggml_backend backend{}; backend.context = &context;
                    ggml_tensor * nodes[] = {out}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 1;
                    const int before = launches;
                    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS && launches == before + 1, "dispatch");
                    check(std::all_of(result.begin() + 8, result.end() - 24, [](float v) { return v == -123.0f; }), "writes");
                    check(std::all_of(result.begin(), result.begin() + 8, [](float v) { return v == 37.0f; }) &&
                          std::all_of(result.end() - 24, result.end(), [](float v) { return v == 37.0f; }), "guards");
                    check(std::all_of(packed.begin(), packed.end(), [](uint16_t v) { return v == 0; }) &&
                          std::all_of(x.begin(), x.end(), [](float v) { return v == 3.0f; }), "readonly inputs");
                    expected_node = nullptr;
                }
                ++checks; ggml_free(ctx);
            }
        };
        test("valid", true, [](auto *) {}, true);
        test("unknown caps", true, [&](auto *) { device.props = {}; });
        test("grid", false, [&](auto *) { device.props.maxGridSize[0] = 64; });
        test("threads", false, [&](auto *) { device.props.maxThreadsPerBlock = 16; });
        test("wave", false, [&](auto *) { device.props.warpSize = 64; });
        test("static LDS", false, [&](auto *) { metadata->static_shared_memory = 65537; });
        test("dynamic LDS", false, [&](auto *) { metadata->shared_memory = 65537; });
        test("scratch", false, [&](auto *) { metadata->global_scratch_size = 1; });
        test("profile scratch", false, [&](auto *) { metadata->profile_scratch_size = 1; });
        test("argument count", false, [&](auto *) { metadata->argument_count = 7; });
        test("missing ABI", false, [&](auto *) { metadata->argument_abi.pop_back(); });
        test("pointer kind", false, [&](auto *) { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("pointer size", false, [&](auto *) { metadata->argument_abi[0].size = 4; });
        test("scalar kind", false, [&](auto *) { metadata->argument_abi[4].kind = flagos_amd::kernel_argument_kind::global_buffer; });
        test("scalar size", false, [&](auto *) { metadata->argument_abi[4].size = 8; });
        test("block zero", false, [&](auto *) { metadata->block_size = 0; });
        test("weight alignment", false, [](auto * out) { out->src[0]->data = reinterpret_cast<void *>(UINT64_C(0x100000001)); });
        test("input alignment", false, [](auto * out) { out->src[1]->data = reinterpret_cast<void *>(UINT64_C(0x200000002)); });
        test("output alignment", false, [](auto * out) { out->data = reinterpret_cast<void *>(UINT64_C(0x300000002)); });
        test("weight dtype", false, [](auto * out) { out->src[0]->type = GGML_TYPE_F32; });
        test("output layout", false, [](auto * out) { out->nb[0] += 4; });
        *metadata = saved;
    }
    // Q4_K candidate priorities and resource fallback are checked without bindings.
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 64);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1); auto * out = ggml_mul_mat(ctx, weights, input);
    auto * eight = const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_mul_mat_q4_k_f32_narrow8"));
    auto * four = const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_mul_mat_q4_k_f32_narrow"));
    check(eight && four, "narrow profiles"); const auto saved8 = *eight, saved4 = *four;
    setenv("FLAGOS_AMD_Q4_GEMV_NARROW", "8", 1); hipGetDeviceProperties(&device.props, 0);
    check(amd_describe_quantized_gemv(&device, out, flagos_quantized_matmul_kind::q4_k, 256, 64, &expected) && expected.row_tile == 8, "eight selected"); ++checks;
    eight->shared_memory = 65537;
    check(amd_describe_quantized_gemv(&device, out, flagos_quantized_matmul_kind::q4_k, 256, 64, &expected) && expected.row_tile == 4, "four fallback"); ++checks;
    four->shared_memory = 65537;
    check(amd_describe_quantized_gemv(&device, out, flagos_quantized_matmul_kind::q4_k, 256, 64, &expected) && expected.row_tile == 1, "ordinary fallback"); ++checks;
    *eight = saved8; *four = saved4; device.props.maxGridSize[0] = 8;
    check(amd_device_supports_op(&device.device_iface, out) &&
          amd_describe_quantized_gemv(&device, out, flagos_quantized_matmul_kind::q4_k, 256, 64, &expected) && expected.grid_x == 8, "narrow fits when ordinary does not"); ++checks;
    eight->global_scratch_size = 1;
    check(!amd_device_supports_op(&device.device_iface, out), "all remaining choices exceed grid"); ++checks;
    *eight = saved8; *four = saved4; ggml_free(ctx); device.aot.reset(); check(loaded == unloaded, "module lifecycle");
    std::printf("GEMV_HOST_CONTRACT_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches);
}
