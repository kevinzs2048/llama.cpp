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
static int allocations = 0, releases = 0, state_writes = 0, quantized_writes = 0;
static bool reject_dequant_grid = false;
static unsigned short * expected_dequant = nullptr;
static float * expected_output = nullptr;
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
extern "C" hipError_t hipMalloc(void ** ptr, size_t size) { *ptr = std::malloc(size); ++allocations; return *ptr ? hipSuccess : hipErrorOutOfMemory; }
extern "C" hipError_t hipFree(void * ptr) { std::free(ptr); ++releases; return hipSuccess; }
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    const auto & name = symbols.at(reinterpret_cast<uintptr_t>(function)); ++launches;
    if (name.find("flagos_dequant_q4_k_f16") == 0) {
        if (reject_dequant_grid && gx > 128) return hipErrorInvalidValue;
        auto * output = *static_cast<unsigned short **>(args[2]); expected_dequant = output;
        std::fill_n(output, gx * 256, 0x3c00);
    } else if (name.find("flagos_copy_strided_f16_bits") == 0) {
        auto * input = *static_cast<unsigned short **>(args[0]); auto * output = *static_cast<unsigned short **>(args[1]);
        std::copy_n(input, *static_cast<int *>(args[18]), output); ++state_writes;
    } else if (name.find("flagos_mul_mat_q4_k_f32_batched") == 0) {
        auto * output = *static_cast<float **>(args[3]); check(output == expected_output, "quantized output binding");
        std::fill_n(output, *static_cast<int *>(args[5]) * *static_cast<int *>(args[6]), 42.f); ++quantized_writes;
    } else check(false, "unexpected GPU launch in host fixture");
    return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "package and stage"); const bool after = argc == 2 || std::atoi(argv[2]) != 0;
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "ffn_swiglu,ffn_swiglu_down", 1);
    setenv("FLAGOS_AMD_PREFILL_F16_GEMM", "1", 1); setenv("FLAGOS_AMD_FFN_DOWN_F16", "1", 1);
    setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "1", 1);
    amd_device_context device; device.arch = "gfx1150"; device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock modules");
    hipGetDeviceProperties(&device.props, 0); amd_fill_device_profile(device, &device.profile);
    int checks = 0;
    for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * w0 = ggml_new_tensor_2d(ctx, type, 1024, 64); auto * w1 = ggml_new_tensor_2d(ctx, type, 1024, 64);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1);
        auto * gate = ggml_mul_mat(ctx, w0, x); auto * up = ggml_mul_mat(ctx, w1, x); auto * glu = ggml_swiglu_split(ctx, gate, up);
        const char * name = amd_ffn_swiglu_q4_decode_kernel(&device, amd_quantized_kind(type));
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name)); check(metadata, "decode metadata"); const auto saved = *metadata;
        auto test = [&](const char * label, bool legal, const std::function<void()> & mutate) {
            *metadata = saved; hipGetDeviceProperties(&device.props, 0); mutate();
            check(amd_supports_ffn_swiglu_q4_decode(&device, gate, up, glu) == (legal || !after), label); ++checks;
            ggml_tensor * nodes[] = {gate, up, glu}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 3;
            flagos_pattern_candidate candidate; candidate.id = flagos_pattern_id::ffn_swiglu;
            candidate.node_indices = {0, 1, 2}; candidate.required_output_node_indices = {2};
            amd_backend_context context(&device, nullptr);
            check(amd_query_fusion(&context, &graph, candidate).supported == (legal || !after), "lowering acceptance agrees with legality"); ++checks;
        };
        test("decode legal", true, [] {});
        test("decode grid", false, [&] { device.props.maxGridSize[0] = 7; });
        test("decode wave", false, [&] { device.props.warpSize = 64; });
        test("decode threads", false, [&] { device.props.maxThreadsPerBlock = 16; });
        test("decode LDS", false, [&] { metadata->shared_memory = 65537; });
        test("decode static LDS", false, [&] { metadata->static_shared_memory = 65537; });
        test("decode scratch", false, [&] { metadata->global_scratch_size = 1; });
        test("decode profile scratch", false, [&] { metadata->profile_scratch_size = 1; });
        test("decode count", false, [&] { metadata->argument_count = 9; });
        test("decode missing ABI", false, [&] { metadata->argument_abi.pop_back(); });
        test("decode pointer kind", false, [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("decode scalar size", false, [&] { metadata->argument_abi[6].size = 8; });
        test("decode weight alignment", false, [&] { w0->data = reinterpret_cast<void *>(UINT64_C(0x100001)); }); w0->data = nullptr;
        test("decode activation alignment", false, [&] { x->data = reinterpret_cast<void *>(UINT64_C(0x200002)); }); x->data = nullptr;
        test("decode output alignment", false, [&] { glu->data = reinterpret_cast<void *>(UINT64_C(0x300002)); }); glu->data = nullptr;
        *metadata = saved; ggml_free(ctx);
    }
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * w0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 1024, 1024); auto * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 1024, 1024);
    auto * wd = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 1024, 64); auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 32);
    auto * gate = ggml_mul_mat(ctx, w0, x); auto * up = ggml_mul_mat(ctx, w1, x); auto * glu = ggml_swiglu_split(ctx, gate, up); auto * down = ggml_mul_mat(ctx, wd, glu);
    for (const char * name : {"flagos_ffn_swiglu_f16_f16_grouped", "flagos_mul_mat_f16_f16_grouped"}) {
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name)); check(metadata, "down metadata"); const auto saved = *metadata;
        auto test = [&](const char * label, bool legal, const std::function<void()> & mutate) {
            *metadata = saved; hipGetDeviceProperties(&device.props, 0); mutate();
            check(amd_supports_ffn_swiglu_down_f16(&device, gate, up, glu, down) == (legal || !after), label); ++checks;
        };
        test("down legal", true, [] {});
        test("down wave", false, [&] { device.props.warpSize = 64; });
        test("down threads", false, [&] { device.props.maxThreadsPerBlock = 16; });
        test("down LDS", false, [&] { metadata->shared_memory = 65537; });
        test("down static LDS", false, [&] { metadata->static_shared_memory = 65537; });
        test("down scratch", false, [&] { metadata->global_scratch_size = 1; });
        test("down profile scratch", false, [&] { metadata->profile_scratch_size = 1; });
        test("down count", false, [&] { ++metadata->argument_count; });
        test("down missing ABI", false, [&] { metadata->argument_abi.pop_back(); });
        test("down pointer kind", false, [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("down scalar size", false, [&] { metadata->argument_abi.back().size = 8; });
        *metadata = saved;
    }
    hipGetDeviceProperties(&device.props, 0);device.props.maxGridSize[0] = 31;
    check(amd_supports_ffn_swiglu_down_f16(&device, gate, up, glu, down) == !after, "flattened down grid cap"); ++checks;
    ggml_free(ctx); device.aot.reset();check(loaded == unloaded && launches == 0 && allocations == 0, "zero submissions and allocation");
    std::printf("FFN_FUSION_RESOURCE_COMPLETE API=%d stage=%s checks=%d actual_GPU_calls=0 allocations=0 launches=0\n", GGML_BACKEND_API_VERSION, after ? "after" : "before", checks);
}
