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
    setenv("FLAGOS_AMD_F16_COPY", "1", 1); setenv("FLAGOS_AMD_PREFILL_F16_GEMM", "1", 1);
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1); setenv("FLAGOS_AMD_FUSIONS", "none", 1);
    setenv("FLAGOS_AMD_QUANT_TILED_GEMM", "0", 1); setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "0", 1);
    setenv("FLAGOS_AMD_ALIGNED_F16_GEMM", "0", 1);
    amd_device_context device; device.arch = "gfx1150"; device.ordinal = 0; device.device_iface.context = &device;
    device.buffer_type_iface.context = &device; device.buffer_type_iface.iface = g_buffer_type_iface;
    device.buffer_type_iface.device = &device.device_iface;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock modules"); hipGetDeviceProperties(&device.props, 0);
    int contract_checks = 0;
    if (after) {
        amd_fill_device_profile(device, &device.profile);
        const char * profiles[] = {"flagos_mul_mat_f16_f32_batched", "flagos_mul_mat_f16_f32_grouped", "flagos_mul_mat_f16_f32_batched_aligned"};
        flagos_amd::kernel_metadata * metadata[3];
        std::vector<flagos_amd::kernel_metadata> saved;
        for (int i = 0; i < 3; ++i) { metadata[i] = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(profiles[i])); check(metadata[i] != nullptr, "projection profile"); saved.push_back(*metadata[i]); }
        alignas(16) unsigned char binding[32]{};
        amd_f16_projection_launch projection;
        for (int selected = 0; selected < 3; ++selected) {
            auto test = [&](const char * label, bool accepted, const std::function<void()> & change) {
                for (int i = 0; i < 3; ++i) { *metadata[i] = saved[i]; if (i != selected) metadata[i]->global_scratch_size = 1; }
                hipGetDeviceProperties(&device.props, 0);
                setenv("FLAGOS_AMD_GROUPED_F16_GEMM", selected == 1 ? "1" : "0", 1);
                setenv("FLAGOS_AMD_ALIGNED_F16_GEMM", selected == 2 ? "1" : "0", 1);
                change();
                check(amd_describe_f16_projection(&device, binding, binding, binding, 1024, 144, 257, &projection) == accepted, label);
                if (accepted) check(std::strcmp(projection.kernel, profiles[selected]) == 0 && projection.grid_x == (selected == 1 ? 9u : 3u) && projection.grid_y == (selected == 1 ? 1u : 3u), "metadata tile selection");
                ++contract_checks;
            };
            test("legal profile", true, [] {});
            test("LDS", false, [&] { metadata[selected]->shared_memory = 65537; });
            test("static LDS", false, [&] { metadata[selected]->static_shared_memory = 65537; });
            test("threads", false, [&] { device.props.maxThreadsPerBlock = 64; });
            test("grid", false, [&] { device.props.maxGridSize[0] = 2; });
            test("wave", false, [&] { device.props.warpSize = 64; });
            test("argument count", false, [&] { metadata[selected]->argument_count = 7; });
            test("missing ABI", false, [&] { metadata[selected]->argument_abi.pop_back(); });
            test("pointer kind", false, [&] { metadata[selected]->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
            test("scalar size", false, [&] { metadata[selected]->argument_abi[3].size = 8; });
            test("scratch", false, [&] { metadata[selected]->global_scratch_size = 1; });
            test("profile scratch", false, [&] { metadata[selected]->profile_scratch_size = 1; });
            test("tile M", false, [&] { metadata[selected]->tile_m = 0; });
            test("tile N", false, [&] { metadata[selected]->tile_n = 0; });
            test("tile K", false, [&] { metadata[selected]->tile_k = 0; });
        }
        for (int i = 0; i < 3; ++i) *metadata[i] = saved[i];
        hipGetDeviceProperties(&device.props, 0);
        setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "1", 1); setenv("FLAGOS_AMD_ALIGNED_F16_GEMM", "1", 1);
        check(amd_describe_f16_projection(&device, binding, binding, binding, 1024, 144, 257, &projection) && projection.grouped, "grouped priority"); ++contract_checks;
        metadata[1]->shared_memory = 65537;
        check(amd_describe_f16_projection(&device, binding, binding, binding, 1024, 144, 257, &projection) && projection.aligned, "aligned fallback"); ++contract_checks;
        metadata[2]->shared_memory = 65537;
        check(amd_describe_f16_projection(&device, binding, binding, binding, 1024, 144, 257, &projection) && !projection.grouped && !projection.aligned, "ordinary fallback"); ++contract_checks;
        metadata[0]->shared_memory = 65537;
        check(!amd_describe_f16_projection(&device, binding, binding, binding, 1024, 144, 257, &projection), "all profiles rejected"); ++contract_checks;
        for (int i = 0; i < 3; ++i) *metadata[i] = saved[i];
        setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "0", 1); setenv("FLAGOS_AMD_ALIGNED_F16_GEMM", "0", 1);
    }
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 1024, 64);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 16); auto * out = ggml_mul_mat(ctx, weights, input);
    auto * source = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 17);
    auto * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 17); auto * copy = ggml_cpy(ctx, source, state);
    auto * storage = amd_buffer_alloc(&device.buffer_type_iface, ggml_nbytes(weights));
    check(storage != nullptr, "weight buffer"); ggml_backend_buffer_set_usage(storage, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    weights->buffer = storage; weights->data = ggml_backend_buffer_get_base(storage);
    std::memset(weights->data, 0, ggml_nbytes(weights));
    std::vector<float> x(1024 * 16, 1.f), result(64 * 16, -99.f);
    std::vector<unsigned short> from(32 * 17, 0x3c00), persistent(32 * 17, 0x7c00);
    input->data = x.data(); out->data = result.data(); source->data = from.data(); state->data = copy->data = persistent.data(); expected_output = result.data();
    flagos_quantized_matmul_signature signature; check(flagos_describe_quantized_matmul(out, &signature), "signature");
    amd_backend_context context(&device, nullptr); ggml_backend backend{}; backend.context = &context;
    // Inject invalid launch metadata after package loading. This exercises provider
    // selection independently of the loader, which rejects invalid packages.
    // The ordinary quantized profile fits; the optional F16 profile does not.
    device.props.sharedMemPerBlock = 65536;
    auto * f16 = const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_mul_mat_f16_f32_batched"));
    check(f16 != nullptr, "F16 metadata");
    f16->shared_memory = 70000;
    check(amd_get_dequantized_weight(&context, weights, signature) != nullptr, "warm dequant cache");
    check(expected_dequant != nullptr && !context.dequant_cache.empty(), "cached binding");
    ggml_tensor * nodes[] = {copy, out}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 2;
    check(amd_device_supports_op(&device.device_iface, out), "ordinary placement legal");
    const auto status = amd_backend_graph_compute(&backend, &graph);
    check(status == (after ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED), "F16 resource fallback result");
    check(state_writes == 1 && persistent == from, "earlier state copy");
    check(quantized_writes == (after ? 1 : 0), "quantized fallback submissions");
    check(std::all_of(result.begin(), result.end(), [after](float v) { return v == (after ? 42.f : -99.f); }), "output contents");
    amd_release_dequant_cache(&context); context.graph_plans.clear();
    // An impossible dequant grid should decline before private allocation.
    device.props.sharedMemPerBlock = 65536; device.props.maxGridSize[0] = 128; reject_dequant_grid = true;
    const int before_allocations = allocations;
    check(amd_get_dequantized_weight(&context, weights, signature) == nullptr, "dequant grid rejected");
    check(allocations - before_allocations == (after ? 0 : 1), "private allocation avoided");
#if GGML_BACKEND_API_VERSION >= 3
    check(ggml_prec_set_src(out, GGML_PREC_F32, 1), "activation minimum precision");
    check(!amd_allows_f16_matmul(out), "activation rank respected");
#endif
    ggml_backend_buffer_free(storage); device.aot.reset(); ggml_free(ctx);
    check(loaded == unloaded && allocations == releases, "module and allocation lifecycle");
    std::printf("F16_RESOURCE_REPRO API=%d stage=%s contract_checks=%d state_writes=%d quantized_fallback_writes=%d dequant_rejected_grid_allocations=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, after ? "after" : "before", contract_checks, state_writes, quantized_writes, allocations - before_allocations);
}
