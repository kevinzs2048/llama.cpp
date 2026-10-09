// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Host contract checks only; the HIP functions below do not call the driver.
#ifndef FLAGOS_PROVIDER_SOURCE
#define FLAGOS_PROVIDER_SOURCE "../flagos-amd.cpp"
#endif
#include FLAGOS_PROVIDER_SOURCE
#include <cstdio>
#include <functional>
#include <map>

static int loaded = 0, unloaded = 0;
static std::map<uintptr_t, std::string> symbols;
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
static int launches = 0;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find("flagos_gated_delta_net_scalar_f32_cache") == 0, "cache symbol");
    check(gx == 2 && gy == 1 && gz > 0, "cache grid");
    check(*static_cast<int *>(args[8]) == 128 && *static_cast<int *>(args[9]) == 2 && *static_cast<int *>(args[10]) == 1, "cache dimensions");
    auto * output = *static_cast<float **>(args[6]);
    auto * cache = *static_cast<float **>(args[7]);
    std::fill_n(output, 256, 42.f); std::fill_n(cache, 32768, 43.f); ++launches;
    return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "package and stage");
    const bool after = argc == 2 || std::atoi(argv[2]) != 0;
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    amd_device_context device; device.arch = "gfx1150"; device.ordinal = 0;
    device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "modules");
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 2, 1, 1);
    auto * k = ggml_dup_tensor(ctx, q); auto * v = ggml_dup_tensor(ctx, q);
    auto * gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 2, 1, 1);
    auto * beta = ggml_dup_tensor(ctx, gate);
    auto * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 2, 1);
    auto * gdn = ggml_gated_delta_net(ctx, q, k, v, gate, beta, state, 1);
    auto * snapshot = ggml_view_1d(ctx, gdn, 32768, 256 * sizeof(float));
    auto * persistent = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32768);
    auto * destination = ggml_view_3d(ctx, persistent, 32768, 1, 1, 32768 * 4, 32768 * 4, 0);
    auto * copy = ggml_cpy(ctx, snapshot, destination);
    std::vector<std::vector<float>> inputs;
    for (auto * tensor : { q, k, v, gate, beta, state }) {
        inputs.emplace_back(ggml_nelements(tensor), 1.f); tensor->data = inputs.back().data();
    }
    std::vector<float> result(ggml_nelements(gdn), -99.f), cache(32768, -98.f);
    gdn->data = result.data(); snapshot->data = result.data() + 256;
    persistent->data = destination->data = copy->data = cache.data();
    const char * names[] = { "flagos_gated_delta_net_scalar_f32_cache",
        "flagos_gated_delta_net_scalar_f32_cache_only", "flagos_gated_delta_net_scalar_f32_cache_only_decode" };
    int checks = 0;
    hipGetDeviceProperties(&device.props, 0);
    check(amd_supports_gated_delta_net(&device, gdn), "direct GDN legal");
    amd_backend_context context(&device, nullptr);
    ggml_tensor * nodes[] = { gdn, snapshot, copy }; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 3;
    for (int index = 0; index < 3; ++index) {
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[index]));
        check(metadata != nullptr, "cache profile"); const auto saved = *metadata;
        auto support = [&] { return amd_supports_gated_delta_net_cache(&device, gdn, snapshot, copy, names[index]); };
        auto test = [&](const char * label, const std::function<void()> & change) {
            *metadata = saved; hipGetDeviceProperties(&device.props, 0); change();
            check(support() == !after, label); ++checks;
        };
        check(support(), "legal cache profile"); ++checks;
        test("argument count", [&] { ++metadata->argument_count; });
        test("typed ABI missing", [&] { metadata->argument_abi.pop_back(); });
        test("scalar width", [&] { metadata->argument_abi.back().size = 8; });
        test("pointer kind", [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("global scratch", [&] { metadata->global_scratch_size = 1; });
        test("profile scratch", [&] { metadata->profile_scratch_size = 1; });
        test("dynamic LDS", [&] { metadata->shared_memory = 65537; });
        test("static LDS", [&] { metadata->static_shared_memory = 65537; });
        test("threads", [&] { metadata->num_warps = 64; });
        test("wave", [&] { metadata->warp_size = 64; });
        test("cache grid Z", [&] { metadata->tile_n = 1; device.props.maxGridSize[2] = 64; });
        *metadata = saved; hipGetDeviceProperties(&device.props, 0);
        flagos_plan_step step; step.kind = flagos_execution_kind::pattern;
        step.candidate.id = flagos_pattern_id::gated_delta_net_decode;
        step.candidate.node_indices = { 0, 1, 2 };
        step.implementation_id = 12 + index;
        if (index == 0) {
            check(amd_execute_fusion(&context, &graph, step), "legal fusion dispatch"); ++checks;
            check(std::all_of(result.begin(), result.begin() + 256, [](float x) { return x == 42.f; }), "attention binding");
            check(std::all_of(cache.begin(), cache.end(), [](float x) { return x == 43.f; }), "cache binding");
            std::fill(result.begin(), result.end(), -99.f); std::fill(cache.begin(), cache.end(), -98.f);
        }
    }
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "gated_delta_net_cache", 1);
    flagos_pattern_candidate candidate;
    candidate.id = flagos_pattern_id::gated_delta_net_decode;
    candidate.node_indices = { 0, 1, 2 };
    candidate.required_output_node_indices = { 0, 2 };
    check(amd_query_fusion(&context, &graph, candidate).supported, "legal fusion query"); ++checks;
    for (const char * name : names) {
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name));
        metadata->argument_abi.back().size = 8;
    }
    check(amd_supports_gated_delta_net(&device, gdn), "direct GDN remains supported"); ++checks;
    check(amd_query_fusion(&context, &graph, candidate).supported == !after,
        "reject invalid fusion profiles without rejecting the direct operator"); ++checks;
    device.aot.reset(); ggml_free(ctx);
    check(loaded == unloaded && launches == 1, "module and launch lifecycle");
    std::printf("GDN_CACHE_CONTRACT API=%d stage=%s checks=%d mock_launches=%d actual_GPU_calls=0\n",
        GGML_BACKEND_API_VERSION, after ? "after" : "before", checks, launches);
}
