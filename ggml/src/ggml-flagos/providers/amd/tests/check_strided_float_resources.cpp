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

static int loaded = 0, unloaded = 0, state_writes = 0;
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
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned, unsigned, unsigned,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(symbols.at(reinterpret_cast<uintptr_t>(f)) == "flagos_copy_strided_f16_bits", "only state copy submitted");
    auto * input = *static_cast<unsigned short **>(args[0]);
    auto * output = *static_cast<unsigned short **>(args[1]);
    std::copy_n(input, *static_cast<int *>(args[18]), output); ++state_writes;
    return hipSuccess;
}

int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "package and optional before stage");
    const bool after = argc == 2 || std::atoi(argv[2]) != 0;
    setenv("FLAGOS_AMD_F16_COPY", "1", 1);
    setenv("FLAGOS_AMD_STRIDED_BATCHED_F16_GEMM", "1", 1);
    setenv("FLAGOS_AMD_DENSE_FLOAT_OPS", "1", 1);
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1); setenv("FLAGOS_AMD_FUSIONS", "none", 1);
    amd_device_context device; device.arch = "gfx1150"; device.ordinal = 0;
    device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock module initialization");
    hipGetDeviceProperties(&device.props, 0);
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 64);
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 32, 2);
    auto * output = ggml_mul_mat(ctx, weights, input);
    std::vector<float> w(32 * 64, 1.f), x(32 * 32 * 2, 1.f), result(64 * 32 * 2, -99.f);
    weights->data = w.data(); input->data = x.data(); output->data = result.data();
    const char * names[] = { "flagos_mul_mat_f16_f32_strided_batched",
        "flagos_mul_mat_f16_f32_strided_batched_ieee", "flagos_mul_mat_f32_f32_strided_batched_ieee" };
    int contract_checks = 0;
    if (after) {
        for (int index = 0; index < 3; ++index) {
            auto * half_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 64);
            auto * node = index == 2 ? output : ggml_mul_mat(ctx, half_weights, input);
            std::vector<unsigned short> half(32 * 64, 0x3c00);
            half_weights->data = half.data(); node->data = result.data();
            if (index == 1) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            check(std::strcmp(amd_strided_batched_float_kernel_name(node), names[index]) == 0, "precision profile");
            auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[index]));
            check(metadata != nullptr, "strided profile present"); const auto saved = *metadata;
            auto test = [&](const char * label, bool accepted, const std::function<void()> & change) {
                *metadata = saved; hipGetDeviceProperties(&device.props, 0); change();
                check(amd_device_supports_op(&device.device_iface, node) == accepted, label);
                ++contract_checks;
            };
            test("legal profile", true, [] {});
            test("grid X", false, [&] { device.props.maxGridSize[0] = 1; });
            test("grid Y", false, [&] { device.props.maxGridSize[1] = 1; });
            test("grid Z", false, [&] { device.props.maxGridSize[2] = 1; });
            test("threads", false, [&] { device.props.maxThreadsPerBlock = 64; });
            test("wave", false, [&] { device.props.warpSize = 64; });
            test("dynamic LDS", false, [&] { device.props.sharedMemPerBlock = 2048; });
            test("missing typed ABI", false, [&] { metadata->argument_abi.pop_back(); });
            test("pointer kind", false, [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
            test("pointer size", false, [&] { metadata->argument_abi[1].size = 4; });
            test("scalar kind", false, [&] { metadata->argument_abi[3].kind = flagos_amd::kernel_argument_kind::global_buffer; });
            test("scalar size", false, [&] { metadata->argument_abi[3].size = 8; });
            *metadata = saved;
        }
    }
    hipGetDeviceProperties(&device.props, 0);
    auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[2]));
    check(metadata != nullptr, "F32 metadata"); metadata->argument_abi[3].size = 8;
    check(amd_device_supports_op(&device.device_iface, output) == !after, "typed ABI checked at placement");
    auto * source = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 16);
    auto * state = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 16);
    auto * copy = ggml_cpy(ctx, source, state);
    std::vector<unsigned short> from(16, 0x3c00), persistent(16, 0x7c00);
    source->data = from.data(); copy->data = state->data = persistent.data();
    amd_backend_context context(&device, nullptr); ggml_backend backend{}; backend.context = &context;
    ggml_tensor * nodes[] = { copy, output }; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 2;
    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED, "bad graph rejected");
    check(state_writes == (after ? 0 : 1), "reject before earlier state write");
    check(persistent == (after ? std::vector<unsigned short>(16, 0x7c00) : from), "state contents");
    check(std::all_of(result.begin(), result.end(), [](float value) { return value == -99.f; }), "output unchanged");
    context.graph_plans.clear(); device.aot.reset(); ggml_free(ctx);
    check(loaded == unloaded, "module lifecycle");
    std::printf("STRIDED_FLOAT_CONTRACT API=%d stage=%s checks=%d state_writes=%d actual_GPU_calls=0\n",
        GGML_BACKEND_API_VERSION, after ? "after" : "before", contract_checks, state_writes);
}
