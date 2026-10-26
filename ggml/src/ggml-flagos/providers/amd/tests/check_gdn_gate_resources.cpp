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
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned, unsigned, unsigned,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void **, void **) {
    check(false, "no dispatch in capability fixture"); return hipErrorInvalidValue;
}
int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "package and stage");
    const bool after = argc == 2 || std::atoi(argv[2]) != 0;
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "attention_output_gate,gdn_gate_projections", 1);
    amd_device_context device; device.arch = "gfx1150"; device.ordinal = 0;
    device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "modules");
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 17);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    auto * scale = ggml_dup_tensor(ctx, bias);
    auto * add = ggml_add(ctx, x, bias);
    auto * activation = ggml_softplus(ctx, add);
    auto * mul = ggml_mul(ctx, activation, scale);
    auto * alpha_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 2560, 32);
    auto * beta_weights = ggml_dup_tensor(ctx, alpha_weights);
    auto * gate_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
    auto * alpha_projection = ggml_mul_mat(ctx, alpha_weights, gate_x);
    auto * beta_projection = ggml_mul_mat(ctx, beta_weights, gate_x);
    auto * beta_sigmoid = ggml_sigmoid(ctx, beta_projection);
    auto * gate_add = ggml_add(ctx, alpha_projection, bias);
    auto * gate_softplus = ggml_softplus(ctx, gate_add);
    auto * gate_mul = ggml_mul(ctx, gate_softplus, scale);
    amd_gdn_gate_projection_match match;
    match.alpha_projection = alpha_projection; match.beta_projection = beta_projection;
    match.alpha_bias = bias; match.alpha_scale = scale;
    match.alpha_mul = gate_mul; match.beta_sigmoid = beta_sigmoid;
    std::vector<ggml_tensor *> tensors { x, bias, scale, add, activation, mul, alpha_weights,
        beta_weights, gate_x, alpha_projection, beta_projection, beta_sigmoid, gate_add, gate_softplus, gate_mul };
    std::vector<std::vector<uint8_t>> buffers;
    for (auto * tensor : tensors) {
        buffers.emplace_back(ggml_nbytes(tensor) + 64, 0);
        tensor->data = buffers.back().data();
        check(reinterpret_cast<uintptr_t>(tensor->data) % 16 == 0, "aligned host allocation");
    }
    ggml_tensor * alpha_nodes[] = { add, activation, mul };
    ggml_cgraph alpha_graph{}; alpha_graph.nodes = alpha_nodes; alpha_graph.n_nodes = 3;
    flagos_pattern_candidate alpha_candidate;
    alpha_candidate.id = flagos_pattern_id::attention_output_gate;
    alpha_candidate.node_indices = { 0, 1, 2 }; alpha_candidate.required_output_node_indices = { 2 };
    ggml_tensor * gate_nodes[] = { beta_projection, beta_sigmoid, alpha_projection, gate_add, gate_softplus, gate_mul };
    ggml_cgraph gate_graph{}; gate_graph.nodes = gate_nodes; gate_graph.n_nodes = 6;
    flagos_pattern_candidate gate_candidate;
    gate_candidate.id = flagos_pattern_id::gdn_gate_projections;
    gate_candidate.node_indices = { 0, 1, 2, 3, 4, 5 }; gate_candidate.required_output_node_indices = { 1, 5 };
    amd_backend_context context(&device, nullptr);
    int checks = 0;
    for (int index = 0; index < 2; ++index) {
        const char * name = index == 0 ? "flagos_gdn_alpha_gate_f32" : "flagos_gdn_q8_gate_projections_f32";
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name));
        check(metadata != nullptr, "gate metadata"); const auto saved = *metadata;
        auto support = [&] { return index == 0 ? amd_supports_gdn_alpha_gate(&device, add, activation, mul)
            : amd_supports_gdn_gate_projections(&device, match); };
        auto query = [&] { return amd_query_fusion(&context, index == 0 ? &alpha_graph : &gate_graph,
            index == 0 ? alpha_candidate : gate_candidate).supported; };
        auto reset = [&] {
            *metadata = saved; hipGetDeviceProperties(&device.props, 0);
            for (size_t i = 0; i < tensors.size(); ++i) tensors[i]->data = buffers[i].data();
        };
        auto test = [&](const char * label, const std::function<void()> & change) {
            reset(); change(); check(support() == !after, label); check(query() == !after, "query agrees with capability"); checks += 2;
        };
        reset(); check(support() && query(), "legal bound gate"); checks += 2;
        test("argument count", [&] { ++metadata->argument_count; });
        test("missing typed ABI", [&] { metadata->argument_abi.pop_back(); });
        test("pointer kind", [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("scalar size", [&] { metadata->argument_abi.back().size = 8; });
        test("global scratch", [&] { metadata->global_scratch_size = 1; });
        test("profile scratch", [&] { metadata->profile_scratch_size = 1; });
        test("dynamic LDS", [&] { metadata->shared_memory = 65537; });
        test("static LDS", [&] { metadata->static_shared_memory = 65537; });
        test("device grid X", [&] { device.props.maxGridSize[0] = 1; });
        test("device threads", [&] { device.props.maxThreadsPerBlock = 16; });
        test("device wave", [&] { device.props.warpSize = 64; });
        auto alignment = [&](ggml_tensor * tensor) {
            test("compiled pointer alignment", [&] {
                tensor->data = static_cast<uint8_t *>(tensor->data) + (tensor->type == GGML_TYPE_Q8_0 ? 2 : 4);
            });
        };
        for (auto * tensor : index == 0 ? std::vector<ggml_tensor *> { x, bias, scale, mul }
            : std::vector<ggml_tensor *> { gate_x, alpha_weights, beta_weights, bias, scale, gate_mul, beta_sigmoid }) {
            alignment(tensor);
        }
        reset(); for (auto * tensor : tensors) tensor->data = nullptr;
        check(support() && query(), "legal before allocation"); checks += 2;
        reset();
    }
    device.aot.reset(); ggml_free(ctx);
    check(loaded == unloaded, "module lifecycle");
    std::printf("GDN_GATE_CONTRACT API=%d stage=%s checks=%d actual_GPU_calls=0\n",
        GGML_BACKEND_API_VERSION, after ? "after" : "before", checks);
}
