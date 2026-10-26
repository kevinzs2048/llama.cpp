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
        unsigned bx, unsigned by, unsigned bz, unsigned, hipStream_t, void ** args, void **) {
    check(expected_node && gx == (ggml_nelements(expected_node) + 255) / 256 && gy == 1 && gz == 1, "grid");
    check(bx == 128 && by == 1 && bz == 1, "threads");
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find("flagos_tanh_f32") == 0, "symbol");
    check(*static_cast<void **>(args[0]) == expected_node->src[0]->data &&
          *static_cast<void **>(args[1]) == expected_node->data &&
          *static_cast<int *>(args[2]) == ggml_nelements(expected_node), "ordered ABI bindings");
    std::fill_n(static_cast<float *>(expected_node->data), ggml_nelements(expected_node), -123.0f);
    ++launches; return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2, "package"); setenv("FLAGOS_AMD_TANH", "1", 1);
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1); setenv("FLAGOS_AMD_FUSIONS", "none", 1);
    amd_device_context device; device.arch = "gfx1150"; device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock modules");
    auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_tanh_f32"));
    check(metadata != nullptr, "metadata"); const auto saved = *metadata; int checks = 0;
    using mutation = std::function<void(ggml_tensor *)>;
    auto test = [&](const char * name, bool accepted, const mutation & modify, bool compute = false) {
        for (bool bound : {false, true}) {
            auto * ctx = ggml_init({1024 * 1024, nullptr, true});
            auto * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 257, 3, 2, 1);
            auto * out = ggml_tanh(ctx, input);
            std::vector<float> source(ggml_nelements(input) + 32, 3.0f), result(ggml_nelements(out) + 32, 37.0f);
            if (bound) { input->data = source.data() + 8; out->data = result.data() + 8; }
            *metadata = saved; check(hipGetDeviceProperties(&device.props, 0) == hipSuccess, "properties"); modify(out);
            check(amd_device_supports_op(&device.device_iface, out) == accepted, name);
            if (compute && bound) {
                expected_node = out; amd_backend_context context(&device, nullptr); ggml_backend backend{}; backend.context = &context;
                ggml_tensor * nodes[] = {out}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 1;
                const int before = launches;
                check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS && launches == before + 1, "compute");
                const bool alias = out->data == input->data; const auto & writes = alias ? source : result;
                check(std::all_of(writes.begin() + 8, writes.end() - 24, [](float v) { return v == -123.0f; }), "bound writes");
                check(std::all_of(writes.begin(), writes.begin() + 8, [alias](float v) { return v == (alias ? 3.0f : 37.0f); }) &&
                      std::all_of(writes.end() - 24, writes.end(), [alias](float v) { return v == (alias ? 3.0f : 37.0f); }), "guards");
                check(alias || std::all_of(source.begin(), source.end(), [](float v) { return v == 3.0f; }), "readonly source");
                setenv("FLAGOS_AMD_TANH", "0", 1);
                check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED && launches == before + 1, "policy rejection before writes");
                setenv("FLAGOS_AMD_TANH", "1", 1); expected_node = nullptr;
            }
            ++checks; ggml_free(ctx);
        }
    };
    test("valid", true, [](auto *) {}, true);
    test("exact alias", true, [](auto * out) { out->view_src = out->src[0]; if (out->src[0]->data) out->data = out->src[0]->data; }, true);
    test("shifted alias", false, [](auto * out) { out->view_src = out->src[0]; out->view_offs = 4; if (out->src[0]->data) out->data = static_cast<float *>(out->src[0]->data) + 1; });
    test("default off", false, [](auto *) { setenv("FLAGOS_AMD_TANH", "0", 1); }); setenv("FLAGOS_AMD_TANH", "1", 1);
    test("unknown caps", true, [&](auto *) { device.props = {}; });
    test("grid cap", false, [&](auto *) { device.props.maxGridSize[0] = 1; });
    test("thread cap", false, [&](auto *) { device.props.maxThreadsPerBlock = 32; });
    test("wave size", false, [&](auto *) { device.props.warpSize = 64; });
    test("static LDS", false, [&](auto *) { metadata->static_shared_memory = 65537; });
    test("dynamic LDS", false, [&](auto *) { metadata->shared_memory = 65537; });
    test("scratch", false, [&](auto *) { metadata->global_scratch_size = 1; });
    test("profile scratch", false, [&](auto *) { metadata->profile_scratch_size = 1; });
    test("block", false, [&](auto *) { metadata->block_size = 128; });
    test("argument count", false, [&](auto *) { metadata->argument_count = 4; });
    test("missing ABI", false, [&](auto *) { metadata->argument_abi.pop_back(); });
    test("pointer kind", false, [&](auto *) { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
    test("pointer size", false, [&](auto *) { metadata->argument_abi[0].size = 4; });
    test("scalar kind", false, [&](auto *) { metadata->argument_abi[2].kind = flagos_amd::kernel_argument_kind::global_buffer; });
    test("scalar size", false, [&](auto *) { metadata->argument_abi[2].size = 8; });
    test("input type", false, [](auto * out) { out->src[0]->type = GGML_TYPE_F16; });
    test("output type", false, [](auto * out) { out->type = GGML_TYPE_F16; });
    test("shape", false, [](auto * out) { ++out->ne[1]; });
    test("input stride", false, [](auto * out) { out->src[0]->nb[1] += 4; });
    test("output stride", false, [](auto * out) { out->nb[1] += 4; });
    test("input misalignment", false, [](auto * out) { out->src[0]->data = reinterpret_cast<void *>(UINT64_C(0x100000002)); });
    test("output misalignment", false, [](auto * out) { out->data = reinterpret_cast<void *>(UINT64_C(0x200000002)); });
    *metadata = saved; device.aot.reset(); check(loaded == unloaded, "module lifecycle");
    std::printf("TANH_HOST_CONTRACT_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches);
}
