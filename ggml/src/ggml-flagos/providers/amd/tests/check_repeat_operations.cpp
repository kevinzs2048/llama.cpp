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
static amd_repeat_operation expected;
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
    check(expected_node != nullptr && gx == expected.grid_x && gy == 1 && gz == 1, "launch grid");
    check(bx == 128 && by == 1 && bz == 1, "launch threads");
    check(*static_cast<void **>(args[0]) == expected_node->src[0]->data, "source binding");
    if (expected.is_binary) {
        check(symbols.at(reinterpret_cast<uintptr_t>(function)).find("flagos_binary_repeat_strided_f32") == 0, "binary symbol");
        check(*static_cast<void **>(args[1]) == expected_node->src[1]->data, "repeat source binding");
        check(*static_cast<void **>(args[2]) == expected_node->data, "output binding");
        int slot = 3;
        for (const auto * values : {expected.binary.dimensions, expected.binary.repeat,
                                   expected.binary.source_strides, expected.binary.repeat_strides}) {
            for (int dim = 0; dim < 4; ++dim) check(*static_cast<int *>(args[slot++]) == values[dim], "binary scalar order");
        }
        check(*static_cast<int *>(args[19]) == expected.binary.elements, "element count");
    } else {
        check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected.kernel) == 0, "scalar/reduction symbol");
        check(*static_cast<void **>(args[1]) == expected_node->data, "output binding");
        check(*static_cast<int *>(args[2]) == (expected.columns ? expected.columns : expected.elements), "scalar count");
        if (!expected.columns) {
            check(std::memcmp(args[3], &expected.minimum, 4) == 0 && std::memcmp(args[4], &expected.maximum, 4) == 0, "clamp bounds bits");
        }
    }
    std::fill_n(static_cast<float *>(expected_node->data), ggml_nelements(expected_node), -123.0f);
    ++launches; return hipSuccess;
}

int main(int argc, char ** argv) {
    check(argc == 2, "repeat package");
    setenv("FLAGOS_AMD_REPEAT_OPS", "1", 1); setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    setenv("FLAGOS_AMD_FUSIONS", "none", 1); setenv("FLAGOS_AMD_KERNEL_PROFILE", "0", 1);
    amd_device_context device; device.arch = "gfx1150"; device.ordinal = 0; device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock modules");
    const char * names[] = {"flagos_add_repeat_strided_f32", "flagos_mul_repeat_strided_f32", "flagos_div_repeat_strided_f32",
                           "flagos_clamp_compare_f32", "flagos_sum_rows_f32_acc_f64"};
    int checks = 0;
    for (int kind = 0; kind < 5; ++kind) {
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[kind]));
        check(metadata != nullptr, "module"); const auto saved = *metadata;
        using mutation = std::function<void(ggml_tensor *)>;
        auto test = [&](const char * label, bool accepted, const mutation & modify, bool compute = false) {
            for (bool bound : {false, true}) {
                auto * ctx = ggml_init({1024 * 1024, nullptr, true});
                auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kind == 2 || kind == 4 ? 2 : 32,
                                             kind == 1 ? 2 : kind == 2 ? 512 : 16, kind == 1 ? 16 : 1, 1);
                ggml_tensor * out = nullptr;
                if (kind < 3) {
                    auto * y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kind == 0 ? 32 : 1, kind == 0 ? 1 : x->ne[1], x->ne[2], 1);
                    out = kind == 0 ? ggml_add(ctx, x, y) : kind == 1 ? ggml_mul(ctx, x, y) : ggml_div(ctx, x, y);
                    if (kind == 0) { x->nb[1] *= 2; x->nb[2] *= 2; x->nb[3] *= 2; }
                } else out = kind == 3 ? ggml_clamp(ctx, x, 1e-4f, INFINITY) : ggml_sum_rows(ctx, x);
                std::vector<float> a(ggml_nbytes(x) / 4 + 32, 3.0f), b(kind < 3 ? ggml_nbytes(out->src[1]) / 4 + 32 : 0, 2.0f);
                std::vector<float> result(ggml_nelements(out) + 32, 37.0f);
                if (kind == 3) { out->view_src = nullptr; out->view_offs = 0; }
                if (bound) { x->data = a.data() + 8; out->data = result.data() + 8; if (kind < 3) out->src[1]->data = b.data() + 8; }
                *metadata = saved; check(hipGetDeviceProperties(&device.props, 0) == hipSuccess, "fake properties"); modify(out);
                check(amd_device_supports_op(&device.device_iface, out) == accepted, label);
                if (compute && bound) {
                    check(amd_describe_repeat_operation(&device, out, &expected), "execution description");
                    check(std::strcmp(expected.kernel, names[kind]) == 0, "kernel selection"); expected_node = out;
                    amd_backend_context context(&device, nullptr); ggml_backend backend{}; backend.context = &context;
                    ggml_tensor * nodes[] = {out}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 1;
                    const int before = launches;
                    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS && launches == before + 1, "direct dispatch");
                    const bool in_place = out->data == x->data;
                    const auto & writes = in_place ? a : result;
                    check(std::all_of(writes.begin() + 8, writes.begin() + 8 + ggml_nelements(out), [](float v) { return v == -123.0f; }), "bound output writes");
                    check(std::all_of(writes.begin(), writes.begin() + 8, [in_place](float v) { return v == (in_place ? 3.0f : 37.0f); }) &&
                          std::all_of(writes.end() - 24, writes.end(), [in_place](float v) { return v == (in_place ? 3.0f : 37.0f); }), "output guards");
                    check((in_place || std::all_of(a.begin(), a.end(), [](float v) { return v == 3.0f; })) &&
                          std::all_of(b.begin(), b.end(), [](float v) { return v == 2.0f; }), "inputs untouched");
                    setenv("FLAGOS_AMD_REPEAT_OPS", "0", 1);
                    check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED && launches == before + 1, "policy change before writes");
                    setenv("FLAGOS_AMD_REPEAT_OPS", "1", 1); context.graph_plans.clear(); expected_node = nullptr;
                }
                ++checks; ggml_free(ctx);
            }
        };
        test("valid", true, [](auto *) {}, true);
        test("default off", false, [](auto *) { setenv("FLAGOS_AMD_REPEAT_OPS", "0", 1); });
        setenv("FLAGOS_AMD_REPEAT_OPS", "1", 1);
        test("unknown caps", true, [&](auto *) { device.props = {}; });
        test("known grid cap", false, [&](auto *) { device.props.maxGridSize[0] = 1; });
        test("wave mismatch", false, [&](auto *) { device.props.warpSize = 64; });
        test("thread cap", false, [&](auto *) { device.props.maxThreadsPerBlock = 32; });
        test("static LDS", false, [&](auto *) { metadata->static_shared_memory = 65537; });
        test("dynamic LDS", false, [&](auto *) { metadata->shared_memory = 65537; });
        test("wrong block", false, [&](auto *) { ++metadata->block_size; });
        test("wrong argument count", false, [&](auto *) { ++metadata->argument_count; });
        test("missing ABI slot", false, [&](auto *) { metadata->argument_abi.pop_back(); });
        test("pointer kind", false, [&](auto *) { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("pointer size", false, [&](auto *) { metadata->argument_abi[0].size = 4; });
        test("scalar kind", false, [&](auto *) { metadata->argument_abi[kind < 3 ? 3 : 2].kind = flagos_amd::kernel_argument_kind::global_buffer; });
        test("scalar size", false, [&](auto *) { metadata->argument_abi[kind < 3 ? 3 : 2].size = 8; });
        test("scratch", false, [&](auto *) { metadata->global_scratch_size = 1; });
        test("source dtype", false, [](auto * out) { out->src[0]->type = GGML_TYPE_F16; });
        test("output layout", false, [](auto * out) { out->nb[1] += 4; });
        test("source misalignment", false, [](auto * out) { out->src[0]->data = reinterpret_cast<void *>(UINT64_C(0x100000002)); });
        test("output misalignment", false, [](auto * out) { out->data = reinterpret_cast<void *>(UINT64_C(0x200000002)); });
        test("unallocated shared root", kind == 3, [](auto * out) { out->view_src = out->src[0]; if (out->data) out->data = out->src[0]->data; });
        test("partial output overlap", false, [](auto * out) { out->view_src = out->src[0]; out->view_offs = 4; if (out->src[0]->data) out->data = static_cast<float *>(out->src[0]->data) + 1; });
        if (kind < 3) {
            test("invalid repeat dimension", false, [](auto * out) { out->src[1]->ne[0] = 3; });
            test("row span overflow", false, [](auto * out) { out->src[0]->nb[1] = size_t(INT32_MAX) * 4; });
            test("fractional stride", false, [](auto * out) { out->src[0]->nb[1] += 1; });
            test("leading element stride", false, [](auto * out) { out->src[0]->nb[0] = 8; });
        } else if (kind == 3) {
            test("exact clamp in-place dispatch", true, [](auto * out) { out->view_src = out->src[0]; if (out->src[0]->data) out->data = out->src[0]->data; }, true);
            test("NaN bound", false, [](auto * out) { ggml_set_op_params_f32(out, 0, NAN); });
            test("reversed bound", false, [](auto * out) { ggml_set_op_params_f32(out, 0, INFINITY); ggml_set_op_params_f32(out, 1, -INFINITY); });
        } else {
            test("row beyond profile", false, [](auto * out) { out->src[0]->ne[0] = 1025; });
            test("output row width", false, [](auto * out) { out->ne[0] = 2; });
        }
        *metadata = saved;
    }
    device.aot.reset(); check(loaded == unloaded, "module lifecycle");
    std::printf("REPEAT_HOST_CONTRACT_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches);
}
