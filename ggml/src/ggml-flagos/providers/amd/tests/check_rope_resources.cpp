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
static int allocations = 0, releases = 0;
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

static ggml_tensor * expected_node = nullptr;
static int expected_kind = 0;
static const char * names[] = { "flagos_rope_ggml_recurrence_f32", "flagos_rope_ggml_recurrence_f32",
    "flagos_rope_ggml_neox_recurrence_f32", "flagos_rope_neox_f32", "flagos_mrope_f32",
    "flagos_rope_ggml_recurrence_f32" };
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_node != nullptr, "expected node");
    const auto & name = symbols.at(reinterpret_cast<uintptr_t>(function));
    check(name.find(names[expected_kind]) == 0, "selected symbol");
    check(gx == (expected_kind == 3 ? 2 : 6) && gy == 1 && gz == 1, "row or pair grid");
    check(*static_cast<void **>(args[0]) == expected_node->src[0]->data, "input binding");
    check(*static_cast<void **>(args[1]) == expected_node->src[1]->data, "position binding");
    if (expected_kind == 3 || expected_kind == 4) {
        check(*static_cast<void **>(args[2]) == expected_node->data, "legacy output binding");
        check(*static_cast<int *>(args[3]) == (expected_kind == 3 ? 768 : 128), "legacy count or width");
    } else {
        check(*static_cast<void **>(args[2]) == (expected_node->src[2] ? expected_node->src[2]->data : expected_node->src[0]->data), "factor binding");
        check(*static_cast<void **>(args[3]) == expected_node->data, "recurrence output binding");
        check(*static_cast<int *>(args[4]) == 128 && *static_cast<int *>(args[5]) == 2 && *static_cast<int *>(args[6]) == 3, "recurrence shape");
        check(*static_cast<int *>(args[7]) == (expected_kind == 2 ? 64 : 128), "rotary dimensions");
    }
    std::fill_n(static_cast<float *>(expected_node->data), ggml_nelements(expected_node), 42.f);
    ++launches;
    return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2, "package");
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    setenv("FLAGOS_AMD_FUSIONS", "none", 1);
    amd_device_context device;
    device.arch = "gfx1150";device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "modules");
    int checks = 0;
    for (int kind = 0; kind < 6; ++kind) {
        setenv("FLAGOS_AMD_ROPE_NEOX_RECURRENCE", kind == 2 ? "1" : "0", 1);
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 3);
        auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kind == 4 || kind == 5 ? 12 : 3);
        auto * factors = kind == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64) : nullptr;
        ggml_tensor * out;
        if (kind == 4 || kind == 5) {
            int sections[4] = {16, 16, 16, 16};
            out = ggml_rope_multi(ctx, x, positions, nullptr, 128, sections,
                kind == 4 ? GGML_ROPE_TYPE_MROPE : GGML_ROPE_TYPE_IMROPE, 0, 10000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
        } else {
            out = ggml_rope_ext(ctx, x, positions, factors, kind == 2 ? 64 : 128,
                kind == 2 || kind == 3 ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL,
                0, 10000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
        }
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[kind]));
        check(metadata != nullptr, "metadata");const auto saved = *metadata;
        auto supported = [&] { return kind == 3 ? amd_supports_rope_neox(&device, out)
            : kind == 4 ? amd_supports_mrope(&device, out) : amd_supports_ggml_rope(&device, out); };
        hipGetDeviceProperties(&device.props, 0);check(supported(), "valid unbound");++checks;
        std::vector<float> input(800, 37.f), output(800, 47.f), f(96, 1.f);
        std::vector<int32_t> pos(44, 1);
        x->data = input.data() + 8;out->data = output.data() + 8;positions->data = pos.data() + 8;
        if (factors) factors->data = f.data() + 8;
        check(supported(), "valid bound");++checks;
        expected_node = out;expected_kind = kind;
        {
            amd_backend_context context(&device, nullptr);ggml_backend backend{};backend.context = &context;
            ggml_tensor * nodes[] = {out};ggml_cgraph graph{};graph.nodes = nodes;graph.n_nodes = 1;
            check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_SUCCESS, "dispatch");
        }
        expected_node = nullptr;
        check(std::all_of(output.begin() + 8, output.begin() + 776, [](float v) { return v == 42.f; }), "output writes");
        check(std::all_of(output.begin(), output.begin() + 8, [](float v) { return v == 47.f; }) &&
            std::all_of(output.begin() + 776, output.end(), [](float v) { return v == 47.f; }), "output guards");
        check(std::all_of(input.begin(), input.end(), [](float v) { return v == 37.f; }) &&
            std::all_of(f.begin(), f.end(), [](float v) { return v == 1.f; }) &&
            std::all_of(pos.begin(), pos.end(), [](int32_t v) { return v == 1; }), "readonly inputs");
        auto test = [&](const char * label, bool old_accept, const std::function<void()> & mutate) {
            for (bool bound : {false, true}) {
                *metadata = saved;hipGetDeviceProperties(&device.props, 0);
                x->data = bound ? reinterpret_cast<void *>(UINT64_C(0x100000000)) : nullptr;
                out->data = bound ? reinterpret_cast<void *>(UINT64_C(0x200000000)) : nullptr;
                positions->data = bound ? reinterpret_cast<void *>(UINT64_C(0x300000000)) : nullptr;
                if (factors) factors->data = bound ? reinterpret_cast<void *>(UINT64_C(0x400000000)) : nullptr;
                mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
                check(supported() == old_accept, label);
#else
                (void)old_accept;check(!supported(), label);
                check(!amd_device_supports_op(&device.device_iface, out), "placement refusal");
#endif
                ++checks;
            }
        };
        test("threads", true, [&] { device.props.maxThreadsPerBlock = 64; });
        test("wave", true, [&] { device.props.warpSize = 64; });
        test("grid", true, [&] { device.props.maxGridSize[0] = 1; });
        test("dynamic LDS", kind != 2, [&] { metadata->shared_memory = 70000; });
        test("static LDS", kind != 2, [&] { metadata->static_shared_memory = 70000; });
        test("scratch", kind != 2, [&] { metadata->global_scratch_size = 1; });
        test("profile scratch", kind != 2, [&] { metadata->profile_scratch_size = 1; });
        test("argument count", kind == 3 || kind == 4, [&] { metadata->argument_count += 1; });
        test("missing ABI", true, [&] { metadata->argument_abi.pop_back(); });
        test("pointer ABI", true, [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("scalar width", true, [&] { metadata->argument_abi.back().size = 8; });
        if (kind == 3 || kind == 4) {
            test("input alignment", true, [&] { x->data = reinterpret_cast<void *>(UINT64_C(0x100000002)); });
            test("position alignment", true, [&] { positions->data = reinterpret_cast<void *>(UINT64_C(0x300000002)); });
            test("output alignment", true, [&] { out->data = reinterpret_cast<void *>(UINT64_C(0x200000002)); });
        }
        if (kind == 3) test("zero block", true, [&] { metadata->block_size = 0; });
        *metadata = saved;ggml_free(ctx);
    }
    {
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        const int64_t tokens = static_cast<int64_t>(INT32_MAX) / 4 + 1;
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, tokens);
        auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens * 4);
        int sections[4] = {1, 0, 0, 0};
        auto * out = ggml_rope_multi(ctx, x, positions, nullptr, 2, sections,
            GGML_ROPE_TYPE_MROPE, 0, 10000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
        for (bool bound : {false, true}) {
            hipGetDeviceProperties(&device.props, 0);
            x->data = bound ? reinterpret_cast<void *>(UINT64_C(0x1000000000)) : nullptr;
            out->data = bound ? reinterpret_cast<void *>(UINT64_C(0x2000000000)) : nullptr;
            positions->data = bound ? reinterpret_cast<void *>(UINT64_C(0x3000000000)) : nullptr;
#ifdef FLAGOS_BEFORE_FIXTURE
            check(amd_supports_mrope(&device, out), "old oversized position span");
#else
            check(!amd_supports_mrope(&device, out), "oversized position span");
            check(!amd_device_supports_op(&device.device_iface, out), "oversized positions placement");
#endif
            ++checks;
        }
        ggml_free(ctx);
    }
    device.aot.reset();check(loaded == unloaded && launches == 6 && allocations == 0, "lifecycle");
    std::printf("ROPE_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches);
}
