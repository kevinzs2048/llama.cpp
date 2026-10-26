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


static ggml_tensor * expected_norm = nullptr;
static ggml_tensor * expected_mul = nullptr;
static ggml_tensor * expected_add = nullptr;
static int expected_kind = 0;
static const char * expected_symbol = nullptr;
static bool fail_submission = false;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_norm && expected_mul, "expected tensors");
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected_symbol) == 0, "selected symbol");
    check(gx == 3 && gy == 1 && gz == 1, "row grid");
    auto * weight = expected_mul->src[0] == expected_norm ? expected_mul->src[1] : expected_mul->src[0];
    auto pointer = [&](int slot, void * value) { check(*static_cast<void **>(args[slot]) == value, "fusion pointer ABI"); };
    int scalars;
    if (expected_kind == 2 || expected_kind == 3) {
        pointer(0, expected_mul->data);pointer(1, expected_add ? expected_add->src[0]->data : expected_norm->src[0]->data);
        if (expected_add) pointer(2, expected_add->src[1]->data);
        pointer(expected_add ? 3 : 2, weight->data);scalars = expected_add ? 4 : 3;
    } else {
        pointer(0, expected_kind == 4 ? expected_add->data : expected_norm->data);
        pointer(1, expected_mul->data);pointer(2, expected_add ? expected_add->src[0]->data : expected_norm->src[0]->data);
        if (expected_add) pointer(3, expected_add->src[1]->data);
        pointer(expected_add ? 4 : 3, weight->data);scalars = expected_add ? 5 : 4;
    }
    check(*static_cast<int *>(args[scalars]) == 256 && *static_cast<float *>(args[scalars + 1]) == 1e-5f, "fusion scalar ABI");
    ++launches;
    if (fail_submission) {
        static_cast<float *>(expected_norm->data)[0] = -98.f;
        static_cast<float *>(expected_mul->data)[0] = -99.f;
        return hipErrorInvalidValue;
    }
    if (expected_kind != 2 && expected_kind != 3) {
        std::fill_n(static_cast<float *>(expected_kind == 4 ? expected_add->data : expected_norm->data), 768, 41.f);
    }
    std::fill_n(static_cast<float *>(expected_mul->data), 768, 42.f);
    return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "package");
    const bool dedicated_only = argc == 3;
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "rms_norm_mul,add_rms_norm_mul", 1);
    amd_device_context device;device.arch = "gfx1150";device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "modules");int checks = 0;
    for (int kind = 0; kind < 5; ++kind) {
        if (dedicated_only && kind < 2) continue;
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);
        auto * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);
        auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        auto * add = kind == 1 || kind == 3 || kind == 4 ? ggml_add(ctx, x, bias) : nullptr;
        auto * norm = ggml_rms_norm(ctx, add ? add : x, 1e-5f);
        auto * mul = kind == 2 || kind == 3 ? ggml_mul_inplace(ctx, norm, weight) : ggml_mul(ctx, norm, weight);
        std::vector<float> xv(800, 31.f), bv(800, 33.f), wv(288, 35.f), av(800, 37.f), nv(800, 39.f), mv(800, 43.f);
        x->data = xv.data() + 8;bias->data = bv.data() + 8;weight->data = wv.data() + 8;
        if (add) add->data = av.data() + 8;
        norm->data = nv.data() + 8;mul->data = kind == 2 || kind == 3 ? norm->data : mv.data() + 8;
        auto supported = [&] {
            if (kind == 0) return amd_supports_rms_norm_mul(&device, norm, mul);
            if (kind == 1) return amd_supports_add_rms_norm_mul(&device, add, norm, mul);
            if (kind == 2) return amd_supports_rms_norm_mul_inplace(&device, norm, mul);
            if (kind == 3) return amd_supports_add_rms_norm_mul_inplace(&device, add, norm, mul);
            return amd_supports_add_rms_norm_mul_residual(&device, add, norm, mul);
        };
        const char * name = kind == 0 ? "flagos_rms_norm_mul_f32" : kind == 1 ? "flagos_add_rms_norm_mul_f32"
            : kind == 2 ? amd_rms_norm_mul_inplace_kernel(&device, 256) : kind == 3 ? "flagos_add_rms_norm_mul_inplace_f32"
            : amd_add_rms_norm_mul_residual_kernel(&device, 256);
        check(name != nullptr, "kernel selection");
        std::vector<flagos_amd::kernel_metadata *> profiles;
        profiles.push_back(const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name)));
        if (kind == 4) {
            for (const char * alternative : { "flagos_add_rms_norm_mul_residual_f32_narrow", "flagos_add_rms_norm_mul_residual_f32" }) {
                auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(alternative));
                if (metadata && std::find(profiles.begin(), profiles.end(), metadata) == profiles.end()) profiles.push_back(metadata);
            }
        }
        std::vector<flagos_amd::kernel_metadata> saved;
        for (auto * p : profiles) { check(p != nullptr, "metadata");saved.push_back(*p); }
        hipGetDeviceProperties(&device.props, 0);
#ifdef FLAGOS_BEFORE_FIXTURE
        if (dedicated_only && kind != 4) {
            check(!supported(), "old missing unselected profile dependency");++checks;ggml_free(ctx);continue;
        }
#endif
        check(supported(), "valid bound");++checks;
        auto * full = ggml_new_graph(ctx);ggml_build_forward_expand(full, mul);
        check(full->n_nodes == (add ? 3 : 2), "graph topology");auto graph = ggml_graph_view(full, 0, full->n_nodes);
        flagos_pattern_candidate candidate;candidate.id = add ? flagos_pattern_id::add_rms_norm_mul : flagos_pattern_id::rms_norm_mul;
        candidate.node_indices = add ? std::vector<int>{0, 1, 2} : std::vector<int>{0, 1};
        candidate.required_output_node_indices = kind == 0 ? std::vector<int>{0, 1} : kind == 1 ? std::vector<int>{1, 2}
            : kind == 2 ? std::vector<int>{1} : kind == 3 ? std::vector<int>{2} : std::vector<int>{0, 2};
        amd_backend_context context(&device, nullptr);
        auto query = [&] { return amd_query_fusion(&context, &graph, candidate).supported; };
        check(query(), "valid required outputs");++checks;
        flagos_plan_step step;step.kind = flagos_execution_kind::pattern;step.candidate = candidate;
        step.implementation_id = kind == 4 ? 10 : add ? 2 : 1;
        expected_norm = norm;expected_mul = mul;expected_add = add;expected_kind = kind;expected_symbol = name;
        check(amd_execute_fusion(&context, &graph, step), "selected fusion dispatch");
        const auto & output = kind == 2 || kind == 3 ? nv : mv;
        check(std::all_of(output.begin() + 8, output.begin() + 776, [](float value) { return value == 42.f; }), "terminal output");
        check(std::all_of(output.begin(), output.begin() + 8, [&](float value) { return value == (kind == 2 || kind == 3 ? 39.f : 43.f); }) &&
            std::all_of(output.begin() + 776, output.end(), [&](float value) { return value == (kind == 2 || kind == 3 ? 39.f : 43.f); }), "terminal guards");
        if (kind == 0 || kind == 1) check(std::all_of(nv.begin() + 8, nv.begin() + 776, [](float value) { return value == 41.f; }), "required norm output");
        if (kind == 4) check(std::all_of(av.begin() + 8, av.begin() + 776, [](float value) { return value == 41.f; }) &&
            std::all_of(nv.begin(), nv.end(), [](float value) { return value == 39.f; }), "residual and unused norm");
        check(std::all_of(xv.begin(), xv.end(), [](float value) { return value == 31.f; }) &&
            std::all_of(bv.begin(), bv.end(), [](float value) { return value == 33.f; }) &&
            std::all_of(wv.begin(), wv.end(), [](float value) { return value == 35.f; }), "readonly inputs");
        auto restore = [&] {
            for (size_t i = 0; i < profiles.size(); ++i) *profiles[i] = saved[i];
            hipGetDeviceProperties(&device.props, 0);
            x->data = reinterpret_cast<void *>(UINT64_C(0x100000000));bias->data = reinterpret_cast<void *>(UINT64_C(0x200000000));
            weight->data = reinterpret_cast<void *>(UINT64_C(0x300000000));norm->data = reinterpret_cast<void *>(UINT64_C(0x400000000));
            mul->data = kind == 2 || kind == 3 ? norm->data : reinterpret_cast<void *>(UINT64_C(0x500000000));
            if (add) add->data = reinterpret_cast<void *>(UINT64_C(0x600000000));
        };
        auto test = [&](const char * label, const std::function<void(flagos_amd::kernel_metadata *)> & mutate) {
            restore();for (auto * p : profiles) mutate(p);
#ifdef FLAGOS_BEFORE_FIXTURE
            check(supported(), label);check(query(), "old lowering accepted invalid resources");
#else
            check(!supported(), label);check(!query(), "lowering rejected before writes");
#endif
            ++checks;
        };
        test("threads", [&](auto *) { device.props.maxThreadsPerBlock = 64; });
        test("wave", [&](auto *) { device.props.warpSize = 64; });
        test("grid", [&](auto *) { device.props.maxGridSize[0] = 1; });
        test("LDS", [&](auto * p) { p->shared_memory = 70000; });
        test("static LDS", [&](auto * p) { p->static_shared_memory = 70000; });
        test("scratch", [&](auto * p) { p->global_scratch_size = 1; });
        test("profile scratch", [&](auto * p) { p->profile_scratch_size = 1; });
        test("argument count", [&](auto * p) { p->argument_count += 1; });
        test("missing ABI", [&](auto * p) { p->argument_abi.pop_back(); });
        test("pointer ABI", [&](auto * p) { p->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("scalar width", [&](auto * p) { p->argument_abi.back().size = 8; });
        test("input alignment", [&](auto *) { x->data = reinterpret_cast<void *>(UINT64_C(0x100000002)); });
        test("weight alignment", [&](auto *) { weight->data = reinterpret_cast<void *>(UINT64_C(0x300000002)); });
        test("output alignment", [&](auto *) { mul->data = reinterpret_cast<void *>(UINT64_C(0x500000002));if (kind == 2 || kind == 3) norm->data = mul->data; });
        restore();
#ifndef FLAGOS_BEFORE_FIXTURE
        if (kind == 4 && profiles.size() == 2) {
            profiles[0]->global_scratch_size = 1;
            check(supported() && query(), "legal residual alternative");
            check(std::strcmp(amd_add_rms_norm_mul_residual_kernel(&device, 256), profiles[1]->symbol.c_str()) == 0, "resource-aware alternative selection");
            ++checks;restore();
        }
#endif
        if (!dedicated_only && kind >= 1 && kind <= 3) {
            auto * unrelated = const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_rms_norm_mul_f32"));
            const auto original = *unrelated;unrelated->block_size = 0;
#ifdef FLAGOS_BEFORE_FIXTURE
            check(!supported() && !query(), "old unselected profile coupling");
#else
            check(supported() && query(), "unselected profile does not invalidate selected implementation");
#endif
            *unrelated = original;++checks;
        }
        if (kind == 0) {
            std::fill(nv.begin(), nv.end(), 39.f);std::fill(mv.begin(), mv.end(), 43.f);
            x->data = xv.data() + 8;weight->data = wv.data() + 8;norm->data = nv.data() + 8;mul->data = mv.data() + 8;
            norm->flags |= GGML_TENSOR_FLAG_OUTPUT;
            expected_kind = 0;expected_symbol = "flagos_rms_norm_mul_f32";fail_submission = true;
            ggml_backend backend{};backend.context = &context;const int prior = launches;
            check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED, "post-submit graph failure");
            check(launches == prior + 1 && mv[8] == -99.f && nv[8] == -98.f &&
                mv[9] == 43.f && nv[9] == 39.f, "no retry after partial writes");
            fail_submission = false;++checks;
        }
        restore();expected_norm = expected_mul = expected_add = nullptr;ggml_free(ctx);
    }
    device.aot.reset();check(loaded == unloaded && launches == (dedicated_only ?
#ifdef FLAGOS_BEFORE_FIXTURE
        1
#else
        3
#endif
        : 6) && allocations == 0, "lifecycle");
    std::printf("NORMALIZATION_FUSION_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches);
}
