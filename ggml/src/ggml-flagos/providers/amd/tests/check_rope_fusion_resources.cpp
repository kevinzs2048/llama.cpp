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
static ggml_tensor * expected_rope = nullptr;
static ggml_tensor * expected_store = nullptr;
static const char * expected_symbol = nullptr;
static int expected_kind = 0;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_rope && symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected_symbol) == 0, "selected symbol");
    check(gx == (expected_kind == 0 ? 3 : 6) && gy == 1 && gz == 1, "fusion grid");
    auto ptr = [&](int i, void * value) { check(*static_cast<void **>(args[i]) == value, "pointer ABI"); };
    if (expected_kind == 0) {
        ptr(0, expected_rope->src[0]->data);ptr(1, expected_rope->src[1]->data);
        ptr(2, expected_store->src[1]->data);ptr(3, expected_store->data);
        check(*static_cast<int *>(args[4]) == 256 && *static_cast<int *>(args[5]) == 3 &&
            *static_cast<int *>(args[6]) == 7 && *static_cast<int *>(args[7]) == 128, "store geometry ABI");
    } else {
        auto * mul = expected_rope->src[0];auto * weight = mul->src[0] == expected_norm ? mul->src[1] : mul->src[0];
        ptr(0, expected_norm->src[0]->data);ptr(1, weight->data);ptr(2, expected_rope->src[1]->data);
        if (expected_kind == 2) ptr(3, expected_store->src[1]->data);
        ptr(expected_kind == 2 ? 4 : 3, expected_kind == 2 ? expected_store->data : expected_rope->data);
        const int scalar = expected_kind == 2 ? 5 : 4;
        check(*static_cast<int *>(args[scalar]) == 128 && *static_cast<int *>(args[scalar + 1]) == 2 &&
            *static_cast<int *>(args[scalar + 2]) == 3, "norm rope geometry ABI");
    }
    if (expected_kind == 1) std::fill_n(static_cast<float *>(expected_rope->data), 768, 42.f);
    else std::fill_n(static_cast<uint16_t *>(expected_store->data), 1792, uint16_t(0x7777));
    ++launches;return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2, "package");setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "rope_kv_store,rms_norm_mul_rope", 1);
    setenv("FLAGOS_AMD_ROPE_NEOX_RECURRENCE", "0", 1);
    amd_device_context device;device.arch = "gfx1150";device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1], 0, "gfx1150", true), "modules");int checks = 0;
    for (int kind = 0; kind < 3; ++kind) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 3);
        auto * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
        auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
        auto * norm = kind ? ggml_rms_norm(ctx, x, 1e-5f) : nullptr;
        auto * mul = kind ? ggml_mul(ctx, norm, weight) : nullptr;
        auto * rope = ggml_rope(ctx, kind ? mul : x, pos, 128, GGML_ROPE_TYPE_NEOX);
        auto * view = ggml_view_2d(ctx, rope, 256, 3, 256 * 4, 0);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3);
        auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 7);
        auto * store = ggml_set_rows(ctx, dst, view, ids);
        std::vector<float> xv(800, 31.f), wv(160, 33.f), nv(800, 35.f), mv(800, 37.f), rv(800, 39.f);
        std::vector<int32_t> pv(35, 1);std::vector<int64_t> iv(35, 1);std::vector<uint16_t> cv(1824, 0x3333);
        x->data = xv.data() + 8;weight->data = wv.data() + 8;pos->data = pv.data() + 8;
        if (norm) norm->data = nv.data() + 8;if (mul) mul->data = mv.data() + 8;
        rope->data = view->data = rv.data() + 8;ids->data = iv.data() + 8;store->data = dst->data = cv.data() + 8;
        const char * name = kind == 0 ? "flagos_rope_kv_store_f32_f16" : kind == 1 ? "flagos_rms_norm_mul_rope_neox_f32" : "flagos_rms_norm_mul_rope_kv_store_neox_f32_f16";
        auto * metadata = const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name));check(metadata, "metadata");const auto saved = *metadata;
        auto supported = [&] { return kind == 0 ? amd_supports_rope_kv_store(&device, rope, view, store)
            : amd_supports_rms_norm_mul_rope(&device, norm, mul, rope, kind == 2 ? view : nullptr, kind == 2 ? store : nullptr); };
        hipGetDeviceProperties(&device.props, 0);check(supported(), "valid fusion");++checks;
        auto * full = ggml_new_graph(ctx);ggml_build_forward_expand(full, kind == 1 ? rope : store);auto graph = ggml_graph_view(full, 0, full->n_nodes);
        flagos_pattern_candidate candidate;candidate.id = kind == 0 ? flagos_pattern_id::rope_kv_store : kind == 1 ? flagos_pattern_id::rms_norm_mul_rope : flagos_pattern_id::rms_norm_mul_rope_kv_store;
        candidate.node_indices = kind == 2 ? std::vector<int>{0, 1, 2, 3, 4} : std::vector<int>{0, 1, 2};candidate.required_output_node_indices = {kind == 2 ? 4 : 2};
        check(full->n_nodes == (kind == 2 ? 5 : 3), "topology");amd_backend_context context(&device, nullptr);
        auto query = [&] { return amd_query_fusion(&context, &graph, candidate).supported; };check(query(), "valid lowering");++checks;
        flagos_plan_step step;step.kind = flagos_execution_kind::pattern;step.candidate = candidate;step.implementation_id = kind == 0 ? 5 : kind == 1 ? 8 : 9;
        expected_norm = norm;expected_rope = rope;expected_store = store;expected_symbol = name;expected_kind = kind;
        check(amd_execute_fusion(&context, &graph, step), "fusion dispatch");
        if (kind == 1) check(std::all_of(rv.begin() + 8, rv.begin() + 776, [](float value) { return value == 42.f; }), "rope output");
        else check(std::all_of(cv.begin() + 8, cv.begin() + 1800, [](uint16_t value) { return value == 0x7777; }), "cache output");
        check(std::all_of(xv.begin(), xv.end(), [](float value) { return value == 31.f; }) &&
            std::all_of(wv.begin(), wv.end(), [](float value) { return value == 33.f; }) &&
            std::all_of(pv.begin(), pv.end(), [](int32_t value) { return value == 1; }) &&
            std::all_of(iv.begin(), iv.end(), [](int64_t value) { return value == 1; }), "readonly inputs");
        auto test = [&](const char * label, const std::function<void()> & mutate) {
            *metadata = saved;hipGetDeviceProperties(&device.props, 0);x->data = xv.data() + 8;weight->data = wv.data() + 8;mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
            check(supported() && query(), label);
#else
            check(!supported() && !query(), label);
#endif
            ++checks;
        };
        test("kernel threads", [&] { metadata->num_warps = 64; });
        test("kernel wave", [&] { metadata->warp_size = 64; });
        test("kernel LDS", [&] { metadata->shared_memory = 70000; });
        test("kernel static LDS", [&] { metadata->static_shared_memory = 70000; });
        test("kernel scratch", [&] { metadata->global_scratch_size = 1; });
        test("profile scratch", [&] { metadata->profile_scratch_size = 1; });
        test("argument count", [&] { metadata->argument_count += 1; });
        test("missing ABI", [&] { metadata->argument_abi.pop_back(); });
        test("pointer ABI", [&] { metadata->argument_abi[0].kind = flagos_amd::kernel_argument_kind::by_value; });
        test("scalar width", [&] { metadata->argument_abi.back().size = 8; });
        if (kind) {
            test("original norm input alignment", [&] { x->data = reinterpret_cast<void *>(UINT64_C(0x100000002)); });
            test("weight alignment", [&] { weight->data = reinterpret_cast<void *>(UINT64_C(0x200000002)); });
            test("fusion row grid", [&] { device.props.maxGridSize[0] = 3; });
        } else test("zero block", [&] { metadata->block_size = 0; });
        *metadata = saved;hipGetDeviceProperties(&device.props, 0);
        x->data = xv.data() + 8;weight->data = wv.data() + 8;
        if (kind == 2) {
            auto * unselected = const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_rope_kv_store_f32_f16"));
            const auto original = *unselected;unselected->global_scratch_size = 1;
            check(supported() && query(), "unselected simpler fusion profile does not invalidate selected RMS fusion");
            *unselected = original;++checks;
        }
        check(std::all_of(cv.begin(), cv.begin() + 8, [](uint16_t value) { return value == 0x3333; }) &&
            std::all_of(cv.begin() + 1800, cv.end(), [](uint16_t value) { return value == 0x3333; }), "cache guards");
        check(std::all_of(rv.begin(), rv.begin() + 8, [](float value) { return value == 39.f; }) &&
            std::all_of(rv.begin() + 776, rv.end(), [](float value) { return value == 39.f; }), "rope guards");
        expected_norm = expected_rope = expected_store = nullptr;ggml_free(ctx);
    }
    device.aot.reset();check(loaded == unloaded && launches == 3 && allocations == 0, "lifecycle");
    std::printf("ROPE_FUSION_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches);
}
