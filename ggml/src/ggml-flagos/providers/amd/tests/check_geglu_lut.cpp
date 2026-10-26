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
static int allocations = 0, releases = 0, uploads = 0;
static bool fail_allocation = false, fail_upload = false, fail_launch = false;
static std::map<void *, size_t> live_allocations;
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
extern "C" hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
extern "C" hipError_t hipMalloc(void ** ptr, size_t size) {
    if (fail_allocation) { *ptr = nullptr; return hipErrorOutOfMemory; }
    *ptr = std::malloc(size); if (*ptr) { ++allocations; live_allocations[*ptr] = size; }
    return *ptr ? hipSuccess : hipErrorOutOfMemory;
}
extern "C" hipError_t hipMemcpy(void * dst, const void * src, size_t size, hipMemcpyKind kind) {
    check(live_allocations.count(dst) && live_allocations.at(dst) == size, "asset allocation");
    check(kind == hipMemcpyHostToDevice && size == 131072, "asset upload contract");
    if (fail_upload) { std::memcpy(dst, src, size / 2); return hipErrorUnknown; }
    std::memcpy(dst, src, size); ++uploads; return hipSuccess;
}
extern "C" hipError_t hipFree(void * ptr) { check(live_allocations.erase(ptr) == 1, "live allocation release"); std::free(ptr); ++releases; return hipSuccess; }
struct captured_call { void * table; void * output; size_t count; };
static bool mock_capture_active = false;
static captured_call pending_capture {};
static std::map<uintptr_t, captured_call> captured_graphs, graph_executables;
static int captures = 0, replays = 0, graph_destroys = 0, exec_destroys = 0;
extern "C" hipError_t hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode) {
    check(!mock_capture_active, "capture nesting"); mock_capture_active = true; ++captures; return hipSuccess;
}
extern "C" hipError_t hipStreamEndCapture(hipStream_t, hipGraph_t * graph) {
    check(mock_capture_active, "active capture"); mock_capture_active = false;
    captured_graphs[captures] = pending_capture; *graph = reinterpret_cast<hipGraph_t>(uintptr_t(captures));
    return hipSuccess;
}
extern "C" hipError_t hipGraphInstantiate(hipGraphExec_t * executable, hipGraph_t graph, hipGraphNode_t *, char *, size_t) {
    const auto id = reinterpret_cast<uintptr_t>(graph);
    graph_executables[id] = captured_graphs.at(id); *executable = reinterpret_cast<hipGraphExec_t>(id); return hipSuccess;
}
extern "C" hipError_t hipGraphDestroy(hipGraph_t graph) {
    check(captured_graphs.erase(reinterpret_cast<uintptr_t>(graph)) == 1, "graph destruction"); ++graph_destroys; return hipSuccess;
}
extern "C" hipError_t hipGraphLaunch(hipGraphExec_t executable, hipStream_t) {
    const auto & call = graph_executables.at(reinterpret_cast<uintptr_t>(executable));
    check(live_allocations.count(call.table) && live_allocations.at(call.table) == 131072, "replay retains table");
    std::fill_n(static_cast<float *>(call.output), call.count, -123.f); ++replays; return hipSuccess;
}
extern "C" hipError_t hipGraphExecDestroy(hipGraphExec_t executable) {
    check(live_allocations.count(graph_executables.at(reinterpret_cast<uintptr_t>(executable)).table) == 1, "table remains live through executable destruction");
    check(graph_executables.erase(reinterpret_cast<uintptr_t>(executable)) == 1, "executable destruction"); ++exec_destroys; return hipSuccess;
}
static ggml_tensor * expected_node = nullptr;
static std::string expected_kernel;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_node != nullptr, "unexpected submission");
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected_kernel) == 0, "symbol");
    check(gx == (ggml_nelements(expected_node)+255)/256 && gy==1 && gz==1, "flat grid");
    check(*static_cast<void **>(args[0]) == expected_node->src[0]->data, "gate binding");
    check(*static_cast<void **>(args[1]) == expected_node->src[1]->data, "up binding");
    check(*static_cast<void **>(args[2]) == expected_node->data, "output binding");
    void * table = *static_cast<void **>(args[3]);
    check(live_allocations.count(table) && live_allocations.at(table) == 131072, "live table binding");
    check(*static_cast<int *>(args[4]) == ggml_nelements(expected_node), "element count");
    if (mock_capture_active) {
        pending_capture = {table, expected_node->data, static_cast<size_t>(ggml_nelements(expected_node))};
    } else {
        std::fill_n(static_cast<float *>(expected_node->data),ggml_nelements(expected_node),-123.f);
    }
    ++launches;return fail_launch ? hipErrorUnknown : hipSuccess;
}
int main(int argc,char **argv) {
    check(argc==2,"package");setenv("FLAGOS_AMD_GEGLU_LUT","1",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");int checks=0;
    for(int variant=0;variant<3;++variant) {
        auto*ctx=ggml_init({1024*1024,nullptr,true});auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);auto*y=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);ggml_tensor*out=nullptr;const char*name=nullptr;
        out=ggml_geglu_split(ctx,x,y);name=AMD_GEGLU_LUT_KERNEL;
        auto*metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(name));check(metadata,"metadata");const auto saved=*metadata;
        auto supported=[&](){return amd_supports_geglu_lut(&device,out);};
        hipGetDeviceProperties(&device.props,0);setenv("FLAGOS_AMD_GEGLU_LUT","0",1);check(!supported(),"default off");setenv("FLAGOS_AMD_GEGLU_LUT","1",1);check(supported(),"valid unbound");++checks;
        std::vector<float> a(ggml_nelements(x)+32,41.f),b(ggml_nelements(y)+32,43.f),result(ggml_nelements(out)+32,37.f);x->data=a.data()+8;y->data=b.data()+8;out->data=result.data()+8;check(supported(),"valid bound");++checks;
        if(variant==1)out->data=x->data;
        if(variant==2)out->data=y->data;
        expected_node=out;expected_kernel=name;
        auto * context = new amd_backend_context(&device, nullptr);
        auto * backend = new ggml_backend{}; backend->context = context; device.backends.push_back(context);
        ggml_tensor * nodes[] = {out}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 1;
        const int old_allocations = allocations, old_uploads = uploads;
        const uint64_t cold_key = amd_fusion_config_key(context);
        check(amd_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS, "valid dispatch");
        check(context->geglu_lut_ready && context->geglu_lut_generation == 1, "asset ready generation");
        check(amd_fusion_config_key(context) != cold_key, "asset changes plan generation");
        const auto * asset = amd_geglu_lut_asset(&device);
        check(std::memcmp(context->geglu_lut, asset->bytes.data(), asset->bytes.size()) == 0, "exact uploaded bytes");
        check(amd_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS, "repeated dispatch");
        check(allocations == old_allocations + 1 && uploads == old_uploads + 1, "one upload per backend");
        amd_backend_free(backend);

        if(variant==0)check(std::all_of(result.begin()+8,result.end()-24,[](float v){return v==-123.f;}),"output");check(std::all_of(result.begin(),result.begin()+8,[](float v){return v==37.f;})&&std::all_of(result.end()-24,result.end(),[](float v){return v==37.f;}),"guards");check(variant==1 || std::all_of(a.begin(),a.end(),[](float v){return v==41.f;}),"first readonly input");check(variant==2 || std::all_of(b.begin(),b.end(),[](float v){return v==43.f;}),"second readonly input");
        if(variant!=0){auto &written=variant==1?a:b;const float untouched=variant==1?41.f:43.f;check(std::all_of(written.begin()+8,written.end()-24,[](float v){return v==-123.f;}),"exact inplace output");check(std::all_of(written.begin(),written.begin()+8,[&](float v){return v==untouched;})&&std::all_of(written.end()-24,written.end(),[&](float v){return v==untouched;}),"inplace guards");}expected_node=nullptr;
        auto test=[&](const char*label,const std::function<void()>&mutate) {
          for(bool bound:{false,true}) {
            *metadata=saved;hipGetDeviceProperties(&device.props,0);x->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)):nullptr;y->data=bound?reinterpret_cast<void*>(UINT64_C(0x200000000)):nullptr;out->data=bound?reinterpret_cast<void*>(UINT64_C(0x300000000)):nullptr;mutate();
            check(!supported(),label);check(!amd_device_supports_op(&device.device_iface,out),"placement refusal");
            ++checks;
          }
        };
        test("grid",[&]{device.props.maxGridSize[0]=1;});test("threads",[&]{device.props.maxThreadsPerBlock=64;});test("wave",[&]{device.props.warpSize=64;});
        test("dynamic LDS",[&]{metadata->shared_memory=70000;});test("static LDS",[&]{metadata->static_shared_memory=70000;});test("scratch",[&]{metadata->global_scratch_size=1;});test("profile scratch",[&]{metadata->profile_scratch_size=1;});
        test("argument count",[&]{metadata->argument_count+=1;});test("missing ABI",[&]{metadata->argument_abi.pop_back();});test("pointer ABI",[&]{metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("scalar width",[&]{metadata->argument_abi.back().size=8;});test("block",[&]{metadata->block_size=128;});
        test("input alignment",[&]{x->data=reinterpret_cast<void*>(UINT64_C(0x100000002));});test("output alignment",[&]{out->data=reinterpret_cast<void*>(UINT64_C(0x300000002));});
        test("second input alignment",[&]{y->data=reinterpret_cast<void*>(UINT64_C(0x200000002));});
        *metadata=saved;ggml_free(ctx);
    }
    check(launches == 6 && allocations == 3 && uploads == 3 && releases == 3, "successful lifecycle");
    auto * ctx = ggml_init({1024*1024,nullptr,true});
    auto * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 257);
    auto * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 257);
    auto * out = ggml_geglu_split(ctx, x, y);
    std::vector<float> a(257, 41.f), b(257, 43.f), result(257, 37.f);
    x->data = a.data(); y->data = b.data(); out->data = result.data();
    ggml_tensor * nodes[] = {out}; ggml_cgraph graph{}; graph.nodes = nodes; graph.n_nodes = 1;
    hipGetDeviceProperties(&device.props, 0);
    auto * mutable_asset = const_cast<flagos_amd::constant_asset *>(device.aot->find_asset(AMD_GEGLU_LUT_NAME));
    const auto saved_asset = *mutable_asset;
    for (int mismatch = 0; mismatch < 3; ++mismatch) {
        if (mismatch == 0) mutable_asset->sha256 = std::string(64, 'a');
        if (mismatch == 1) mutable_asset->bytes.pop_back();
        if (mismatch == 2) mutable_asset->name = "other-precision-table";
        check(!amd_supports_geglu_lut(&device, out), "precision asset refusal");
        check(!amd_device_supports_op(&device.device_iface, out), "placement rejects precision mismatch");
        amd_backend_context context(&device, nullptr); ggml_backend backend{}; backend.context = &context;
        const int old_allocations = allocations, old_launches = launches;
        check(amd_backend_graph_compute(&backend, &graph) == GGML_STATUS_FAILED, "asset rejection before graph writes");
        check(allocations == old_allocations && launches == old_launches, "no work for wrong precision asset");
        *mutable_asset = saved_asset;
        ++checks;
    }
    for (int failure = 0; failure < 3; ++failure) {
        auto * context = new amd_backend_context(&device, nullptr);
        auto * backend = new ggml_backend{}; backend->context = context; device.backends.push_back(context);
        const int before = launches, before_allocations = allocations, before_uploads = uploads;
        fail_allocation = failure == 0; fail_upload = failure == 1; fail_launch = failure == 2;
        expected_node = out; expected_kernel = AMD_GEGLU_LUT_KERNEL;
        std::fill(result.begin(), result.end(), 37.f);
        check(amd_backend_graph_compute(backend, &graph) == GGML_STATUS_FAILED, "injected failure");
        check(launches == before + (failure == 2), "no fallback or extra submission");
        if (failure < 2) check(std::all_of(result.begin(), result.end(), [](float v){return v == 37.f;}), "preparation leaves output unchanged");
        else check(std::all_of(result.begin(), result.end(), [](float v){return v == -123.f;}), "post-write error remains terminal");
        if (failure == 1) {
            check(context->geglu_lut != nullptr && !context->geglu_lut_ready, "failed upload retained");
            fail_upload = false;
            check(amd_backend_graph_compute(backend, &graph) == GGML_STATUS_FAILED, "failed asset cannot replay");
            check(allocations == before_allocations + 1 && uploads == before_uploads && launches == before, "failed upload not retried or submitted");
        }
        fail_allocation = fail_upload = fail_launch = false;
        amd_backend_free(backend);
        ++checks;
    }
    {
        setenv("FLAGOS_AMD_GRAPH_CAPTURE", "all", 1);
        auto * context = new amd_backend_context(&device, reinterpret_cast<hipStream_t>(uintptr_t(1)));
        auto * backend = new ggml_backend{}; backend->context = context; device.backends.push_back(context);
        auto * first = ggml_view_tensor(ctx, out), * second = ggml_view_tensor(ctx, out), * third = ggml_view_tensor(ctx, out);
        first->data = second->data = third->data = out->data;
        ggml_tensor * capture_nodes[] = {out, first, second, third};
        ggml_cgraph capture_graph{}; capture_graph.nodes = capture_nodes; capture_graph.n_nodes = 4;
        expected_node = out; expected_kernel = AMD_GEGLU_LUT_KERNEL;
        const int old_allocations = allocations, old_uploads = uploads, old_launches = launches;
        for (int pass = 0; pass < 3; ++pass) {
            std::fill(result.begin(), result.end(), 37.f);
            check(amd_backend_graph_compute(backend, &capture_graph) == GGML_STATUS_SUCCESS, "capture execution");
            check(std::all_of(result.begin(), result.end(), [](float v){return v == -123.f;}), "eager captured replay output");
        }
        check(captures == 1 && replays == 2 && launches == old_launches + 2, "warmup capture replay");
        check(allocations == old_allocations + 1 && uploads == old_uploads + 1, "capture never reallocates table");
        check(context->stats.graph_replays.load() == 1 && context->graph_captures.size() == 1, "replay cache");
        ++context->geglu_lut_generation;
        check(amd_backend_graph_compute(backend, &capture_graph) == GGML_STATUS_SUCCESS, "asset generation invalidates capture");
        check(exec_destroys == 1 && context->graph_captures.empty() && launches == old_launches + 3, "old executable not replayed");
        check(amd_backend_graph_compute(backend, &capture_graph) == GGML_STATUS_SUCCESS, "recapture generation");
        check(captures == 2 && replays == 3, "new capture");
        setenv("FLAGOS_AMD_GEGLU_LUT", "0", 1);
        std::fill(result.begin(), result.end(), 37.f);
        const int rejected_launches = launches, rejected_replays = replays;
        check(amd_backend_graph_compute(backend, &capture_graph) == GGML_STATUS_FAILED, "placement policy must be rebuilt");
        check(launches == rejected_launches && replays == rejected_replays, "disabled policy cannot replay or dispatch");
        check(std::all_of(result.begin(), result.end(), [](float v){return v == 37.f;}), "policy refusal before writes");
        setenv("FLAGOS_AMD_GEGLU_LUT", "1", 1);
        amd_backend_free(backend);
        check(captured_graphs.empty() && graph_executables.empty() && graph_destroys == 2 && exec_destroys == 2, "capture teardown before table release");
        setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
        checks += 6;
    }
    ggml_free(ctx); device.aot.reset();
    check(loaded == unloaded && allocations == releases && live_allocations.empty(), "mock lifecycle");
    std::printf("GEGLU_LUT_HOST_COMPLETE API=%d checks=%d mock_launches=%d uploads=%d actual_GPU_calls=0\n", GGML_BACKEND_API_VERSION, checks, launches, uploads);
}
