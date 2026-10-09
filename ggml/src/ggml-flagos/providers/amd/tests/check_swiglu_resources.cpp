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
static std::string expected_kernel;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_node != nullptr, "unexpected submission");
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected_kernel) == 0, "symbol");
    check(gx == (ggml_nelements(expected_node)+255)/256 && gy==1 && gz==1, "flat grid");
    const int pointers = expected_node->src[1] ? 3 : 2;
    check(*static_cast<void **>(args[0])==expected_node->src[0]->data, "first input");
    if(pointers==3)check(*static_cast<void **>(args[1])==expected_node->src[1]->data,"second input");
    check(*static_cast<void **>(args[pointers-1])==expected_node->data,"output");
    if(expected_node->op==GGML_OP_SCALE) {
        check(*static_cast<float *>(args[2])==1.25f && *static_cast<float *>(args[3])==-.5f,"scale bias ABI");
        check(*static_cast<int *>(args[4])==ggml_nelements(expected_node),"scale count");
    } else {
        check(*static_cast<int *>(args[pointers])==ggml_nelements(expected_node),"element count");
        if(expected_node->op==GGML_OP_MUL || expected_kernel=="flagos_add_repeat_f32")check(*static_cast<int *>(args[4])==ggml_nelements(expected_node->src[1]),"repeat count");
    }
    std::fill_n(static_cast<float *>(expected_node->data),ggml_nelements(expected_node),-123.f);
    ++launches;return hipSuccess;
}
int main(int argc,char **argv) {
    check(argc==2,"package");setenv("FLAGOS_AMD_REPEAT_OPS","0",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");int checks=0;
    for(int variant=0;variant<3;++variant) {
        auto*ctx=ggml_init({1024*1024,nullptr,true});auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);auto*y=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);ggml_tensor*out=nullptr;const char*name=nullptr;
        out=ggml_swiglu_split(ctx,x,y);name="flagos_swiglu_split_f32";
        auto*metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(name));check(metadata,"metadata");const auto saved=*metadata;
        auto supported=[&](){return amd_supports_swiglu(&device,out);};
        hipGetDeviceProperties(&device.props,0);check(supported(),"valid unbound");++checks;
        std::vector<float> a(ggml_nelements(x)+32,41.f),b(ggml_nelements(y)+32,43.f),result(ggml_nelements(out)+32,37.f);x->data=a.data()+8;y->data=b.data()+8;out->data=result.data()+8;check(supported(),"valid bound");++checks;
        if(variant==1)out->data=x->data;
        if(variant==2)out->data=y->data;
        expected_node=out;expected_kernel=name;
        {amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;ggml_tensor*nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"valid dispatch");}
        if(variant==0)check(std::all_of(result.begin()+8,result.end()-24,[](float v){return v==-123.f;}),"output");check(std::all_of(result.begin(),result.begin()+8,[](float v){return v==37.f;})&&std::all_of(result.end()-24,result.end(),[](float v){return v==37.f;}),"guards");check(variant==1 || std::all_of(a.begin(),a.end(),[](float v){return v==41.f;}),"first readonly input");check(variant==2 || std::all_of(b.begin(),b.end(),[](float v){return v==43.f;}),"second readonly input");
        if(variant!=0){auto &written=variant==1?a:b;const float untouched=variant==1?41.f:43.f;check(std::all_of(written.begin()+8,written.end()-24,[](float v){return v==-123.f;}),"exact inplace output");check(std::all_of(written.begin(),written.begin()+8,[&](float v){return v==untouched;})&&std::all_of(written.end()-24,written.end(),[&](float v){return v==untouched;}),"inplace guards");}expected_node=nullptr;
        auto test=[&](const char*label,const std::function<void()>&mutate) {
          for(bool bound:{false,true}) {
            *metadata=saved;hipGetDeviceProperties(&device.props,0);x->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)):nullptr;y->data=bound?reinterpret_cast<void*>(UINT64_C(0x200000000)):nullptr;out->data=bound?reinterpret_cast<void*>(UINT64_C(0x300000000)):nullptr;mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
            check(supported(),label);
#else
            check(!supported(),label);check(!amd_device_supports_op(&device.device_iface,out),"placement refusal");
#endif
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
    device.aot.reset();check(loaded==unloaded&&launches==3&&allocations==0,"mock lifecycle");std::printf("SWIGLU_DIRECT_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches);
}
