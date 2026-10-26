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
static ggml_tensor*expected_node=nullptr;
static int expected_kind=0;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function,unsigned gx,unsigned gy,unsigned gz,unsigned,unsigned,unsigned,unsigned,hipStream_t,void**args,void**){
    check(expected_node&&gx==(expected_kind==3?4:3)&&gy==1&&gz==1,"grid");const auto&name=symbols.at(reinterpret_cast<uintptr_t>(function));
    if(expected_kind==3){check(name.find("flagos_set_rows_f32_f16")==0,"SET_ROWS symbol");check(*static_cast<void**>(args[0])==expected_node->src[0]->data&&*static_cast<void**>(args[1])==expected_node->src[1]->data&&*static_cast<void**>(args[2])==expected_node->data,"state bindings");check(*static_cast<int*>(args[3])==257&&*static_cast<int*>(args[4])==3&&*static_cast<int*>(args[5])==7,"state ABI");std::fill_n(static_cast<uint16_t*>(expected_node->data),ggml_nelements(expected_node),uint16_t(0x7777));}
    else {check(name.find(expected_kind==2?"flagos_rms_norm_ggml_strided_f32":"flagos_rms_norm_f32")==0,"norm symbol");check(*static_cast<void**>(args[0])==expected_node->data&&*static_cast<void**>(args[1])==expected_node->src[0]->data&&*static_cast<int*>(args[2])==256,"norm ABI");if(expected_kind==2){check(*static_cast<int*>(args[3])==3&&*static_cast<int*>(args[4])==1&&*static_cast<int*>(args[5])==512,"strided norm dimensions");}std::fill_n(static_cast<float*>(expected_node->data),ggml_nelements(expected_node),42.f);}
    ++launches;return hipSuccess;
}
int main(int argc,char**argv){
    check(argc==2,"package");setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);setenv("FLAGOS_AMD_STRIDED_RMS_NORM","1",1);amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");int checks=0;
    for(int kind=0;kind<4;++kind){
        auto*ctx=ggml_init({1024*1024,nullptr,true});auto*root=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,kind==2?512:kind==3?257:256,3);auto*x=kind==2?ggml_view_2d(ctx,root,256,3,512*4,0):root;ggml_tensor*out=nullptr;ggml_tensor*ids=nullptr;ggml_tensor*destination=nullptr;
        if(kind==3){ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I64,3);destination=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,257,7);out=ggml_set_rows(ctx,destination,x,ids);}else out=kind==1?ggml_rms_norm_inplace(ctx,x,1e-5f):ggml_rms_norm(ctx,x,1e-5f);
        const char*name=kind==3?"flagos_set_rows_f32_f16":kind==2?"flagos_rms_norm_ggml_strided_f32":"flagos_rms_norm_f32";auto*metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(name));check(metadata,"metadata");const auto saved=*metadata;auto supported=[&](){return kind==3?amd_supports_set_rows(&device,out):amd_supports_rms_norm(&device,out);};
        hipGetDeviceProperties(&device.props,0);check(supported(),"valid unbound");++checks;
        std::vector<float>a(ggml_nelements(root)+32,37.f),result(ggml_nelements(out)+32,47.f);std::vector<uint16_t>cache(ggml_nelements(out)+32,0x3333);std::vector<int64_t>indices(35,1);root->data=a.data()+8;x->data=root->data;out->data=kind==1?x->data:kind==3?static_cast<void*>(cache.data()+8):static_cast<void*>(result.data()+8);if(kind==3){ids->data=indices.data()+8;destination->data=out->data;}check(supported(),"valid bound");++checks;
        expected_node=out;expected_kind=kind;{amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;ggml_tensor*nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"dispatch");}
        if(kind==3){check(std::all_of(cache.begin()+8,cache.end()-24,[](uint16_t v){return v==0x7777;}),"state writes");check(std::all_of(cache.begin(),cache.begin()+8,[](uint16_t v){return v==0x3333;})&&std::all_of(cache.end()-24,cache.end(),[](uint16_t v){return v==0x3333;}),"state guards");check(std::all_of(indices.begin(),indices.end(),[](int64_t v){return v==1;}),"readonly indices");}
        else {const auto &r=kind==1?a:result;check(std::all_of(r.begin()+8,r.begin()+8+ggml_nelements(out),[](float v){return v==42.f;}),"norm output");}
        if(kind!=1)check(std::all_of(a.begin(),a.end(),[](float v){return v==37.f;}),"readonly input");expected_node=nullptr;
        auto test=[&](const char*label,bool old_accept,const std::function<void()>&mutate){for(bool bound:{false,true}){*metadata=saved;hipGetDeviceProperties(&device.props,0);root->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)):nullptr;x->data=root->data;out->data=kind==1?x->data:bound?reinterpret_cast<void*>(UINT64_C(0x200000000)):nullptr;if(kind==3){ids->data=bound?reinterpret_cast<void*>(UINT64_C(0x300000000)):nullptr;destination->data=out->data;}mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
            check(supported()==old_accept,label);
#else
            (void)old_accept;check(!supported(),label);check(!amd_device_supports_op(&device.device_iface,out),"placement refusal");
#endif
            ++checks;}};
        test("threads",true,[&]{device.props.maxThreadsPerBlock=64;});test("wave",true,[&]{device.props.warpSize=64;});test("grid",true,[&]{device.props.maxGridSize[0]=1;});test("LDS",true,[&]{metadata->shared_memory=70000;});test("static LDS",true,[&]{metadata->static_shared_memory=70000;});test("scratch",true,[&]{metadata->global_scratch_size=1;});test("profile scratch",true,[&]{metadata->profile_scratch_size=1;});test("argument count",kind!=2,[&]{metadata->argument_count+=1;});test("missing ABI",true,[&]{metadata->argument_abi.pop_back();});test("pointer ABI",true,[&]{metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("scalar width",true,[&]{metadata->argument_abi.back().size=8;});
        if(kind!=2)test("source alignment",true,[&]{root->data=reinterpret_cast<void*>(UINT64_C(0x100000002));x->data=root->data;if(kind==1)out->data=x->data;});
        if(kind==3){test("index alignment",true,[&]{ids->data=reinterpret_cast<void*>(UINT64_C(0x300000004));});test("destination alignment",true,[&]{out->data=reinterpret_cast<void*>(UINT64_C(0x200000001));destination->data=out->data;});test("block",true,[&]{metadata->block_size=128;});}
        *metadata=saved;ggml_free(ctx);
    }
    device.aot.reset();check(loaded==unloaded&&launches==4&&allocations==0,"lifecycle");std::printf("NORM_STATE_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches);
}
