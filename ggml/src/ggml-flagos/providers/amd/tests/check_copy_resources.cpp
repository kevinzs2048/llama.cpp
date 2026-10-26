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
static ggml_tensor* expected_node=nullptr;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function,unsigned gx,unsigned gy,unsigned gz,unsigned,unsigned,unsigned,unsigned,hipStream_t,void **args,void**){
    check(expected_node&&gx==4&&gy==1&&gz==1,"copy grid");const int es=ggml_type_size(expected_node->type);const char*name=expected_node->type==GGML_TYPE_F16?"flagos_copy_strided_f16_bits":"flagos_copy_strided_f32";check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(name)==0,"copy symbol");check(*static_cast<void**>(args[0])==expected_node->src[0]->data&&*static_cast<void**>(args[1])==expected_node->data,"copy pointers");
    for(int d=0;d<4;++d){check(*static_cast<int*>(args[2+d])==expected_node->src[0]->ne[d],"source dims");check(*static_cast<int*>(args[6+d])==expected_node->ne[d],"output dims");check(*static_cast<int*>(args[10+d])==expected_node->src[0]->nb[d]/es,"source strides");check(*static_cast<int*>(args[14+d])==expected_node->nb[d]/es,"output strides");}check(*static_cast<int*>(args[18])==771,"copy count");std::memset(expected_node->data,0x77,ggml_nbytes(expected_node));++launches;return hipSuccess;
}
int main(int argc,char**argv){check(argc==2,"package");setenv("FLAGOS_AMD_F16_COPY","1",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");int checks=0;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16})for(int kind=0;kind<3;++kind)for(bool strided:{false,true}){
        const int es=ggml_type_size(type);auto*ctx=ggml_init({1024*1024,nullptr,true});auto*root=ggml_new_tensor_2d(ctx,type,strided?512:257,3);auto*x=strided?ggml_view_2d(ctx,root,257,3,512*es,0):root;auto*y=ggml_new_tensor_2d(ctx,type,257,3);auto*out=kind==0?ggml_cpy(ctx,x,y):kind==1?ggml_cont(ctx,x):ggml_dup(ctx,x);auto*metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(amd_copy_kernel_name(out)));check(metadata,"copy metadata");const auto saved=*metadata;
        hipGetDeviceProperties(&device.props,0);check(amd_supports_copy(&device,out),"valid unbound");++checks;
        std::vector<uint8_t>a(ggml_nbytes(root)+128,0x55),b(ggml_nbytes(out)+128,0x33);root->data=a.data()+64;x->data=root->data;y->data=b.data()+64;out->data=y->data;check(amd_supports_copy(&device,out),"valid bound");++checks;expected_node=out;
        {amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;ggml_tensor*nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"dispatch");}
        check(std::all_of(a.begin(),a.end(),[](uint8_t v){return v==0x55;}),"readonly source");check(std::all_of(b.begin()+64,b.end()-64,[](uint8_t v){return v==0x77;}),"output");check(std::all_of(b.begin(),b.begin()+64,[](uint8_t v){return v==0x33;})&&std::all_of(b.end()-64,b.end(),[](uint8_t v){return v==0x33;}),"guards");expected_node=nullptr;
        auto test=[&](const char*label,bool old_accept,const std::function<void()>&mutate){for(bool bound:{false,true}){*metadata=saved;hipGetDeviceProperties(&device.props,0);root->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)):nullptr;x->data=root->data;y->data=bound?reinterpret_cast<void*>(UINT64_C(0x200000000)):nullptr;out->data=y->data;mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
            check(amd_supports_copy(&device,out)==old_accept,label);
#else
            (void)old_accept;check(!amd_supports_copy(&device,out),label);check(!amd_device_supports_op(&device.device_iface,out),"placement refusal");
#endif
            ++checks;}};
        const bool f32=type==GGML_TYPE_F32;
        test("grid",true,[&]{device.props.maxGridSize[0]=1;});test("threads",true,[&]{device.props.maxThreadsPerBlock=64;});test("wave",true,[&]{device.props.warpSize=64;});test("dynamic LDS",f32,[&]{metadata->shared_memory=70000;});test("static LDS",f32,[&]{metadata->static_shared_memory=70000;});test("scratch",f32,[&]{metadata->global_scratch_size=1;});test("profile scratch",f32,[&]{metadata->profile_scratch_size=1;});test("argument count",f32,[&]{metadata->argument_count=18;});test("missing ABI",true,[&]{metadata->argument_abi.pop_back();});test("pointer ABI",true,[&]{metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("scalar width",true,[&]{metadata->argument_abi.back().size=8;});test("block",f32,[&]{metadata->block_size=128;});
        test("source alignment",f32,[&]{root->data=reinterpret_cast<void*>(UINT64_C(0x100000001));x->data=root->data;});test("output alignment",f32,[&]{y->data=reinterpret_cast<void*>(UINT64_C(0x200000001));out->data=y->data;});
        *metadata=saved;ggml_free(ctx);
    }
    device.aot.reset();check(loaded==unloaded&&launches==12&&allocations==0,"lifecycle");std::printf("COPY_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches);
}
