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
static ggml_tensor * expected_node=nullptr;
static bool expected_prefill=false;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function,unsigned gx,unsigned gy,unsigned gz,
        unsigned,unsigned,unsigned,unsigned,hipStream_t,void ** args,void **) {
    check(expected_node,"unexpected submission");auto * out=expected_node;
    const std::string prefix=expected_prefill?"flagos_flash_attn_prefill_f32_f16":"flagos_flash_attn_decode_f32_f16";
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(prefix)==0,"symbol");
    check(gx==out->src[0]->ne[2] && gy==(expected_prefill?(out->src[0]->ne[1]+15)/16:1) && gz==1,"attention grid");
    for(int i=0;i<4;++i)check(*static_cast<void **>(args[i])==out->src[i]->data,"input pointers");
    check(*static_cast<void **>(args[4])==out->data,"output pointer");
    int slot=5;
    if(expected_prefill)check(*static_cast<int *>(args[slot++])==out->src[0]->ne[1],"query length");
    check(*static_cast<int *>(args[slot++])==out->src[1]->ne[1],"key length");
    check(*static_cast<int *>(args[slot++])==out->src[0]->ne[2]/out->src[1]->ne[2],"GQA ratio");
    for(int i=0;i<3;++i)for(int d=1;d<3;++d)check(*static_cast<int *>(args[slot++])==out->src[i]->nb[d]/(i==0?4:2),"QKV stride");
    if(expected_prefill)for(int d=0;d<2;++d)check(*static_cast<int *>(args[slot++])==out->src[3]->nb[d]/2,"mask stride");
    check(*static_cast<int *>(args[slot++])==out->nb[2]/4,"output token stride");check(*static_cast<int *>(args[slot++])==out->nb[1]/4,"output head stride");
    check(*static_cast<float *>(args[slot++])==.125f,"scale");check(slot==(expected_prefill?19:16),"ABI length");
    std::fill_n(static_cast<float *>(out->data),ggml_nelements(out),-123.f);++launches;return hipSuccess;
}
int main(int argc,char **argv) {
    check(argc==2,"package");setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS","1",1);setenv("FLAGOS_AMD_FUSIONS","flash_attn_decode,flash_attn_prefill",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_SPLITKV_DECODE","0",1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");int checks=0;
    for(int variant=0;variant<4;++variant) {
        bool prefill=variant>=2;int head=variant%2?256:128;int tokens=prefill?17:1;auto * ctx=ggml_init({1024*1024,nullptr,true});auto * q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,head,tokens,8);auto * kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,head,2,33);auto * vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,head,2,33);auto * k=ggml_permute(ctx,kb,0,2,1,3);auto * v=ggml_permute(ctx,vb,0,2,1,3);auto * mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,33,tokens);auto * out=ggml_flash_attn_ext(ctx,q,k,v,mask,.125f,0.f,0.f);
        auto * metadata=const_cast<flagos_amd::kernel_metadata*>(amd_flash_attn_kernel(&device,out,prefill));check(metadata,"metadata");auto saved=*metadata;auto supported=[&](){return prefill?amd_supports_flash_attn_prefill(&device,out):amd_supports_flash_attn_decode(&device,out);};hipGetDeviceProperties(&device.props,0);check(supported(),"valid unbound");++checks;
        std::vector<float> query(ggml_nelements(q)+32,41.f),result(ggml_nelements(out)+32,37.f);std::vector<ggml_fp16_t> keys(ggml_nelements(k)+32,43),values(ggml_nelements(v)+32,47),masks(ggml_nelements(mask)+32,53);q->data=query.data()+8;kb->data=k->data=keys.data()+8;vb->data=v->data=values.data()+8;mask->data=masks.data()+8;out->data=result.data()+8;check(supported(),"valid bound");++checks;expected_node=out;expected_prefill=prefill;
        {amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;ggml_tensor * nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"valid dispatch");}
        check(std::all_of(result.begin()+8,result.end()-24,[](float x){return x==-123.f;}),"output");check(std::all_of(result.begin(),result.begin()+8,[](float x){return x==37.f;})&&std::all_of(result.end()-24,result.end(),[](float x){return x==37.f;}),"guards");check(std::all_of(query.begin(),query.end(),[](float x){return x==41.f;})&&std::all_of(keys.begin(),keys.end(),[](ggml_fp16_t x){return x==43;})&&std::all_of(values.begin(),values.end(),[](ggml_fp16_t x){return x==47;})&&std::all_of(masks.begin(),masks.end(),[](ggml_fp16_t x){return x==53;}),"readonly inputs");expected_node=nullptr;
        auto test=[&](const char * label,const std::function<void()> & mutate) {
            for(bool bound:{false,true}) {
                *metadata=saved;hipGetDeviceProperties(&device.props,0);for(int i=0;i<4;++i)out->src[i]->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)*(i+1)):nullptr;out->data=bound?reinterpret_cast<void*>(UINT64_C(0x600000000)):nullptr;mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
                check(supported(),label);
#else
                check(!supported(),label);check(!amd_device_supports_op(&device.device_iface,out),"placement refusal");
#endif
                ++checks;
            }
        };
        test("head grid",[&]{device.props.maxGridSize[0]=1;});if(prefill)test("query grid",[&]{device.props.maxGridSize[1]=1;});test("threads",[&]{device.props.maxThreadsPerBlock=64;});test("wave",[&]{device.props.warpSize=64;});test("LDS capacity",[&]{device.props.sharedMemPerBlock=1;});test("dynamic LDS",[&]{metadata->shared_memory=70000;});test("static LDS",[&]{metadata->static_shared_memory=70000;});test("scratch",[&]{metadata->global_scratch_size=1;});test("profile scratch",[&]{metadata->profile_scratch_size=1;});test("missing ABI",[&]{metadata->argument_abi.pop_back();});test("pointer ABI",[&]{metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("scalar width",[&]{metadata->argument_abi.back().size=8;});test("zero warps",[&]{metadata->num_warps=0;});*metadata=saved;ggml_free(ctx);
    }
    int split_checks=0;
    for(int variant=0;variant<6;++variant) {
        bool grouped=variant>=2;int head=(variant==1||variant==5)?256:128;int group=variant<2?4:(variant==2?2:variant==3?4:8);setenv("FLAGOS_AMD_SPLITKV_DECODE","1",1);setenv("FLAGOS_AMD_GROUPED_SPLITKV_DECODE",grouped?"1":"0",1);
        auto * ctx=ggml_init({1024*1024,nullptr,true});auto * q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,head,1,16);auto * kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,head,16/group,4096);auto * vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,head,16/group,4096);auto * k=ggml_permute(ctx,kb,0,2,1,3);auto * v=ggml_permute(ctx,vb,0,2,1,3);auto * mask=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,4096);auto * out=ggml_flash_attn_ext(ctx,q,k,v,mask,.125f,0.f,0.f);hipGetDeviceProperties(&device.props,0);amd_backend_context context(&device,nullptr);flagos_amd::splitkv_decode_geometry geometry;check(amd_describe_splitkv(&context,out,&geometry),"valid optional selection");++split_checks;
        auto * partial=const_cast<flagos_amd::kernel_metadata*>(amd_splitkv_kernel(&device,out,false));auto * merge=const_cast<flagos_amd::kernel_metadata*>(amd_splitkv_kernel(&device,out,true));check(partial&&merge,"split profiles");auto saved_partial=*partial,saved_merge=*merge;
        auto test=[&](const char * label,const std::function<void()> & mutate) {
            for(bool bound:{false,true}) {
                *partial=saved_partial;*merge=saved_merge;hipGetDeviceProperties(&device.props,0);for(int j=0;j<4;++j)out->src[j]->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)*(j+1)):nullptr;out->data=bound?reinterpret_cast<void*>(UINT64_C(0x600000000)):nullptr;mutate();check(amd_supports_flash_attn_decode(&device,out),"legal baseline preserved");
#ifdef FLAGOS_SPLITKV_BEFORE_FIXTURE
                check(amd_describe_splitkv(&context,out,&geometry),label);
#else
                check(!amd_describe_splitkv(&context,out,&geometry),label);
#endif
                ++split_checks;
            }
        };
        test("partial pointer ABI",[&]{partial->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("partial scalar width",[&]{partial->argument_abi.back().size=8;});test("partial ABI missing",[&]{partial->argument_abi.pop_back();});test("merge pointer ABI",[&]{merge->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("merge scalar width",[&]{merge->argument_abi.back().size=8;});test("merge ABI missing",[&]{merge->argument_abi.pop_back();});test("split grid",[&]{device.props.maxGridSize[1]=1;});test("partial static LDS",[&]{partial->static_shared_memory=65536;});test("merge static LDS",[&]{merge->static_shared_memory=65537;});*partial=saved_partial;*merge=saved_merge;
#ifndef FLAGOS_SPLITKV_BEFORE_FIXTURE
        partial->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;
        std::vector<float> query(ggml_nelements(q)+32,41.f),result(ggml_nelements(out)+32,37.f);std::vector<ggml_fp16_t> keys(ggml_nelements(k)+32,43),values(ggml_nelements(v)+32,47),masks(ggml_nelements(mask)+32,53);q->data=query.data()+8;kb->data=k->data=keys.data()+8;vb->data=v->data=values.data()+8;mask->data=masks.data()+8;out->data=result.data()+8;hipGetDeviceProperties(&device.props,0);expected_node=out;expected_prefill=false;int previous_launches=launches;ggml_backend backend{};backend.context=&context;ggml_tensor * nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"invalid optional profile uses baseline");check(launches==previous_launches+1&&allocations==0&&context.splitkv_workspace==nullptr,"one baseline submission without workspace");check(std::all_of(result.begin()+8,result.end()-24,[](float x){return x==-123.f;}),"fallback output");check(std::all_of(result.begin(),result.begin()+8,[](float x){return x==37.f;})&&std::all_of(result.end()-24,result.end(),[](float x){return x==37.f;}),"fallback guards");check(std::all_of(query.begin(),query.end(),[](float x){return x==41.f;})&&std::all_of(keys.begin(),keys.end(),[](ggml_fp16_t x){return x==43;})&&std::all_of(values.begin(),values.end(),[](ggml_fp16_t x){return x==47;})&&std::all_of(masks.begin(),masks.end(),[](ggml_fp16_t x){return x==53;}),"fallback readonly inputs");expected_node=nullptr;*partial=saved_partial;++split_checks;
#endif
        ggml_free(ctx);
    }
    setenv("FLAGOS_AMD_SPLITKV_DECODE","0",1);setenv("FLAGOS_AMD_GROUPED_SPLITKV_DECODE","0",1);std::printf("SPLITKV_SELECTION_RESOURCES_HOST_COMPLETE API=%d checks=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,split_checks);
#ifdef FLAGOS_SPLITKV_BEFORE_FIXTURE
    const int expected_launches=4;
#else
    const int expected_launches=10;
#endif
    device.aot.reset();check(loaded==unloaded&&launches==expected_launches&&allocations==0,"mock lifecycle");std::printf("FLASH_ATTN_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches);
}
