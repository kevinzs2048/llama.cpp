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
static ggml_tensor * expected_activation = nullptr, * expected_mul = nullptr;
static std::vector<std::string> submitted;
static bool fail_fused_submission=false;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function,unsigned gx,unsigned gy,unsigned gz,
 unsigned,unsigned,unsigned,unsigned,hipStream_t,void **args,void **) {
    check(expected_activation && expected_mul && gx==4 && gy==1 && gz==1,"grid");
    const std::string name=symbols.at(reinterpret_cast<uintptr_t>(function));submitted.push_back(name);
    const bool fused=name.find("flagos_swiglu_split_f32")==0;
    const bool silu=name.find("flagos_silu_f32")==0;
    const bool mul=name.find("flagos_mul_f32")==0;check(fused||silu||mul,"selected library kernel");
    void *output=nullptr;
    if(fused) {
        check(*static_cast<void **>(args[0])==expected_activation->src[0]->data && *static_cast<void **>(args[1])==expected_mul->src[1]->data && *static_cast<void **>(args[2])==expected_mul->data,"fused input/output bindings");
        check(*static_cast<int*>(args[3])==771,"fused count");output=*static_cast<void **>(args[2]);
    } else if(silu) {check(*static_cast<void **>(args[0])==expected_activation->src[0]->data && *static_cast<void **>(args[1])==expected_activation->data,"direct silu bindings");output=*static_cast<void **>(args[1]);}
    else {check(*static_cast<void **>(args[0])==expected_activation->data && *static_cast<void **>(args[1])==expected_mul->src[1]->data && *static_cast<void **>(args[2])==expected_mul->data,"direct mul bindings");output=*static_cast<void **>(args[2]);}
    if(fused&&fail_fused_submission){static_cast<float*>(output)[0]=-99.f;++launches;return hipErrorInvalidValue;}
    std::fill_n(static_cast<float*>(output),771,42.f);++launches;return hipSuccess;
}
int main(int argc,char **argv) {
    check(argc==2,"package");setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","attention_output_gate",1);setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS","1",1);setenv("FLAGOS_AMD_REPEAT_OPS","0",1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");
    const char * selected_profile=device.aot->find("flagos_silu_mul_f32")?"flagos_silu_mul_f32":"flagos_swiglu_split_f32";
    auto*metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(selected_profile));check(metadata,"library gate profile");const auto saved=*metadata;int checks=0;
#ifdef FLAGOS_BEFORE_FIXTURE
    const bool after=false;
#else
    const bool after=true;
#endif
    using mutation=std::function<void(ggml_tensor*,ggml_tensor*)>;
    auto test=[&](const char*label,bool legal,const mutation&change,bool execute=false,bool retain_activation=false,bool experimental=true,bool unknown_visibility=false,bool external_consumer=false) {
      for(bool bound:{false,true}) {
        *metadata=saved;hipGetDeviceProperties(&device.props,0);setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS",experimental?"1":"0",1);
        auto*ctx=ggml_init({1024*1024,nullptr,true});auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);auto*up=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);auto*activation=ggml_silu(ctx,x);auto*mul=ggml_mul(ctx,activation,up);
        std::vector<float> xv(803,37.f),uv(803,43.f),av(803,47.f),ov(803,53.f);
        if(bound){x->data=xv.data()+8;up->data=uv.data()+8;activation->data=av.data()+8;mul->data=ov.data()+8;}
        if(retain_activation)ggml_set_output(activation);
        change(activation,mul);
#ifndef FLAGOS_BEFORE_FIXTURE
        check(std::strcmp(amd_attention_output_gate_kernel(&device,activation),selected_profile)==0,"dedicated package compatibility selection");
#endif
        check(amd_supports_attention_output_gate(&device,activation,mul)==(legal&&after),label);++checks;
        auto*full=ggml_new_graph(ctx);ggml_build_forward_expand(full,mul);check(full->n_nodes==2,"real graph topology");if(external_consumer){auto*later=ggml_add(ctx,activation,up);ggml_build_forward_expand(full,later);}
        ggml_cgraph graph=ggml_graph_view(full,0,2);
        if(unknown_visibility){graph.use_counts=nullptr;graph.visited_hash_set={};}
        amd_backend_context context(&device,nullptr);flagos_pattern_candidate candidate;candidate.id=flagos_pattern_id::attention_output_gate;candidate.node_indices={0,1};candidate.required_output_node_indices=(retain_activation||unknown_visibility||external_consumer)?std::vector<int>{0,1}:std::vector<int>{1};
        check(amd_query_fusion(&context,&graph,candidate).supported==(legal&&after&&experimental&&!retain_activation&&!unknown_visibility&&!external_consumer),"lowering agrees with required outputs");++checks;
        if(execute&&bound) {
            expected_activation=activation;expected_mul=mul;submitted.clear();ggml_backend backend{};backend.context=&context;
            check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"graph compute");
            const bool fused=after&&legal&&experimental&&!retain_activation&&!unknown_visibility&&!external_consumer;
            if(submitted.size()!=(fused?1:2)){std::fprintf(stderr,"FUSION_COUNT_DIAGNOSTIC label=%s expected=%d actual=%zu\n",label,fused?1:2,submitted.size());for(const auto&name:submitted)std::fprintf(stderr,"submitted=%s\n",name.c_str());}
            check(submitted.size()==(fused?1:2),"one fused or two direct launches");
            if(fused)check(submitted[0].find("flagos_swiglu_split_f32")==0,"existing library reuse");
            else check(submitted[0].find("flagos_silu_f32")==0&&submitted[1].find("flagos_mul_f32")==0,"direct fallback order");
            check(std::all_of(ov.begin()+8,ov.end()-24,[](float v){return v==42.f;}),"final output");
            check(std::all_of(ov.begin(),ov.begin()+8,[](float v){return v==53.f;})&&std::all_of(ov.end()-24,ov.end(),[](float v){return v==53.f;}),"output guards");
            check(std::all_of(xv.begin(),xv.end(),[](float v){return v==37.f;})&&std::all_of(uv.begin(),uv.end(),[](float v){return v==43.f;}),"readonly inputs");
            if(fused)check(std::all_of(av.begin(),av.end(),[](float v){return v==47.f;}),"unused intermediate untouched");
            else check(std::all_of(av.begin()+8,av.end()-24,[](float v){return v==42.f;}),"required direct intermediate");expected_activation=expected_mul=nullptr;
        }
        ggml_free(ctx);
      }
    };
    test("valid fusion",true,[](auto*,auto*){},true);
    test("explicit disable",true,[](auto*,auto*){},true,false,false);
    test("required intermediate",true,[](auto*,auto*){},true,true);
    test("unknown whole graph",true,[](auto*,auto*){},true,false,true,true);
    test("external consumer",true,[](auto*,auto*){},true,false,true,false,true);
    test("default stays disabled",true,[](auto*,auto*){unsetenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS");},true,false,false);
    test("exact inplace",true,[](auto*a,auto*m){a->src[0]->data=reinterpret_cast<void*>(UINT64_C(0x100000000));m->data=a->src[0]->data;});
    test("partial overlap",false,[](auto*a,auto*m){a->src[0]->data=reinterpret_cast<void*>(UINT64_C(0x100000000));m->data=reinterpret_cast<void*>(UINT64_C(0x100000004));});
    test("missing ABI",false,[&](auto*,auto*){metadata->argument_abi.pop_back();},true);
    test("wrong count",false,[&](auto*,auto*){metadata->argument_count=3;},true);
    test("pointer kind",false,[&](auto*,auto*){metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;},true);
    test("scalar width",false,[&](auto*,auto*){metadata->argument_abi.back().size=8;},true);
    test("block",false,[&](auto*,auto*){metadata->block_size=128;},true);
    test("warps",false,[&](auto*,auto*){metadata->num_warps=8;},true);
    test("LDS",false,[&](auto*,auto*){metadata->shared_memory=70000;},true);
    test("static LDS",false,[&](auto*,auto*){metadata->static_shared_memory=70000;},true);
    test("scratch",false,[&](auto*,auto*){metadata->global_scratch_size=1;},true);
    test("profile scratch",false,[&](auto*,auto*){metadata->profile_scratch_size=1;},true);
    test("wave",false,[&](auto*,auto*){device.props.warpSize=64;});
    test("thread cap",false,[&](auto*,auto*){device.props.maxThreadsPerBlock=64;});
    test("grid cap",false,[&](auto*,auto*){device.props.maxGridSize[0]=1;});
    test("input alignment",false,[](auto*a,auto*){a->src[0]->data=reinterpret_cast<void*>(UINT64_C(0x100000002));});
    test("up alignment",false,[](auto*,auto*m){m->src[1]->data=reinterpret_cast<void*>(UINT64_C(0x200000002));});
    test("output alignment",false,[](auto*,auto*m){m->data=reinterpret_cast<void*>(UINT64_C(0x300000002));});
    if(after) {
        *metadata=saved;hipGetDeviceProperties(&device.props,0);setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS","1",1);
        auto*ctx=ggml_init({1024*1024,nullptr,true});auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);auto*up=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,257,3);auto*a=ggml_silu(ctx,x);auto*m=ggml_mul(ctx,a,up);
        std::vector<float>xv(803,37.f),uv(803,43.f),av(803,47.f),ov(803,53.f);x->data=xv.data()+8;up->data=uv.data()+8;a->data=av.data()+8;m->data=ov.data()+8;auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,m);ggml_cgraph split=ggml_graph_view(g,0,2);
        expected_activation=a;expected_mul=m;submitted.clear();fail_fused_submission=true;
        {amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;check(amd_backend_graph_compute(&backend,&split)==GGML_STATUS_FAILED,"post-submission error");}
        check(submitted.size()==1&&submitted[0].find("flagos_swiglu_split_f32")==0,"no direct retry after partial fusion write");check(ov[8]==-99.f&&std::all_of(ov.begin()+9,ov.end(),[](float v){return v==53.f;})&&std::all_of(ov.begin(),ov.begin()+8,[](float v){return v==53.f;}),"partial output remains failed");check(std::all_of(av.begin(),av.end(),[](float v){return v==47.f;})&&std::all_of(xv.begin(),xv.end(),[](float v){return v==37.f;})&&std::all_of(uv.begin(),uv.end(),[](float v){return v==43.f;}),"no intermediate or input rewrite");fail_fused_submission=false;expected_activation=expected_mul=nullptr;++checks;ggml_free(ctx);
    }
    *metadata=saved;device.aot.reset();check(loaded==unloaded&&allocations==0,"lifecycle");std::printf("selected_profile=%s\n",selected_profile);std::printf("SILU_MUL_FUSION_HOST_COMPLETE API=%d checks=%d mock_launches=%d after=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches,int(after));
}
