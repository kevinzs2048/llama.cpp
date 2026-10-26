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
static ggml_tensor * expected_node = nullptr;
static amd_batched_gather_launch expected;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    check(expected_node && gx == expected.grid_x && gy == 1 && gz == 1, "selected grid");
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected.kernel) == 0, "selected symbol");
    check(*static_cast<void **>(args[0]) == expected_node->src[0]->data &&
        *static_cast<void **>(args[1]) == expected_node->src[1]->data &&
        *static_cast<void **>(args[2]) == expected_node->data, "pointer bindings");
    for (int i = 0; i < 13; ++i) check(*static_cast<int *>(args[3+i]) == expected.values[i], "sixteen slot ABI");
    std::fill_n(static_cast<float *>(expected_node->data), ggml_nelements(expected_node), -123.f);
    ++launches;return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc==2,"package");setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);setenv("FLAGOS_AMD_PREFILL_F16_GEMM","0",1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;
    device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"mock modules");int checks=0;
    using mutation=std::function<void(ggml_tensor*)>;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) {
        const char*name=type==GGML_TYPE_F32?"flagos_get_rows_batched_f32":type==GGML_TYPE_F16?"flagos_get_rows_batched_f16_f32":"flagos_get_rows_batched_bf16_bits_f32";
        auto*metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(name));check(metadata,"profile");const auto saved=*metadata;
        for (const auto batches : {std::pair<int,int>(16,1),std::pair<int,int>(3,2)}) {
            auto test=[&](const char*label,bool accepted,const mutation&change,bool execute=false){
                for(bool bound:{false,true}) {
                    auto*ctx=ggml_init({1024*1024,nullptr,true});auto*table=ggml_new_tensor_4d(ctx,type,257,4,batches.first,batches.second);auto*indices=ggml_new_tensor_3d(ctx,GGML_TYPE_I32,2,batches.first,batches.second);
                    table->nb[1]+=3*ggml_type_size(type);table->nb[2]=table->nb[1]*4+5*ggml_type_size(type);table->nb[3]=table->nb[2]*batches.first+7*ggml_type_size(type);
                    indices->nb[1]=16;indices->nb[2]=indices->nb[1]*batches.first+12;indices->nb[3]=indices->nb[2]*batches.second;
                    auto*out=ggml_get_rows(ctx,table,indices);
                    std::vector<uint16_t> storage((ggml_nbytes(table)+1)/2+64,0x4aaa);std::vector<int> ids(ggml_nbytes(indices)/4+64,3);std::vector<float> results(ggml_nelements(out)+32,37.f);
                    if(bound){table->data=storage.data()+8;indices->data=ids.data()+8;out->data=results.data()+8;}
                    *metadata=saved;hipGetDeviceProperties(&device.props,0);setenv("FLAGOS_AMD_BATCHED_GATHER","1",1);change(out);
                    amd_batched_gather_launch launch;
                    check(amd_describe_batched_gather(&device,out,&launch)==accepted,label);
                    if(accepted)check(amd_device_supports_op(&device.device_iface,out),"public supports op");
                    if(execute&&bound){expected=launch;expected_node=out;amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;ggml_tensor*nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;int before=launches;
                        check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS&&launches==before+1,"dispatch");
                        check(std::all_of(results.begin()+8,results.end()-24,[](float v){return v==-123.f;}),"output writes");
                        check(std::all_of(results.begin(),results.begin()+8,[](float v){return v==37.f;})&&std::all_of(results.end()-24,results.end(),[](float v){return v==37.f;}),"guards");
                        check(std::all_of(storage.begin(),storage.end(),[](uint16_t v){return v==0x4aaa;})&&std::all_of(ids.begin(),ids.end(),[](int v){return v==3;}),"read only inputs");expected_node=nullptr;
                    }
                    ++checks;ggml_free(ctx);
                }
            };
            test("valid batched",true,[](auto*){},true);
            test("default off",false,[](auto*){unsetenv("FLAGOS_AMD_BATCHED_GATHER");});
            test("grid",false,[&](auto*){device.props.maxGridSize[0]=1;});
            test("threads",false,[&](auto*){device.props.maxThreadsPerBlock=64;});
            test("wave",false,[&](auto*){device.props.warpSize=64;});
            test("scratch",false,[&](auto*){metadata->global_scratch_size=1;});
            test("profile scratch",false,[&](auto*){metadata->profile_scratch_size=1;});
            test("shared",false,[&](auto*){metadata->shared_memory=1;});
            test("argument count",false,[&](auto*){metadata->argument_count=15;});
            test("missing ABI",false,[&](auto*){metadata->argument_abi.pop_back();});
            test("pointer type",false,[&](auto*){metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});
            test("scalar width",false,[&](auto*){metadata->argument_abi[3].size=8;});
            test("block",false,[&](auto*){metadata->block_size=128;});
            test("inner stride",false,[](auto*out){out->src[0]->nb[0]*=2;});
            test("index fractional stride",false,[](auto*out){out->src[1]->nb[0]+=1;});
            test("table batch shape",false,[](auto*out){out->src[0]->ne[2]+=1;});
            test("index batch shape",false,[](auto*out){out->src[1]->ne[1]+=1;});
            test("output shape",false,[](auto*out){out->ne[2]+=1;});
            test("output stride",false,[](auto*out){out->nb[0]+=4;});
            test("output type",false,[](auto*out){out->type=GGML_TYPE_F16;});
            test("index type",false,[](auto*out){out->src[1]->type=GGML_TYPE_I64;});
            test("source span",false,[](auto*out){out->src[0]->nb[2]=size_t(INT32_MAX)*8;});
            test("source pointer alignment",false,[](auto*out){out->src[0]->data=reinterpret_cast<void*>(UINT64_C(0x100000001));});
            test("index pointer alignment",false,[](auto*out){out->src[1]->data=reinterpret_cast<void*>(UINT64_C(0x200000002));});
            test("output pointer alignment",false,[](auto*out){out->data=reinterpret_cast<void*>(UINT64_C(0x300000002));});
            test("shared root",false,[](auto*out){out->view_src=out->src[0];out->view_offs=0;});
        }
        *metadata=saved;
    }
    device.aot.reset();check(loaded==unloaded,"module lifecycle");
    std::printf("BATCHED_GATHER_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches);
}
