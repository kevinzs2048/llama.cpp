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
    const auto * table = expected_node->src[0];
    const bool quantized = ggml_is_quantized(table->type);
    const int pointers = quantized ? 4 : 3;
    const int block = quantized ? ggml_blck_size(table->type) : 256;
    check(symbols.at(reinterpret_cast<uintptr_t>(function)).find(expected_kernel) == 0, "kernel symbol");
    check(gx == expected_node->ne[1] && gy == (expected_node->ne[0] + block - 1) / block && gz == 1, "gather grid");
    check(*static_cast<void **>(args[0]) == table->data, "table binding");
    if (quantized) check(*static_cast<void **>(args[1]) == table->data, "F16 weight alias");
    check(*static_cast<void **>(args[pointers-2]) == expected_node->src[1]->data &&
        *static_cast<void **>(args[pointers-1]) == expected_node->data, "indices and output bindings");
    check(*static_cast<int *>(args[pointers]) == expected_node->ne[0], "column scalar");
    if (!quantized && table->type != GGML_TYPE_F32) check(*static_cast<int *>(args[4]) == table->ne[1], "table row scalar");
    std::fill_n(static_cast<float *>(expected_node->data), ggml_nelements(expected_node), -123.f);
    ++launches; return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc==2,"package");setenv("FLAGOS_AMD_DENSE_FLOAT_OPS","1",1);setenv("FLAGOS_AMD_BATCHED_GATHER","0",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;
    device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");int checks=0;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q8_0,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}) {
        const bool quant=ggml_is_quantized(type);
        auto *ctx=ggml_init({1024*1024,nullptr,true});auto*table=ggml_new_tensor_2d(ctx,type,512,7);auto*ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,3);auto*out=ggml_get_rows(ctx,table,ids);
        const char *name=quant?amd_quantized_get_rows_kernel(amd_quantized_kind(type)):type==GGML_TYPE_F32?"flagos_get_rows_f32":amd_half_get_rows_kernel_name(out);
        auto *metadata=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(name));check(metadata,"metadata");const auto saved=*metadata;
        auto supported=[&](){return quant?amd_supports_get_rows(&device,out):type==GGML_TYPE_F32?amd_supports_dense_get_rows(&device,out):amd_supports_half_get_rows(&device,out);};
        auto test=[&](const char*label,bool old_accept,const std::function<void()>&mutate){
            for(bool bound:{false,true}) {
                *metadata=saved;hipGetDeviceProperties(&device.props,0);table->data=bound?reinterpret_cast<void*>(UINT64_C(0x100000000)):nullptr;ids->data=bound?reinterpret_cast<void*>(UINT64_C(0x200000000)):nullptr;out->data=bound?reinterpret_cast<void*>(UINT64_C(0x300000000)):nullptr;
                mutate();
#ifdef FLAGOS_BEFORE_FIXTURE
                check(supported()==old_accept,label);
#else
                (void)old_accept;check(!supported(),label);check(!amd_device_supports_op(&device.device_iface,out),"placement rejection");
#endif
                ++checks;
            }
        };
        hipGetDeviceProperties(&device.props,0);check(supported(),"valid unbound");++checks;
        std::vector<uint16_t> source(ggml_nbytes(table)/2+32, 0x1234);std::vector<int> index_data(ggml_nelements(ids)+32, 1);std::vector<float> results(ggml_nelements(out)+32, 37.f);
        table->data=source.data()+8;ids->data=index_data.data()+8;out->data=results.data()+8;
        check(supported() && amd_device_supports_op(&device.device_iface,out),"valid bound");++checks;
        expected_node=out;expected_kernel=name;
        { amd_backend_context context(&device,nullptr);ggml_backend backend{};backend.context=&context;ggml_tensor*nodes[]={out};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=1;
          check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"valid dispatch"); }
        check(std::all_of(results.begin()+8,results.end()-24,[](float v){return v==-123.f;}),"output");
        check(std::all_of(results.begin(),results.begin()+8,[](float v){return v==37.f;})&&std::all_of(results.end()-24,results.end(),[](float v){return v==37.f;}),"guards");
        check(std::all_of(source.begin(),source.end(),[](uint16_t v){return v==0x1234;})&&std::all_of(index_data.begin(),index_data.end(),[](int v){return v==1;}),"readonly sources");expected_node=nullptr;
        test("thread cap",true,[&]{device.props.maxThreadsPerBlock=64;});
        test("wave cap",true,[&]{device.props.warpSize=64;});
        test("grid x cap",true,[&]{device.props.maxGridSize[0]=1;});
        test("grid y cap",true,[&]{device.props.maxGridSize[1]=1;});
        test("dynamic LDS",type==GGML_TYPE_F32||quant,[&]{metadata->shared_memory=70000;});
        test("static LDS",type==GGML_TYPE_F32||quant,[&]{metadata->static_shared_memory=70000;});
        test("scratch",type==GGML_TYPE_F32||quant,[&]{metadata->global_scratch_size=1;});
        test("profile scratch",type==GGML_TYPE_F32||quant,[&]{metadata->profile_scratch_size=1;});
        test("wrong count",quant,[&]{metadata->argument_count+=1;});
        test("missing ABI",true,[&]{metadata->argument_abi.pop_back();});
        test("pointer ABI",true,[&]{metadata->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});
        test("scalar width",true,[&]{metadata->argument_abi.back().size=8;});
        test("wrong block",quant,[&]{metadata->block_size=128;});
        test("weight alignment",type==GGML_TYPE_F32||quant,[&]{table->data=reinterpret_cast<void*>(UINT64_C(0x100000001));});
        test("index alignment",type==GGML_TYPE_F32||quant,[&]{ids->data=reinterpret_cast<void*>(UINT64_C(0x200000002));});
        test("output alignment",type==GGML_TYPE_F32||quant,[&]{out->data=reinterpret_cast<void*>(UINT64_C(0x300000002));});
        *metadata=saved;ggml_free(ctx);
    }
    device.aot.reset();check(loaded==unloaded&&launches==9&&allocations==0,"no GPU submission");
    std::printf("GATHER_RESOURCES_HOST_COMPLETE API=%d checks=%d mock_launches=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,launches);
}
