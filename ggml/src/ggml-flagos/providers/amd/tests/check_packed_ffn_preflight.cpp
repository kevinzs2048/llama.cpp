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
static int allocations = 0, releases = 0, state_writes = 0, quantized_writes = 0;
static bool reject_dequant_grid = false;
static unsigned short * expected_dequant = nullptr;
static float * expected_output = nullptr;
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
static int pack_writes = 0, regular_writes = 0;
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned, unsigned,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    const auto & name = symbols.at(reinterpret_cast<uintptr_t>(function)); ++launches;
    if (name.find("flagos_dequant_q4_k_f16") == 0) {
        auto * output = *static_cast<unsigned short **>(args[2]);
        std::fill_n(output, gx * 256, 0x3c00);
    } else if (name == "flagos_pack_f32_f16") {
        auto * output = *static_cast<unsigned short **>(args[1]);
        std::fill_n(output, *static_cast<int *>(args[2]), 0x3c00); ++pack_writes;
    } else if (name == "flagos_ffn_swiglu_f16_f32_batched_tiled") {
        auto * output = *static_cast<float **>(args[3]);
        std::fill_n(output, *static_cast<int *>(args[5]) * *static_cast<int *>(args[6]), 42.f); ++regular_writes;
    } else check(false, "unexpected kernel in packed host fixture");
    return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2 || argc == 3, "package and optional stage");const bool after = argc == 2 || std::atoi(argv[2]) != 0;
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "ffn_swiglu", 1);setenv("FLAGOS_AMD_PREFILL_F16_GEMM", "1", 1);
    setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "0", 1);setenv("FLAGOS_AMD_TILED_F16_FFN", "1", 1);
    setenv("FLAGOS_AMD_PACKED_F16_FFN", "1", 1);setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.buffer_type_iface.context=&device;device.buffer_type_iface.iface=g_buffer_type_iface;device.buffer_type_iface.device=&device.device_iface;
    device.aot=std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1],0,"gfx1150",true),"mock modules");
    hipGetDeviceProperties(&device.props,0);amd_fill_device_profile(device,&device.profile);
    auto * ctx=ggml_init({1024*1024,nullptr,true});
    auto * w0=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,2048,6144);
    auto * w1=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,2048,6144);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2048,512);
    auto * gate=ggml_mul_mat(ctx,w0,x);auto * up=ggml_mul_mat(ctx,w1,x);auto * glu=ggml_swiglu_split(ctx,gate,up);
    auto * storage=amd_buffer_alloc(&device.buffer_type_iface,ggml_nbytes(w0)*2);
    check(storage!=nullptr,"weight storage");ggml_backend_buffer_set_usage(storage,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    w0->buffer=w1->buffer=storage;w0->data=ggml_backend_buffer_get_base(storage);
    w1->data=static_cast<uint8_t *>(w0->data)+ggml_nbytes(w0);
    std::vector<float> input(2048*512,1.f),output(6144*512,-99.f);
    x->data=input.data();glu->data=output.data();
    // Covered projection outputs are never touched by either fused implementation.
    gate->data=reinterpret_cast<void *>(uintptr_t(0x100000));up->data=reinterpret_cast<void *>(uintptr_t(0x4000000));
    amd_backend_context context(&device,nullptr);flagos_quantized_matmul_signature signature;
    check(flagos_describe_quantized_matmul(gate,&signature),"signature");
    check(amd_get_dequantized_weight(&context,w0,signature)!=nullptr,"warm gate cache");
    check(amd_get_dequantized_weight(&context,w1,signature)!=nullptr,"warm up cache");
    auto * packed=const_cast<flagos_amd::kernel_metadata *>(device.aot->find("flagos_ffn_swiglu_f16_packed_activation"));
    check(packed!=nullptr,"packed metadata");packed->shared_memory=70000;
    ggml_tensor * nodes[]={gate,up,glu};ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=3;
    flagos_plan_step step;step.candidate.id=flagos_pattern_id::ffn_swiglu;
    step.candidate.node_indices={0,1,2};step.candidate.required_output_node_indices={2};step.implementation_id=6;
    const int allocations_before=allocations;
    check(amd_execute_fusion(&context,&graph,step)==after,"packed failure declines before write");
    check(pack_writes==(after?0:1)&&regular_writes==(after?1:0),"submission order and fallback");
    check(allocations-allocations_before==(after?0:1),"illegal packed kernel avoids workspace allocation");
    check(std::all_of(output.begin(),output.end(),[&](float v){return v==(after?42.f:-99.f);}),"terminal output");
    amd_release_dequant_cache(&context);
    if(context.packed_ffn_workspace!=nullptr){hipFree(context.packed_ffn_workspace);context.packed_ffn_workspace=nullptr;}
    ggml_backend_buffer_free(storage);device.aot.reset();ggml_free(ctx);
    check(loaded==unloaded&&allocations==releases,"balanced module and allocation lifecycle");
    std::printf("PACKED_FFN_REPRO API=%d stage=%s pack_writes=%d regular_writes=%d workspace_allocations=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,after?"after":"before",pack_writes,regular_writes,allocations-allocations_before);
}
