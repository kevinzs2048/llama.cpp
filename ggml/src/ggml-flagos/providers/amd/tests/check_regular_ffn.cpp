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
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
        unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    (void) function; (void) gx; (void) gy; (void) gz; (void) args;
    ++launches; check(false, "unexpected launch in resource-only host fixture");
    return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc == 2, "package");
    setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "1", 1);
    setenv("FLAGOS_AMD_FUSIONS", "ffn_swiglu", 1);
    setenv("FLAGOS_AMD_PREFILL_F16_GEMM", "1", 1);
    setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "1", 1);
    setenv("FLAGOS_AMD_TILED_F16_FFN", "0", 1);
    setenv("FLAGOS_AMD_PACKED_F16_FFN", "1", 1);
    setenv("FLAGOS_AMD_GRAPH_CAPTURE", "0", 1);
    amd_device_context device; device.arch = "gfx1150"; device.device_iface.context = &device;
    device.aot = std::make_unique<flagos_amd::kernel_registry>();
    check(device.aot->initialize(argv[1], 0, "gfx1150", true), "mock modules");
    hipGetDeviceProperties(&device.props, 0); amd_fill_device_profile(device, &device.profile);
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    auto * w0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 1024, 64);
    auto * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 1024, 64);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 32);
    auto * gate = ggml_mul_mat(ctx, w0, x); auto * up = ggml_mul_mat(ctx, w1, x);
    auto * glu = ggml_swiglu_split(ctx, gate, up);
    const char * names[] = {"flagos_ffn_swiglu_f16_f32_grouped", "flagos_ffn_swiglu_f16_f32_batched_tiled", "flagos_ffn_swiglu_f16_f32_batched"};
    std::vector<flagos_amd::kernel_metadata> saved;
    for (const char * name : names) {auto * m=const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name));check(m,"FFN metadata");saved.push_back(*m);m->shared_memory=70000;}
#ifdef FLAGOS_BEFORE_FIXTURE
    check(amd_supports_ffn_swiglu_f16(&device, gate, up, glu), "before advertises illegal FFN");
    std::printf("REGULAR_FFN_BEFORE API=%d advertises_all_illegal=true GPU_calls=0\n",GGML_BACKEND_API_VERSION);
#else
    int checks=0;
    check(!amd_supports_ffn_swiglu_f16(&device, gate, up, glu), "after rejects all-illegal FFN");++checks;
    auto restore=[&] {for(size_t i=0;i<saved.size();++i)*const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[i]))=saved[i];hipGetDeviceProperties(&device.props,0);};
    restore();
    for (ggml_tensor * tensor : {w0, w1, x, glu}) {
        tensor->data = reinterpret_cast<void *>(uintptr_t(tensor == w0 || tensor == w1 ? 0x100001 : 0x200002));
        check(!amd_supports_ffn_swiglu_f16(&device, gate, up, glu), "fusion binding natural alignment");
        tensor->data = nullptr; ++checks;
    }
    void * weights=reinterpret_cast<void *>(uintptr_t(0x100000));void * input=reinterpret_cast<void *>(uintptr_t(0x200000));void * output=reinterpret_cast<void *>(uintptr_t(0x300000));
    amd_f16_ffn_launch launch;
    for(int variant=0;variant<3;++variant) {
        setenv("FLAGOS_AMD_GROUPED_F16_GEMM",variant==0?"1":"0",1);setenv("FLAGOS_AMD_TILED_F16_FFN",variant==1?"1":"0",1);
        auto test=[&](const char * label,bool legal,const std::function<void(flagos_amd::kernel_metadata *)> & mutate) {
            restore();for(int i=0;i<3;++i)if(i!=variant)const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[i]))->shared_memory=70000;
            auto * m=const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[variant]));mutate(m);
            const bool found=amd_describe_f16_ffn(&device,weights,weights,input,output,1024,64,32,&launch);
            check(found==legal,label);if(found)check(std::strcmp(launch.kernel,names[variant])==0,"selected variant");++checks;
        };
        test("legal",true,[](auto *){});
        test("wave",false,[&](auto *){device.props.warpSize=64;});
        test("threads",false,[&](auto *){device.props.maxThreadsPerBlock=16;});
        test("shared",false,[](auto *m){m->shared_memory=70000;});
        test("static shared",false,[](auto *m){m->static_shared_memory=70000;});
        test("scratch",false,[](auto *m){m->global_scratch_size=1;});
        test("profile scratch",false,[](auto *m){m->profile_scratch_size=1;});
        test("argument count",false,[](auto *m){++m->argument_count;});
        test("ABI missing",false,[](auto *m){m->argument_abi.pop_back();});
        test("pointer",false,[](auto *m){m->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});
        test("scalar",false,[](auto *m){m->argument_abi.back().size=8;});
        test("tile M",false,[](auto *m){m->tile_m=0;});
        test("tile N",false,[](auto *m){m->tile_n=0;});
        test("tile K",false,[](auto *m){m->tile_k=0;});
        test("grid X",false,[&](auto *){device.props.maxGridSize[0]=1;});
    }
    restore();setenv("FLAGOS_AMD_GROUPED_F16_GEMM","1",1);setenv("FLAGOS_AMD_TILED_F16_FFN","0",1);
    const_cast<flagos_amd::kernel_metadata *>(device.aot->find(names[0]))->shared_memory=70000;
    check(amd_describe_f16_ffn(&device,weights,weights,input,output,1024,64,32,&launch)&&!launch.grouped&&!launch.tiled,"grouped declines to ordinary");++checks;
    setenv("FLAGOS_AMD_TILED_F16_FFN","1",1);
    check(amd_describe_f16_ffn(&device,weights,weights,input,output,1024,64,32,&launch)&&launch.tiled,"grouped declines to tiled");++checks;
    restore();device.props.maxGridSize[0]=2;
    check(amd_describe_f16_ffn(&device,weights,weights,input,output,1024,64,1024,&launch)&&!launch.grouped&&launch.grid_x==2,"flattened grid decline to 2D");++checks;
    restore();
    for(int which=0;which<4;++which){void * ptrs[]={weights,weights,input,output};ptrs[which]=reinterpret_cast<void *>(uintptr_t(which<2?0x100001:0x200002));check(!amd_describe_f16_ffn(&device,ptrs[0],ptrs[1],ptrs[2],ptrs[3],1024,64,32,&launch),"natural alignment");++checks;}
    check(!amd_describe_f16_ffn(&device,weights,weights,input,output,INT32_MAX,64,32,&launch),"i32 products");++checks;
    setenv("FLAGOS_AMD_GROUPED_F16_GEMM","0",1);setenv("FLAGOS_AMD_TILED_F16_FFN","1",1);
    check(amd_describe_f16_ffn(&device,weights,weights,input,output,2048,6144,512,&launch)&&launch.tiled,"regular tiled");++checks;
    const char * packed_names[]={"flagos_ffn_swiglu_f16_packed_activation","flagos_pack_f32_f16"};
    for(const char * name:packed_names){auto * m=const_cast<flagos_amd::kernel_metadata *>(device.aot->find(name));check(m,"packed metadata");const auto original=*m;
        auto test=[&](const char * label,bool legal,const std::function<void()> & mutate){*m=original;hipGetDeviceProperties(&device.props,0);mutate();amd_f16_ffn_launch packed;check(amd_describe_packed_f16_ffn(&device,weights,weights,input,output,nullptr,2048,6144,512,launch,&packed)==legal,label);if(legal)check(packed.packed&&packed.pack_grid==4096&&packed.grid_x==192&&packed.grid_y==4,"packed geometry");++checks;};
        test("packed legal",true,[]{});test("packed shared",false,[&]{m->shared_memory=70000;});test("packed static shared",false,[&]{m->static_shared_memory=70000;});test("packed scratch",false,[&]{m->global_scratch_size=1;});test("packed profile scratch",false,[&]{m->profile_scratch_size=1;});test("packed count",false,[&]{++m->argument_count;});test("packed ABI",false,[&]{m->argument_abi.pop_back();});test("packed pointer",false,[&]{m->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;});test("packed scalar",false,[&]{m->argument_abi.back().size=8;});test("packed wave",false,[&]{device.props.warpSize=64;});test("packed threads",false,[&]{device.props.maxThreadsPerBlock=16;});test("packed grid",false,[&]{device.props.maxGridSize[0]=191;});*m=original;
    }
    hipGetDeviceProperties(&device.props,0);amd_f16_ffn_launch packed;
    check(!amd_describe_packed_f16_ffn(&device,weights,weights,input,output,reinterpret_cast<void *>(uintptr_t(0x400002)),2048,6144,512,launch,&packed),"workspace alignment");++checks;
    device.props.maxGridSize[0]=4000;
    check(!amd_describe_packed_f16_ffn(&device,weights,weights,input,output,nullptr,2048,6144,512,launch,&packed),"pack grid checked before write");++checks;
    check(amd_describe_f16_ffn(&device,weights,weights,input,output,2048,6144,512,&launch)&&launch.tiled,"regular remains legal when packed cannot launch");++checks;
    std::printf("REGULAR_FFN_AFTER API=%d checks=%d actual_GPU_calls=0 allocations=0 launches=0\n",GGML_BACKEND_API_VERSION,checks);
#endif
    ggml_free(ctx);device.aot.reset();check(loaded==unloaded&&launches==0&&allocations==0,"no hardware and balanced modules");
}
