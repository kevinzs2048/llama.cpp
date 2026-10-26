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
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,hipStream_t,void**,void**){check(false,"no submissions in alias fixture");return hipErrorInvalidValue;}
int main(int argc,char**argv){
    check(argc==2,"package");setenv("FLAGOS_AMD_REPEAT_OPS","0",1);amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"modules");hipGetDeviceProperties(&device.props,0);int checks=0;
#ifdef FLAGOS_BEFORE_FIXTURE
    const bool after=false;
#else
    const bool after=true;
#endif
    for(int kind=0;kind<6;++kind)for(int offset:{0,4,1024})for(int binding=0;binding<4;++binding){
        auto*ctx=ggml_init({1024*1024,nullptr,true});auto*root=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1024);auto*a=ggml_view_1d(ctx,root,256,0);auto*b=ggml_view_1d(ctx,root,256,offset);ggml_tensor*out=nullptr;
        switch(kind){case 0:out=ggml_cpy(ctx,a,b);break;case 1:out=ggml_add_inplace(ctx,a,b);break;case 2:out=ggml_mul_inplace(ctx,a,b);break;case 3:out=ggml_silu_inplace(ctx,a);break;case 4:out=ggml_scale_inplace(ctx,a,1.f);break;default:out=ggml_silu(ctx,a);break;}
        // GGML flattens nested views to the allocation root; known offsets are available before allocation.
        check(a->view_src==root&&b->view_src==root,"public view roots");
        if(binding){root->data=reinterpret_cast<void*>(UINT64_C(0x100000000));if(binding&1){a->data=root->data;b->data=static_cast<char*>(root->data)+offset;}if(binding&2)out->data=kind==0?static_cast<char*>(root->data)+offset:kind==5?reinterpret_cast<void*>(UINT64_C(0x200000000)):root->data;}
        const bool public_alias=kind<=2;const bool overlap=offset<1024;const bool partial=offset==4;const bool fully_bound=binding==3;
        bool legal=true;if(kind==0)legal=!overlap;else if(kind==1||kind==2)legal=!partial;
        const bool expected=kind==0&&(binding==1||binding==2)?false:legal||(!after&&(!fully_bound||!public_alias));
        bool supported=false;switch(kind){case 0:supported=amd_supports_copy(&device,out);break;case 1:supported=amd_supports_add(&device,out);break;case 2:supported=amd_supports_mul(&device,out);break;case 3:supported=amd_supports_silu(&device,out);break;case 4:supported=amd_supports_scale(&device,out);break;default:supported=amd_supports_silu(&device,out);break;}
        check(supported==expected,"known alias capability before allocation");++checks;ggml_free(ctx);
    }
    device.aot.reset();check(loaded==unloaded&&launches==0&&allocations==0,"lifecycle");std::printf("TENSOR_ALIAS_HOST_COMPLETE API=%d checks=%d after=%d actual_GPU_calls=0\n",GGML_BACKEND_API_VERSION,checks,int(after));
}
