// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGOS_PROVIDER_SOURCE
#define FLAGOS_PROVIDER_SOURCE "../flagos-amd.cpp"
#endif
#include FLAGOS_PROVIDER_SOURCE
#include <cstdio>
#include <functional>
#include <map>
static int checks=0,launches=0,loaded=0,unloaded=0;
static bool fail_launch=false;
static std::map<uintptr_t,std::string> symbols;
static ggml_tensor * expected=nullptr;
static void check(bool value,const char*name){++checks;if(!value){std::fprintf(stderr,"CHECK_FAILED %s\n",name);std::exit(2);}}
extern "C" hipError_t hipSetDevice(int){return hipSuccess;}
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t*p,int){*p={};p->warpSize=32;p->maxThreadsPerBlock=1024;p->sharedMemPerBlock=65536;p->maxGridSize[0]=INT32_MAX;p->maxGridSize[1]=p->maxGridSize[2]=65535;std::strcpy(p->gcnArchName,"gfx1150");return hipSuccess;}
extern "C" hipError_t hipModuleLoadData(hipModule_t*m,const void*){*m=reinterpret_cast<hipModule_t>(uintptr_t(++loaded));return hipSuccess;}
extern "C" hipError_t hipModuleUnload(hipModule_t){++unloaded;return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t*f,hipModule_t,const char*name){uintptr_t id=symbols.size()+1;symbols[id]=name;*f=reinterpret_cast<hipFunction_t>(id);return hipSuccess;}
extern "C" hipError_t hipStreamSynchronize(hipStream_t){return hipSuccess;}
extern "C" hipError_t hipStreamDestroy(hipStream_t){return hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,hipStream_t,void**args,void**){
 check(expected!=nullptr,"submission expectation");auto*device=static_cast<amd_device_context*>(expected->extra);amd_argsort_launch l;check(amd_describe_argsort(device,expected,&l),"launch description");check(symbols.at(reinterpret_cast<uintptr_t>(f))==device->aot->find(l.kernel)->symbol,"symbol");check(gx==l.rows&&gy==1&&gz==1&&bx==unsigned(device->aot->find(l.kernel)->num_warps*32)&&by==1&&bz==1,"grid and workgroup");check(*static_cast<const void**>(args[0])==expected->src[0]->data,"source pointer");check(*static_cast<void**>(args[1])==expected->data,"output pointer");check(*static_cast<int*>(args[2])==l.columns,"columns");check(*static_cast<int*>(args[3])==l.source_row_stride,"element stride");check(*static_cast<int*>(args[4])==l.descending,"sort order");check(shared==unsigned(device->aot->find(l.kernel)->shared_memory),"LDS request");std::fill_n(static_cast<int32_t*>(expected->data),ggml_nelements(expected),-321);++launches;return fail_launch?hipErrorUnknown:hipSuccess;
}
int main(int argc,char**argv){
 check(argc==2,"package");unsetenv("FLAGOS_AMD_ARGSORT");setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;hipGetDeviceProperties(&device.props,0);device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"package initialization");
 for(int columns:{1,40,64,129,256,257,1024,2049,4096})for(int desc:{0,1})for(int padding:{0,7}){
  ggml_init_params params={1048576,nullptr,true};auto*ctx=ggml_init(params);auto*x=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,columns,2,3,2);x->nb[1]=(columns+padding)*4;x->nb[2]=x->nb[1]*2;x->nb[3]=x->nb[2]*3;auto*out=ggml_argsort(ctx,x,desc?GGML_SORT_ORDER_DESC:GGML_SORT_ORDER_ASC);out->extra=&device;
  unsetenv("FLAGOS_AMD_ARGSORT");check(!amd_device_supports_op(&device.device_iface,out),"default policy off");setenv("FLAGOS_AMD_ARGSORT","1",1);check(amd_device_supports_op(&device.device_iface,out),"unbound placement");std::vector<float>ib(ggml_nbytes(x)/4+32,13579.25f);const auto before=ib;std::vector<int32_t>ob(ggml_nelements(out)+32,1357911);x->data=ib.data()+5;out->data=ob.data()+5;expected=out;check(amd_device_supports_op(&device.device_iface,out),"bound placement");auto copy_x=*x;auto copy_out=*out;
  auto reject=[&](const char*name,const std::function<void()>&modify,const std::function<void()>&restore){modify();check(!amd_device_supports_op(&device.device_iface,out),name);restore();check(amd_device_supports_op(&device.device_iface,out),"restored valid contract");};
  reject("input dtype",[&]{x->type=GGML_TYPE_F16;},[&]{*x=copy_x;});reject("output dtype",[&]{out->type=GGML_TYPE_F32;},[&]{*out=copy_out;});reject("output layout",[&]{out->nb[1]+=4;},[&]{*out=copy_out;});reject("shape mismatch",[&]{out->ne[1]+=1;},[&]{*out=copy_out;});reject("invalid order",[&]{out->op_params[0]=2;},[&]{*out=copy_out;});reject("unknown params",[&]{out->op_params[1]=1;},[&]{*out=copy_out;});reject("inner stride",[&]{x->nb[0]=8;},[&]{*x=copy_x;});reject("overlapping rows",[&]{x->nb[1]=size_t(columns-1)*4;},[&]{*x=copy_x;});reject("outer stride gap",[&]{x->nb[2]+=4;},[&]{*x=copy_x;});reject("unaligned row stride",[&]{x->nb[1]+=1;},[&]{*x=copy_x;});reject("unaligned source pointer",[&]{x->data=static_cast<uint8_t*>(x->data)+1;},[&]{*x=copy_x;});reject("unaligned output pointer",[&]{out->data=static_cast<uint8_t*>(out->data)+1;},[&]{*out=copy_out;});reject("exact alias",[&]{out->data=x->data;},[&]{*out=copy_out;});reject("partial alias",[&]{out->data=static_cast<float*>(x->data)+1;},[&]{*out=copy_out;});reject("disabled policy",[]{setenv("FLAGOS_AMD_ARGSORT","0",1);},[]{setenv("FLAGOS_AMD_ARGSORT","1",1);});
  amd_argsort_launch l;check(amd_describe_argsort(&device,out,&l),"descriptor");auto*m=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(l.kernel));auto saved=*m;
  reject("wrong block",[&]{m->block_size*=2;},[&]{*m=saved;});reject("wrong warps",[&]{m->num_warps=m->num_warps==4?8:4;},[&]{*m=saved;});reject("wrong wave",[&]{m->warp_size=64;},[&]{*m=saved;});reject("wrong ABI count",[&]{m->argument_count=4;},[&]{*m=saved;});reject("scalar ABI width",[&]{m->argument_abi[2].size=8;},[&]{*m=saved;});reject("pointer ABI kind",[&]{m->argument_abi[0].kind=flagos_amd::kernel_argument_kind::by_value;},[&]{*m=saved;});reject("scratch",[&]{m->global_scratch_size=4;},[&]{*m=saved;});auto props=device.props;reject("grid capacity",[&]{device.props.maxGridSize[0]=11;},[&]{device.props=props;});reject("LDS capacity",[&]{device.props.sharedMemPerBlock=1;},[&]{device.props=props;});reject("thread capacity",[&]{device.props.maxThreadsPerBlock=127;},[&]{device.props=props;});
  amd_backend_context context(&device,reinterpret_cast<hipStream_t>(uintptr_t(1)));ggml_backend backend{};backend.context=&context;backend.device=&device.device_iface;auto*g=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(g,out);int n_before=launches;check(amd_backend_graph_compute(&backend,g)==GGML_STATUS_SUCCESS,"direct graph execution");check(launches==n_before+1,"one launch per node");check(ob[5]==-321&&ob[4]==1357911&&ob[5+ggml_nelements(out)]==1357911,"bounded write");check(ib==before,"input unchanged");
  if(columns==4096&&desc==1&&padding==7){fail_launch=true;check(amd_backend_graph_compute(&backend,g)==GGML_STATUS_FAILED,"post-write error");check(launches==n_before+2,"no retry after write");fail_launch=false;device.aot->reset();check(!amd_device_supports_op(&device.device_iface,out),"missing package refuses placement");}
  ggml_free(ctx);
 }
 check(loaded==unloaded,"module teardown");std::printf("ARGSORT_HOST_COMPLETE checks=%d mock_submissions=%d actual_GPU_calls=0\n",checks,launches);
}
