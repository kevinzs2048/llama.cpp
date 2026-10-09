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
 check(expected!=nullptr,"submission expectation");
 const char * kernel=nullptr;
 switch(expected->src[0]->type){
  case GGML_TYPE_Q4_0:kernel="flagos_mul_mat_id_q4_0_f32";break;
  case GGML_TYPE_Q4_K:kernel="flagos_mul_mat_id_q4_k_f32";break;
  case GGML_TYPE_Q4_1:kernel="flagos_mul_mat_id_q4_1_f32";break;
  case GGML_TYPE_Q8_0:kernel="flagos_mul_mat_id_q8_0_f32_finite_compensation";break;
  default:check(false,"expected packed weight dtype");
 }
 check(symbols.at(reinterpret_cast<uintptr_t>(f))==kernel,"dtype-specific symbol");check(gx==unsigned(expected->ne[0])&&gy==unsigned(expected->ne[1]*expected->ne[2])&&gz==1&&bx==128&&by==1&&bz==1,"independent grid and workgroup");check(*static_cast<const void**>(args[0])==expected->src[0]->data&&*static_cast<const void**>(args[1])==expected->src[0]->data,"aliased packed weight views");for(int i=1;i<3;++i)check(*static_cast<const void**>(args[i+1])==expected->src[i]->data,"activation and ID pointers");check(*static_cast<void**>(args[4])==expected->data,"output pointer order");
 const auto*w=expected->src[0];const auto*x=expected->src[1];const auto*id=expected->src[2];
 const int values[]={int(w->ne[0]),int(w->ne[1]),int(id->ne[0]),int(id->ne[1]),int(x->ne[1]),int(w->ne[2]),int(w->nb[1]),int(w->nb[2]),int(x->nb[1]/4),int(x->nb[2]/4),int(id->nb[0]/4),int(id->nb[1]/4)};for(int i=0;i<12;++i)check(*static_cast<int*>(args[i+5])==values[i],"scalar argument order");
 const auto*metadata=static_cast<amd_device_context*>(expected->extra)->aot->find(kernel);check(shared==unsigned(metadata->shared_memory),"LDS request");std::fill_n(static_cast<float*>(expected->data),ggml_nelements(expected),-321.f);++launches;return fail_launch?hipErrorUnknown:hipSuccess;
}
int main(int argc,char**argv){
 check(argc==2||(argc==3&&std::strcmp(argv[2],"--stale-q8")==0),"package");setenv("FLAGOS_AMD_MOE","1",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE","0",1);setenv("FLAGOS_AMD_FUSIONS","none",1);
 amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;hipGetDeviceProperties(&device.props,0);device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"package initialization");
 if(argc==3){
  auto *ctx=ggml_init({1048576,nullptr,true});auto *w=ggml_new_tensor_3d(ctx,GGML_TYPE_Q8_0,64,3,4);auto *x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,64,1,2);auto *ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,2);auto *out=ggml_mul_mat_id(ctx,w,x,ids);
  check(device.aot->find("flagos_mul_mat_id_q8_0_f32")!=nullptr,"old unsafe implementation present");check(device.aot->find("flagos_mul_mat_id_q8_0_f32_finite_compensation")==nullptr,"safe implementation absent");check(!amd_device_supports_op(&device.device_iface,out),"old Q8 package rejected at placement");check(launches==0,"no stale kernel submission");ggml_free(ctx);device.aot->reset();check(loaded==unloaded,"stale package module teardown");std::printf("STALE_Q8_PACKAGE_REJECTED checks=%d mock_submissions=%d actual_GPU_calls=0\n",checks,launches);return 0;
 }
 for(int kind=0;kind<4;++kind)for(int padded=0;padded<2;++padded){
  ggml_init_params params={4*1024*1024,nullptr,true};auto*ctx=ggml_init(params);int k=kind==1?768:96;auto*w=ggml_new_tensor_3d(ctx,kind==1?GGML_TYPE_Q4_K:kind==2?GGML_TYPE_Q4_1:kind==3?GGML_TYPE_Q8_0:GGML_TYPE_Q4_0,k,7,4);auto*x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,2,3);auto*ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,3);size_t row_bytes=ggml_row_size(w->type,k);
  if(padded){w->nb[1]=row_bytes+6;w->nb[2]=w->nb[1]*7+10;w->nb[3]=w->nb[2]*4;x->nb[1]=(k+7)*4;x->nb[2]=x->nb[1]*2+9*4;x->nb[3]=x->nb[2]*3;ids->nb[0]=8;ids->nb[1]=28;ids->nb[2]=ids->nb[3]=84;}
  auto*out=ggml_mul_mat_id(ctx,w,x,ids);out->extra=&device;check(amd_device_supports_op(&device.device_iface,out),"unbound placement");
  std::vector<uint8_t>wb(ggml_nbytes(w)+64);std::vector<float>xb(ggml_nbytes(x)/4+32),ob(ggml_nelements(out)+32,13579.25f);std::vector<int32_t>ib(ggml_nbytes(ids)/4+32);
  w->data=wb.data()+6;x->data=xb.data()+5;ids->data=ib.data()+5;out->data=ob.data()+5;check(amd_device_supports_op(&device.device_iface,out),"bound placement");expected=out;
  auto reject=[&](const char*name,const std::function<void()>&modify,const std::function<void()>&restore){modify();check(!amd_device_supports_op(&device.device_iface,out),name);restore();check(amd_device_supports_op(&device.device_iface,out),"restored valid contract");};
  auto copy_w=*w;auto copy_x=*x;auto copy_ids=*ids;auto copy_out=*out;
  reject("wrong weight dtype",[&]{w->type=GGML_TYPE_Q5_K;},[&]{*w=copy_w;});
  reject("wrong activation dtype",[&]{x->type=GGML_TYPE_F16;},[&]{*x=copy_x;});
  reject("wrong ID dtype",[&]{ids->type=GGML_TYPE_I64;},[&]{*ids=copy_ids;});
  reject("wrong output dtype",[&]{out->type=GGML_TYPE_F16;},[&]{*out=copy_out;});
  reject("wrong K",[&]{x->ne[0]=31;},[&]{*x=copy_x;});
  reject("nondivisible input picks",[&]{x->ne[1]=3;},[&]{*x=copy_x;});
  reject("wrong tokens",[&]{ids->ne[1]=4;},[&]{*ids=copy_ids;});
  reject("wrong output rows",[&]{out->ne[0]=8;},[&]{*out=copy_out;});
  reject("unsupported outer weights dimension",[&]{w->ne[3]=2;},[&]{*w=copy_w;});
  reject("unknown op parameters",[&]{out->op_params[0]=1;},[&]{*out=copy_out;});
#if GGML_BACKEND_API_VERSION >= 3
  check(ggml_prec_set_acc(out,GGML_PREC_F32),"set actual accumulator precision API");check(amd_device_supports_op(&device.device_iface,out),"F32 accumulator contract");*out=copy_out;
  check(ggml_prec_set_src(out,GGML_PREC_F32,1),"set actual activation precision API");check(amd_device_supports_op(&device.device_iface,out),"packed weights preserve F32 activation");
  expected=out;amd_backend_context precision_context(&device,reinterpret_cast<hipStream_t>(uintptr_t(1)));check(amd_launch_moe(&precision_context,out),"F32 precision uses qualified packed ABI");*out=copy_out;
#else
  reject("API2 reserved accumulator precision",[&]{out->op_params[0]=GGML_PREC_F32;},[&]{*out=copy_out;});
  reject("API2 reserved activation precision",[&]{out->op_params[3]=GGML_PREC_F32;},[&]{*out=copy_out;});
#endif
  reject("unknown activation precision",[&]{out->op_params[3]=12345;},[&]{*out=copy_out;});

  reject("overlapping weight rows",[&]{w->nb[1]=row_bytes-2;},[&]{*w=copy_w;});
  reject("odd packed row stride",[&]{w->nb[1]+=1;},[&]{*w=copy_w;});
  reject("odd packed expert stride",[&]{w->nb[2]+=1;},[&]{*w=copy_w;});
  reject("non-block K",[&]{w->ne[0]-=1;x->ne[0]-=1;},[&]{*w=copy_w;*x=copy_x;});
  reject("packed span exceeds i32",[&]{w->nb[2]=size_t(INT32_MAX-1);},[&]{*w=copy_w;});
  reject("output product exceeds i32",[&]{out->ne[0]=INT32_MAX;out->ne[1]=INT32_MAX;out->ne[2]=INT32_MAX;},[&]{*out=copy_out;});
  reject("misaligned stride",[&]{x->nb[1]+=1;},[&]{*x=copy_x;});
  reject("misaligned pointer",[&]{ids->data=static_cast<uint8_t*>(ids->data)+1;},[&]{*ids=copy_ids;});
  reject("output alias",[&]{out->data=x->data;},[&]{*out=copy_out;});
  reject("disabled policy",[]{setenv("FLAGOS_AMD_MOE","0",1);},[]{setenv("FLAGOS_AMD_MOE","1",1);});
  amd_moe_launch l;check(amd_describe_moe(&device,out,&l),"descriptor");auto*m=const_cast<flagos_amd::kernel_metadata*>(device.aot->find(l.kernel));auto saved=*m;
  reject("wrong tile M",[&]{m->tile_m=8;},[&]{*m=saved;});reject("wrong tile K",[&]{m->tile_k=512;},[&]{*m=saved;});reject("wrong block",[&]{m->block_size=512;},[&]{*m=saved;});
  reject("wrong ABI scalar width",[&]{m->argument_abi[5].size=8;},[&]{*m=saved;});reject("wrong ABI pointer kind",[&]{m->argument_abi[2].kind=flagos_amd::kernel_argument_kind::by_value;},[&]{*m=saved;});reject("scratch requirement",[&]{m->global_scratch_size=4;},[&]{*m=saved;});
  auto saved_props=device.props;reject("grid capacity",[&]{device.props.maxGridSize[1]=5;},[&]{device.props=saved_props;});if(m->shared_memory+m->static_shared_memory>0)reject("LDS capacity",[&]{device.props.sharedMemPerBlock=1;},[&]{device.props=saved_props;});
  amd_backend_context context(&device,reinterpret_cast<hipStream_t>(uintptr_t(1)));ggml_backend backend{};backend.context=&context;backend.device=&device.device_iface;auto*g=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(g,out);check(amd_backend_graph_compute(&backend,g)==GGML_STATUS_SUCCESS,"direct graph execution");check(launches==(kind*2+padded+1)*(GGML_BACKEND_API_VERSION>=3?2:1),"one launch per requested op");check(ob[5]==-321.f&&ob[4]==13579.25f&&ob[5+ggml_nelements(out)]==13579.25f,"bounded destination");
  if(kind==3&&padded==1){fail_launch=true;check(amd_backend_graph_compute(&backend,g)==GGML_STATUS_FAILED,"post-write submission error");check(launches==(GGML_BACKEND_API_VERSION>=3?17:9),"no backend retry after write");fail_launch=false;}
  ggml_free(ctx);
 }
 device.aot->reset();check(loaded==unloaded,"module teardown");std::printf("QUANTIZED_MOE_HOST_COMPLETE checks=%d mock_submissions=%d actual_GPU_calls=0\n",checks,launches);
}
