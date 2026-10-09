// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGOS_PROVIDER_SOURCE
#define FLAGOS_PROVIDER_SOURCE "../flagos-amd.cpp"
#endif
#include FLAGOS_PROVIDER_SOURCE
#include <cstdio>
#include <map>
static int checks=0,submitted=0,copy_writes=0,late_writes=0,loaded=0,unloaded=0;
static int captures=0,replays=0,exec_destroys=0;
static bool active=false;
static std::map<uintptr_t,std::string> symbols;
static void check(bool b,const char*n){++checks;if(!b){std::fprintf(stderr,"CHECK_FAILED %s\n",n);std::exit(2);}}
struct call { int kind; const uint16_t * source; void * destination; size_t count; };
static std::vector<call> pending;
static std::map<uintptr_t,std::vector<call>> graphs,executables;
static void write(const call &c){if(c.kind==0){std::copy_n(c.source,c.count,static_cast<uint16_t*>(c.destination));++copy_writes;}else if(c.kind==1){std::fill_n(static_cast<float*>(c.destination),c.count,42.f);++late_writes;}else{std::fill_n(static_cast<int32_t*>(c.destination),c.count,42);++late_writes;}}
extern "C" hipError_t hipSetDevice(int){return hipSuccess;}
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t*p,int){*p={};p->warpSize=32;p->maxThreadsPerBlock=1024;p->sharedMemPerBlock=65536;p->maxGridSize[0]=INT32_MAX;p->maxGridSize[1]=p->maxGridSize[2]=65535;p->integrated=1;std::strcpy(p->gcnArchName,"gfx1150");return hipSuccess;}
extern "C" hipError_t hipModuleLoadData(hipModule_t*m,const void*){*m=reinterpret_cast<hipModule_t>(uintptr_t(++loaded));return hipSuccess;}
extern "C" hipError_t hipModuleUnload(hipModule_t){++unloaded;return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t*f,hipModule_t,const char*n){auto id=symbols.size()+1;symbols[id]=n;*f=reinterpret_cast<hipFunction_t>(id);return hipSuccess;}
extern "C" hipError_t hipStreamSynchronize(hipStream_t){return hipSuccess;}
extern "C" hipError_t hipStreamDestroy(hipStream_t){return hipSuccess;}
extern "C" hipError_t hipStreamBeginCapture(hipStream_t,hipStreamCaptureMode){check(!active,"capture nesting");active=true;pending.clear();++captures;return hipSuccess;}
extern "C" hipError_t hipStreamEndCapture(hipStream_t,hipGraph_t*g){check(active,"active capture");active=false;graphs[captures]=pending;*g=reinterpret_cast<hipGraph_t>(uintptr_t(captures));return hipSuccess;}
extern "C" hipError_t hipGraphInstantiate(hipGraphExec_t*e,hipGraph_t g,hipGraphNode_t*,char*,size_t){auto id=reinterpret_cast<uintptr_t>(g);executables[id]=graphs.at(id);*e=reinterpret_cast<hipGraphExec_t>(id);return hipSuccess;}
extern "C" hipError_t hipGraphDestroy(hipGraph_t g){check(graphs.erase(reinterpret_cast<uintptr_t>(g))==1,"destroy graph");return hipSuccess;}
extern "C" hipError_t hipGraphExecDestroy(hipGraphExec_t e){check(executables.erase(reinterpret_cast<uintptr_t>(e))==1,"destroy executable");++exec_destroys;return hipSuccess;}
extern "C" hipError_t hipGraphLaunch(hipGraphExec_t e,hipStream_t){for(const auto&c:executables.at(reinterpret_cast<uintptr_t>(e)))write(c);++replays;return hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,hipStream_t,void**a,void**){
 auto name=symbols.at(reinterpret_cast<uintptr_t>(f));call c{};
 if(name.find("flagos_copy_strided_f16_bits")==0)c={0,*static_cast<const uint16_t**>(a[0]),*static_cast<void**>(a[1]),size_t(*static_cast<int*>(a[18]))};
 else if(name.find("flagos_mul_mat_id_f16_f32")==0)c={1,nullptr,*static_cast<void**>(a[3]),size_t(*static_cast<int*>(a[5]))*size_t(*static_cast<int*>(a[6]))*size_t(*static_cast<int*>(a[7]))};
 else if(name.find("flagos_argsort_f32_i32")==0)c={2,nullptr,*static_cast<void**>(a[1]),size_t(gx)*size_t(*static_cast<int*>(a[2]))};
 else check(false,"unexpected kernel");
 if(active)pending.push_back(c);else write(c);++submitted;return hipSuccess;
}
int main(int argc,char**argv){
 check(argc==2,"package");setenv("FLAGOS_AMD_F16_COPY","1",1);setenv("FLAGOS_AMD_FUSIONS","none",1);setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS","0",1);amd_device_context device;device.arch="gfx1150";device.device_iface.context=&device;hipGetDeviceProperties(&device.props,0);device.aot=std::make_unique<flagos_amd::kernel_registry>();check(device.aot->initialize(argv[1],0,"gfx1150",true),"package");
 for(int kind:{1,2})for(int capture:{0,1}){
  const char*flag=kind==1?"FLAGOS_AMD_MOE":"FLAGOS_AMD_ARGSORT";const char*other=kind==1?"FLAGOS_AMD_ARGSORT":"FLAGOS_AMD_MOE";setenv(flag,"1",1);setenv(other,"0",1);setenv("FLAGOS_AMD_GRAPH_CAPTURE",capture?"all":"0",1);
  auto*ctx=ggml_init({1048576,nullptr,true});auto*source=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,32,2);std::vector<uint16_t>source_data(64,0x3c00);source->data=source_data.data();std::vector<std::vector<uint16_t>>states(3,std::vector<uint16_t>(64,0x7c00));ggml_tensor*nodes[4];
  for(int i=0;i<3;++i){auto*s=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,32,2);s->data=states[i].data();nodes[i]=ggml_cpy(ctx,source,s);nodes[i]->data=s->data;}
  std::vector<uint16_t>wdata(32*7*4,0x3c00);std::vector<float>xdata(96,1.f),fout(42,-99.f);std::vector<int32_t>iddata(6,0),iout(96,-99);
  if(kind==1){auto*w=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,32,7,4);auto*x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,32,1,3);auto*ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,3);w->data=wdata.data();x->data=xdata.data();ids->data=iddata.data();nodes[3]=ggml_mul_mat_id(ctx,w,x,ids);nodes[3]->data=fout.data();}
  else{auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,3);x->data=xdata.data();nodes[3]=ggml_argsort(ctx,x,GGML_SORT_ORDER_DESC);nodes[3]->data=iout.data();}
  ggml_cgraph graph{};graph.nodes=nodes;graph.n_nodes=4;
  {
   amd_backend_context context(&device,reinterpret_cast<hipStream_t>(uintptr_t(1)));ggml_backend backend{};backend.context=&context;backend.device=&device.device_iface;
   for(int i=0;i<3;++i)check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"initial execute/capture/replay");
   if(capture)check(!context.graph_captures.empty(),"replay cache established");
   for(auto&s:states)std::fill(s.begin(),s.end(),0x7c00);std::fill(fout.begin(),fout.end(),-99.f);std::fill(iout.begin(),iout.end(),-99);int cw=copy_writes,lw=late_writes,rp=replays,sl=submitted;auto key=amd_execution_policy_key();setenv(flag,"0",1);auto result=amd_backend_graph_compute(&backend,&graph);
#ifdef EXPECT_OLD_POLICY_GAP
   check(amd_execution_policy_key()==key,"old selector omitted from key");check(copy_writes-cw==3,"old earlier state writes reproduced");check(capture?(result==GGML_STATUS_SUCCESS&&late_writes-lw==1&&replays-rp==1):(result==GGML_STATUS_FAILED&&late_writes==lw),"old direct/capture policy gap");
#else
   check(amd_execution_policy_key()!=key,"selector changes policy key");check(result==GGML_STATUS_FAILED,"disabled graph rejected");check(copy_writes==cw&&late_writes==lw&&replays==rp&&submitted==sl,"reject before writes/submissions/replay");for(const auto&s:states)check(std::all_of(s.begin(),s.end(),[](uint16_t v){return v==0x7c00;}),"state unchanged");check(std::all_of(fout.begin(),fout.end(),[](float v){return v==-99.f;})&&std::all_of(iout.begin(),iout.end(),[](int v){return v==-99;}),"late output unchanged");
#endif
   std::printf("SELECTOR_POLICY_CASE kind=%d capture=%d earlier_writes=%d late_writes=%d replays=%d result=%d actual_GPU_calls=0\n",kind,capture,copy_writes-cw,late_writes-lw,replays-rp,int(result));setenv(flag,"1",1);check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"reenable original accepted policy");
#ifndef EXPECT_OLD_POLICY_GAP
   int destroyed=exec_destroys;auto built=context.stats.graph_plans_built.load();setenv(other,"1",1);check(amd_backend_graph_compute(&backend,&graph)==GGML_STATUS_SUCCESS,"valid policy change");check(context.stats.graph_plans_built.load()==built+1,"plan invalidated for valid selector change");if(capture)check(exec_destroys==destroyed+1,"old executable invalidated");
#endif
   check(hipStreamSynchronize(context.stream)==hipSuccess,"fixture stream idle");amd_clear_graph_captures(&context);context.graph_plans.clear();
  }
  ggml_free(ctx);
 }
 device.aot->reset();check(loaded==unloaded&&graphs.empty()&&executables.empty()&&!active,"balanced lifetime");std::printf("MOE_ARGSORT_POLICY_CHECK_COMPLETE checks=%d captures=%d replays=%d actual_GPU_calls=0\n",checks,captures,replays);
}
