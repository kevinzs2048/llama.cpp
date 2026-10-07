#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-flagos.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
using reference_fn = int (*)(const float *, const float *, const float *, const float *, const float *, const float *, float *, int, int, int, int, int, int, int);

int main(int argc, char ** argv) {
    CHECK(argc == 2);
    void * library = dlopen(argv[1], RTLD_NOW);
    CHECK(library != nullptr);
    auto reference = reinterpret_cast<reference_fn>(dlsym(library, "reference_gdn"));
    CHECK(reference != nullptr);
    ggml_backend_load_all();
    auto registry = ggml_backend_flagos_reg();
    CHECK(registry != nullptr && ggml_backend_reg_dev_count(registry) > 0);
    auto device = ggml_backend_reg_dev_get(registry, 0);
    auto backend = ggml_backend_dev_init(device, nullptr);
    auto buft = ggml_backend_dev_buffer_type(device);
    CHECK(backend != nullptr && buft != nullptr);
    const int shapes[][5] = {
        {128,32,16,512,1}, {128,32,16,37,3}, {128,32,16,1,3},
        {128,8,4,37,3}, {64,4,2,37,3}, {128,32,16,31,1},
    };
    for (const auto & shape : shapes) {
        const int size=shape[0], heads=shape[1], q_heads=shape[2], tokens=shape[3], snapshots=shape[4];
        ggml_init_params params { 1u << 20, nullptr, true };
        auto ctx=ggml_init(params);
        CHECK(ctx != nullptr);
        ggml_tensor * inputs[] = {
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, size, q_heads, tokens, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, size, q_heads, tokens, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, size, heads, tokens, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, tokens, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, heads, tokens, 1),
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, size, size, heads, 1),
        };
        auto output=ggml_gated_delta_net(ctx,inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],snapshots);
        std::vector<ggml_backend_buffer_t> buffers;
        std::vector<std::vector<float>> host;
        for (int index=0;index<7;++index) {
            auto tensor=index<6?inputs[index]:output;
            auto buffer=ggml_backend_buft_alloc_buffer(buft,ggml_nbytes(tensor));
            CHECK(buffer != nullptr);
            CHECK(ggml_backend_tensor_alloc(buffer,tensor,ggml_backend_buffer_get_base(buffer))==GGML_STATUS_SUCCESS);
            buffers.push_back(buffer);
            if (index<6) {
                host.emplace_back(ggml_nelements(tensor));
                for (size_t i=0;i<host.back().size();++i) {
                    float value=float(i%31)*0.001f-0.015f;
                    if (index==3) value=-0.02f+float(i%7)*0.001f;
                    if (index==4) value=0.4f+float(i%5)*0.03f;
                    host.back()[i]=value;
                }
                ggml_backend_tensor_set(tensor,host.back().data(),0,ggml_nbytes(tensor));
            }
        }
        CHECK(ggml_backend_dev_supports_op(device,output));
        void * saved=output->data;
        output->data=inputs[0]->data;
        CHECK(!ggml_backend_dev_supports_op(device,output));
        output->data=saved;
        size_t stride=inputs[0]->nb[1];
        inputs[0]->nb[1]+=sizeof(float);
        CHECK(!ggml_backend_dev_supports_op(device,output));
        inputs[0]->nb[1]=stride;
        std::vector<float> cpu(ggml_nelements(output),-123.0f);
        CHECK(reference(host[0].data(),host[1].data(),host[2].data(),host[3].data(),host[4].data(),host[5].data(),cpu.data(),size,heads,q_heads,tokens,1,1,snapshots)==0);
        auto graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,output);
        std::vector<float> baseline;
        float worst=0.0f;
        for (const char * enabled : {"0","1","0","1"}) {
            CHECK(setenv("FLAGOS_AMD_TILED_GDN_PREFILL",enabled,1)==0);
            std::vector<float> actual(cpu.size(),-123.0f);
            ggml_backend_tensor_set(output,actual.data(),0,ggml_nbytes(output));
            CHECK(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
            if (baseline.empty()) baseline=actual;
            CHECK(std::memcmp(actual.data(),baseline.data(),actual.size()*sizeof(float))==0);
            for (size_t i=0;i<actual.size();++i) {
                CHECK(std::isfinite(actual[i]));
                const float error=std::fabs(actual[i]-cpu[i]);
                worst=std::max(worst,error);
                CHECK(error<=3e-5f+3e-5f*std::fabs(cpu[i]));
            }
            for (int i=0;i<6;++i) {
                std::vector<float> after(host[i].size());
                ggml_backend_tensor_get(inputs[i],after.data(),0,ggml_nbytes(inputs[i]));
                CHECK(std::memcmp(after.data(),host[i].data(),after.size()*sizeof(float))==0);
            }
        }
        std::printf("GDN_PROVIDER_CASE size=%d heads=%d q_heads=%d tokens=%d snapshots=%d CPU_max_abs=%.9g bitwise=1 inputs_unchanged=1\n",size,heads,q_heads,tokens,snapshots,worst);
        for (auto buffer:buffers) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    dlclose(library);
    std::puts("GDN_PROVIDER_COMPLETE cases=6 toggle_trials=24");
}
