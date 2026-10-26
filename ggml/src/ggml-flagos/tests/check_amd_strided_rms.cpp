#include <cstdio>
#include <cstdlib>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "CHECK_FAILED line=%d %s\n", __LINE__, #condition); std::exit(2); } } while (false)

#include "ggml.h"
#include "ggml-backend.h"
#include <cstring>
#include <algorithm>

extern "C" int reference_rms_norm(const float * input, float * output,
        int ne0, int ne1, int ne2, int ne3, int s1, int s2, int s3, float eps) {
    if (!input || !output || ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0 ||
        s1 < 0 || s2 < 0 || s3 < 0 || !(eps > 0.0f)) return -1;
    const int64_t span = ne0 + int64_t(ne1 - 1) * s1 + int64_t(ne2 - 1) * s2 + int64_t(ne3 - 1) * s3;
    ggml_init_params params { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return -2;
    ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, std::max(span, int64_t(ne0) * ne1 * ne2 * ne3));
    std::memcpy(storage->data, input, sizeof(float) * span);
    ggml_tensor * x = ggml_view_4d(ctx, storage, ne0, ne1, ne2, ne3,
        sizeof(float) * s1, sizeof(float) * s2, sizeof(float) * s3, 0);
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) { ggml_free(ctx); return -3; }
    ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
        ggml_backend_reg_get_proc_address(registry, "ggml_backend_set_n_threads"));
    if (!set_threads) { ggml_backend_free(backend); ggml_free(ctx); return -4; }
    set_threads(backend, 1);
    const int status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    ggml_backend_free(backend);
    if (status == GGML_STATUS_SUCCESS) std::memcpy(output, y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return status;
}

#include "../providers/amd/flagos-amd-op-contracts.h"

#include <limits>
#include <iostream>

static void check_contracts() {
    ggml_init_params params { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * original = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, 8, 3, 2);
    ggml_tensor * norm = ggml_rms_norm(ctx, original, 1e-6f);
    int passed = 0;
    const auto check = [&](auto mutate, bool expected) {
        ggml_tensor x = *original;
        ggml_tensor y = *norm;
        x.nb[1] *= 2; x.nb[2] *= 2; x.nb[3] *= 2;
        y.src[0] = &x;
        mutate(x, y);
        CHECK(flagos_amd::supports_strided_rms_norm_layout(&y, 256) == expected);
        ++passed;
    };
    check([](auto &, auto &) {}, true);
    check([](auto & x, auto &) { x.nb[1] = 0; }, true);
    check([](auto & x, auto &) { std::swap(x.nb[1], x.nb[2]); }, true);
    check([](auto & x, auto & y) { x.data = (void *) 0x10000; y.data = (void *) 0x100000; }, true);
    check([](auto & x, auto & y) { x.data = (void *) 0x10000; y.data = x.data; }, false);
    check([](auto & x, auto & y) { x.data = (void *) 0x10000; y.data = (void *) 0x10004; }, false);
    check([](auto & x, auto &) { x.nb[0] = 8; }, false);
    check([](auto & x, auto &) { x.nb[1] += 1; }, false);
    check([](auto & x, auto &) { x.type = GGML_TYPE_F16; }, false);
    check([](auto &, auto & y) { y.type = GGML_TYPE_F16; }, false);
    check([](auto &, auto & y) { y.ne[0] = 257; }, false);
    check([](auto & x, auto & y) { x.ne[0] = y.ne[0] = 0; }, false);
    check([](auto &, auto & y) { y.nb[1] *= 2; }, false);
    check([](auto &, auto & y) { y.src[0] = nullptr; }, false);
    check([](auto & x, auto &) { x.nb[2] = (size_t) INT32_MAX * 4; }, false);
    check([](auto & x, auto &) { x.data = (void *) 0x10001; }, false);
    for (float eps : { 0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        check([&](auto &, auto & y) { std::memcpy(y.op_params, &eps, sizeof(eps)); }, false);
    }
    ggml_free(ctx);
    std::cout << passed << " contract cases passed\n";
}

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" int reference_rms_norm(const float *, float *, int, int, int, int, int, int, int, float);

int main(int argc, char ** argv) {
    check_contracts();
    CHECK(argc == 2 || (argc == 3 && std::strcmp(argv[2], "reject") == 0));
    ggml_backend_load_all_from_path(argv[1]);
    auto dev = ggml_backend_dev_by_name("FlagOS:AMD:0");
    CHECK(dev != nullptr);
    auto backend = ggml_backend_dev_init(dev, nullptr);
    CHECK(backend != nullptr);
    float worst = 0.0f;
    int cases = 0;
    for (int cols : { 3, 128, 256 }) {
        for (int tokens : { 1, 5, 512 }) {
          for (int layout = 0; layout < 3; ++layout) {
            const int heads = 8, batches = 2;
            int s1 = cols * 2, s2 = s1 * heads;
            const int s3 = s2 * tokens;
            if (layout == 1) s1 = 0;
            if (layout == 2) std::swap(s1, s2);
            const int span = cols + (heads - 1) * s1 + (tokens - 1) * s2 + (batches - 1) * s3;
            const float eps = 1e-6f;
            std::vector<float> host(span), expected(cols * heads * tokens * batches), actual(expected.size());
            for (int i = 0; i < span; ++i) host[i] = std::sin(i * 0.017f) + std::cos(i * 0.073f) * 0.25f;
            CHECK(reference_rms_norm(host.data(), expected.data(), cols, heads, tokens, batches, s1, s2, s3, eps) == 0);
            ggml_init_params params { 1024 * 1024, nullptr, true };
            auto ctx = ggml_init(params);
            CHECK(ctx);
            auto storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, span);
            auto x = ggml_view_4d(ctx, storage, cols, heads, tokens, batches,
                sizeof(float) * s1, sizeof(float) * s2, sizeof(float) * s3, 0);
            auto y = ggml_rms_norm(ctx, x, eps);
            ggml_set_name(y, "strided-rms-provider-test");
            auto graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, y);
            if (argc == 3) {
                CHECK(!ggml_backend_supports_op(backend, y));
                ggml_free(ctx);
                ++cases;
                continue;
            }
            CHECK(ggml_backend_supports_op(backend, y));
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            CHECK(buffer);
            ggml_backend_tensor_set(storage, host.data(), 0, host.size() * sizeof(float));
            CHECK(ggml_backend_supports_op(backend, y));
            void * output_data = y->data;
            y->data = x->data;
            CHECK(!ggml_backend_supports_op(backend, y));
            y->data = static_cast<char *>(x->data) + sizeof(float);
            CHECK(!ggml_backend_supports_op(backend, y));
            y->data = output_data;
            const size_t inner_stride = x->nb[0];
            x->nb[0] *= 2;
            CHECK(!ggml_backend_supports_op(backend, y));
            x->nb[0] = inner_stride;
            float invalid_eps = 0.0f;
            std::memcpy(y->op_params, &invalid_eps, sizeof(float));
            CHECK(!ggml_backend_supports_op(backend, y));
            std::memcpy(y->op_params, &eps, sizeof(float));
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                CHECK(std::isfinite(actual[i]));
                worst = std::max(worst, std::abs(actual[i] - expected[i]));
                CHECK(std::abs(actual[i] - expected[i]) <= 3e-6f + 3e-6f * std::abs(expected[i]));
            }
            std::vector<float> restored(host.size());
            ggml_backend_tensor_get(storage, restored.data(), 0, restored.size() * sizeof(float));
            CHECK(restored == host);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ++cases;
          }
        }
    }
    ggml_backend_free(backend);
    std::cout << "RMS_PROVIDER_COMPLETE cases=" << cases << " max_abs=" << worst << "\n";
}
