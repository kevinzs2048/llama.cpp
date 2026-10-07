#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "CHECK_FAILED %s\n", message); std::exit(2); }
}

static void check(ggml_backend_t backend, int k, int rows, int columns, bool forced_f32) {
    ggml_init_params params { 4u << 20, nullptr, true };
    auto * wc = ggml_init(params); auto * ec = ggml_init(params);
    require(wc && ec, "contexts");
    auto * gate_weights = ggml_new_tensor_2d(wc, GGML_TYPE_Q4_K, k, rows);
    auto * up_weights = ggml_new_tensor_2d(wc, GGML_TYPE_Q4_K, k, rows);
    auto * x = ggml_new_tensor_2d(ec, GGML_TYPE_F32, k, columns);
    auto * gate = ggml_mul_mat(ec, gate_weights, x);
    auto * up = ggml_mul_mat(ec, up_weights, x);
    if (forced_f32) {
        ggml_mul_mat_set_prec(gate, GGML_PREC_F32);
        ggml_mul_mat_set_prec(up, GGML_PREC_F32);
    }
    auto * output = ggml_swiglu_split(ec, gate, up);
    auto * graph = ggml_new_graph(ec); ggml_build_forward_expand(graph, output);
    auto wb = ggml_backend_alloc_ctx_tensors(wc, backend);
    auto eb = ggml_backend_alloc_ctx_tensors(ec, backend);
    require(wb && eb, "buffers");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weights(size_t(k) * rows), activation(size_t(k) * columns);
    std::vector<unsigned char> quantized(ggml_nbytes(gate_weights));
    for (int projection = 0; projection < 2; ++projection) {
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) * 0.017f + projection) * 0.013f;
        require(ggml_quantize_chunk(GGML_TYPE_Q4_K, weights.data(), quantized.data(), 0, rows, k, nullptr) == quantized.size(), "quantization");
        ggml_backend_tensor_set(projection ? up_weights : gate_weights, quantized.data(), 0, quantized.size());
    }
    std::vector<float> reference(size_t(rows) * columns), actual(reference.size());
    for (int input_version = 0; input_version < 2; ++input_version) {
        for (size_t i = 0; i < activation.size(); ++i) activation[i] = std::cos(float(i) * 0.013f + input_version) * 0.10013f;
        ggml_backend_tensor_set(x, activation.data(), 0, ggml_nbytes(x));
        for (int packed = 0; packed < 2; ++packed) {
            setenv("FLAGOS_AMD_PACKED_F16_FFN", packed ? "1" : "0", 1);
            std::fprintf(stderr, "PACKED_FFN_CASE k=%d rows=%d columns=%d forced_F32=%d input_version=%d packed=%d\n", k, rows, columns, forced_f32, input_version, packed);
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
            auto & result = packed ? actual : reference;
            ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
            require(std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }), "finite output");
        }
        require(std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0, "packed output bitwise equality");
    }
    ggml_backend_buffer_free(eb); ggml_backend_buffer_free(wb);
    ggml_free(ec); ggml_free(wc);
}

int main(int argc, char ** argv) {
    require(argc == 2, "backend directory");
    ggml_backend_load_all_from_path(argv[1]);
    auto * backend = ggml_backend_init_by_name("FlagOS:AMD:0", nullptr);
    require(backend != nullptr, "AMD backend");
    check(backend, 2048, 6144, 512, false);
    check(backend, 2560, 9216, 512, false);
    check(backend, 2048, 6144, 33, false);
    check(backend, 2048, 6144, 512, true);
    ggml_backend_free(backend);
    std::puts("AMD packed FFN: model shapes, changed inputs, workspace reuse and F32 precision passed");
    return 0;
}
