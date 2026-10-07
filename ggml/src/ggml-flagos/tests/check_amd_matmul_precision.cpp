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

static std::vector<float> run(bool forced_f32, bool prefill_f16, bool ffn = false) {
    constexpr int k = 1024, rows = 64, columns = 8;
    setenv("FLAGOS_AMD_PREFILL_F16_GEMM", prefill_f16 ? "1" : "0", 1);
    auto * backend = ggml_backend_init_by_name("FlagOS:AMD:0", nullptr);
    require(backend != nullptr, "AMD backend");
    ggml_init_params params { 4u << 20, nullptr, true };
    auto * wc = ggml_init(params); auto * ec = ggml_init(params);
    require(wc && ec, "contexts");
    auto * w = ggml_new_tensor_2d(wc, GGML_TYPE_Q4_K, k, rows);
    auto * x = ggml_new_tensor_2d(ec, GGML_TYPE_F32, k, columns);
    auto * out = ggml_mul_mat(ec, w, x);
    auto * up_weights = ffn ? ggml_new_tensor_2d(wc, GGML_TYPE_Q4_K, k, rows) : nullptr;
    auto * up = ffn ? ggml_mul_mat(ec, up_weights, x) : nullptr;
    ggml_mul_mat_set_prec(out, forced_f32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
    if (up) ggml_mul_mat_set_prec(up, forced_f32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
    require(ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out), "supports matrix");
    if (ffn) out = ggml_swiglu_split(ec, out, up);
    auto * graph = ggml_new_graph(ec); ggml_build_forward_expand(graph, out);
    auto wb = ggml_backend_alloc_ctx_tensors(wc, backend);
    auto eb = ggml_backend_alloc_ctx_tensors(ec, backend);
    require(wb && eb, "buffers");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weights(k*rows), activation(k*columns);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i)*0.017f)*0.13f;
    for (size_t i = 0; i < activation.size(); ++i) activation[i] = std::cos(float(i)*0.013f)*1.0013f;
    std::vector<unsigned char> quantized(ggml_nbytes(w));
    auto n = ggml_quantize_chunk(GGML_TYPE_Q4_K, weights.data(), quantized.data(), 0, rows, k, nullptr);
    require(n == quantized.size(), "quantization size");
    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    if (up_weights) ggml_backend_tensor_set(up_weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(x, activation.data(), 0, activation.size()*sizeof(float));
    std::fprintf(stderr, "PRECISION_CASE forced_f32=%d prefill_f16=%d ffn=%d\n", forced_f32, prefill_f16, ffn);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    ggml_backend_synchronize(backend);
    std::vector<float> result(rows*columns);
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    require(std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }), "finite output");
    ggml_backend_buffer_free(eb); ggml_backend_buffer_free(wb);
    ggml_free(ec); ggml_free(wc); ggml_backend_free(backend);
    return result;
}

int main(int argc, char ** argv) {
    require(argc == 2, "backend directory argument");
    ggml_backend_load_all_from_path(argv[1]);
    const auto default_plain = run(false, false);
    const auto default_half = run(false, true);
    const auto forced_plain = run(true, false);
    const auto forced_half = run(true, true);
    const auto same = [](const auto & a, const auto & b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
    };
    const auto ffn_default_plain = run(false, false, true);
    const auto ffn_default_half = run(false, true, true);
    const auto ffn_forced_plain = run(true, false, true);
    const auto ffn_forced_half = run(true, true, true);
    float difference = 0.0f;
    for (size_t i = 0; i < forced_plain.size(); ++i) difference = std::max(difference, std::fabs(forced_plain[i]-forced_half[i]));
    std::printf("{\"forced_F32_ignores_F16_toggle\":%s,\"default_routes_differ\":%s,\"forced_max_abs_diff\":%.9g}\n",
        same(forced_plain, forced_half) ? "true" : "false",
        same(default_plain, default_half) ? "false" : "true", difference);
    std::printf("{\"forced_F32_FFN_ignores_F16_toggle\":%s,\"default_FFN_routes_differ\":%s}\n",
        same(ffn_forced_plain, ffn_forced_half) ? "true" : "false",
        same(ffn_default_plain, ffn_default_half) ? "false" : "true");
    require(same(forced_plain, forced_half), "forced F32 matrix changed with F16 optimization");
    require(same(ffn_forced_plain, ffn_forced_half), "forced F32 FFN changed with F16 optimization");
    require(!same(default_plain, default_half), "fixture did not exercise the optional F16 matrix path");
    require(!same(ffn_default_plain, ffn_default_half), "fixture did not exercise the optional F16 FFN path");
    return 0;
}
