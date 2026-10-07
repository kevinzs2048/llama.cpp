#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "CHECK_FAILED %s\n", message); std::exit(2); }
}

struct policy_observations {
    std::atomic<size_t> plans { 0 };
    std::atomic<size_t> captures { 0 };
};

static void observe_policy_log(enum ggml_log_level, const char * text, void * user_data) {
    auto & counts = *static_cast<policy_observations *>(user_data);
    if (std::strstr(text, "FlagOS AMD: built plan ") != nullptr) {
        ++counts.plans;
    }
    if (std::strstr(text, "FlagOS AMD: captured HIP graph ") != nullptr) {
        ++counts.captures;
    }
    std::fputs(text, stderr);
}

static std::vector<float> run(bool forced_f32, bool prefill_f16, bool ffn = false, bool switch_policy = false) {
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
    if (switch_policy) {
        for (int index = 0; index < 4; ++index) {
            out = ggml_scale(ec, out, 1.01f);
        }
    }
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
    std::vector<float> result(rows*columns);
    if (switch_policy) {
        setenv("FLAGOS_AMD_GRAPH_CAPTURE", "all", 1);
        setenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS", "0", 1);
        setenv("FLAGOS_LOG_GRAPH_PLAN", "1", 1);
        policy_observations counts;
        ggml_log_set(observe_policy_log, &counts);
        std::vector<float> decoded(weights.size());
        ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(quantized.data(), decoded.data(), k*rows);
        std::vector<double> expected(result.size());
        for (int col = 0; col < columns; ++col) {
            for (int row = 0; row < rows; ++row) {
                double sum = 0.0;
                for (int inner = 0; inner < k; ++inner) {
                    sum += double(decoded[row*k + inner])*activation[col*k + inner];
                }
                for (int index = 0; index < 4; ++index) {
                    sum *= double(1.01f);
                }
                expected[col*rows + row] = sum;
            }
        }
        // Each switch must retire the old executable, even if the package
        // currently falls back because the optional tiled kernel is absent.
        for (const char * policy : {"0", "1", "0"}) {
            setenv("FLAGOS_AMD_QUANT_TILED_GEMM", policy, 1);
            for (int repeat = 0; repeat < 4; ++repeat) {
                require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "policy compute");
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
                for (size_t index = 0; index < result.size(); ++index) {
                    require(std::isfinite(result[index]) &&
                        std::abs(result[index] - expected[index]) <= 1e-3 + 3e-5*std::abs(expected[index]),
                        "policy output differs from CPU reference");
                }
            }
        }
        ggml_log_set(nullptr, nullptr);
        unsetenv("FLAGOS_LOG_GRAPH_PLAN");
        unsetenv("FLAGOS_AMD_QUANT_TILED_GEMM");
        unsetenv("FLAGOS_AMD_GRAPH_CAPTURE");
        unsetenv("FLAGOS_AMD_ENABLE_EXPERIMENTAL_FUSIONS");
        std::printf("AMD policy invalidation plans=%zu captures=%zu expected=3/3\n", counts.plans.load(), counts.captures.load());
        require(counts.plans.load() == 3, "quantized policy reused old plan");
        require(counts.captures.load() == 3, "quantized policy reused old HIP graph");
    } else {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
        require(std::all_of(result.begin(), result.end(), [](float v) { return std::isfinite(v); }), "finite output");
    }
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
    run(true, false, false, true);
    return 0;
}
