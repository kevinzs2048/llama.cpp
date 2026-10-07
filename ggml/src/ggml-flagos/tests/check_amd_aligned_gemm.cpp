#include "../providers/amd/flagos-amd-op-contracts.h"
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
    setenv("FLAGOS_AMD_GROUPED_F16_GEMM", "0", 1);
    for (bool forced_f32 : {false, true}) {
        setenv("FLAGOS_AMD_ALIGNED_F16_GEMM", "0", 1);
        const auto generic = run(forced_f32, true);
        setenv("FLAGOS_AMD_ALIGNED_F16_GEMM", "1", 1);
        const auto aligned = run(forced_f32, true);
        require(generic.size() == aligned.size() && std::memcmp(generic.data(), aligned.data(), generic.size()*sizeof(float)) == 0,
            "aligned matrix arithmetic changed");
    }
    const auto ptr = [](uintptr_t value) { return reinterpret_cast<const void *>(value); };
    require(flagos_amd::supports_aligned_f16_gemm_launch(ptr(16), ptr(32), ptr(48), 1024, 64), "valid launch rejected");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(nullptr, ptr(32), ptr(48), 1024, 64), "null pointer accepted");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(ptr(17), ptr(32), ptr(48), 1024, 64), "unaligned pointer accepted");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(ptr(16), ptr(33), ptr(48), 1024, 64), "unaligned input accepted");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(ptr(16), ptr(32), ptr(49), 1024, 64), "unaligned output accepted");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(ptr(16), ptr(32), ptr(48), 1025, 64), "K tail accepted");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(ptr(16), ptr(32), ptr(48), 1024, 65), "rows tail accepted");
    require(!flagos_amd::supports_aligned_f16_gemm_launch(ptr(16), ptr(32), ptr(48), 0, 64), "empty K accepted");
    std::puts("AMD aligned GEMM default/F32 bitwise and launch contracts passed");
    return 0;
}
