#include "ggml-flagos.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); std::abort(); } } while (0)

static void equal_bytes(ggml_tensor * t, const std::vector<uint8_t> & expected) {
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(t, actual.data(), 0, actual.size());
    CHECK(actual == expected);
    if (actual.size() > 31) {
        ggml_backend_tensor_get(t, actual.data(), 7, 23);
        CHECK(std::memcmp(actual.data(), expected.data() + 7, 23) == 0);
    }
}

static void test_type(ggml_backend_t backend, ggml_type type, int64_t k, int64_t rows = 32) {
    auto ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    CHECK(ctx);
    auto w = ggml_new_tensor_2d(ctx, type, k, rows);
    auto copy = ggml_dup_tensor(ctx, w);
    auto view = ggml_view_1d(ctx, w, k, w->nb[1]);
    auto middle = ggml_view_2d(ctx, w, k, 3, w->nb[1], w->nb[1]);
    auto nested = ggml_view_1d(ctx, middle, k, w->nb[1]);
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 2);
    auto y = ggml_mul_mat(ctx, w, x);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto buft = ggml_backend_get_default_buffer_type(backend);
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    CHECK(buffer);
    const bool cpu_only_weight = type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q4_K ||
        type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q5_K ||
        (type == GGML_TYPE_Q5_0 && rows % 32 == 0) ||
        ((type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0) && rows % 32 != 0);
    CHECK((w->extra != nullptr) != cpu_only_weight);
    CHECK((ggml_backend_buft_get_alloc_size(buft, w) > ggml_nbytes(w)) != cpu_only_weight);
    CHECK(!view->extra);
    CHECK(!nested->extra);
    std::vector<float> data(ggml_nelements(w));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(float(i) * 0.17f) * 3.0f;
    }
    std::vector<uint8_t> expected(ggml_nbytes(w));
    CHECK(ggml_quantize_chunk(type, data.data(), expected.data(), 0, rows, k, nullptr) == expected.size());
    ggml_backend_tensor_set(w, expected.data(), 0, expected.size());
    equal_bytes(w, expected);

    auto cpu_ctx = ggml_init({ 1024 * 1024, nullptr, true });
    auto cpu_w = ggml_dup_tensor(cpu_ctx, w);
    auto cpu_buffer = ggml_backend_alloc_ctx_tensors_from_buft(cpu_ctx, ggml_backend_cpu_buffer_type());
    CHECK(cpu_buffer);
    ggml_backend_tensor_copy(w, cpu_w);
    equal_bytes(cpu_w, expected);
    ggml_backend_tensor_copy(cpu_w, copy);
    equal_bytes(copy, expected);
    ggml_backend_tensor_copy(w, copy);
    equal_bytes(copy, expected);

    const size_t row = w->nb[1];
    std::vector<uint8_t> expected_view(expected.begin() + row, expected.begin() + 2 * row);
    equal_bytes(view, expected_view);
    std::vector<uint8_t> expected_nested(expected.begin() + 2 * row, expected.begin() + 3 * row);
    equal_bytes(nested, expected_nested);
    expected[3 * row - 5] ^= 0x04;
    ggml_backend_tensor_set(nested, expected.data() + 3 * row - 5, row - 5, 1);
    equal_bytes(w, expected);
    // Modify payload bytes without changing quantization scales.
    const size_t start = row - 5;
    expected[row + start] ^= 0x12;
    ggml_backend_tensor_set(view, expected.data() + row + start, start, 1);
    equal_bytes(w, expected);
    ggml_backend_tensor_memset(view, 0x23, start, 3);
    std::fill(expected.begin() + row + start, expected.begin() + row + start + 3, 0x23);
    equal_bytes(w, expected);
    // A partial write across blocks must refresh the packed representation too.
    ggml_backend_tensor_set(copy, expected.data() + 5, 5, expected.size() - 10);
    equal_bytes(copy, expected);

    if (type == GGML_TYPE_Q4_0) {
        std::vector<float> inputs(ggml_nelements(x));
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = std::cos(float(i) * 0.11f);
        }
        ggml_backend_tensor_set(x, inputs.data(), 0, ggml_nbytes(x));
        CHECK(ggml_backend_supports_op(backend, y));
        auto cpu = ggml_backend_cpu_init();
        auto ref = ggml_backend_graph_copy(cpu, graph);
        CHECK(ref.graph);
        CHECK(ggml_backend_graph_compute(cpu, ref.graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(ggml_nelements(y)), reference(actual.size());
        ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
        auto ref_y = ggml_graph_node(ref.graph, ggml_graph_n_nodes(ref.graph) - 1);
        ggml_backend_tensor_get(ref_y, reference.data(), 0, ggml_nbytes(y));
        double error = 0, scale = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            CHECK(std::isfinite(actual[i]) && std::isfinite(reference[i]));
            error += double(actual[i] - reference[i]) * (actual[i] - reference[i]);
            scale += double(reference[i]) * reference[i];
        }
        CHECK(error / std::max(scale, 1e-20) < 5e-4);
        ggml_backend_graph_copy_free(ref);

        // A packed weight view must transfer canonical bytes to a CPU split.
        auto fallback_ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
        auto weight_view = ggml_view_2d(fallback_ctx, w, k, 16, w->nb[1], 8 * w->nb[1]);
        auto fallback_y = ggml_mul_mat(fallback_ctx, weight_view, x);
        auto fallback_graph = ggml_new_graph(fallback_ctx);
        ggml_build_forward_expand(fallback_graph, fallback_y);
        CHECK(!ggml_backend_supports_op(backend, fallback_y));
        ggml_backend_t backends[] = { backend, cpu };
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        CHECK(sched);
        CHECK(ggml_backend_sched_graph_compute(sched, fallback_graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_sched_get_tensor_backend(sched, fallback_y) == cpu);
        // Build the reference from the original canonical weight bytes.
        auto ref_ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
        auto ref_w = ggml_new_tensor_2d(ref_ctx, type, k, 16);
        auto ref_x = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, k, 2);
        auto ref_out = ggml_mul_mat(ref_ctx, ref_w, ref_x);
        auto ref_graph = ggml_new_graph(ref_ctx);
        ggml_build_forward_expand(ref_graph, ref_out);
        auto ref_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ref_ctx, ggml_backend_cpu_buffer_type());
        CHECK(ref_buffer);
        ggml_backend_tensor_set(ref_w, expected.data() + 8 * row, 0, ggml_nbytes(ref_w));
        ggml_backend_tensor_set(ref_x, inputs.data(), 0, ggml_nbytes(ref_x));
        CHECK(ggml_backend_graph_compute(cpu, ref_graph) == GGML_STATUS_SUCCESS);
        actual.resize(ggml_nelements(fallback_y));
        reference.resize(actual.size());
        ggml_backend_tensor_get(fallback_y, actual.data(), 0, ggml_nbytes(fallback_y));
        ggml_backend_tensor_get(ref_out, reference.data(), 0, ggml_nbytes(ref_out));
        CHECK(actual == reference);
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(ref_buffer);
        ggml_free(ref_ctx);
        ggml_free(fallback_ctx);
        ggml_backend_free(cpu);
    }

    if (cpu_only_weight) {
        // The provider rejects these lossy-repacked matmuls. The scheduler must copy the
        // canonical weight bytes to CPU, rather than the native packed form.
        std::vector<float> inputs(ggml_nelements(x));
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = std::cos(float(i) * 0.11f);
        }
        ggml_backend_tensor_set(x, inputs.data(), 0, ggml_nbytes(x));
        CHECK(!ggml_backend_supports_op(backend, y));
        // Build an unallocated output as model graphs do. The y tensor above
        // already belongs to the SpaceMiT buffer and cannot move to CPU.
        auto fallback_ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
        auto fallback_y = ggml_mul_mat(fallback_ctx, w, x);
        auto fallback_graph = ggml_new_graph(fallback_ctx);
        ggml_build_forward_expand(fallback_graph, fallback_y);
        auto cpu = ggml_backend_cpu_init();
        auto ref_ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
        auto ref_w = ggml_new_tensor_2d(ref_ctx, type, k, rows);
        auto ref_x = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, k, 2);
        auto ref_y = ggml_mul_mat(ref_ctx, ref_w, ref_x);
        auto ref_graph = ggml_new_graph(ref_ctx);
        ggml_build_forward_expand(ref_graph, ref_y);
        auto ref_buffer = ggml_backend_alloc_ctx_tensors_from_buft(ref_ctx, ggml_backend_cpu_buffer_type());
        CHECK(ref_buffer);
        ggml_backend_tensor_set(ref_w, expected.data(), 0, expected.size());
        ggml_backend_tensor_set(ref_x, inputs.data(), 0, ggml_nbytes(ref_x));
        CHECK(ggml_backend_graph_compute(cpu, ref_graph) == GGML_STATUS_SUCCESS);
        ggml_backend_t backends[] = { backend, cpu };
        auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        CHECK(sched);
        CHECK(ggml_backend_sched_graph_compute(sched, fallback_graph) == GGML_STATUS_SUCCESS);
        CHECK(ggml_backend_sched_get_tensor_backend(sched, fallback_y) == cpu);
        std::vector<float> actual(ggml_nelements(fallback_y)), reference(actual.size());
        ggml_backend_tensor_get(fallback_y, actual.data(), 0, ggml_nbytes(fallback_y));
        ggml_backend_tensor_get(ref_y, reference.data(), 0, ggml_nbytes(fallback_y));
        CHECK(actual == reference);
        std::printf("%s MUL_MAT scheduler CPU fallback matches canonical reference\n", ggml_type_name(type));
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(ref_buffer);
        ggml_free(ref_ctx);
        ggml_backend_free(cpu);
        ggml_free(fallback_ctx);
    }

    ggml_backend_buffer_clear(buffer, 0x12);
    std::fill(expected.begin(), expected.end(), 0x12);
    equal_bytes(w, expected);
    equal_bytes(copy, expected);
    ggml_backend_buffer_clear(buffer, 0);
    std::fill(expected.begin(), expected.end(), 0);
    equal_bytes(w, expected);
    // clear must not dereference a destroyed GGML tensor descriptor.
    void * reused_address = w->data;
    ggml_free(ctx);
    ggml_backend_buffer_clear(buffer, 0);
    auto reuse_ctx = ggml_init({ 1024 * 1024, nullptr, true });
    auto plain = ggml_new_tensor_1d(reuse_ctx, GGML_TYPE_F32, 128);
    CHECK(ggml_backend_tensor_alloc(buffer, plain, reused_address) == GGML_STATUS_SUCCESS);
    ggml_backend_buffer_clear(buffer, 0x12);
    std::vector<uint8_t> plain_expected(ggml_nbytes(plain), 0x12);
    equal_bytes(plain, plain_expected);
    ggml_backend_buffer_reset(buffer);
    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(reuse_ctx);
    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(cpu_ctx);
    std::printf("%s k=%lld rows=%lld canonical transfer checks passed\n", ggml_type_name(type), (long long) k, (long long) rows);
}

static void test_shared_session(ggml_backend_dev_t dev) {
    struct instance {
        ggml_backend_t backend;
        ggml_context * ctx;
        ggml_backend_buffer_t buffer;
        ggml_cgraph * graph;
        ggml_tensor * output;
        struct ggml_backend_graph_copy reference;
    } instances[2] {};

    for (int i = 0; i < 2; ++i) {
        auto & current = instances[i];
        current.backend = ggml_backend_dev_init(dev, nullptr);
        CHECK(current.backend);
        current.ctx = ggml_init({ 1024 * 1024, nullptr, true });
        CHECK(current.ctx);
        auto weight = ggml_new_tensor_2d(current.ctx, GGML_TYPE_Q4_0, 256, 32);
        auto input = ggml_new_tensor_2d(current.ctx, GGML_TYPE_F32, 256, 2);
        current.output = ggml_mul_mat(current.ctx, weight, input);
        current.graph = ggml_new_graph(current.ctx);
        ggml_build_forward_expand(current.graph, current.output);
        current.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            current.ctx, ggml_backend_get_default_buffer_type(current.backend));
        CHECK(current.buffer);
        std::vector<float> values(ggml_nelements(weight));
        std::vector<float> activations(ggml_nelements(input));
        for (size_t j = 0; j < values.size(); ++j) {
            values[j] = std::sin(float(j + i * 71) * 0.13f);
        }
        for (size_t j = 0; j < activations.size(); ++j) {
            activations[j] = std::cos(float(j + i * 53) * 0.09f);
        }
        std::vector<uint8_t> quantized(ggml_nbytes(weight));
        CHECK(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(), 0, 32, 256, nullptr) == quantized.size());
        ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
        ggml_backend_tensor_set(input, activations.data(), 0, ggml_nbytes(input));
        auto cpu = ggml_backend_cpu_init();
        current.reference = ggml_backend_graph_copy(cpu, current.graph);
        CHECK(current.reference.graph);
        CHECK(ggml_backend_graph_compute(cpu, current.reference.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_free(cpu);
    }

    // Both handles refer to the registry-owned native session. The native
    // stream mutex must serialize concurrent graph launches on that session.
    CHECK(instances[0].backend->context == instances[1].backend->context);
    std::thread first([&] {
        for (int i = 0; i < 3; ++i) {
            CHECK(ggml_backend_graph_compute(instances[0].backend, instances[0].graph) == GGML_STATUS_SUCCESS);
        }
    });
    std::thread second([&] {
        for (int i = 0; i < 3; ++i) {
            CHECK(ggml_backend_graph_compute(instances[1].backend, instances[1].graph) == GGML_STATUS_SUCCESS);
        }
    });
    first.join();
    second.join();

    for (auto & current : instances) {
        std::vector<float> actual(ggml_nelements(current.output));
        std::vector<float> expected(actual.size());
        ggml_backend_tensor_get(current.output, actual.data(), 0, ggml_nbytes(current.output));
        auto ref_output = ggml_graph_node(current.reference.graph, ggml_graph_n_nodes(current.reference.graph) - 1);
        ggml_backend_tensor_get(ref_output, expected.data(), 0, ggml_nbytes(ref_output));
        double error = 0.0;
        double scale = 0.0;
        for (size_t j = 0; j < actual.size(); ++j) {
            CHECK(std::isfinite(actual[j]) && std::isfinite(expected[j]));
            error += double(actual[j] - expected[j]) * (actual[j] - expected[j]);
            scale += double(expected[j]) * expected[j];
        }
        CHECK(error / std::max(scale, 1e-20) < 5e-4);
        ggml_backend_graph_copy_free(current.reference);
        ggml_backend_buffer_free(current.buffer);
        ggml_free(current.ctx);
        ggml_backend_free(current.backend);
    }
    std::puts("two concurrent SpaceMiT backend handles share a serialized native session");
}

static void test_batched_weight_canonical(ggml_backend_t backend) {
    auto weights_ctx = ggml_init({ 1024 * 1024, nullptr, true });
    CHECK(weights_ctx);
    auto weight = ggml_new_tensor_4d(weights_ctx, GGML_TYPE_Q4_0, 256, 32, 4, 2);
    auto input = ggml_new_tensor_4d(weights_ctx, GGML_TYPE_F32, 256, 1, 4, 2);
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        weights_ctx, ggml_backend_get_default_buffer_type(backend));
    CHECK(buffer);

    std::vector<float> values(ggml_nelements(weight));
    std::vector<float> activations(ggml_nelements(input));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(float(i) * 0.021f);
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = std::cos(float(i) * 0.017f);
    }
    std::vector<uint8_t> quantized(ggml_nbytes(weight));
    CHECK(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(), 0, 256, 256, nullptr) == quantized.size());
    ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(input, activations.data(), 0, ggml_nbytes(input));
    equal_bytes(weight, quantized);

    auto graph_ctx = ggml_init({ 1024 * 1024, nullptr, true });
    CHECK(graph_ctx);
    auto result = ggml_mul_mat(graph_ctx, weight, input);
    auto graph = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(graph, result);
    CHECK(ggml_backend_supports_op(backend, result));
    auto cpu = ggml_backend_cpu_init();
    auto reference_ctx = ggml_init({ 1024 * 1024, nullptr, true });
    CHECK(reference_ctx);
    auto reference_weight = ggml_new_tensor_4d(reference_ctx, GGML_TYPE_Q4_0, 256, 32, 4, 2);
    auto reference_input = ggml_new_tensor_4d(reference_ctx, GGML_TYPE_F32, 256, 1, 4, 2);
    auto reference_result = ggml_mul_mat(reference_ctx, reference_weight, reference_input);
    auto reference_graph = ggml_new_graph(reference_ctx);
    ggml_build_forward_expand(reference_graph, reference_result);
    auto reference_buffer = ggml_backend_alloc_ctx_tensors_from_buft(reference_ctx, ggml_backend_cpu_buffer_type());
    CHECK(reference_buffer);
    ggml_backend_tensor_set(reference_weight, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(reference_input, activations.data(), 0, ggml_nbytes(reference_input));
    CHECK(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_t backends[] = { backend, cpu };
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    CHECK(sched);
    CHECK(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_sched_get_tensor_backend(sched, result) == backend);
    std::vector<float> actual(ggml_nelements(result)), expected(actual.size());
    ggml_backend_tensor_get(result, actual.data(), 0, ggml_nbytes(result));
    ggml_backend_tensor_get(reference_result, expected.data(), 0, ggml_nbytes(reference_result));
    double error = 0.0;
    double scale = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        error += double(actual[i] - expected[i]) * (actual[i] - expected[i]);
        scale += double(expected[i]) * expected[i];
    }
    CHECK(error / std::max(scale, 1e-20) < 5e-4);
    std::puts("batched Q4_0 weight uses canonical scalar MUL_MAT on SpaceMiT");

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(reference_buffer);
    ggml_free(reference_ctx);
    ggml_backend_free(cpu);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights_ctx);
}

int main() {
    auto reg = ggml_backend_flagos_reg();
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        auto dev = ggml_backend_reg_dev_get(reg, i);
        if (std::strcmp(ggml_backend_dev_name(dev), "SPACEMIT0") == 0) {
            device = dev;
            backend = ggml_backend_dev_init(dev, nullptr);
        }
    }
    CHECK(backend);
    for (auto type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                       GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K,
                       GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
        test_type(backend, type, 256);
    }
    test_type(backend, GGML_TYPE_Q4_0, 288);
    test_type(backend, GGML_TYPE_Q4_1, 2048, 2048);
    test_type(backend, GGML_TYPE_Q4_K, 2048, 2048);
    test_type(backend, GGML_TYPE_Q5_0, 288, 288);
    test_type(backend, GGML_TYPE_Q5_1, 2048, 2048);
    test_type(backend, GGML_TYPE_Q5_K, 2048, 2048);
    test_type(backend, GGML_TYPE_Q6_K, 256, 17);
    test_type(backend, GGML_TYPE_Q8_0, 256, 17);
    test_shared_session(device);
    test_batched_weight_canonical(backend);
    ggml_backend_free(backend);
}
