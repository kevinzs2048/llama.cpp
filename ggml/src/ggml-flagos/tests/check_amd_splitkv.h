#pragma once

#include "providers/amd/flagos-amd-op-contracts.h"
#include <memory>

struct splitkv_check_case {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> tensors;
    std::vector<ggml_backend_buffer_t> buffers;
    int width, heads, keys, group;

    splitkv_check_case(ggml_backend_t backend_, ggml_backend_dev_t dev,
            ggml_backend_buffer_type_t buft, int width_, int group_, int keys_, bool padded):
        backend(backend_), width(width_), heads(group_ * 2), keys(keys_), group(group_) {
        ctx = ggml_init({2u << 20, nullptr, true});
        CHECK(ctx);
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, heads);
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, keys, 2);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, keys, 2);
        mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, keys);
        q->nb[1] = (width + (padded ? 4 : 0)) * heads * sizeof(float);
        q->nb[2] = (width + (padded ? 4 : 0)) * sizeof(float);
        k->nb[1] = (2 * width + (padded ? 8 : 0)) * sizeof(ggml_fp16_t);
        k->nb[2] = width * sizeof(ggml_fp16_t);
        v->nb[1] = k->nb[1];
        v->nb[2] = k->nb[2];
        output = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(width)), 0.0f, 0.0f);
        auto * end = output;
        tensors = {q, k, v, mask, output};
        for (int i = 0; i < 3; ++i) {
            end = ggml_scale(ctx, end, 1.0f);
            tensors.push_back(end);
        }
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, end);
        CHECK(graph->n_nodes == 4);
        for (auto * t : tensors) {
            CHECK(ggml_backend_dev_supports_op(dev, t) || t->op == GGML_OP_NONE);
            auto buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t) + 512);
            CHECK(buffer);
            ggml_backend_buffer_clear(buffer, 0x5a);
            CHECK(ggml_backend_tensor_alloc(buffer, t,
                static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + 256) == GGML_STATUS_SUCCESS);
            buffers.push_back(buffer);
        }
        CHECK(ggml_backend_dev_supports_op(dev, output));
        flagos_amd::splitkv_decode_geometry geometry;
        CHECK(flagos_amd::describe_splitkv_decode(output, &geometry));
        CHECK(geometry.heads == heads && geometry.splits == 8);
        CHECK(geometry.keys_per_split * geometry.splits >= keys);
        auto invalid = *output;
        invalid.ne[1] = 17;
        CHECK(!flagos_amd::describe_splitkv_decode(&invalid, &geometry));
    }

    ~splitkv_check_case() {
        for (auto buffer : buffers) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    double run(int state) {
        std::vector<float> q_host(ggml_nbytes(q) / sizeof(float), 37.25f);
        std::vector<ggml_fp16_t> k_host(ggml_nbytes(k) / sizeof(ggml_fp16_t), ggml_fp32_to_fp16(37.25f));
        auto v_host = k_host;
        std::vector<ggml_fp16_t> mask_host(keys, ggml_fp32_to_fp16(0.0f));
        const size_t qh = q->nb[2] / sizeof(float), kt = k->nb[1] / sizeof(ggml_fp16_t);
        const size_t kh = k->nb[2] / sizeof(ggml_fp16_t);
        for (int h = 0; h < heads; ++h) {
            for (int d = 0; d < width; ++d) {
                q_host[h * qh + d] = 0.7f * std::sin(float(d + 17 * h + 11 * state) * 0.173f);
            }
        }
        for (int t = 0; t < keys; ++t) {
            for (int h = 0; h < 2; ++h) {
                for (int d = 0; d < width; ++d) {
                    k_host[t * kt + h * kh + d] = ggml_fp32_to_fp16(
                        0.5f * std::sin(float(d + 7 * t + h + state) * 0.113f));
                    v_host[t * kt + h * kh + d] = ggml_fp32_to_fp16(
                        0.5f * std::cos(float(d + 3 * t + 11 * h + 7 * state) * 0.079f));
                }
            }
            if (state == 1) mask_host[t] = ggml_fp32_to_fp16(t % 3 == 0 ? -INFINITY : -float(t % 17) * 0.1f);
            if (state == 2) mask_host[t] = ggml_fp32_to_fp16(-INFINITY);
        }
        ggml_backend_tensor_set(q, q_host.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, k_host.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v, v_host.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(mask, mask_host.data(), 0, ggml_nbytes(mask));
        std::vector<double> expected(width * heads, 0.0);
        if (state != 2) {
            const double scale = double(1.0f / std::sqrt(float(width)));
            for (int h = 0; h < heads; ++h) {
                std::vector<double> scores(keys);
                double high = -INFINITY;
                for (int t = 0; t < keys; ++t) {
                    double sum = 0.0;
                    for (int d = 0; d < width; ++d) {
                        sum += double(ggml_fp16_to_fp32(ggml_fp32_to_fp16(q_host[h * qh + d]))) *
                            double(ggml_fp16_to_fp32(k_host[t * kt + (h / group) * kh + d]));
                    }
                    scores[t] = sum * scale + double(ggml_fp16_to_fp32(mask_host[t]));
                    high = std::max(high, scores[t]);
                }
                double denominator = 0.0;
                for (double & s : scores) { s = std::exp(s - high); denominator += s; }
                for (int d = 0; d < width; ++d) {
                    double sum = 0.0;
                    for (int t = 0; t < keys; ++t) {
                        sum += scores[t] * double(ggml_fp16_to_fp32(v_host[t * kt + (h / group) * kh + d]));
                    }
                    expected[h * width + d] = sum / denominator;
                }
            }
        }
        std::vector<float> first(width * heads), actual(first.size());
        double max_error = 0.0;
        for (int repeat = 0; repeat < 3; ++repeat) {
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
            for (size_t i = 0; i < actual.size(); ++i) {
                CHECK(std::isfinite(actual[i]));
                const double error = std::fabs(double(actual[i]) - expected[i]);
                CHECK(error <= 1e-3 + 5e-3 * std::fabs(expected[i]));
                max_error = std::max(max_error, error);
                if (state == 2) CHECK(actual[i] == 0.0f && !std::signbit(actual[i]));
            }
            if (repeat == 0) first = actual;
            else CHECK(std::memcmp(first.data(), actual.data(), actual.size() * sizeof(float)) == 0);
        }
        const void * inputs[] = {q_host.data(), k_host.data(), v_host.data(), mask_host.data()};
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto * t = tensors[i];
            std::vector<uint8_t> bytes(ggml_backend_buffer_get_size(buffers[i]));
            CHECK(hipMemcpy(bytes.data(), ggml_backend_buffer_get_base(buffers[i]), bytes.size(),
                hipMemcpyDeviceToHost) == hipSuccess);
            CHECK(std::all_of(bytes.begin(), bytes.begin() + 256, [](uint8_t x) { return x == 0x5a; }));
            CHECK(std::all_of(bytes.begin() + 256 + ggml_nbytes(t), bytes.end(),
                [](uint8_t x) { return x == 0x5a; }));
            if (i < 4) CHECK(std::memcmp(bytes.data() + 256, inputs[i], ggml_nbytes(t)) == 0);
        }
        return max_error;
    }
};

static void check_amd_splitkv_decode(ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    std::unique_ptr<splitkv_check_case> retained;
    int cases = 0;
    double max_error = 0.0;
    for (int width : {128, 256}) {
        for (int group : {2, 4, 8}) {
            for (int keys : {4096, 4097, 8192}) {
                for (bool padded : {false, true}) {
                    auto test = std::make_unique<splitkv_check_case>(backend, dev, buft, width, group, keys, padded);
                    for (int state = 0; state < 3; ++state) {
                        max_error = std::max(max_error, test->run(state));
                        ++cases;
                    }
                    if (!retained) retained = std::move(test);
                }
            }
        }
    }
    // Reuse the live small graph after larger heads forced workspace replacement.
    for (int state = 0; state < 3; ++state) {
        max_error = std::max(max_error, retained->run(state));
    }
    CHECK(cases == 108);
    std::printf("GGML_SPLITKV_COMPLETE cases=108 old_graph_after_growth=3 full_FP64=1 "
        "guards_inputs_unchanged=1 eager_repeat_bitwise=1 max_error=%.9g\n", max_error);
}
