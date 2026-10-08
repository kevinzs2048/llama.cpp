#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-flagos.h"
#include "ggml-impl.h"
#include "check_closed_graph.h"
#include "../providers/amd/flagos-amd-op-contracts.h"
#include <cstdint>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check_rope_layout_contracts(ggml_context * ctx, ggml_backend_dev_t dev) {
    int checks = 0;
    for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_IMROPE }) {
        const int width = mode == GGML_ROPE_TYPE_NORMAL ? 128 : 256;
        const int dims = mode == GGML_ROPE_TYPE_NORMAL ? 128 : 64;
        auto * backing = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8192);
        auto * source = ggml_view_3d(ctx, backing, width, 3, 5,
            width * sizeof(float), width * 3 * sizeof(float), 0);
        auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mode == GGML_ROPE_TYPE_NORMAL ? 5 : 20);
        auto * factors = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width / 2);
        int sections[] = { 11, 11, 10, 0 };
        auto * output = mode == GGML_ROPE_TYPE_NORMAL
            ? ggml_rope_ext(ctx, source, positions, factors, dims, mode, 8192,
                5000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)
            : ggml_rope_multi(ctx, source, positions, factors, dims, sections, mode, 8192,
                10000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        const auto accepts = [&](const ggml_tensor * tensor) {
            return dev != nullptr ? ggml_backend_dev_supports_op(dev, tensor)
                : flagos_amd::supports_ggml_rope_layout(tensor, 128);
        };
        CHECK(source->data == nullptr && output->data == nullptr);
        CHECK(accepts(output));
        ++checks;
        const auto reject = [&](const char * label, const auto & change) {
            ggml_tensor y = *output, x = *source, p = *positions, f = *factors;
            y.src[0] = &x;
            y.src[1] = &p;
            y.src[2] = &f;
            change(y, x, p, f);
            if (accepts(&y)) {
                std::fprintf(stderr, "RoPE contract accepted %s (mode %d)\n", label, mode);
                std::abort();
            }
            ++checks;
        };
        reject("unaligned source view", [](auto &, auto & x, auto &, auto &) { x.view_offs = 1; });
        reject("unaligned output view", [&](auto & y, auto &, auto &, auto &) { y.view_src = backing; y.view_offs = 2; });
        reject("unaligned positions view", [&](auto &, auto &, auto & p, auto &) { p.view_src = backing; p.view_offs = 3; });
        reject("unaligned factors view", [&](auto &, auto &, auto &, auto & f) { f.view_src = backing; f.view_offs = 1; });
        reject("partial source alias", [&](auto & y, auto &, auto &, auto &) { y.view_src = backing; y.view_offs = sizeof(float); });
        reject("position storage alias", [](auto & y, auto &, auto & p, auto &) { y.view_src = &p; });
        reject("factor storage alias", [](auto & y, auto &, auto &, auto & f) { y.view_src = &f; });
        reject("unaligned bound output", [](auto & y, auto &, auto &, auto &) { y.data = reinterpret_cast<void *>(uintptr_t(0x1001)); });
        reject("unaligned bound source", [](auto &, auto & x, auto &, auto &) { x.data = reinterpret_cast<void *>(uintptr_t(0x1002)); });
        reject("unaligned bound positions", [](auto &, auto &, auto & p, auto &) { p.data = reinterpret_cast<void *>(uintptr_t(0x1003)); });
        reject("unaligned bound factors", [](auto &, auto &, auto &, auto & f) { f.data = reinterpret_cast<void *>(uintptr_t(0x1001)); });
        reject("F16 source", [](auto &, auto & x, auto &, auto &) { x.type = GGML_TYPE_F16; });
        reject("F16 output", [](auto & y, auto &, auto &, auto &) { y.type = GGML_TYPE_F16; });
        reject("I64 positions", [](auto &, auto &, auto & p, auto &) { p.type = GGML_TYPE_I64; });
        reject("short positions", [](auto &, auto &, auto & p, auto &) { --p.ne[0]; });
        reject("strided positions", [](auto &, auto &, auto & p, auto &) { p.nb[0] = 2 * sizeof(int32_t); });
        reject("short factors", [&](auto &, auto &, auto &, auto & f) { f.ne[0] = width / 2 - 1; });
        reject("wrong factor type", [](auto &, auto &, auto &, auto & f) { f.type = GGML_TYPE_F16; });
        reject("strided factors", [](auto &, auto &, auto &, auto & f) { f.nb[0] = 2 * sizeof(float); });
        reject("byte source stride", [](auto &, auto & x, auto &, auto &) { ++x.nb[1]; });
        reject("wide element stride", [](auto &, auto & x, auto &, auto &) { x.nb[0] = 2 * sizeof(float); });
        reject("strided output", [](auto & y, auto &, auto &, auto &) { y.nb[1] += sizeof(float); });
        reject("zero rotation", [](auto & y, auto &, auto &, auto &) { y.op_params[1] = 0; });
        reject("odd rotation", [](auto & y, auto &, auto &, auto &) { y.op_params[1] = 3; });
        reject("rotation wider than head", [&](auto & y, auto &, auto &, auto &) { y.op_params[1] = width + 2; });
        reject("vision RoPE", [](auto & y, auto &, auto &, auto &) { y.op_params[2] = GGML_ROPE_TYPE_VISION; });
        const auto frequency = [](ggml_tensor & y, int index, float value) {
            std::memcpy(y.op_params + 5 + index, &value, sizeof(value));
        };
        reject("zero frequency base", [&](auto & y, auto &, auto &, auto &) { frequency(y, 0, 0.0f); });
        reject("infinite frequency base", [&](auto & y, auto &, auto &, auto &) { frequency(y, 0, INFINITY); });
        reject("zero frequency scale", [&](auto & y, auto &, auto &, auto &) { frequency(y, 1, 0.0f); });
        reject("YaRN extension", [&](auto & y, auto &, auto &, auto &) { frequency(y, 2, 1.0f); });
        reject("NaN attention factor", [&](auto & y, auto &, auto &, auto &) { frequency(y, 3, NAN); });
        if (mode == GGML_ROPE_TYPE_NORMAL) {
            reject("frequency recurrence overflow", [&](auto & y, auto &, auto &, auto &) {
                y.op_params[1] = 2;
                frequency(y, 0, std::numeric_limits<float>::denorm_min());
            });
            reject("denormal frequency recurrence", [&](auto & y, auto &, auto &, auto &) {
                y.op_params[1] = 2;
                frequency(y, 0, 1.0e38f);
            });
        }
        if (mode == GGML_ROPE_TYPE_IMROPE) {
            reject("negative IMROPE section", [](auto & y, auto &, auto &, auto &) { y.op_params[11] = -1; });
            reject("wrong IMROPE section total", [](auto & y, auto &, auto &, auto &) { ++y.op_params[13]; });
        }
        ggml_tensor in_place = *output;
        in_place.view_src = backing;
        in_place.view_offs = source->view_offs;
        CHECK(accepts(&in_place));
        ++checks;
        auto * padded = ggml_view_3d(ctx, backing, width, 3, 5,
            (width + 8) * sizeof(float), (width + 8) * 3 * sizeof(float), 0);
        ggml_tensor strided = *output;
        strided.src[0] = padded;
        CHECK(accepts(&strided));
        ++checks;
        strided.view_src = backing;
        CHECK(!accepts(&strided));
        ++checks;
        ggml_tensor no_factors = *output;
        no_factors.src[2] = nullptr;
        CHECK(accepts(&no_factors));
        ++checks;
        CHECK(!flagos_amd::supports_ggml_rope_layout(output, 64));
        CHECK(!flagos_amd::supports_ggml_rope_layout(output, 256));
        checks += 2;
    }
    std::printf("AMD NORMAL/IMROPE %d pre-allocation binding contracts passed\n", checks);
}

static void check_ggml_rope_device(ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft, ggml_context * ctx) {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr);
    struct rope_case { int width, dims, mode; float base; int sections[4]; };
    const rope_case cases[] = {
        { 128, 32, GGML_ROPE_TYPE_NORMAL, 10000.0f, { 0, 0, 0, 0 } },
        { 128, 128, GGML_ROPE_TYPE_NORMAL, 5000000.0f, { 0, 0, 0, 0 } },
        { 256, 64, GGML_ROPE_TYPE_IMROPE, 10000000.0f, { 11, 11, 10, 0 } },
        { 128, 48, GGML_ROPE_TYPE_IMROPE, 10000.0f, { 9, 6, 6, 3 } },
    };
    for (const auto & spec : cases) {
        const int width = spec.width, dims = spec.dims, mode = spec.mode;
        for (bool use_factors : { false, true }) {
            constexpr int heads = 3, tokens = 5;
            ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, heads, tokens);
            ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                tokens * (mode == GGML_ROPE_TYPE_IMROPE ? 4 : 1));
            ggml_tensor * factors = use_factors ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width / 2) : nullptr;
            int sections[4];
            std::memcpy(sections, spec.sections, sizeof(sections));
            ggml_tensor * y = mode == GGML_ROPE_TYPE_NORMAL
                ? ggml_rope_ext(ctx, x, pos, factors, dims, mode, 8192,
                    spec.base, 0.5f, 0.0f, 1.125f, 0.0f, 0.0f)
                : ggml_rope_multi(ctx, x, pos, factors, dims, sections, mode, 8192,
                    spec.base, 0.5f, 0.0f, 1.125f, 0.0f, 0.0f);
            std::vector<ggml_backend_buffer_t> buffers;
            for (ggml_tensor * tensor : { x, pos, factors, y }) {
                if (tensor == nullptr) continue;
                ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(tensor));
                CHECK(buffer != nullptr);
                CHECK(ggml_backend_tensor_alloc(buffer, tensor,
                    ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
                buffers.push_back(buffer);
            }
            CHECK(ggml_backend_dev_supports_op(dev, y));
            ggml_tensor alias = *y;
            alias.data = static_cast<char *>(x->data) + sizeof(float);
            CHECK(!ggml_backend_dev_supports_op(dev, &alias));
            alias = *y;
            alias.data = pos->data;
            CHECK(!ggml_backend_dev_supports_op(dev, &alias));
            std::vector<float> input(width * heads * tokens), expected, actual(input.size());
            std::vector<float> factor_values(width / 2);
            std::vector<int32_t> positions(ggml_nelements(pos));
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i) * 0.01f);
            for (size_t i = 0; i < factor_values.size(); ++i) factor_values[i] = 0.75f + i * 0.05f;
            const int32_t position_values[tokens] = { -3, 4095, 8191, 131071, 262143 };
            for (size_t i = 0; i < positions.size(); ++i) {
                positions[i] = position_values[i % tokens] + static_cast<int32_t>(i / tokens) * 7;
            }
            ggml_init_params cpu_init { 4u << 20, nullptr, true };
            ggml_context * cpu_ctx = ggml_init(cpu_init);
            CHECK(cpu_ctx != nullptr);
            ggml_tensor * cpu_x = ggml_dup_tensor(cpu_ctx, x);
            ggml_tensor * cpu_pos = ggml_dup_tensor(cpu_ctx, pos);
            ggml_tensor * cpu_factors = use_factors ? ggml_dup_tensor(cpu_ctx, factors) : nullptr;
            ggml_tensor * cpu_y = mode == GGML_ROPE_TYPE_NORMAL
                ? ggml_rope_ext(cpu_ctx, cpu_x, cpu_pos, cpu_factors, dims, mode, 8192,
                    spec.base, 0.5f, 0.0f, 1.125f, 0.0f, 0.0f)
                : ggml_rope_multi(cpu_ctx, cpu_x, cpu_pos, cpu_factors, dims, sections, mode, 8192,
                    spec.base, 0.5f, 0.0f, 1.125f, 0.0f, 0.0f);
            ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
            ggml_build_forward_expand(cpu_graph, cpu_y);
            ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);
            CHECK(cpu_buffer != nullptr);
            ggml_backend_tensor_set(cpu_x, input.data(), 0, ggml_nbytes(cpu_x));
            ggml_backend_tensor_set(cpu_pos, positions.data(), 0, ggml_nbytes(cpu_pos));
            if (cpu_factors) ggml_backend_tensor_set(cpu_factors, factor_values.data(), 0, ggml_nbytes(cpu_factors));
            CHECK(ggml_backend_graph_compute(cpu, cpu_graph) == GGML_STATUS_SUCCESS);
            expected.resize(input.size());
            ggml_backend_tensor_get(cpu_y, expected.data(), 0, ggml_nbytes(cpu_y));
            ggml_backend_buffer_free(cpu_buffer);
            ggml_free(cpu_ctx);
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(pos, positions.data(), 0, ggml_nbytes(pos));
            if (factors) ggml_backend_tensor_set(factors, factor_values.data(), 0, ggml_nbytes(factors));
            ggml_tensor * nodes[] = { y };
            ggml_cgraph graph {};
            graph.n_nodes = 1;
            graph.nodes = nodes;
            flagos_check_closed_graph(ctx, graph);
            CHECK(ggml_backend_graph_compute(backend, &graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(y, actual.data(), 0, ggml_nbytes(y));
            for (size_t i = 0; i < actual.size(); ++i) CHECK(std::fabs(actual[i] - expected[i]) < 0.002f);
            for (auto buffer : buffers) ggml_backend_buffer_free(buffer);
        }
    }
    ggml_backend_free(cpu);
    std::puts("FlagOS AMD normal/IMROPE device checks passed (actual GGML CPU oracle)");
}
