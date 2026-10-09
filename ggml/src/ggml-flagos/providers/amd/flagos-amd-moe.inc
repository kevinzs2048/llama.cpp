// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
struct amd_moe_launch {
    const char * kernel = nullptr;
    bool quantized = false;
    int k = 0, rows = 0, picks = 0, tokens = 0, input_picks = 0, experts = 0;
    int sw1 = 0, sw2 = 0, sx1 = 0, sx2 = 0, si0 = 0, si1 = 0;
    unsigned int grid_x = 0, grid_y = 0;
};

static bool amd_describe_moe(const amd_device_context * device,
        const ggml_tensor * op, amd_moe_launch * launch) {
    if (launch == nullptr) {
        return false;
    }
    *launch = {};
    if (!amd_env_enabled("FLAGOS_AMD_MOE") || device == nullptr || device->aot == nullptr ||
        op == nullptr || op->op != GGML_OP_MUL_MAT_ID || op->src[0] == nullptr ||
        op->src[1] == nullptr || op->src[2] == nullptr || !amd_tensor_is_contiguous_f32(op)) {
        return false;
    }
    const auto * weights = op->src[0];
    const auto * input = op->src[1];
    const auto * ids = op->src[2];
    const bool quantized = weights->type == GGML_TYPE_Q4_0 || weights->type == GGML_TYPE_Q4_K;
    if ((weights->type != GGML_TYPE_F16 && weights->type != GGML_TYPE_F32 && weights->type != GGML_TYPE_BF16 && !quantized) ||
        input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        weights->ne[3] != 1 || input->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        op->ne[3] != 1) {
        return false;
    }
    for (const auto * tensor : {weights, input, ids, op}) {
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->ne[dim] <= 0 || tensor->ne[dim] > INT32_MAX) {
                return false;
            }
        }
    }
    uint64_t output_elements = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (output_elements > static_cast<uint64_t>(INT32_MAX) / static_cast<uint64_t>(op->ne[dim])) return false;
        output_elements *= static_cast<uint64_t>(op->ne[dim]);
    }
    if (weights->ne[0] > 16384 || weights->ne[0] != input->ne[0] ||
        ids->ne[1] != input->ne[2] || ids->ne[0] % input->ne[1] != 0 ||
        op->ne[0] != weights->ne[1] || op->ne[1] != ids->ne[0] ||
        op->ne[2] != ids->ne[1]) {
        return false;
    }
    const size_t weight_size = (weights->type == GGML_TYPE_F16 || weights->type == GGML_TYPE_BF16) ? sizeof(uint16_t) : sizeof(float);
    const size_t weight_stride_unit = quantized ? 1 : weight_size;
    const int block = weights->type == GGML_TYPE_Q4_K ? 256 : 32;
    if (quantized && weights->ne[0] % block != 0) return false;
    const size_t row_bytes = quantized ? static_cast<size_t>(weights->ne[0] / block) * ggml_type_size(weights->type)
                                      : static_cast<size_t>(weights->ne[0]) * weight_size;
    if (quantized) {
        if (weights->nb[1] > INT32_MAX || weights->nb[2] > INT32_MAX ||
            weights->nb[1] % sizeof(uint16_t) != 0 || weights->nb[2] % sizeof(uint16_t) != 0) return false;
        const uint64_t span = static_cast<uint64_t>(weights->ne[2]-1) * weights->nb[2] +
            static_cast<uint64_t>(weights->ne[1]-1) * weights->nb[1] + row_bytes;
        if (span > INT32_MAX) return false;
    }
    if (weights->nb[0] != (quantized ? ggml_type_size(weights->type) : weight_size) || input->nb[0] != sizeof(float) ||
        (!quantized && !amd_tensor_element_offsets_fit_i32(weights, weight_size)) ||
        !amd_tensor_element_offsets_fit_i32(input, sizeof(float)) ||
        !amd_tensor_element_offsets_fit_i32(ids, sizeof(int32_t)) ||
        weights->nb[1] < row_bytes ||
        weights->nb[2] < static_cast<size_t>(weights->ne[1]) * weights->nb[1] ||
        input->nb[1] < static_cast<size_t>(input->ne[0]) * sizeof(float) ||
        input->nb[2] < static_cast<size_t>(input->ne[1]) * input->nb[1] ||
        ids->nb[0] < sizeof(int32_t) || ids->nb[1] < static_cast<size_t>(ids->ne[0]) * ids->nb[0]) {
        return false;
    }
    for (const auto * tensor : {weights, input, ids}) {
        const size_t alignment = tensor == weights ? (quantized ? sizeof(uint16_t) : weight_size) : sizeof(float);
        if (!amd_tensor_data_is_aligned(tensor, alignment) || amd_tensor_data_overlaps(op, tensor)) {
            return false;
        }
    }
    if (!amd_tensor_data_is_aligned(op, sizeof(float))) {
        return false;
    }
    for (size_t index = 0; index < sizeof(op->op_params) / sizeof(op->op_params[0]); ++index) {
        const auto value = op->op_params[index];
#if GGML_BACKEND_API_VERSION >= 3
        // API 3 precision contracts: slot 0 is accumulator; slot 3 is src1.
        // Every profile accumulates in F32; only these profiles preserve F32 src1.
        const bool preserves_f32 = quantized || weights->type == GGML_TYPE_F32;
        if (value == GGML_PREC_F32 && (index == 0 || (index == 3 && preserves_f32))) {
            continue;
        }
#endif
        if (value != 0) {
            return false;
        }
    }
    launch->quantized = quantized;
    launch->kernel = weights->type == GGML_TYPE_F16 ? "flagos_mul_mat_id_f16_f32" :
        weights->type == GGML_TYPE_F32 ? "flagos_mul_mat_id_f32_f32" :
        weights->type == GGML_TYPE_BF16 ? "flagos_mul_mat_id_bf16_bits_f32" :
        weights->type == GGML_TYPE_Q4_K ? "flagos_mul_mat_id_q4_k_f32" : "flagos_mul_mat_id_q4_0_f32";
    launch->k = static_cast<int>(weights->ne[0]);
    launch->rows = static_cast<int>(weights->ne[1]);
    launch->picks = static_cast<int>(ids->ne[0]);
    launch->tokens = static_cast<int>(ids->ne[1]);
    launch->input_picks = static_cast<int>(input->ne[1]);
    launch->experts = static_cast<int>(weights->ne[2]);
    launch->sw1 = static_cast<int>(weights->nb[1] / weight_stride_unit);
    launch->sw2 = static_cast<int>(weights->nb[2] / weight_stride_unit);
    launch->sx1 = static_cast<int>(input->nb[1] / sizeof(float));
    launch->sx2 = static_cast<int>(input->nb[2] / sizeof(float));
    launch->si0 = static_cast<int>(ids->nb[0] / sizeof(int32_t));
    launch->si1 = static_cast<int>(ids->nb[1] / sizeof(int32_t));
    launch->grid_x = quantized ? static_cast<unsigned>(launch->rows) : static_cast<unsigned>((static_cast<uint64_t>(launch->rows) + 3) / 4);
    const uint64_t grid_y = static_cast<uint64_t>(launch->picks) * launch->tokens;
    const auto * metadata = device->aot->find(launch->kernel);
    const int tile_k = quantized ? block : 64;
    if (metadata == nullptr || (quantized && !metadata->exact_block_size) || metadata->block_size != tile_k || metadata->tile_m != (quantized ? 1 : 4) ||
        metadata->tile_n != 1 || metadata->tile_k != tile_k || metadata->num_warps != 4 || metadata->warp_size != 32 ||
        !amd_kernel_launch_contract_fits(device, metadata, quantized ? 17 : 16, quantized ? 5 : 4, launch->grid_x, grid_y, 1)) {
        return false;
    }
    launch->grid_y = static_cast<unsigned>(grid_y);
    return true;
}

static bool amd_supports_moe(const amd_device_context * device, const ggml_tensor * op) {
    amd_moe_launch launch;
    return amd_describe_moe(device, op, &launch);
}

static bool amd_launch_moe(amd_backend_context * context, ggml_tensor * node) {
    amd_moe_launch launch;
    if (!amd_describe_moe(context->device, node, &launch)) {
        return false;
    }
    const void * weights = node->src[0]->data;
    const void * input = node->src[1]->data;
    const void * ids = node->src[2]->data;
    void * output = node->data;
    // Quantized weights expose byte and half views of one read-only allocation.
    flagos_amd::kernel_arguments arguments;
    arguments.push_back(&weights);
    if (launch.quantized) arguments.push_back(&weights);
    for (const void ** pointer : { &input, &ids }) arguments.push_back(pointer);
    arguments.push_back(&output);
    for (int * scalar : { &launch.k, &launch.rows, &launch.picks, &launch.tokens, &launch.input_picks, &launch.experts,
        &launch.sw1, &launch.sw2, &launch.sx1, &launch.sx2, &launch.si0, &launch.si1 }) arguments.push_back(scalar);
    return context->device->aot->launch(launch.kernel, context->stream,
        launch.grid_x, launch.grid_y, 1, arguments);
}
