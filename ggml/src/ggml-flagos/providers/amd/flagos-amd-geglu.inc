// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
static constexpr char AMD_GEGLU_LUT_NAME[] = "ggml-gelu-fp16-v1";
static constexpr char AMD_GEGLU_LUT_SHA256[] = "5eceda6f287dec9dc0bcfa24516d6744a031a5c52d1984688b4558aeb1daf909";
static constexpr char AMD_GEGLU_LUT_KERNEL[] = "flagos_geglu_fp16_lut_split_f32";

static const flagos_amd::constant_asset * amd_geglu_lut_asset(const amd_device_context * device) {
    if (device == nullptr || device->aot == nullptr) {
        return nullptr;
    }
    const auto * asset = device->aot->find_asset(AMD_GEGLU_LUT_NAME);
    return asset != nullptr && asset->sha256 == AMD_GEGLU_LUT_SHA256 &&
        asset->bytes.size() == (size_t(1) << 17) ? asset : nullptr;
}

static bool amd_supports_geglu_lut(const amd_device_context * device, const ggml_tensor * op) {
    if (!amd_env_enabled("FLAGOS_AMD_GEGLU_LUT") || amd_geglu_lut_asset(device) == nullptr ||
        op == nullptr || op->op != GGML_OP_GLU || ggml_get_glu_op(op) != GGML_GLU_OP_GEGLU ||
        op->src[0] == nullptr || op->src[1] == nullptr ||
        !amd_tensor_is_contiguous_f32(op) || !amd_tensor_is_contiguous_f32(op->src[0]) ||
        !amd_tensor_is_contiguous_f32(op->src[1]) ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_are_same_shape(op, op->src[1]) ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX ||
        !amd_tensor_output_can_reuse_input(op, op->src[0]) ||
        !amd_tensor_output_can_reuse_input(op, op->src[1]) ||
        !amd_tensor_data_is_aligned(op, sizeof(float)) ||
        !amd_tensor_data_is_aligned(op->src[0], sizeof(float)) ||
        !amd_tensor_data_is_aligned(op->src[1], sizeof(float))) {
        return false;
    }
    const auto * metadata = device->aot->find(AMD_GEGLU_LUT_KERNEL);
    const uint64_t grid = (static_cast<uint64_t>(ggml_nelements(op)) + 255) / 256;
    return metadata != nullptr && metadata->block_size == 256 && metadata->num_warps == 4 &&
        metadata->warp_size == 32 && metadata->shared_memory == 0 && metadata->static_shared_memory == 0 &&
        amd_kernel_launch_contract_fits(device, metadata, 5, 4, grid, 1, 1);
}

static bool amd_prepare_geglu_lut(amd_backend_context * context, const ggml_cgraph * graph) {
    bool required = false;
    for (int index = 0; index < graph->n_nodes; ++index) {
        const auto * node = graph->nodes[index];
        if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_GEGLU) {
            if (!amd_supports_geglu_lut(context->device, node)) {
                return false;
            }
            required = true;
        }
    }
    if (!required || context->geglu_lut_ready) {
        return true;
    }
    if (context->geglu_lut != nullptr) {
        return false;
    }
    const auto * asset = amd_geglu_lut_asset(context->device);
    if (!amd_hip_check(hipMalloc(&context->geglu_lut, asset->bytes.size()), "hipMalloc GEGLU LUT")) {
        return false;
    }
    // A failed upload retains its allocation until teardown proves the stream idle.
    if (!amd_hip_check(hipMemcpy(context->geglu_lut, asset->bytes.data(), asset->bytes.size(),
            hipMemcpyHostToDevice), "hipMemcpy GEGLU LUT")) {
        return false;
    }
    context->geglu_lut_ready = true;
    ++context->geglu_lut_generation;
    return true;
}
