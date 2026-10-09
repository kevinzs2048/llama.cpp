// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

static bool amd_supports_tanh(const amd_device_context * device, const ggml_tensor * op) {
    if (!amd_env_enabled("FLAGOS_AMD_TANH") ||
        !amd_supports_unary_f32(device, op, GGML_UNARY_OP_TANH, "flagos_tanh_f32") ||
        !amd_tensor_data_is_aligned(op, sizeof(float)) ||
        !amd_tensor_data_is_aligned(op->src[0], sizeof(float))) {
        return false;
    }
    const auto * input = op->src[0];
    const auto * output_root = op->view_src == nullptr ? op : op->view_src;
    const auto * input_root = input->view_src == nullptr ? input : input->view_src;
    // Pointwise load precedes the store in each program. Exact same-layout
    // aliases are safe; shifted views can race other programs.
    if (output_root == input_root && op->data == nullptr && input->data == nullptr &&
        op->view_offs != input->view_offs) {
        return false;
    }
    const auto * metadata = device->aot->find("flagos_tanh_f32");
    const uint64_t grid = (static_cast<uint64_t>(ggml_nelements(op)) + 255) / 256;
    return metadata->block_size == 256 && metadata->num_warps == 4 && metadata->warp_size == 32 &&
        amd_kernel_launch_contract_fits(device, metadata, 3, 2, grid, 1, 1);
}

static bool amd_launch_tanh(amd_backend_context * context, ggml_tensor * node) {
    int elements = static_cast<int>(ggml_nelements(node));
    flagos_amd::kernel_arguments arguments = {&node->src[0]->data, &node->data, &elements};
    return context->device->aot->launch("flagos_tanh_f32", context->stream,
        static_cast<unsigned int>((static_cast<int64_t>(elements) + 255) / 256), 1, 1, arguments);
}
