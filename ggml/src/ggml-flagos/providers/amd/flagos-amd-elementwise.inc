// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Baseline flat F32 kernels. Tensor semantics and alias rules remain in each
// supports_op path; this helper checks their common fixed launch profile.
static bool amd_elementwise_launch_contract_fits(const amd_device_context * device,
        const ggml_tensor * op, const char * kernel, size_t arguments, size_t pointers) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        op->src[0] == nullptr || kernel == nullptr || ggml_nelements(op) < 0 ||
        ggml_nelements(op) > INT32_MAX || !amd_tensor_data_is_aligned(op, sizeof(float)) ||
        !amd_tensor_data_is_aligned(op->src[0], sizeof(float)) ||
        (pointers == 3 && (op->src[1] == nullptr ||
            !amd_tensor_data_is_aligned(op->src[1], sizeof(float))))) {
        return false;
    }
    const auto * metadata = device->aot->find(kernel);
    const uint64_t grid = std::max<uint64_t>(1,
        (static_cast<uint64_t>(ggml_nelements(op)) + 255) / 256);
    return metadata != nullptr && metadata->block_size == 256 &&
        metadata->num_warps == 4 && metadata->warp_size == 32 &&
        metadata->shared_memory == 0 && metadata->static_shared_memory == 0 &&
        amd_kernel_launch_contract_fits(device, metadata, arguments, pointers, grid, 1, 1);
}

static bool amd_supports_unary_f32(
        const amd_device_context * device, const ggml_tensor * op,
        ggml_unary_op unary, const char * kernel_name) {
    return device != nullptr && device->aot != nullptr && kernel_name != nullptr &&
        device->aot->find(kernel_name) != nullptr && op != nullptr &&
        op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == unary &&
        op->src[0] != nullptr && amd_tensor_is_contiguous_f32(op) &&
        amd_tensor_is_contiguous_f32(op->src[0]) &&
        ggml_are_same_shape(op, op->src[0]) && ggml_nelements(op) > 0 &&
        ggml_nelements(op) <= INT32_MAX &&
        amd_tensor_output_can_reuse_input(op, op->src[0]) &&
        amd_elementwise_launch_contract_fits(device, op, kernel_name, 3, 2);
}

