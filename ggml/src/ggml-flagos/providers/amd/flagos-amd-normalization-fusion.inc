// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

static bool amd_norm_fusion_launch_contract_fits(const amd_device_context * device,
        const char * kernel, const ggml_tensor * norm, size_t arguments, size_t pointers,
        std::initializer_list<const ggml_tensor *> tensors) {
    const auto * metadata = device->aot->find(kernel);
    return metadata != nullptr && metadata->block_size >= norm->ne[0] &&
        amd_kernel_launch_contract_fits(device, metadata, arguments, pointers,
            static_cast<uint64_t>(ggml_nelements(norm)) / norm->ne[0], 1, 1) &&
        std::all_of(tensors.begin(), tensors.end(), [](const ggml_tensor * tensor) {
            return amd_tensor_data_is_aligned(tensor, sizeof(float));
        });
}

static bool amd_supports_rms_norm_mul_tensors(const amd_device_context * device,
                                      const ggml_tensor * norm,
                                      const ggml_tensor * mul) {
    if (device == nullptr || device->aot == nullptr || norm == nullptr || mul == nullptr ||
        norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || norm->src[0] == nullptr ||
        norm->type != GGML_TYPE_F32 || norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(norm, norm->src[0]) || !ggml_are_same_shape(mul, norm) ||
        norm->ne[0] <= 0 || norm->ne[0] > INT32_MAX || norm->ne[0] % 4 != 0 ||
        norm->ne[0] > 4096 || ggml_nelements(norm) > INT32_MAX ||
        !ggml_is_contiguous(norm) || !ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(mul)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] :
        (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (weight == nullptr || weight->type != GGML_TYPE_F32 ||
        weight->ne[0] != norm->ne[0] || weight->ne[1] != 1 ||
        weight->ne[2] != 1 || weight->ne[3] != 1 ||
        !ggml_is_contiguous(weight) || norm->data == nullptr || mul->data == nullptr ||
        norm->src[0]->data == nullptr || weight->data == nullptr) {
        return false;
    }
    if (amd_tensor_data_overlaps(norm, norm->src[0]) || amd_tensor_data_overlaps(norm, weight) ||
        amd_tensor_data_overlaps(mul, norm->src[0]) || amd_tensor_data_overlaps(mul, weight)) {
        return false;
    }
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps < 0.0f) {
        return false;
    }
    return true;
}

static bool amd_supports_rms_norm_mul(const amd_device_context * device,
        const ggml_tensor * norm, const ggml_tensor * mul) {
    if (!amd_supports_rms_norm_mul_tensors(device, norm, mul)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    return amd_norm_fusion_launch_contract_fits(device, "flagos_rms_norm_mul_f32", norm, 6, 4,
        {norm, mul, norm->src[0], weight});
}

static bool amd_supports_add_rms_norm_mul_tensors(const amd_device_context * device,
                                          const ggml_tensor * add,
                                          const ggml_tensor * norm,
                                          const ggml_tensor * mul) {
    if (device == nullptr || device->aot == nullptr || add == nullptr ||
        !amd_supports_rms_norm_mul_tensors(device, norm, mul) ||
        add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 ||
        add->src[0] == nullptr || add->src[1] == nullptr ||
        norm->src[0] != add ||
        !ggml_are_same_shape(add, norm) || !ggml_is_contiguous(add) ||
        add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add->src[0], norm) || !ggml_are_same_shape(add->src[1], norm) ||
        !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
        add->src[0]->data == nullptr || add->src[1]->data == nullptr) {
        return false;
    }
    if (amd_tensor_data_overlaps(norm, add->src[0]) ||
        amd_tensor_data_overlaps(norm, add->src[1]) ||
        amd_tensor_data_overlaps(mul, add->src[0]) ||
        amd_tensor_data_overlaps(mul, add->src[1])) {
        return false;
    }
    return true;
}

static bool amd_supports_add_rms_norm_mul(const amd_device_context * device,
        const ggml_tensor * add, const ggml_tensor * norm, const ggml_tensor * mul) {
    if (!amd_supports_add_rms_norm_mul_tensors(device, add, norm, mul)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    return amd_norm_fusion_launch_contract_fits(device, "flagos_add_rms_norm_mul_f32", norm, 7, 5,
        {norm, mul, add->src[0], add->src[1], weight});
}

static const char * amd_rms_norm_mul_inplace_kernel(
        const amd_device_context * device, int64_t n_cols) {
    if (device == nullptr || device->aot == nullptr || n_cols <= 0) {
        return nullptr;
    }
    if (const auto * narrow = device->aot->find(
            "flagos_rms_norm_mul_inplace_f32_narrow")) {
        if (narrow->exact_block_size && narrow->block_size == n_cols &&
            amd_kernel_launch_contract_fits(device, narrow, 5, 3, 1, 1, 1)) {
            return "flagos_rms_norm_mul_inplace_f32_narrow";
        }
    }
    const auto * generic = device->aot->find("flagos_rms_norm_mul_inplace_f32");
    return generic != nullptr && generic->block_size >= n_cols &&
        amd_kernel_launch_contract_fits(device, generic, 5, 3, 1, 1, 1)
        ? "flagos_rms_norm_mul_inplace_f32" : nullptr;
}

static bool amd_supports_rms_norm_mul_inplace(const amd_device_context * device,
                                              const ggml_tensor * norm,
                                              const ggml_tensor * mul) {
    const ggml_tensor * weight = mul != nullptr && mul->src[0] == norm ? mul->src[1] :
        (mul != nullptr ? mul->src[0] : nullptr);
    return device != nullptr && device->aot != nullptr && norm != nullptr && mul != nullptr &&
        norm->data != nullptr && norm->data == mul->data &&
        amd_rms_norm_mul_inplace_kernel(device, norm->ne[0]) != nullptr &&
        amd_supports_rms_norm_mul_tensors(device, norm, mul) &&
        norm->src[0]->data != norm->data && weight != nullptr && weight->data != norm->data &&
        amd_norm_fusion_launch_contract_fits(device,
            amd_rms_norm_mul_inplace_kernel(device, norm->ne[0]), norm, 5, 3,
            {mul, norm->src[0], weight});
}

static bool amd_supports_add_rms_norm_mul_inplace(const amd_device_context * device,
                                                  const ggml_tensor * add,
                                                  const ggml_tensor * norm,
                                                  const ggml_tensor * mul) {
    if (device == nullptr || add == nullptr || norm == nullptr || mul == nullptr ||
        norm->data == nullptr || norm->data != mul->data ||
        device->aot == nullptr || device->aot->find("flagos_add_rms_norm_mul_inplace_f32") == nullptr ||
        !amd_supports_add_rms_norm_mul_tensors(device, add, norm, mul) ||
        add->src[0]->data == norm->data || add->src[1]->data == norm->data) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    return weight != nullptr && weight->data != norm->data &&
        amd_norm_fusion_launch_contract_fits(device, "flagos_add_rms_norm_mul_inplace_f32", norm, 6, 4,
            {mul, add->src[0], add->src[1], weight});
}

static const char * amd_add_rms_norm_mul_residual_kernel(
        const amd_device_context * device, int64_t n_cols) {
    if (device == nullptr || device->aot == nullptr || n_cols <= 0) {
        return nullptr;
    }
    const char * best_name = nullptr;
    int best_capacity = INT_MAX;
    for (const char * name : {
            "flagos_add_rms_norm_mul_residual_f32_narrow",
            "flagos_add_rms_norm_mul_residual_f32" }) {
        const auto * metadata = device->aot->find(name);
        if (metadata != nullptr && metadata->block_size >= n_cols &&
            metadata->block_size < best_capacity &&
            amd_kernel_launch_contract_fits(device, metadata, 7, 5, 1, 1, 1)) {
            best_name = name;
            best_capacity = metadata->block_size;
        }
    }
    return best_name;
}

static bool amd_supports_add_rms_norm_mul_residual(const amd_device_context * device,
                                                   const ggml_tensor * add,
                                                   const ggml_tensor * norm,
                                                   const ggml_tensor * mul) {
    if (device == nullptr || device->aot == nullptr || add == nullptr || norm == nullptr || mul == nullptr ||
        add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL ||
        add->src[0] == nullptr || add->src[1] == nullptr || norm->src[0] != add ||
        add->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
        add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(add, add->src[0]) || !ggml_are_same_shape(add, add->src[1]) ||
        !ggml_are_same_shape(norm, add) || !ggml_are_same_shape(mul, norm) ||
        !ggml_is_contiguous(add) || !ggml_is_contiguous(norm) || !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
        add->ne[0] <= 0 || add->ne[0] > 4096 || add->ne[0] % 4 != 0 ||
        ggml_nelements(add) > INT32_MAX || add->data == nullptr || mul->data == nullptr ||
        add->src[0]->data == nullptr || add->src[1]->data == nullptr) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] :
        (mul->src[1] == norm ? mul->src[0] : nullptr);
    float eps = 0.0f;
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    return weight != nullptr && weight->type == GGML_TYPE_F32 &&
        weight->ne[0] == norm->ne[0] && weight->ne[1] == 1 &&
        weight->ne[2] == 1 && weight->ne[3] == 1 && ggml_is_contiguous(weight) &&
        weight->data != nullptr &&
        amd_add_rms_norm_mul_residual_kernel(device, norm->ne[0]) != nullptr &&
        amd_norm_fusion_launch_contract_fits(device,
            amd_add_rms_norm_mul_residual_kernel(device, norm->ne[0]), norm, 7, 5,
            {add, mul, add->src[0], add->src[1], weight}) &&
        std::isfinite(eps) && eps >= 0.0f &&
        !amd_tensor_data_overlaps(add, mul) &&
        !amd_tensor_data_overlaps(add, weight) &&
        !amd_tensor_data_overlaps(mul, weight) &&
        amd_tensor_output_can_reuse_input(add, add->src[0]) &&
        amd_tensor_output_can_reuse_input(add, add->src[1]) &&
        amd_tensor_output_can_reuse_input(mul, add->src[0]) &&
        amd_tensor_output_can_reuse_input(mul, add->src[1]);
}

