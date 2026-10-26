// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
struct amd_argsort_launch {
    const char * kernel = nullptr;
    int columns = 0, source_row_stride = 0, descending = 0;
    unsigned rows = 0;
};

static bool amd_describe_argsort(const amd_device_context * device,
        const ggml_tensor * op, amd_argsort_launch * launch) {
    if (launch == nullptr) {
        return false;
    }
    *launch = {};
    if (!amd_env_enabled("FLAGOS_AMD_ARGSORT") || device == nullptr || device->aot == nullptr ||
        op == nullptr || op->op != GGML_OP_ARGSORT || op->src[0] == nullptr ||
        op->type != GGML_TYPE_I32 || !ggml_is_contiguous(op)) {
        return false;
    }
    const auto * input = op->src[0];
    if (input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) ||
        !amd_tensor_element_offsets_fit_i32(input, sizeof(float)) ||
        !amd_tensor_data_is_aligned(input, sizeof(float)) ||
        !amd_tensor_data_is_aligned(op, sizeof(int32_t)) || amd_tensor_data_overlaps(op, input)) {
        return false;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (input->ne[dim] <= 0 || input->ne[dim] > INT32_MAX || op->ne[dim] != input->ne[dim]) {
            return false;
        }
    }
    uint64_t elements = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (elements > static_cast<uint64_t>(INT32_MAX) / static_cast<uint64_t>(input->ne[dim])) {
            return false;
        }
        elements *= static_cast<uint64_t>(input->ne[dim]);
    }
    if (input->ne[0] > 4096 ||
        input->nb[1] < static_cast<size_t>(input->ne[0]) * sizeof(float) ||
        (input->ne[2] > 1 && input->nb[2] != input->nb[1] * static_cast<size_t>(input->ne[1])) ||
        (input->ne[3] > 1 && input->nb[3] != input->nb[1] * static_cast<size_t>(input->ne[1]) * static_cast<size_t>(input->ne[2]))) {
        return false;
    }
    std::memcpy(&launch->descending, op->op_params, sizeof(int));
    if (launch->descending != GGML_SORT_ORDER_ASC && launch->descending != GGML_SORT_ORDER_DESC) {
        return false;
    }
    for (size_t i = sizeof(int); i < sizeof(op->op_params); ++i) {
        if (reinterpret_cast<const uint8_t *>(op->op_params)[i] != 0) {
            return false;
        }
    }
    launch->columns = static_cast<int>(input->ne[0]);
    launch->source_row_stride = static_cast<int>(input->nb[1] / sizeof(float));
    launch->rows = static_cast<unsigned>(elements / static_cast<uint64_t>(input->ne[0]));
    const int capacities[] = {64, 256, 1024, 4096};
    const char * names[] = {"flagos_argsort_f32_i32_b64", "flagos_argsort_f32_i32_b256",
        "flagos_argsort_f32_i32_b1024", "flagos_argsort_f32_i32_b4096"};
    for (size_t i = 0; i < 4; ++i) {
        if (launch->columns > capacities[i]) {
            continue;
        }
        const auto * metadata = device->aot->find(names[i]);
        if (metadata != nullptr && metadata->block_size == capacities[i] &&
            metadata->num_warps == (capacities[i] == 4096 ? 8 : 4) && metadata->warp_size == 32 &&
            amd_kernel_launch_contract_fits(device, metadata, 5, 2, launch->rows, 1, 1)) {
            launch->kernel = names[i];
            return true;
        }
        // A missing or invalid selected profile is a refusal, not a larger-tile retry.
        return false;
    }
    return false;
}

static bool amd_supports_argsort(const amd_device_context * device, const ggml_tensor * op) {
    amd_argsort_launch launch;
    return amd_describe_argsort(device, op, &launch);
}

static bool amd_launch_argsort(amd_backend_context * context, ggml_tensor * op) {
    amd_argsort_launch launch;
    if (!amd_describe_argsort(context->device, op, &launch)) {
        return false;
    }
    const void * input = op->src[0]->data;
    void * output = op->data;
    flagos_amd::kernel_arguments arguments { &input, &output, &launch.columns,
        &launch.source_row_stride, &launch.descending };
    return context->device->aot->launch(launch.kernel, context->stream, launch.rows, 1, 1, arguments);
}
