// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

struct amd_repeat_operation {
    const char * kernel = nullptr;
    flagos_amd::repeat_f32_geometry binary;
    unsigned int grid_x = 0;
    int elements = 0;
    int columns = 0;
    float minimum = 0.0f;
    float maximum = 0.0f;
    bool is_binary = false;
};

static bool amd_describe_repeat_operation(const amd_device_context * device,
        const ggml_tensor * op, amd_repeat_operation * description) {
    if (description == nullptr) {
        return false;
    }
    *description = {};
    if (!amd_env_enabled("FLAGOS_AMD_REPEAT_OPS") || device == nullptr || device->aot == nullptr || op == nullptr) {
        return false;
    }
    size_t arguments = 0;
    size_t pointers = 0;
    int block = 256;
    if (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL || op->op == GGML_OP_DIV) {
        if ((op->op == GGML_OP_ADD && amd_supports_add(device, op)) ||
            (op->op == GGML_OP_MUL && amd_supports_mul(device, op)) ||
            !flagos_amd::describe_repeat_f32(op, &description->binary)) {
            return false;
        }
        description->kernel = op->op == GGML_OP_ADD ? "flagos_add_repeat_strided_f32" :
            op->op == GGML_OP_MUL ? "flagos_mul_repeat_strided_f32" : "flagos_div_repeat_strided_f32";
        description->is_binary = true;
        description->elements = description->binary.elements;
        arguments = 20;
        pointers = 3;
    } else if (op->op == GGML_OP_CLAMP || op->op == GGML_OP_SUM_ROWS) {
        const auto * source = op->src[0];
        if (!amd_tensor_is_contiguous_f32(op) || !amd_tensor_is_contiguous_f32(source) ||
            ggml_nelements(source) <= 0 || ggml_nelements(source) > INT32_MAX) {
            return false;
        }
        description->elements = static_cast<int>(ggml_nelements(source));
        if (op->op == GGML_OP_CLAMP) {
            if (!ggml_are_same_shape(op, source)) {
                return false;
            }
            std::memcpy(&description->minimum, op->op_params, sizeof(float));
            std::memcpy(&description->maximum, reinterpret_cast<const uint8_t *>(op->op_params) + sizeof(float), sizeof(float));
            if (std::isnan(description->minimum) || std::isnan(description->maximum) ||
                description->minimum > description->maximum) {
                return false;
            }
            description->kernel = "flagos_clamp_compare_f32";
            arguments = 5;
        } else {
            if (source->ne[0] <= 0 || source->ne[0] > 1024 || op->ne[0] != 1) {
                return false;
            }
            for (int dim = 1; dim < 4; ++dim) {
                if (op->ne[dim] != source->ne[dim]) {
                    return false;
                }
            }
            description->kernel = "flagos_sum_rows_f32_acc_f64";
            description->columns = static_cast<int>(source->ne[0]);
            block = 1024;
            arguments = 3;
        }
        pointers = 2;
    } else {
        return false;
    }
    const auto * output_storage = op->view_src == nullptr ? op : op->view_src;
    for (int input = 0; input < (description->is_binary ? 2 : 1); ++input) {
        const auto * source = op->src[input];
        const auto * input_storage = source->view_src == nullptr ? source : source->view_src;
        const bool exact_clamp_inplace = op->op == GGML_OP_CLAMP &&
            std::memcmp(op->nb, source->nb, sizeof(op->nb)) == 0 &&
            ((op->data != nullptr && op->data == source->data) ||
             (op->data == nullptr && source->data == nullptr && output_storage == input_storage &&
              op->view_offs == source->view_offs));
        if (!amd_tensor_data_is_aligned(source, sizeof(float)) ||
            (!exact_clamp_inplace && (output_storage == input_storage || amd_tensor_data_overlaps(op, source)))) {
            return false;
        }
    }
    if (!amd_tensor_data_is_aligned(op, sizeof(float))) {
        return false;
    }
    const auto * metadata = device->aot->find(description->kernel);
    description->grid_x = description->columns == 0
        ? static_cast<unsigned int>((static_cast<int64_t>(description->elements) + block - 1) / block)
        : static_cast<unsigned int>(description->elements / description->columns);
    return metadata != nullptr && metadata->block_size == block && metadata->num_warps == 4 &&
        metadata->warp_size == 32 && amd_kernel_launch_contract_fits(device, metadata,
            arguments, pointers, description->grid_x, 1, 1);
}

static bool amd_supports_repeat_operation(const amd_device_context * device, const ggml_tensor * op) {
    amd_repeat_operation description;
    return amd_describe_repeat_operation(device, op, &description);
}

static bool amd_launch_repeat_operation(amd_backend_context * context, ggml_tensor * node,
        amd_repeat_operation & repeat_operation) {
    flagos_amd::kernel_arguments arguments;
    if (repeat_operation.is_binary) {
        arguments.assign(&node->src[0]->data, &node->src[1]->data, &node->data);
        for (auto & value : repeat_operation.binary.dimensions) arguments.push_back(&value);
        for (auto & value : repeat_operation.binary.repeat) arguments.push_back(&value);
        for (auto & value : repeat_operation.binary.source_strides) arguments.push_back(&value);
        for (auto & value : repeat_operation.binary.repeat_strides) arguments.push_back(&value);
        arguments.push_back(&repeat_operation.binary.elements);
    } else if (repeat_operation.columns != 0) {
        arguments.assign(&node->src[0]->data, &node->data, &repeat_operation.columns);
    } else {
        arguments.assign(&node->src[0]->data, &node->data, &repeat_operation.elements,
            &repeat_operation.minimum, &repeat_operation.maximum);
    }
    if (!context->device->aot->launch(repeat_operation.kernel, context->stream,
            repeat_operation.grid_x, 1, 1, arguments)) {
        return false;
    }
    return true;
}
