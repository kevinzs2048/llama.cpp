// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
struct amd_batched_gather_launch {
    const char * kernel = nullptr;
    int values[13]{};
    unsigned int grid_x = 0;
};

static bool amd_describe_batched_gather(const amd_device_context * device,
        const ggml_tensor * op, amd_batched_gather_launch * launch) {
    if (!amd_env_enabled("FLAGOS_AMD_BATCHED_GATHER") || device == nullptr || device->aot == nullptr ||
        op == nullptr || launch == nullptr || op->op != GGML_OP_GET_ROWS ||
        op->src[0] == nullptr || op->src[1] == nullptr || op->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I32 || !ggml_is_contiguous(op)) {
        return false;
    }
    *launch = {};
    const auto * table = op->src[0];
    const auto * indices = op->src[1];
    const char * name = table->type == GGML_TYPE_F32 ? "flagos_get_rows_batched_f32"
        : table->type == GGML_TYPE_F16 ? "flagos_get_rows_batched_f16_f32"
        : table->type == GGML_TYPE_BF16 ? "flagos_get_rows_batched_bf16_bits_f32" : nullptr;
    if (name == nullptr || table->nb[0] != ggml_type_size(table->type) ||
        table->ne[2] != indices->ne[1] || table->ne[3] != indices->ne[2] || indices->ne[3] != 1 ||
        op->ne[0] != table->ne[0] || op->ne[1] != indices->ne[0] ||
        op->ne[2] != table->ne[2] || op->ne[3] != table->ne[3] ||
        ggml_nelements(op) <= 0 || ggml_nelements(op) > INT32_MAX ||
        !amd_tensor_element_offsets_fit_i32(table, ggml_type_size(table->type)) ||
        !amd_tensor_element_offsets_fit_i32(indices, sizeof(int32_t)) ||
        !amd_tensor_data_is_aligned(table, ggml_type_size(table->type)) ||
        !amd_tensor_data_is_aligned(indices, sizeof(int32_t)) ||
        !amd_tensor_data_is_aligned(op, sizeof(float))) {
        return false;
    }
    for (const auto * source : {table, indices}) {
        for (int dim = 0; dim < 4; ++dim) {
            if (source->ne[dim] <= 0 || source->ne[dim] > INT32_MAX) return false;
        }
        const auto * output_root = op->view_src == nullptr ? op : op->view_src;
        const auto * source_root = source->view_src == nullptr ? source : source->view_src;
        if (output_root == source_root || amd_tensor_data_overlaps(op, source)) return false;
    }
    const auto * metadata = device->aot->find(name);
    const uint64_t grid = (static_cast<uint64_t>(ggml_nelements(op)) + 255) / 256;
    if (metadata == nullptr || metadata->block_size != 256 || metadata->num_warps != 4 ||
        metadata->warp_size != 32 || metadata->shared_memory != 0 || metadata->static_shared_memory != 0 ||
        !amd_kernel_launch_contract_fits(device, metadata, 16, 3, grid, 1, 1)) {
        return false;
    }
    launch->kernel = name;
    launch->grid_x = static_cast<unsigned int>(grid);
    int * values = launch->values;
    values[0] = static_cast<int>(table->ne[0]);
    values[1] = static_cast<int>(table->ne[1]);
    values[2] = static_cast<int>(indices->ne[0]);
    values[3] = static_cast<int>(table->ne[2]);
    values[4] = static_cast<int>(table->ne[3]);
    for (int dim = 0; dim < 4; ++dim) values[5 + dim] = static_cast<int>(table->nb[dim] / ggml_type_size(table->type));
    for (int dim = 0; dim < 3; ++dim) values[9 + dim] = static_cast<int>(indices->nb[dim] / sizeof(int32_t));
    values[12] = static_cast<int>(ggml_nelements(op));
    return true;
}
