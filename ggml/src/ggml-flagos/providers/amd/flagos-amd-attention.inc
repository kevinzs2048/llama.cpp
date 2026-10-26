// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Private attention profiles, workspace selection, and dispatch.
static bool amd_flash_attn_tensor_is_aligned(const ggml_tensor * tensor, size_t element_size) {
    return amd_tensor_data_is_aligned(tensor, element_size) &&
        (tensor->view_src == nullptr || tensor->view_offs % element_size == 0);
}

static const flagos_amd::kernel_metadata * amd_flash_attn_kernel(
        const amd_device_context * device, const ggml_tensor * op, bool prefill) {
    if (device == nullptr || device->aot == nullptr || op == nullptr ||
        (op->ne[0] != 128 && op->ne[0] != 256)) {
        return nullptr;
    }
    const char * name = prefill
        ? (op->ne[0] == 256 ? "flagos_flash_attn_prefill_f32_f16_generic_h256" : "flagos_flash_attn_prefill_f32_f16_generic_h128")
        : (op->ne[0] == 256 ? "flagos_flash_attn_decode_f32_f16_generic_h256" : "flagos_flash_attn_decode_f32_f16_generic_h128");
    const auto * metadata = device->aot->find(name);
    if (metadata == nullptr || metadata->argument_count != (prefill ? 19 : 16) ||
        metadata->tile_k != op->ne[0] || metadata->tile_m != 16 ||
        metadata->tile_n < 16 || metadata->tile_n > 64 ||
        (metadata->tile_n & (metadata->tile_n - 1)) != 0 ||
        metadata->block_size != (prefill ? metadata->tile_m : metadata->tile_n)) {
        return nullptr;
    }
    const uint64_t query_tiles = prefill
        ? (static_cast<uint64_t>(op->ne[2]) + metadata->block_size - 1) / metadata->block_size : 1;
    return amd_kernel_launch_contract_fits(device, metadata, prefill ? 19 : 16, 5,
        static_cast<uint64_t>(op->ne[1]), query_tiles, 1) ? metadata : nullptr;
}

static bool amd_supports_flash_attn_decode(const amd_device_context * device, const ggml_tensor * op) {
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!amd_fusion_enabled(device, "flash_attn_decode") || device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT || amd_flash_attn_kernel(device, op, false) == nullptr) {
        return false;
    }
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
    std::memcpy(&scale, op_params_bytes, sizeof(scale));
    std::memcpy(&max_bias, op_params_bytes + sizeof(float), sizeof(max_bias));
    std::memcpy(&logit_softcap, op_params_bytes + 2 * sizeof(float), sizeof(logit_softcap));
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    return q && k && v && mask && op->src[4] == nullptr &&
        op->view_src == nullptr && op->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 &&
        (op->ne[0] == 128 || op->ne[0] == 256) && op->ne[1] > 0 && op->ne[1] <= INT32_MAX && op->ne[2] == 1 && op->ne[3] == 1 &&
        q->ne[0] == op->ne[0] && q->ne[1] == 1 && q->ne[2] == op->ne[1] && q->ne[3] == 1 &&
        k->ne[0] == op->ne[0] && k->ne[1] > 0 && k->ne[1] <= INT32_MAX && k->ne[2] > 0 && k->ne[2] <= INT32_MAX && k->ne[3] == 1 &&
        q->ne[2] % k->ne[2] == 0 && ggml_are_same_shape(k, v) &&
        mask->ne[0] == k->ne[1] && mask->ne[1] == 1 && mask->ne[2] == 1 && mask->ne[3] == 1 &&
        q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
        k->nb[1] >= static_cast<size_t>(op->ne[0]) * sizeof(ggml_fp16_t) &&
        k->nb[2] == static_cast<size_t>(op->ne[0]) * sizeof(ggml_fp16_t) &&
        v->nb[1] == k->nb[1] && v->nb[2] == k->nb[2] &&
        amd_tensor_element_offsets_fit_i32(q, sizeof(float)) &&
        amd_tensor_element_offsets_fit_i32(k, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(v, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(mask, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(op, sizeof(float)) &&
        amd_flash_attn_tensor_is_aligned(q, sizeof(float)) &&
        amd_flash_attn_tensor_is_aligned(k, sizeof(ggml_fp16_t)) &&
        amd_flash_attn_tensor_is_aligned(v, sizeof(ggml_fp16_t)) &&
        amd_flash_attn_tensor_is_aligned(mask, sizeof(ggml_fp16_t)) &&
        amd_flash_attn_tensor_is_aligned(op, sizeof(float)) &&
        q->nb[1] / sizeof(float) <= INT32_MAX && q->nb[2] / sizeof(float) <= INT32_MAX &&
        k->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && k->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        v->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && v->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        op->nb[1] / sizeof(float) <= INT32_MAX && op->nb[2] / sizeof(float) <= INT32_MAX &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) &&
        std::isfinite(scale) && max_bias == 0.0f && logit_softcap == 0.0f &&
        !amd_tensor_data_overlaps(op, q) && !amd_tensor_data_overlaps(op, k) &&
        !amd_tensor_data_overlaps(op, v) && !amd_tensor_data_overlaps(op, mask);
}

static const flagos_amd::kernel_metadata * amd_grouped_splitkv_kernel(
        const amd_device_context * device, const ggml_tensor * op) {
    if (!amd_env_enabled("FLAGOS_AMD_GROUPED_SPLITKV_DECODE") || op->ne[1] != 16 ||
        op->src[1] == nullptr || op->src[1]->ne[2] <= 0) {
        return nullptr;
    }
    static const struct { int head; int group; const char * name; } profiles[] = {
        {128, 2, "flagos_flash_decode_splitkv_grouped_f32_f16_h128_g2"},
        {128, 4, "flagos_flash_decode_splitkv_grouped_f32_f16_h128_g4"},
        {128, 8, "flagos_flash_decode_splitkv_grouped_f32_f16_h128_g8"},
        {256, 8, "flagos_flash_decode_splitkv_grouped_f32_f16_h256_g8"},
    };
    const int64_t group = op->ne[1] / op->src[1]->ne[2];
    for (const auto & profile : profiles) {
        if (op->ne[0] != profile.head || group != profile.group) {
            continue;
        }
        const auto * metadata = device->aot->find(profile.name);
        if (metadata != nullptr && metadata->argument_count == 18 &&
            metadata->symbol == "flagos_flash_decode_splitkv_grouped_partial_f32_f16" &&
            metadata->tile_m == 16 && metadata->tile_n == 16 && metadata->tile_k == profile.head &&
            metadata->block_size == 16 && metadata->num_warps == 4 && metadata->warp_size == 32 &&
            metadata->shared_memory == (profile.head == 128 ? 4096 : 8192) &&
            metadata->global_scratch_size == 0 && metadata->profile_scratch_size == 0) {
            return metadata;
        }
    }
    return nullptr;
}

static const flagos_amd::kernel_metadata * amd_splitkv_kernel(
        const amd_device_context * device, const ggml_tensor * op, bool merge) {
    if (!merge) {
        if (const auto * grouped = amd_grouped_splitkv_kernel(device, op)) {
            return grouped;
        }
    }
    const char * name = merge
        ? (op->ne[0] == 256 ? "flagos_flash_decode_splitkv_merge_f32_h256" : "flagos_flash_decode_splitkv_merge_f32_h128")
        : (op->ne[0] == 256 ? "flagos_flash_decode_splitkv_partial_f32_f16_h256" : "flagos_flash_decode_splitkv_partial_f32_f16_h128");
    const auto * metadata = device->aot->find(name);
    if (metadata == nullptr || metadata->argument_count != (merge ? 7 : 18) ||
        metadata->symbol != (merge ? "flagos_flash_decode_splitk_merge_f32" : "flagos_flash_decode_splitk_stage1_f32_f16") ||
        metadata->tile_k != op->ne[0] || metadata->tile_m != 0 ||
        metadata->tile_n != (merge ? 16 : 32) || metadata->block_size != (merge ? 16 : 32) ||
        metadata->num_warps != 4 || metadata->warp_size != 32 ||
        metadata->shared_memory != (merge ? 0 : 512) ||
        metadata->global_scratch_size != 0 || metadata->profile_scratch_size != 0) {
        return nullptr;
    }
    return metadata;
}

static bool amd_describe_splitkv(const amd_backend_context * context, const ggml_tensor * op,
        flagos_amd::splitkv_decode_geometry * geometry) {
    if (!amd_env_enabled("FLAGOS_AMD_SPLITKV_DECODE") || context == nullptr ||
        !amd_supports_flash_attn_decode(context->device, op) ||
        flagos_amd::base_gcn_arch(context->device->arch) != "gfx1150" ||
        !flagos_amd::describe_splitkv_decode(op, geometry)) {
        return false;
    }
    const auto * partial = amd_splitkv_kernel(context->device, op, false);
    const auto * merge = amd_splitkv_kernel(context->device, op, true);
    if (partial == nullptr || merge == nullptr) {
        return false;
    }
    const uint64_t group = static_cast<uint64_t>(op->src[0]->ne[2] / op->src[1]->ne[2]);
    const uint64_t partial_heads = partial->tile_m == 16 ? geometry->heads / group : geometry->heads;
    return amd_kernel_launch_contract_fits(context->device, partial, 18, 7,
            partial_heads, geometry->splits, 1) &&
        amd_kernel_launch_contract_fits(context->device, merge, 7, 4, geometry->heads, 1, 1);
}

static bool amd_splitkv_ready(const amd_backend_context * context, const ggml_tensor * op) {
    flagos_amd::splitkv_decode_geometry geometry;
    return amd_describe_splitkv(context, op, &geometry) && context->splitkv_workspace != nullptr &&
        geometry.workspace_bytes <= context->splitkv_workspace_capacity;
}

static bool amd_prepare_splitkv_workspace(amd_backend_context * context, const ggml_cgraph * graph) {
    if (!amd_env_enabled("FLAGOS_AMD_SPLITKV_DECODE")) {
        return true;
    }
    size_t required = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == nullptr || graph->nodes[i]->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        flagos_amd::splitkv_decode_geometry geometry;
        if (amd_describe_splitkv(context, graph->nodes[i], &geometry)) {
            required = std::max(required, geometry.workspace_bytes);
        }
    }
    if (required <= context->splitkv_workspace_capacity) {
        return true;
    }
    if (!amd_hip_check(hipStreamSynchronize(context->stream), "hipStreamSynchronize before split-KV allocation")) {
        return false;
    }
    void * workspace = nullptr;
    if (!amd_hip_check(hipMalloc(&workspace, required), "hipMalloc split-KV workspace")) {
        GGML_LOG_WARN("FlagOS AMD: split-KV workspace unavailable; using qualified decode baseline\n");
        return true;
    }
    context->graph_plans.clear();
    amd_clear_graph_captures(context);
    if (context->splitkv_workspace != nullptr &&
        !amd_hip_check(hipFree(context->splitkv_workspace), "hipFree retired split-KV workspace")) {
        amd_hip_check(hipFree(workspace), "hipFree unused split-KV workspace");
        return false;
    }
    context->splitkv_workspace = workspace;
    context->splitkv_workspace_capacity = required;
    ++context->splitkv_workspace_generation;
    context->stats.splitkv_workspace_allocations.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static bool amd_supports_flash_attn_prefill(const amd_device_context * device, const ggml_tensor * op) {
    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    if (!amd_fusion_enabled(device, "flash_attn_prefill") || device == nullptr || device->aot == nullptr || op == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT ||
        amd_flash_attn_kernel(device, op, true) == nullptr) {
        return false;
    }
    const auto * op_params_bytes = reinterpret_cast<const uint8_t *>(op->op_params);
    std::memcpy(&scale, op_params_bytes, sizeof(scale));
    std::memcpy(&max_bias, op_params_bytes + sizeof(float), sizeof(max_bias));
    std::memcpy(&logit_softcap, op_params_bytes + 2 * sizeof(float), sizeof(logit_softcap));
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    // The first prefill implementation deliberately uses one contiguous
    // F16 mask plane shared by all heads/batches.  More general broadcast and
    // ALiBi variants remain provider capabilities that can be added later
    // without changing the common pattern ABI.
    return q && k && v && mask && op->src[4] == nullptr &&
        op->view_src == nullptr && op->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        mask->type == GGML_TYPE_F16 && q->ne[1] > 1 && q->ne[1] <= INT32_MAX &&
        (op->ne[0] == 128 || op->ne[0] == 256) && q->ne[0] == op->ne[0] && k->ne[0] == op->ne[0] && v->ne[0] == op->ne[0] &&
        op->ne[1] == q->ne[2] && op->ne[2] == q->ne[1] && op->ne[3] == q->ne[3] &&
        q->ne[2] > 0 && q->ne[2] <= INT32_MAX && q->ne[3] == 1 &&
        k->ne[1] > 0 && k->ne[1] <= INT32_MAX && k->ne[2] > 0 && k->ne[2] <= INT32_MAX &&
        k->ne[3] == 1 && v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2] && v->ne[3] == 1 &&
        q->ne[2] % k->ne[2] == 0 &&
        mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1] &&
        mask->ne[2] == 1 && mask->ne[3] == 1 &&
        q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) &&
        v->nb[0] == sizeof(ggml_fp16_t) && mask->nb[0] == sizeof(ggml_fp16_t) &&
        op->nb[0] == sizeof(float) && q->nb[1] / sizeof(float) <= INT32_MAX &&
        q->nb[2] / sizeof(float) <= INT32_MAX && k->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        k->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        v->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX && v->nb[2] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        mask->nb[1] / sizeof(ggml_fp16_t) <= INT32_MAX &&
        amd_tensor_element_offsets_fit_i32(q, sizeof(float)) &&
        amd_tensor_element_offsets_fit_i32(k, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(v, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(mask, sizeof(ggml_fp16_t)) &&
        amd_tensor_element_offsets_fit_i32(op, sizeof(float)) &&
        amd_flash_attn_tensor_is_aligned(q, sizeof(float)) &&
        amd_flash_attn_tensor_is_aligned(k, sizeof(ggml_fp16_t)) &&
        amd_flash_attn_tensor_is_aligned(v, sizeof(ggml_fp16_t)) &&
        amd_flash_attn_tensor_is_aligned(mask, sizeof(ggml_fp16_t)) &&
        amd_flash_attn_tensor_is_aligned(op, sizeof(float)) &&
        op->nb[1] / sizeof(float) <= INT32_MAX && op->nb[2] / sizeof(float) <= INT32_MAX &&
        ggml_is_contiguous(mask) && ggml_is_contiguous(op) &&
        std::isfinite(scale) && max_bias == 0.0f && logit_softcap == 0.0f &&
        !amd_tensor_data_overlaps(op, q) && !amd_tensor_data_overlaps(op, k) &&
        !amd_tensor_data_overlaps(op, v) && !amd_tensor_data_overlaps(op, mask);
}

static bool amd_launch_flash_attn(amd_backend_context * context, ggml_tensor * node, bool prefill) {
    if (!(prefill ? amd_supports_flash_attn_prefill(context->device, node)
                  : amd_supports_flash_attn_decode(context->device, node)) ||
        !amd_tensor_bindings_ready(node)) {
        return false;
    }
    const auto * metadata = amd_flash_attn_kernel(context->device, node, prefill);
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    int query_length = static_cast<int>(q->ne[1]);
    int key_length = static_cast<int>(k->ne[1]);
    int q_per_kv = static_cast<int>(q->ne[2] / k->ne[2]);
    int stride_q_token = static_cast<int>(q->nb[1] / sizeof(float));
    int stride_q_head = static_cast<int>(q->nb[2] / sizeof(float));
    int stride_k_token = static_cast<int>(k->nb[1] / sizeof(ggml_fp16_t));
    int stride_k_head = static_cast<int>(k->nb[2] / sizeof(ggml_fp16_t));
    int stride_v_token = static_cast<int>(v->nb[1] / sizeof(ggml_fp16_t));
    int stride_v_head = static_cast<int>(v->nb[2] / sizeof(ggml_fp16_t));
    int stride_mask_token = static_cast<int>(mask->nb[0] / sizeof(ggml_fp16_t));
    int stride_mask_query = static_cast<int>(mask->nb[1] / sizeof(ggml_fp16_t));
    int stride_output_token = static_cast<int>(node->nb[2] / sizeof(float));
    int stride_output_head = static_cast<int>(node->nb[1] / sizeof(float));
    float scale = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(scale));
    void * q_data = q->data;
    void * k_data = k->data;
    void * v_data = v->data;
    void * mask_data = mask->data;
    void * output_data = node->data;
    flagos_amd::kernel_arguments arguments;
    if (!prefill && amd_splitkv_ready(context, node)) {
        flagos_amd::splitkv_decode_geometry geometry;
        if (!amd_describe_splitkv(context, node, &geometry)) {
            return false;
        }
        const auto * partial = amd_splitkv_kernel(context->device, node, false);
        const auto * merge = amd_splitkv_kernel(context->device, node, true);
        const bool grouped = partial->tile_m == 16;
        void * partial_m = context->splitkv_workspace;
        void * partial_l = static_cast<float *>(partial_m) + geometry.partial_elements;
        void * partial_a = static_cast<float *>(partial_l) + geometry.partial_elements;
        arguments.assign(&q_data, &k_data, &v_data, &mask_data, &partial_m, &partial_l, &partial_a,
            &key_length, &q_per_kv, &geometry.splits, &geometry.keys_per_split,
            &stride_q_token, &stride_q_head, &stride_k_token, &stride_k_head,
            &stride_v_token, &stride_v_head, &scale);
        if (!context->device->aot->launch(partial->name, context->stream,
                static_cast<unsigned int>(grouped ? geometry.heads / q_per_kv : geometry.heads),
                static_cast<unsigned int>(geometry.splits), 1, arguments)) {
            return false;
        }
        arguments.assign(&partial_m, &partial_l, &partial_a, &output_data,
            &geometry.splits, &stride_output_token, &stride_output_head);
        if (!context->device->aot->launch(merge->name, context->stream,
                static_cast<unsigned int>(geometry.heads), 1, 1, arguments)) {
            return false;
        }
        context->stats.splitkv_decode.fetch_add(1, std::memory_order_relaxed);
        if (grouped) {
            context->stats.splitkv_grouped_decode.fetch_add(1, std::memory_order_relaxed);
        }
        context->stats.kernel_launches.fetch_add(1, std::memory_order_relaxed);
        amd_trace_op(context, node, grouped ? "flash_attn_decode_grouped_splitkv" : "flash_attn_decode_splitkv");
        return true;
    }
    if (prefill) {
        arguments.assign(&q_data, &k_data, &v_data, &mask_data, &output_data,
            &query_length, &key_length, &q_per_kv, &stride_q_token, &stride_q_head,
            &stride_k_token, &stride_k_head, &stride_v_token, &stride_v_head,
            &stride_mask_token, &stride_mask_query, &stride_output_token, &stride_output_head, &scale);
    } else {
        arguments.assign(&q_data, &k_data, &v_data, &mask_data, &output_data,
            &key_length, &q_per_kv, &stride_q_token, &stride_q_head,
            &stride_k_token, &stride_k_head, &stride_v_token, &stride_v_head,
            &stride_output_token, &stride_output_head, &scale);
    }
    const unsigned int grid_y = prefill ? static_cast<unsigned int>(
        (static_cast<uint64_t>(query_length) + metadata->block_size - 1) / metadata->block_size) : 1;
    return context->device->aot->launch(metadata->name, context->stream,
        static_cast<unsigned int>(q->ne[2]), grid_y, 1, arguments);
}

