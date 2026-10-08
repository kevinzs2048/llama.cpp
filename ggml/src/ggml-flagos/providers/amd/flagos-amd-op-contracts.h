#pragma once

#include "ggml.h"
#include "../../flagos-graph-plan.h"
#include "../../../ggml-impl.h"

#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>

namespace flagos_amd {

struct splitkv_decode_geometry {
    int heads = 0;
    int splits = 0;
    int keys_per_split = 0;
    size_t partial_elements = 0;
    size_t workspace_bytes = 0;
};

inline bool supports_aligned_quantized_gemv_bindings(const ggml_tensor * op, int k, int rows) {
    if (op == nullptr || k <= 0 || rows <= 0 || k % 16 != 0 || rows % 16 != 0) {
        return false;
    }
    const ggml_tensor * operands[] = {op, op->src[0], op->src[1]};
    for (const auto * tensor : operands) {
        if (tensor == nullptr || tensor->data == nullptr ||
            reinterpret_cast<uintptr_t>(tensor->data) % 16 != 0 || ggml_nbytes(tensor) > INT32_MAX) {
            return false;
        }
    }
    return true;
}

// The provider also checks the full baseline attention layout and package ABI.
inline bool describe_splitkv_decode(const ggml_tensor * op, splitkv_decode_geometry * geometry) {
    if (geometry == nullptr) {
        return false;
    }
    *geometry = {};
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->op != GGML_OP_FLASH_ATTN_EXT || (op->ne[0] != 128 && op->ne[0] != 256) ||
        op->ne[1] <= 0 || op->ne[1] > 16 || op->ne[2] != 1 || op->ne[3] != 1 ||
        op->src[1]->ne[1] < 4096 || op->src[1]->ne[1] > 8192 || op->src[1]->ne[2] <= 0 ||
        op->ne[1] % op->src[1]->ne[2] != 0) {
        return false;
    }
    const int64_t group = op->ne[1] / op->src[1]->ne[2];
    if (group != 2 && group != 4 && group != 8) {
        return false;
    }
    geometry->heads = static_cast<int>(op->ne[1]);
    geometry->splits = 8;
    geometry->keys_per_split = static_cast<int>((op->src[1]->ne[1] + 7) / 8);
    geometry->partial_elements = static_cast<size_t>(geometry->heads) * geometry->splits;
    geometry->workspace_bytes = geometry->partial_elements * (static_cast<size_t>(op->ne[0]) + 2) * sizeof(float);
    return true;
}

inline bool supports_ggml_rope_layout(const ggml_tensor * op, int block_pairs) {
    if (op == nullptr || op->op != GGML_OP_ROPE || op->src[0] == nullptr ||
        op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->src[1]->type != GGML_TYPE_I32 || !ggml_are_same_shape(op, op->src[0]) ||
        !ggml_is_contiguous(op) || op->src[0]->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(op->src[1]) || op->src[1]->ne[1] != 1 ||
        op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1 ||
        op->src[1]->ne[0] > INT32_MAX ||
        op->ne[0] <= 0 || op->ne[0] % 2 != 0 || block_pairs != 128 || block_pairs < op->ne[0] / 2) {
        return false;
    }
    const ggml_tensor * tensors[] = { op, op->src[0], op->src[1], op->src[2] };
    for (const ggml_tensor * tensor : tensors) {
        if (tensor != nullptr &&
            ((tensor->data != nullptr && reinterpret_cast<uintptr_t>(tensor->data) % sizeof(float) != 0) ||
             (tensor->view_src != nullptr && tensor->view_offs % sizeof(float) != 0))) {
            return false;
        }
    }
    const auto storage = [](const ggml_tensor * tensor) {
        return tensor->view_src == nullptr ? tensor : tensor->view_src;
    };
    const ggml_tensor * output_storage = storage(op);
    for (const ggml_tensor * input : { op->src[1], op->src[2] }) {
        if (input != nullptr && output_storage == storage(input)) {
            return false;
        }
    }
    if (output_storage == storage(op->src[0]) &&
        (op->view_offs != op->src[0]->view_offs || !ggml_is_contiguous(op->src[0]))) {
        return false;
    }
    int64_t elements = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (op->ne[dim] <= 0 || op->ne[dim] > INT32_MAX / elements ||
            op->src[0]->nb[dim] % sizeof(float) != 0 ||
            op->src[0]->nb[dim] / sizeof(float) > INT32_MAX) {
            return false;
        }
        elements *= op->ne[dim];
    }
    if (ggml_nbytes(op->src[0]) / sizeof(float) > INT32_MAX) {
        return false;
    }
    const auto * params = static_cast<const int32_t *>(op->op_params);
    const int mode = params[2];
    if ((mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_IMROPE) ||
        params[1] <= 0 || params[1] % 2 != 0 || params[1] > op->ne[0] ||
        op->src[1]->ne[0] != op->ne[2] * (mode == GGML_ROPE_TYPE_IMROPE ? 4 : 1)) {
        return false;
    }
    if (mode == GGML_ROPE_TYPE_IMROPE) {
        int64_t section_total = 0;
        for (int i = 0; i < GGML_MROPE_SECTIONS; ++i) {
            if (params[11 + i] < 0 || params[11 + i] > INT32_MAX / 3) {
                return false;
            }
            section_total += params[11 + i];
        }
        if (section_total != params[1] / 2 ||
            (params[11] == 0 && params[12] == 0 && params[13] == 0)) {
            return false;
        }
    }
    const ggml_tensor * factors = op->src[2];
    if (factors != nullptr && (factors->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(factors) || factors->ne[0] < op->ne[0] / 2 ||
        factors->ne[1] != 1 || factors->ne[2] != 1 || factors->ne[3] != 1)) {
        return false;
    }
    float frequency[4];
    std::memcpy(frequency, params + 5, sizeof(frequency));
    return std::isfinite(frequency[0]) && frequency[0] > 0.0f &&
        std::isfinite(frequency[1]) && frequency[1] > 0.0f &&
        frequency[2] == 0.0f && std::isfinite(frequency[3]) &&
        // Reject scale overflow/underflow and denormals before GPU launch.
        std::isnormal(std::pow(frequency[0], -2.0f / params[1]));
}


inline bool supports_aligned_f16_gemm_launch(
        const void * weights, const void * x, const void * output,
        int k, int rows) {
    return weights != nullptr && x != nullptr && output != nullptr &&
        k > 0 && rows > 0 && k % 16 == 0 && rows % 16 == 0 &&
        reinterpret_cast<uintptr_t>(weights) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(output) % 16 == 0;
}


inline bool supports_tiled_f16_ffn_launch(
        const void * gate_weights, const void * up_weights,
        const void * x, const void * output, int k, int rows) {
    return supports_aligned_f16_gemm_launch(gate_weights, x, output, k, rows) &&
        up_weights != nullptr && reinterpret_cast<uintptr_t>(up_weights) % 16 == 0;
}

inline bool supports_strided_rms_norm_layout(const ggml_tensor * op, int block_cols) {
    if (op == nullptr || op->op != GGML_OP_RMS_NORM || op->src[0] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, op->src[0]) || !ggml_is_contiguous(op) ||
        op->src[0]->nb[0] != sizeof(float) || op->ne[0] <= 0 || op->ne[0] > block_cols) {
        return false;
    }
    int64_t elements = 1;
    int64_t maximum_offset = 0;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        const int64_t extent = op->ne[dim];
        if (extent <= 0 || extent > INT32_MAX / elements ||
            op->src[0]->nb[dim] % sizeof(float) != 0 ||
            op->src[0]->nb[dim] / sizeof(float) > INT32_MAX) {
            return false;
        }
        const int64_t stride = op->src[0]->nb[dim] / sizeof(float);
        if (extent > 1 && stride > (INT32_MAX - maximum_offset) / (extent - 1)) {
            return false;
        }
        maximum_offset += (extent - 1) * stride;
        elements *= extent;
    }
    float eps = 0.0f;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps <= 0.0f) {
        return false;
    }
    const uintptr_t output = reinterpret_cast<uintptr_t>(op->data);
    const uintptr_t input = reinterpret_cast<uintptr_t>(op->src[0]->data);
    if (output % sizeof(float) != 0 || input % sizeof(float) != 0) {
        return false;
    }
    if (output != 0 && input != 0) {
        const size_t output_size = static_cast<size_t>(elements) * sizeof(float);
        const size_t input_size = static_cast<size_t>(maximum_offset + 1) * sizeof(float);
        if (output > UINTPTR_MAX - output_size || input > UINTPTR_MAX - input_size ||
            (output < input + input_size && input < output + output_size)) {
            return false;
        }
    }
    return true;
}

inline constexpr int dense_gather_block_size = 256;

struct dense_gather_signature {
    int32_t columns = 0;
    int32_t rows = 0;
};

// Runtime bindings and package availability are checked by the provider.
inline bool describe_dense_gather_storage(
        const ggml_tensor * op, dense_gather_signature * signature, ggml_type table_type) {
    if (signature == nullptr) {
        return false;
    }
    *signature = {};
    if (op == nullptr || op->op != GGML_OP_GET_ROWS || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    const auto * table = op->src[0];
    const auto * indices = op->src[1];
    const int64_t max_elements = std::numeric_limits<int32_t>::max() / sizeof(float);
    if ((table_type != GGML_TYPE_F32 && table_type != GGML_TYPE_F16 && table_type != GGML_TYPE_BF16) ||
        op->type != GGML_TYPE_F32 || table->type != table_type || indices->type != GGML_TYPE_I32 ||
        table->ne[0] <= 0 || table->ne[0] > max_elements || table->ne[1] <= 0 ||
        table->ne[1] > max_elements / table->ne[0] || table->ne[2] != 1 || table->ne[3] != 1 ||
        indices->ne[0] < 0 || indices->ne[0] > max_elements / table->ne[0] ||
        indices->ne[1] != 1 || indices->ne[2] != 1 || indices->ne[3] != 1 ||
        op->ne[0] != table->ne[0] || op->ne[1] != indices->ne[0] || op->ne[2] != 1 || op->ne[3] != 1 ||
        !ggml_is_contiguous(table) || !ggml_is_contiguous(indices) || !ggml_is_contiguous(op)) {
        return false;
    }
    signature->columns = static_cast<int32_t>(op->ne[0]);
    signature->rows = static_cast<int32_t>(op->ne[1]);
    return true;
}

inline bool describe_dense_gather(const ggml_tensor * op, dense_gather_signature * signature) {
    return describe_dense_gather_storage(op, signature, GGML_TYPE_F32);
}

struct strided_float_matmul_signature {
    int32_t k = 0, rows = 0, columns = 0, heads = 0;
    int32_t head_ratio = 0, sample_ratio = 0;
    int32_t sw[3] = {}, sx[3] = {};
    uint32_t batches = 0;
    size_t weight_element_size = 0;
};

// Geometry shared by support checks and launch argument construction. Policy,
// precision, runtime bindings and module profiles remain provider decisions.
inline bool describe_strided_float_matmul(
        const ggml_tensor * op, strided_float_matmul_signature * signature) {
    if (signature == nullptr) {
        return false;
    }
    *signature = {};
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr ||
        op->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        (op->src[0]->type != GGML_TYPE_F16 && op->src[0]->type != GGML_TYPE_F32)) {
        return false;
    }
    const auto * weights = op->src[0];
    const auto * input = op->src[1];
    const size_t weight_element_size = ggml_type_size(weights->type);
    int64_t output_elements = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (op->ne[dim] <= 0 || op->ne[dim] > INT32_MAX / output_elements) {
            return false;
        }
        output_elements *= op->ne[dim];
    }
    if (!ggml_is_contiguous(op) || weights->ne[0] != input->ne[0] ||
        op->ne[0] != weights->ne[1] || op->ne[1] != input->ne[1] ||
        op->ne[2] != input->ne[2] || op->ne[3] != input->ne[3] ||
        weights->nb[0] != weight_element_size || input->nb[0] != sizeof(float) ||
        (op->ne[1] + 15) / 16 > 65535 || op->ne[2] * op->ne[3] > 65535) {
        return false;
    }
    for (const auto * source : {weights, input}) {
        const size_t element_size = source == weights ? weight_element_size : sizeof(float);
        int64_t last_element = 0;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (source->ne[dim] <= 0 || source->ne[dim] > INT32_MAX ||
                source->nb[dim] % element_size != 0 || source->nb[dim] / element_size > INT32_MAX) {
                return false;
            }
            const int64_t stride = source->nb[dim] / element_size;
            if (stride != 0 && source->ne[dim] - 1 > (INT32_MAX - last_element) / stride) {
                return false;
            }
            last_element += (source->ne[dim] - 1) * stride;
        }
    }
    if (input->ne[2] % weights->ne[2] != 0 || input->ne[3] % weights->ne[3] != 0) {
        return false;
    }
    signature->k = static_cast<int32_t>(weights->ne[0]);
    signature->rows = static_cast<int32_t>(op->ne[0]);
    signature->columns = static_cast<int32_t>(op->ne[1]);
    signature->heads = static_cast<int32_t>(op->ne[2]);
    signature->head_ratio = static_cast<int32_t>(input->ne[2] / weights->ne[2]);
    signature->sample_ratio = static_cast<int32_t>(input->ne[3] / weights->ne[3]);
    signature->batches = static_cast<uint32_t>(op->ne[2] * op->ne[3]);
    signature->weight_element_size = weight_element_size;
    for (int dim = 0; dim < 3; ++dim) {
        signature->sw[dim] = static_cast<int32_t>(weights->nb[dim + 1] / weight_element_size);
        signature->sx[dim] = static_cast<int32_t>(input->nb[dim + 1] / sizeof(float));
    }
    return true;
}

inline bool gdn_cache_only_outputs_are_safe(
        const ggml_cgraph * cgraph,
        const flagos_pattern_candidate & candidate,
        const ggml_tensor * gdn,
        const ggml_tensor * copy) {
    if (cgraph == nullptr || gdn == nullptr || copy == nullptr || gdn->data == nullptr ||
        static_cast<const int32_t *>(gdn->op_params)[0] != 1 ||
        candidate.node_indices.empty() || candidate.node_indices.front() < 0 ||
        candidate.node_indices.front() >= cgraph->n_nodes ||
        cgraph->nodes[candidate.node_indices.front()] != gdn ||
        candidate.node_indices.back() < 0 ||
        candidate.node_indices.back() >= cgraph->n_nodes ||
        cgraph->nodes[candidate.node_indices.back()] != copy) {
        return false;
    }
    // The audit below only sees this split. If another split consumes the
    // allocation or a covered alias, materialize the complete snapshot instead.
    for (const int index : candidate.node_indices) {
        if (index < 0 || index >= cgraph->n_nodes) {
            return false;
        }
        const ggml_tensor * node = cgraph->nodes[index];
        if ((node == gdn || node->view_src == gdn) &&
            !flagos_graph_exposes_all_consumers(cgraph, node)) {
            return false;
        }
    }
    const ggml_tensor * value = gdn->src[2];
    if (value == nullptr || value->ne[0] <= 0 || value->ne[1] <= 0 ||
        value->ne[2] <= 0 || value->ne[3] <= 0) {
        return false;
    }
    uint64_t attention_elements = 1;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        const uint64_t extent = static_cast<uint64_t>(value->ne[dimension]);
        if (extent > UINT64_MAX / attention_elements) {
            return false;
        }
        attention_elements *= extent;
    }
    if (attention_elements > SIZE_MAX / sizeof(float)) {
        return false;
    }
    const uintptr_t attention_begin = reinterpret_cast<uintptr_t>(gdn->data);
    const size_t attention_bytes = static_cast<size_t>(attention_elements) * sizeof(float);
    if (attention_begin > UINTPTR_MAX - attention_bytes) {
        return false;
    }
    const uintptr_t attention_end = attention_begin + attention_bytes;
    const int copy_index = candidate.node_indices.back();
    const ggml_tensor * destination = copy->src[1];
    if (destination == nullptr) {
        return false;
    }
    std::vector<bool> covered(static_cast<size_t>(cgraph->n_nodes), false);
    for (const int index : candidate.node_indices) {
        if (index < 0 || index >= cgraph->n_nodes || covered[static_cast<size_t>(index)]) {
            return false;
        }
        covered[static_cast<size_t>(index)] = true;
    }
    const auto alias_is_in_attention = [&](const ggml_tensor * tensor) {
        if (tensor == nullptr || tensor->data == nullptr || tensor->view_src != gdn) {
            return false;
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data);
        const size_t bytes = ggml_nbytes(tensor);
        return bytes > 0 && begin >= attention_begin && begin <= UINTPTR_MAX - bytes &&
            begin + bytes <= attention_end;
    };
    // A zero-work view may be absent from cgraph->nodes and appear only as a
    // later node's source.  Required-output bookkeeping is node based, so
    // audit those implicit aliases here before omitting the snapshot suffix.
    // The covered CPY is allowed to read the suffix because this kernel writes
    // its destination directly; every other consumer must stay in the
    // materialized attention prefix.
    for (int consumer_index = 0; consumer_index < cgraph->n_nodes; ++consumer_index) {
        if (covered[static_cast<size_t>(consumer_index)]) {
            continue;
        }
        const ggml_tensor * consumer = cgraph->nodes[consumer_index];
        if (consumer == nullptr ||
            (consumer->view_src == gdn && !alias_is_in_attention(consumer))) {
            return false;
        }
        for (int source = 0; source < GGML_MAX_SRC; ++source) {
            const ggml_tensor * input = consumer->src[source];
            const bool attention_view_input = input == gdn &&
                consumer->view_src == gdn && alias_is_in_attention(consumer);
            if ((input == gdn && !attention_view_input) ||
                (input != nullptr && input->view_src == gdn && !alias_is_in_attention(input))) {
                return false;
            }
        }
    }
    bool copy_required = false;
    for (const int index : candidate.required_output_node_indices) {
        if (index == copy_index) {
            if (copy_required) {
                return false;
            }
            copy_required = true;
            continue;
        }
        if (index < 0 || index >= cgraph->n_nodes ||
            std::find(candidate.node_indices.begin(), candidate.node_indices.end(), index) ==
                candidate.node_indices.end()) {
            return false;
        }
        const ggml_tensor * output = cgraph->nodes[index];
        if (output == nullptr || output->data == nullptr) {
            return false;
        }
        // A scheduled destination VIEW and the terminal CPY expose the same
        // cache storage.  The cache-only kernel materializes both contracts
        // with its direct cache write.
        if (output == destination) {
            continue;
        }
        // A required backing tensor denotes the complete allocation. Partial
        // materialization is only valid for explicitly listed prefix aliases.
        if (output == gdn) {
            return false;
        }
        const uintptr_t begin = reinterpret_cast<uintptr_t>(output->data);
        const size_t bytes = ggml_nbytes(output);
        if (bytes == 0 || begin < attention_begin || begin > UINTPTR_MAX - bytes ||
            begin + bytes > attention_end) {
            return false;
        }
    }
    return copy_required;
}


}
