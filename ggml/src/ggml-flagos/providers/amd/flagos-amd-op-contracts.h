#pragma once

#include "ggml.h"
#include "../../flagos-graph-plan.h"
#include "../../../ggml-impl.h"

#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>

namespace flagos_amd {

inline constexpr int dense_gather_block_size = 256;

struct dense_gather_signature {
    int32_t columns = 0;
    int32_t rows = 0;
};

// Runtime bindings and package availability are checked by the provider.
inline bool describe_dense_gather(const ggml_tensor * op, dense_gather_signature * signature) {
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
    if (op->type != GGML_TYPE_F32 || table->type != GGML_TYPE_F32 || indices->type != GGML_TYPE_I32 ||
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
