#pragma once

#include "ggml.h"

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

}
