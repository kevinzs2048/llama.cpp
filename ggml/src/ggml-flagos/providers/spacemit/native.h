#pragma once

#include "ggml-backend.h"

#include <cstdint>

struct flagos_spacemit_native_info {
    uint32_t vector_bits = 0;
    uint32_t workers = 0;
    const char * target = nullptr;
};

static inline bool flagos_spacemit_cpu_only_weight(const ggml_tensor * tensor, bool ime2) {
    if (tensor->type == GGML_TYPE_Q4_1 || tensor->type == GGML_TYPE_Q4_K ||
        tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q5_K ||
        tensor->type == GGML_TYPE_IQ2_S || tensor->type == GGML_TYPE_IQ3_XXS) {
        return true;
    }
    // Native Q6_K/Q8_0 paths fail selected partial-tile row counts; whole
    // 32-row tiles pass the large-matrix comparison on K3.
    if ((tensor->type == GGML_TYPE_Q6_K || tensor->type == GGML_TYPE_Q8_0) &&
        tensor->ne[1] % 32 != 0) {
        return true;
    }
    // The native IME2 selector repacks Q5_0 when rows are divisible by 32,
    // but that trait's compute_forward does not dispatch Q5_0.
    return tensor->type == GGML_TYPE_Q5_0 && ime2 && tensor->ne[1] % 32 == 0;
}

// Internal source adapter, not a binary plugin ABI.
bool flagos_spacemit_native_query(flagos_spacemit_native_info * info);
ggml_backend_reg_t flagos_spacemit_native_registry();
bool flagos_spacemit_native_is_backend(ggml_backend_t backend);
