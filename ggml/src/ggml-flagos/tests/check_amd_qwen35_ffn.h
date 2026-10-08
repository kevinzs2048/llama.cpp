#pragma once

#include "ggml-backend.h"
#include "ggml.h"

// Check packed MUL_MAT + MUL_MAT + SWIGLU under the configured provider policy.
// Direct execution must produce both projections; fusion must preserve both sentinels.
void flagos_check_amd_qwen35_ffn_case(
        ggml_backend_t backend, ggml_backend_dev_t device,
        ggml_backend_buffer_type_t buft, ggml_context * context,
        ggml_type quant_type);
