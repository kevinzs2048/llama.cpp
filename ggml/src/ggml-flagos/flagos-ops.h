#pragma once

// L2 op adapter: maps a GGML node to a FlagGems kernel call.
//
// Kernel signatures come from the FlagGems export and do not depend on the
// target, so this mapping is written once in the core and used by every
// RUNTIME provider. Target differences (which kernels a package contains,
// BLOCK sizes, argument counts) are read from the package manifest.
//
// Contracts here match the kernels the AMD package already ships
// (flagos_add_f32, flagos_rms_norm_f32, ...), so the same package and the
// same mapping serve the AMD runtime once it implements flagos_runtime_v1.

#include "flagos-package.h"
#include "flagos-runtime.h"

#include "ggml.h"

#include <cstdint>

static constexpr uint32_t FLAGOS_MAX_KERNEL_ARGS = 16;

struct flagos_kernel_call {
    const flagos_kernel_info * kernel = nullptr;
    uint32_t grid[3] = { 1, 1, 1 };
    uint32_t n_args = 0;
    flagos_rt_arg args[FLAGOS_MAX_KERNEL_ARGS] = {};
};

// Ops that need no kernel (views and reshapes). graph_compute skips them.
bool flagos_op_is_noop(const ggml_tensor * op);

// Decide whether `op` maps to a kernel in `package`. When `call` is not
// NULL, also fill in the kernel, grid and arguments; tensor data pointers
// must then be set. supports_op calls this with call == NULL so the decision
// and the launch can never disagree.
bool flagos_op_plan(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call);
