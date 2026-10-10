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

// Matches the 30 explicit arguments the AMD launcher accepts, rounded up.
static constexpr uint32_t FLAGOS_MAX_KERNEL_ARGS = 32;

struct flagos_kernel_call {
    const flagos_kernel_info * kernel = nullptr;
    uint32_t grid[3] = { 1, 1, 1 };
    uint32_t n_args = 0;
    flagos_rt_arg args[FLAGOS_MAX_KERNEL_ARGS] = {};
};

// Ops that need no kernel (views and reshapes). graph_compute skips them.
bool flagos_op_is_noop(const ggml_tensor * op);

// Largest explicit argument count a kernel may take (launchers append
// compiler-managed arguments after these, e.g. two Triton scratch slots).
static constexpr uint32_t FLAGOS_MAX_EXPLICIT_KERNEL_ARGS = 30;

// Decide whether `op` maps to a kernel in `package`. When `call` is not
// NULL, also fill in the kernel, grid and arguments; tensor data pointers
// must then be set. supports_op calls this with call == NULL so the decision
// and the launch can never disagree.
bool flagos_op_plan(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call);

struct ggml_cgraph;
struct flagos_pattern_candidate;

// Fused patterns found by the Common planner (flagos-graph-plan.h). Same
// contract as flagos_op_plan: with call == NULL it only decides; otherwise it
// fills the fused kernel call. The candidate's required outputs must be ones
// the fused kernel writes. Whether a pattern is attempted at all is the
// provider's choice (flagos_runtime_v1::fusion_supported); whether it can run
// depends on the package containing the fused kernel.
bool flagos_fusion_plan(const flagos_kernel_package & package, const ggml_cgraph * graph,
                        const flagos_pattern_candidate & candidate, flagos_kernel_call * call);
