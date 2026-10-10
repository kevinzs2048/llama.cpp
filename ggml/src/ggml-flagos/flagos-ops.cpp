#include "flagos-ops.h"

#include <cmath>
#include <cstring>

bool flagos_op_is_noop(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

namespace {

bool is_contiguous_f32(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

bool fits_i32(int64_t value) {
    return value > 0 && value <= INT32_MAX;
}

// Look the kernel up and check that the package agrees on the argument
// count, when it declares one, and that the 1-D grid fits.
const flagos_kernel_info * find_kernel(const flagos_kernel_package & package, const char * name, uint32_t n_args,
                                       int64_t programs) {
    const flagos_kernel_info * kernel = package.find(name);
    if (kernel == nullptr || (kernel->argument_count >= 0 && static_cast<uint32_t>(kernel->argument_count) != n_args) ||
        programs <= 0 || programs > INT32_MAX) {
        return nullptr;
    }
    return kernel;
}

int64_t programs_for(const flagos_kernel_info * kernel, int64_t elements) {
    const int64_t block = kernel != nullptr && kernel->block_size > 0 ? kernel->block_size : 0;
    return block == 0 ? 0 : (elements + block - 1) / block;
}

// Find a kernel whose grid is ceil(elements / BLOCK).
const flagos_kernel_info * find_elementwise(const flagos_kernel_package & package, const char * name,
                                            uint32_t n_args, int64_t elements) {
    const flagos_kernel_info * kernel = package.find(name);
    return kernel == nullptr ? nullptr : find_kernel(package, name, n_args, programs_for(kernel, elements));
}

flagos_rt_arg ptr(const ggml_tensor * t) {
    flagos_rt_arg arg {};
    arg.kind = FLAGOS_RT_ARG_PTR;
    arg.value.ptr = t->data;
    return arg;
}

flagos_rt_arg i32(int64_t v) {
    flagos_rt_arg arg {};
    arg.kind = FLAGOS_RT_ARG_I32;
    arg.value.i32 = static_cast<int32_t>(v);
    return arg;
}

flagos_rt_arg f32(float v) {
    flagos_rt_arg arg {};
    arg.kind = FLAGOS_RT_ARG_F32;
    arg.value.f32 = v;
    return arg;
}

void set_call(flagos_kernel_call * call, const flagos_kernel_info * kernel, int64_t programs,
              std::initializer_list<flagos_rt_arg> args) {
    if (call == nullptr) {
        return;
    }
    call->kernel = kernel;
    call->grid[0] = static_cast<uint32_t>(programs);
    call->grid[1] = 1;
    call->grid[2] = 1;
    call->n_args = 0;
    for (const flagos_rt_arg & arg : args) {
        call->args[call->n_args++] = arg;
    }
}

// ADD and MUL: src0 has the output shape; src1 is either the same shape or a
// single row repeated over all rows. The kernels broadcast with a flat
// index % src1_elements, which equals GGML's repeat only in those two cases.
bool plan_binary(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!is_contiguous_f32(op) || !is_contiguous_f32(a) || !is_contiguous_f32(b) ||
        !ggml_are_same_shape(op, a) || !fits_i32(ggml_nelements(op)) || !fits_i32(ggml_nelements(b))) {
        return false;
    }
    const bool same_shape = ggml_are_same_shape(a, b);
    const bool row_repeat = b->ne[0] == a->ne[0] && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1;
    if (!same_shape && !row_repeat) {
        return false;
    }
    const int64_t n = ggml_nelements(op);
    if (op->op == GGML_OP_ADD) {
        const char * name = same_shape ? "flagos_add_f32" : "flagos_add_repeat_f32";
        const uint32_t n_args = same_shape ? 4 : 5;
        const flagos_kernel_info * kernel = find_elementwise(package, name, n_args, n);
        if (kernel == nullptr) {
            return false;
        }
        if (same_shape) {
            set_call(call, kernel, programs_for(kernel, n), { ptr(a), ptr(b), ptr(op), i32(n) });
        } else {
            set_call(call, kernel, programs_for(kernel, n), { ptr(a), ptr(b), ptr(op), i32(n), i32(ggml_nelements(b)) });
        }
        return true;
    }
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_mul_f32", 5, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n), { ptr(a), ptr(b), ptr(op), i32(n), i32(ggml_nelements(b)) });
    return true;
}

bool plan_scale(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * src = op->src[0];
    if (!is_contiguous_f32(op) || !is_contiguous_f32(src) || !ggml_are_same_shape(op, src) ||
        !fits_i32(ggml_nelements(op))) {
        return false;
    }
    float scale = 1.0f;
    float bias = 0.0f;
    std::memcpy(&scale, reinterpret_cast<const float *>(op->op_params), sizeof(float));
    std::memcpy(&bias, reinterpret_cast<const float *>(op->op_params) + 1, sizeof(float));
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_scale_f32", 5, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n), { ptr(src), ptr(op), f32(scale), f32(bias), i32(n) });
    return true;
}

// One program per row; the row must fit in one BLOCK.
bool plan_rms_norm(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * src = op->src[0];
    if (!is_contiguous_f32(op) || !is_contiguous_f32(src) || !ggml_are_same_shape(op, src) ||
        !fits_i32(ggml_nelements(op)) || !fits_i32(op->ne[0]) || op->ne[0] % 4 != 0) {
        return false;
    }
    float eps = 0.0f;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    if (!std::isfinite(eps) || eps < 0.0f) {
        return false;
    }
    const int64_t rows = ggml_nelements(op) / op->ne[0];
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_rms_norm_f32", 4, rows);
    if (kernel == nullptr || kernel->block_size < op->ne[0]) {
        return false;
    }
    set_call(call, kernel, rows, { ptr(op), ptr(src), i32(op->ne[0]), f32(eps) });
    return true;
}

bool plan_unary(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const char * name = nullptr;
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_SILU:    name = "flagos_silu_f32"; break;
        case GGML_UNARY_OP_SIGMOID: name = "flagos_sigmoid_f32"; break;
        case GGML_UNARY_OP_TANH:    name = "flagos_tanh_f32"; break;
        default: return false;
    }
    const ggml_tensor * src = op->src[0];
    if (!is_contiguous_f32(op) || !is_contiguous_f32(src) || !ggml_are_same_shape(op, src) ||
        !fits_i32(ggml_nelements(op))) {
        return false;
    }
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, name, 3, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n), { ptr(src), ptr(op), i32(n) });
    return true;
}

} // namespace

bool flagos_op_plan(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    if (op == nullptr || !package.is_open()) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return plan_binary(package, op, call);
        case GGML_OP_SCALE:
            return plan_scale(package, op, call);
        case GGML_OP_RMS_NORM:
            return plan_rms_norm(package, op, call);
        case GGML_OP_UNARY:
            return plan_unary(package, op, call);
        default:
            return false;
    }
}
