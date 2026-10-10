#include "flagos-ops.h"

#include "flagos-graph-plan.h"

#include <cmath>
#include <cstring>
#include <string>

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
    GGML_ASSERT(args.size() <= FLAGOS_MAX_KERNEL_ARGS);
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


// Quantized MUL_MAT with GGUF weights (Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K)
// and F32 activations. Arguments are (weights as bytes, weights as f16 for
// the block scales, x, y, k, rows[, columns]); the kernels address with
// 32-bit offsets, so every pairwise product must fit in int32.
const char * quant_name(flagos_quantized_matmul_kind kind) {
    switch (kind) {
        case flagos_quantized_matmul_kind::q4_0: return "q4_0";
        case flagos_quantized_matmul_kind::q4_1: return "q4_1";
        case flagos_quantized_matmul_kind::q8_0: return "q8_0";
        case flagos_quantized_matmul_kind::q4_k: return "q4_k";
        case flagos_quantized_matmul_kind::q5_k: return "q5_k";
        case flagos_quantized_matmul_kind::q6_k: return "q6_k";
        default: return nullptr;
    }
}

bool plan_mul_mat(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    int32_t precision = 0;
    std::memcpy(&precision, op->op_params, sizeof(precision));
    if (precision != GGML_PREC_DEFAULT && precision != GGML_PREC_F32) {
        return false;
    }
    flagos_quantized_matmul_signature sig;
    if (!flagos_describe_quantized_matmul(op, &sig) || quant_name(sig.weight_kind) == nullptr) {
        return false;
    }
    const int64_t limit = INT32_MAX;
    if (sig.k > limit || sig.rows > limit || sig.columns > limit || sig.k > limit / sig.rows ||
        sig.k > limit / sig.columns || sig.rows > limit / sig.columns) {
        return false;
    }
    const std::string base = std::string("flagos_mul_mat_") + quant_name(sig.weight_kind) + "_f32";
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (sig.columns == 1) {
        // Decode GEMV: one program per output row.
        const flagos_kernel_info * kernel = find_kernel(package, base.c_str(), 6, sig.rows);
        if (kernel == nullptr) {
            return false;
        }
        set_call(call, kernel, sig.rows, { ptr(w), ptr(w), ptr(x), ptr(op), i32(sig.k), i32(sig.rows) });
        return true;
    }
    // Batched (prefill): grid (rows, ceil(columns / BLOCK)).
    const std::string batched = base + "_batched";
    const flagos_kernel_info * kernel = package.find(batched);
    if (kernel == nullptr || kernel->block_size <= 0 ||
        (kernel->argument_count >= 0 && kernel->argument_count != 7)) {
        return false;
    }
    const int64_t col_programs = (sig.columns + kernel->block_size - 1) / kernel->block_size;
    if (col_programs > 65535) {
        return false;
    }
    set_call(call, kernel, sig.rows, { ptr(w), ptr(w), ptr(x), ptr(op), i32(sig.k), i32(sig.rows), i32(sig.columns) });
    if (call != nullptr) {
        call->grid[1] = static_cast<uint32_t>(col_programs);
    }
    return true;
}

// Pointer ranges [data, data + nbytes) overlap but are not identical.
bool partially_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->data == nullptr || b->data == nullptr || a->data == b->data) {
        return false;
    }
    const auto * pa = static_cast<const char *>(a->data);
    const auto * pb = static_cast<const char *>(b->data);
    return pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
}

// Largest element offset reachable through a tensor's strides.
bool element_offsets_fit_i32(const ggml_tensor * t, size_t element_size) {
    int64_t max_offset = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->ne[d] <= 0) {
            return false;
        }
        if (t->nb[d] % element_size != 0) {
            return false;
        }
        max_offset += (t->ne[d] - 1) * static_cast<int64_t>(t->nb[d] / element_size);
    }
    return max_offset < INT32_MAX;
}

// CPY / CONT / DUP of F32 between arbitrarily strided views. The kernel
// walks the logical element index, decomposing it with the source shape for
// the read and with the destination shape for the write, as GGML defines it.
bool plan_copy(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * src = op->src[0];
    if (src == nullptr || op->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 ||
        op->nb[0] != sizeof(float) || src->nb[0] != sizeof(float) ||
        ggml_nelements(op) != ggml_nelements(src) || !fits_i32(ggml_nelements(op)) ||
        !element_offsets_fit_i32(op, sizeof(float)) || !element_offsets_fit_i32(src, sizeof(float))) {
        return false;
    }
    // CPY writes into src[1]; GGML's output is a view of it.
    if (op->op == GGML_OP_CPY &&
        (op->src[1] == nullptr || op->src[1]->type != op->type ||
         ((op->data != nullptr || op->src[1]->data != nullptr) && op->data != op->src[1]->data))) {
        return false;
    }
    if (partially_overlaps(op, src) || (op->data != nullptr && op->data == src->data)) {
        return false;
    }
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_copy_strided_f32", 19, n);
    if (kernel == nullptr) {
        return false;
    }
    static_assert(2 + 4 * GGML_MAX_DIMS + 1 <= FLAGOS_MAX_KERNEL_ARGS, "copy arguments exceed the call capacity");
    if (call != nullptr) {
        set_call(call, kernel, programs_for(kernel, n), { ptr(src), ptr(op) });
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(src->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(op->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(src->nb[d] / sizeof(float));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(op->nb[d] / sizeof(float));
        call->args[call->n_args++] = i32(n);
    }
    return true;
}

// SET_ROWS: write F32 rows into an F16 destination (the KV cache) at I64
// row indices. src[2] is the destination GGML updates in place.
bool plan_set_rows(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * values = op->src[0];
    const ggml_tensor * index = op->src[1];
    const ggml_tensor * dst = op->src[2];
    if (values == nullptr || index == nullptr || dst == nullptr || op->type != GGML_TYPE_F16 ||
        values->type != GGML_TYPE_F32 || index->type != GGML_TYPE_I64 || dst->type != GGML_TYPE_F16 ||
        op->ne[0] <= 0 || op->ne[1] <= 0 || op->ne[2] != 1 || op->ne[3] != 1 ||
        values->ne[0] != op->ne[0] || values->ne[1] <= 0 || values->ne[2] != 1 || values->ne[3] != 1 ||
        index->ne[0] != values->ne[1] || index->ne[1] != 1 || index->ne[2] != 1 || index->ne[3] != 1 ||
        !ggml_are_same_shape(op, dst) || !ggml_is_contiguous(op) || !ggml_is_contiguous(values) ||
        !ggml_is_contiguous(index) || !ggml_is_contiguous(dst) ||
        !fits_i32(ggml_nelements(op)) || !fits_i32(ggml_nelements(values))) {
        return false;
    }
    if (op->data != nullptr && dst->data != nullptr && op->data != dst->data) {
        return false;
    }
    if (partially_overlaps(op, values) || partially_overlaps(op, index)) {
        return false;
    }
    const int64_t n = ggml_nelements(values);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_set_rows_f32_f16", 6, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n),
             { ptr(values), ptr(index), ptr(op), i32(values->ne[0]), i32(values->ne[1]), i32(op->ne[1]) });
    return true;
}

// GET_ROWS from a contiguous F32 table with I32 indices; grid (rows, ceil(columns / BLOCK)).
bool plan_get_rows(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * table = op->src[0];
    const ggml_tensor * index = op->src[1];
    const int64_t max_elements = INT32_MAX / sizeof(float);
    if (table == nullptr || index == nullptr || op->type != GGML_TYPE_F32 || table->type != GGML_TYPE_F32 ||
        index->type != GGML_TYPE_I32 || table->ne[0] <= 0 || table->ne[0] > max_elements || table->ne[1] <= 0 ||
        table->ne[1] > max_elements / table->ne[0] || table->ne[2] != 1 || table->ne[3] != 1 ||
        index->ne[0] <= 0 || index->ne[0] > max_elements / table->ne[0] ||
        index->ne[1] != 1 || index->ne[2] != 1 || index->ne[3] != 1 ||
        op->ne[0] != table->ne[0] || op->ne[1] != index->ne[0] || op->ne[2] != 1 || op->ne[3] != 1 ||
        !ggml_is_contiguous(table) || !ggml_is_contiguous(index) || !ggml_is_contiguous(op) ||
        partially_overlaps(op, table) || partially_overlaps(op, index)) {
        return false;
    }
    const flagos_kernel_info * kernel = package.find("flagos_get_rows_f32");
    if (kernel == nullptr || kernel->block_size <= 0 || (kernel->argument_count >= 0 && kernel->argument_count != 4)) {
        return false;
    }
    const int64_t col_programs = (op->ne[0] + kernel->block_size - 1) / kernel->block_size;
    if (col_programs > 65535 || op->ne[1] > INT32_MAX) {
        return false;
    }
    set_call(call, kernel, op->ne[1], { ptr(table), ptr(index), ptr(op), i32(op->ne[0]) });
    if (call != nullptr) {
        call->grid[1] = static_cast<uint32_t>(col_programs);
    }
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
        case GGML_OP_MUL_MAT:
            return plan_mul_mat(package, op, call);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return plan_copy(package, op, call);
        case GGML_OP_SET_ROWS:
            return plan_set_rows(package, op, call);
        case GGML_OP_GET_ROWS:
            return plan_get_rows(package, op, call);
        default:
            return false;
    }
}
