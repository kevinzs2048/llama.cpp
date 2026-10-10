#include "flagos-ops.h"

#include "flagos-graph-plan.h"

#include "../ggml-impl.h"

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

// ADD and MUL: src0 has the output shape. The kernels broadcast src1 with a
// flat index % src1_elements, which equals GGML's repeat exactly when src1
// matches src0 in its leading dimensions and has extent 1 in the rest
// (same shape, one row repeated, or a single scalar).
bool flat_repeat_matches(const ggml_tensor * a, const ggml_tensor * b) {
    int d = 0;
    while (d < GGML_MAX_DIMS && b->ne[d] == a->ne[d]) {
        ++d;
    }
    for (; d < GGML_MAX_DIMS; ++d) {
        if (b->ne[d] != 1) {
            return false;
        }
    }
    return true;
}

bool plan_binary(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (!is_contiguous_f32(op) || !is_contiguous_f32(a) || !is_contiguous_f32(b) ||
        !ggml_are_same_shape(op, a) || !fits_i32(ggml_nelements(op)) || !fits_i32(ggml_nelements(b))) {
        return false;
    }
    const bool same_shape = ggml_are_same_shape(a, b);
    if (!same_shape && !flat_repeat_matches(a, b)) {
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
        case GGML_UNARY_OP_SOFTPLUS: name = "flagos_softplus_f32"; break;
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

// required_output_node_indices lists graph node indices; compare them with
// the candidate nodes at the given positions.
bool outputs_are(const flagos_pattern_candidate & c, std::initializer_list<size_t> positions) {
    if (c.required_output_node_indices.size() != positions.size()) {
        return false;
    }
    size_t i = 0;
    for (size_t position : positions) {
        if (position >= c.node_indices.size() || c.required_output_node_indices[i++] != c.node_indices[position]) {
            return false;
        }
    }
    return true;
}

// Shared checks for RMS_NORM -> MUL(weight row). Returns the weight.
const ggml_tensor * norm_mul_weight(const ggml_tensor * norm, const ggml_tensor * mul, float & eps) {
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return nullptr;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * weight = mul->src[0] == norm ? mul->src[1] : (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (x == nullptr || weight == nullptr || !is_contiguous_f32(norm) || !is_contiguous_f32(x) ||
        !is_contiguous_f32(mul) || !is_contiguous_f32(weight) || !ggml_are_same_shape(norm, x) ||
        !ggml_are_same_shape(mul, norm) || weight->ne[0] != norm->ne[0] || weight->ne[1] != 1 ||
        weight->ne[2] != 1 || weight->ne[3] != 1 || !fits_i32(norm->ne[0]) || norm->ne[0] % 4 != 0 ||
        !fits_i32(ggml_nelements(norm))) {
        return nullptr;
    }
    std::memcpy(&eps, norm->op_params, sizeof(eps));
    return std::isfinite(eps) && eps >= 0.0f ? weight : nullptr;
}

bool same_or_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    return !partially_overlaps(a, b);
}

bool distinct(const ggml_tensor * a, const ggml_tensor * b) {
    return !partially_overlaps(a, b) && (a->data == nullptr || a->data != b->data);
}

// RMS_NORM followed by MUL with a broadcast row of weights. One program per
// row; the row must fit in one BLOCK.
// - Separate outputs: flagos_rms_norm_mul_f32(norm_out, mul_out, x, weight, n, eps),
//   for "product only" or "norm and product" required outputs.
// - GGML reused the dead norm buffer for the product (mul aliases norm):
//   flagos_rms_norm_mul_inplace_f32(out, x, weight, n, eps), product only.
bool plan_rms_norm_mul(const flagos_kernel_package & package, const ggml_cgraph * graph,
                       const flagos_pattern_candidate & c, flagos_kernel_call * call) {
    if (c.node_indices.size() != 2) {
        return false;
    }
    const ggml_tensor * norm = graph->nodes[c.node_indices[0]];
    const ggml_tensor * mul = graph->nodes[c.node_indices[1]];
    float eps = 0.0f;
    const ggml_tensor * weight = norm_mul_weight(norm, mul, eps);
    if (weight == nullptr) {
        return false;
    }
    const ggml_tensor * x = norm->src[0];
    const int64_t rows = ggml_nelements(norm) / norm->ne[0];
    const bool aliased = norm->data != nullptr && norm->data == mul->data;
    if (aliased) {
        if (!outputs_are(c, { 1 }) || !distinct(mul, x) || !distinct(mul, weight)) {
            return false;
        }
        const flagos_kernel_info * kernel = find_kernel(package, "flagos_rms_norm_mul_inplace_f32", 5, rows);
        if (kernel == nullptr || kernel->block_size < norm->ne[0]) {
            return false;
        }
        set_call(call, kernel, rows, { ptr(mul), ptr(x), ptr(weight), i32(norm->ne[0]), f32(eps) });
        return true;
    }
    if (!(outputs_are(c, { 1 }) || outputs_are(c, { 0, 1 })) || !distinct(norm, mul) || !distinct(norm, x) ||
        !distinct(norm, weight) || !distinct(mul, x) || !distinct(mul, weight)) {
        return false;
    }
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_rms_norm_mul_f32", 6, rows);
    if (kernel == nullptr || kernel->block_size < norm->ne[0]) {
        return false;
    }
    set_call(call, kernel, rows, { ptr(norm), ptr(mul), ptr(x), ptr(weight), i32(norm->ne[0]), f32(eps) });
    return true;
}

// ADD (residual) -> RMS_NORM -> MUL(weight row), keeping the residual sum:
// flagos_add_rms_norm_mul_residual_f32(residual_out, mul_out, x, bias, weight, n, eps).
// Each program reads its row before writing, so the residual may reuse an
// input buffer exactly; any partial overlap is refused.
bool plan_add_rms_norm_mul(const flagos_kernel_package & package, const ggml_cgraph * graph,
                           const flagos_pattern_candidate & c, flagos_kernel_call * call) {
    if (c.node_indices.size() != 3 || !outputs_are(c, { 0, 2 })) {
        return false;
    }
    const ggml_tensor * add = graph->nodes[c.node_indices[0]];
    const ggml_tensor * norm = graph->nodes[c.node_indices[1]];
    const ggml_tensor * mul = graph->nodes[c.node_indices[2]];
    float eps = 0.0f;
    const ggml_tensor * weight = norm_mul_weight(norm, mul, eps);
    if (weight == nullptr || add->op != GGML_OP_ADD || norm->src[0] != add) {
        return false;
    }
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    if (a == nullptr || b == nullptr || !is_contiguous_f32(add) || !is_contiguous_f32(a) || !is_contiguous_f32(b) ||
        !ggml_are_same_shape(add, a) || !ggml_are_same_shape(add, b) ||
        !distinct(add, mul) || !distinct(add, weight) || !distinct(mul, weight) ||
        !same_or_disjoint(add, a) || !same_or_disjoint(add, b) ||
        !same_or_disjoint(mul, a) || !same_or_disjoint(mul, b)) {
        return false;
    }
    const int64_t rows = ggml_nelements(norm) / norm->ne[0];
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_add_rms_norm_mul_residual_f32", 7, rows);
    if (kernel == nullptr || kernel->block_size < norm->ne[0]) {
        return false;
    }
    set_call(call, kernel, rows, { ptr(add), ptr(mul), ptr(a), ptr(b), ptr(weight), i32(norm->ne[0]), f32(eps) });
    return true;
}

// GLU SWIGLU, split form (gate and up are separate tensors):
// flagos_swiglu_split_f32(gate, up, out, n) computes g * sigmoid(g) * up.
bool plan_glu(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * gate = op->src[0];
    const ggml_tensor * up = op->src[1];
    if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || gate == nullptr || up == nullptr ||
        !is_contiguous_f32(op) || !is_contiguous_f32(gate) || !is_contiguous_f32(up) ||
        !ggml_are_same_shape(op, gate) || !ggml_are_same_shape(op, up) || !fits_i32(ggml_nelements(op)) ||
        partially_overlaps(op, gate) || partially_overlaps(op, up)) {
        return false;
    }
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_swiglu_split_f32", 4, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n), { ptr(gate), ptr(up), ptr(op), i32(n) });
    return true;
}

// CONCAT of two F32 tensors along op_params[0]; inputs may be strided, the
// output is dense. 21 arguments: pointers, dim, output and first-input
// extents, input strides, element count.
bool plan_concat(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    int32_t dim = 0;
    std::memcpy(&dim, op->op_params, sizeof(dim));
    if (a == nullptr || b == nullptr || !is_contiguous_f32(op) || a->type != GGML_TYPE_F32 ||
        b->type != GGML_TYPE_F32 || dim < 0 || dim > 3 || !fits_i32(ggml_nelements(op)) ||
        !element_offsets_fit_i32(a, sizeof(float)) || !element_offsets_fit_i32(b, sizeof(float)) ||
        partially_overlaps(op, a) || partially_overlaps(op, b) ||
        (op->data != nullptr && (op->data == a->data || op->data == b->data))) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (op->ne[d] != (d == dim ? a->ne[d] + b->ne[d] : a->ne[d]) || (d != dim && a->ne[d] != b->ne[d])) {
            return false;
        }
    }
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_concat_f32", 21, n);
    if (kernel == nullptr) {
        return false;
    }
    static_assert(3 + 1 + 4 * GGML_MAX_DIMS + 1 <= FLAGOS_MAX_KERNEL_ARGS, "concat arguments exceed the call capacity");
    if (call != nullptr) {
        set_call(call, kernel, programs_for(kernel, n), { ptr(a), ptr(b), ptr(op), i32(dim) });
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(op->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(a->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(a->nb[d] / sizeof(float));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(b->nb[d] / sizeof(float));
        call->args[call->n_args++] = i32(n);
    }
    return true;
}

// L2_NORM over rows of a possibly strided F32 input into a dense output:
// flagos_l2_norm_strided_f32(x, out, n_cols, s1, s2, s3, eps, ne1, ne2), grid (ne1, ne2, ne3).
bool plan_l2_norm(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * src = op->src[0];
    float eps = 0.0f;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    if (src == nullptr || !is_contiguous_f32(op) || src->type != GGML_TYPE_F32 || !ggml_are_same_shape(op, src) ||
        src->nb[0] != sizeof(float) || !element_offsets_fit_i32(src, sizeof(float)) ||
        !fits_i32(ggml_nelements(op)) || op->ne[1] > INT32_MAX || op->ne[2] > 65535 || op->ne[3] > 65535 ||
        !std::isfinite(eps) || eps < 0.0f || partially_overlaps(op, src) ||
        (op->data != nullptr && op->data == src->data)) {
        return false;
    }
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_l2_norm_strided_f32", 9, op->ne[1]);
    if (kernel == nullptr || kernel->block_size < op->ne[0]) {
        return false;
    }
    set_call(call, kernel, op->ne[1], { ptr(src), ptr(op), i32(op->ne[0]), i32(src->nb[1] / sizeof(float)),
             i32(src->nb[2] / sizeof(float)), i32(src->nb[3] / sizeof(float)), f32(eps), i32(op->ne[1]), i32(op->ne[2]) });
    if (call != nullptr) {
        call->grid[1] = static_cast<uint32_t>(op->ne[2]);
        call->grid[2] = static_cast<uint32_t>(op->ne[3]);
    }
    return true;
}

// RMS_NORM of a non-contiguous F32 input (for example per-head norms on a
// view) into a dense output: flagos_rms_norm_ggml_strided_f32(out, x, n_cols,
// ne1, ne2, s1, s2, s3, eps), one program per row. eps must be > 0.
bool plan_rms_norm_strided(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * src = op->src[0];
    float eps = 0.0f;
    std::memcpy(&eps, op->op_params, sizeof(eps));
    if (src == nullptr || ggml_is_contiguous(src) || !is_contiguous_f32(op) || src->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(op, src) || src->nb[0] != sizeof(float) || !element_offsets_fit_i32(src, sizeof(float)) ||
        !fits_i32(ggml_nelements(op)) || !std::isfinite(eps) || eps <= 0.0f || partially_overlaps(op, src) ||
        (op->data != nullptr && op->data == src->data)) {
        return false;
    }
    const int64_t rows = ggml_nelements(op) / op->ne[0];
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_rms_norm_ggml_strided_f32", 9, rows);
    if (kernel == nullptr || kernel->block_size < op->ne[0]) {
        return false;
    }
    set_call(call, kernel, rows, { ptr(op), ptr(src), i32(op->ne[0]), i32(op->ne[1]), i32(op->ne[2]),
             i32(src->nb[1] / sizeof(float)), i32(src->nb[2] / sizeof(float)), i32(src->nb[3] / sizeof(float)), f32(eps) });
    return true;
}

// SSM_CONV (Mamba / Gated DeltaNet short convolution):
// flagos_ssm_conv_f32(s, c, out, d_conv, d_inner, n_tokens, n_seqs, ss1, ss2, sc1, so0, so1),
// grid (n_tokens, n_seqs, ceil(d_inner / BLOCK)).
bool plan_ssm_conv(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * s = op->src[0];
    const ggml_tensor * c = op->src[1];
    if (s == nullptr || c == nullptr || op->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32 ||
        c->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || s->nb[0] != sizeof(float) || c->nb[0] != sizeof(float) ||
        !element_offsets_fit_i32(s, sizeof(float)) || !element_offsets_fit_i32(c, sizeof(float)) ||
        c->ne[2] != 1 || c->ne[3] != 1 || s->ne[0] != c->ne[0] - 1 + op->ne[1] || s->ne[1] != c->ne[1] ||
        s->ne[2] != op->ne[2] || s->ne[3] != 1 || op->ne[0] != c->ne[1] || op->ne[3] != 1 ||
        !fits_i32(ggml_nelements(op)) || op->ne[1] > INT32_MAX || op->ne[2] > 65535 ||
        partially_overlaps(op, s) || partially_overlaps(op, c) ||
        (op->data != nullptr && (op->data == s->data || op->data == c->data))) {
        return false;
    }
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_ssm_conv_f32", 12, op->ne[1]);
    if (kernel == nullptr || kernel->block_size <= 0) {
        return false;
    }
    const int64_t channel_programs = (op->ne[0] + kernel->block_size - 1) / kernel->block_size;
    if (channel_programs > 65535) {
        return false;
    }
    set_call(call, kernel, op->ne[1], { ptr(s), ptr(c), ptr(op), i32(c->ne[0]), i32(op->ne[0]), i32(op->ne[1]),
             i32(op->ne[2]), i32(s->nb[1] / sizeof(float)), i32(s->nb[2] / sizeof(float)), i32(c->nb[1] / sizeof(float)),
             i32(op->nb[1] / sizeof(float)), i32(op->nb[2] / sizeof(float)) });
    if (call != nullptr) {
        call->grid[1] = static_cast<uint32_t>(op->ne[2]);
        call->grid[2] = static_cast<uint32_t>(channel_programs);
    }
    return true;
}

// 16-byte alignment, decided structurally before allocation: allocations
// are aligned to the buffer type (>= 32 bytes), so only a view's byte offset
// can break it.
bool aligned16(const ggml_tensor * t) {
    if (t->data != nullptr) {
        return reinterpret_cast<uintptr_t>(t->data) % 16 == 0;
    }
    return t->view_src == nullptr || t->view_offs % 16 == 0;
}

// GATED_DELTA_NET with a scalar gate per head and token (Qwen3.5/3.6
// linear-attention layers). One program owns tile_n value rows of one
// (head, sequence) state and walks the tokens:
// flagos_gated_delta_net_scalar_f32(q, k, v, g, beta, state, out, S, H, T, B,
//   q strides x3, v strides x3, beta strides x3, q_heads, B/Bq, K, 1/sqrt(S)),
// grid (H, B, ceil(S / tile_n)). k shares q's layout and g shares beta's,
// which the contract checks. The output holds the attention scores followed
// by K state snapshots, most recent first.
bool plan_gated_delta_net(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * g = op->src[3];
    const ggml_tensor * beta = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr || state == nullptr ||
        !is_contiguous_f32(op) || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        !is_contiguous_f32(g) || !is_contiguous_f32(beta) || !is_contiguous_f32(state) ||
        q->nb[0] != sizeof(float) || v->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t S = v->ne[0], H = v->ne[1], T = v->ne[2], B = v->ne[3];
    const int64_t Hq = q->ne[1], Bq = q->ne[3];
    int32_t K = 0;
    std::memcpy(&K, op->op_params, sizeof(K));
    if (S <= 0 || H <= 0 || T <= 0 || B <= 0 || Hq <= 0 || Hq > H || H % Hq != 0 || Bq <= 0 || Bq > B ||
        B % Bq != 0 || K <= 0 || S % 16 != 0 || q->ne[0] != S || q->ne[2] != T ||
        k->ne[0] != S || k->ne[1] != Hq || k->ne[2] != T || k->ne[3] != Bq ||
        g->ne[0] != 1 || g->ne[1] != H || g->ne[2] != T || g->ne[3] != B ||
        !ggml_are_same_shape(g, beta) || state->ne[0] != S || state->ne[1] != S || state->ne[2] != H || state->ne[3] != B ||
        op->ne[0] != S * H || op->ne[1] != T * B + int64_t(K) * S * B || op->ne[2] != 1 || op->ne[3] != 1 ||
        !fits_i32(ggml_nelements(op)) || !fits_i32(S * S * H * B) || !fits_i32(T * B) || H > 65535 || B > 65535 ||
        !element_offsets_fit_i32(q, sizeof(float)) || !element_offsets_fit_i32(v, sizeof(float))) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (q->nb[d] != k->nb[d] || g->nb[d] != beta->nb[d]) {
            return false;
        }
    }
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if ((q->nb[d] / sizeof(float)) % 16 != 0 || (v->nb[d] / sizeof(float)) % 16 != 0) {
            return false;
        }
    }
    for (const ggml_tensor * t : { q, k, v, g, beta, state }) {
        if (!aligned16(t) || partially_overlaps(op, t) || (op->data != nullptr && op->data == t->data)) {
            return false;
        }
    }
    if (!aligned16(op)) {
        return false;
    }
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_gated_delta_net_scalar_f32", 24, H);
    if (kernel == nullptr || kernel->block_size < S || kernel->tile_n <= 0) {
        return false;
    }
    if (call != nullptr) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(S));
        set_call(call, kernel, H, { ptr(q), ptr(k), ptr(v), ptr(g), ptr(beta), ptr(state), ptr(op),
                 i32(S), i32(H), i32(T), i32(B),
                 i32(q->nb[1] / sizeof(float)), i32(q->nb[2] / sizeof(float)), i32(q->nb[3] / sizeof(float)),
                 i32(v->nb[1] / sizeof(float)), i32(v->nb[2] / sizeof(float)), i32(v->nb[3] / sizeof(float)),
                 i32(beta->nb[1] / sizeof(float)), i32(beta->nb[2] / sizeof(float)), i32(beta->nb[3] / sizeof(float)),
                 i32(Hq), i32(B / Bq), i32(K), f32(scale) });
        call->grid[1] = static_cast<uint32_t>(B);
        call->grid[2] = static_cast<uint32_t>((S + kernel->tile_n - 1) / kernel->tile_n);
    }
    return true;
}

// ROPE with the CPU's ordered-multiply theta recurrence, for NORMAL, NEOX
// and IMROPE (Qwen3.5/3.6 text). One program per row, BLOCK pairs:
// flagos_rope_ggml_recurrence_f32 / flagos_rope_ggml_neox_recurrence_f32
// (x, positions, factors|x, out, ne0, ne1, ne2, n_dims, mode, has_factors,
//  sx1, sx2, sx3, sections[4], theta_scale, freq_scale, attn_factor).
// YaRN (ext_factor != 0), MROPE and VISION are not covered.
bool plan_rope(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * pos = op->src[1];
    const ggml_tensor * factors = op->src[2];
    int32_t p[15];
    std::memcpy(p, op->op_params, sizeof(p));
    float freq_base, freq_scale, ext_factor, attn_factor;
    std::memcpy(&freq_base, &p[5], sizeof(float));
    std::memcpy(&freq_scale, &p[6], sizeof(float));
    std::memcpy(&ext_factor, &p[7], sizeof(float));
    std::memcpy(&attn_factor, &p[8], sizeof(float));
    const int32_t n_dims = p[1];
    const int32_t mode = p[2];
    const bool neox = mode == GGML_ROPE_TYPE_NEOX;
    const bool imrope = mode == GGML_ROPE_TYPE_IMROPE;
    if (x == nullptr || pos == nullptr || (mode != 0 && !neox && !imrope) || op->type != GGML_TYPE_F32 ||
        x->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32 || !ggml_are_same_shape(op, x) ||
        !ggml_is_contiguous(op) || x->nb[0] != sizeof(float) || !ggml_is_contiguous(pos) ||
        pos->ne[1] != 1 || pos->ne[2] != 1 || pos->ne[3] != 1 || pos->ne[0] != op->ne[2] * (imrope ? 4 : 1) ||
        op->ne[0] <= 0 || op->ne[0] % 2 != 0 || n_dims <= 0 || n_dims % 2 != 0 || n_dims > op->ne[0] ||
        !element_offsets_fit_i32(x, sizeof(float)) || !fits_i32(ggml_nelements(op)) ||
        !std::isfinite(freq_base) || freq_base <= 0.0f || !std::isfinite(freq_scale) || freq_scale <= 0.0f ||
        ext_factor != 0.0f || !std::isfinite(attn_factor) ||
        !std::isnormal(std::pow(freq_base, -2.0f / n_dims))) {
        return false;
    }
    if (imrope) {
        int64_t sum = 0;
        for (int i = 11; i < 15; ++i) {
            if (p[i] < 0) return false;
            sum += p[i];
        }
        if (sum != n_dims / 2 || (p[11] == 0 && p[12] == 0 && p[13] == 0)) {
            return false;
        }
    }
    if (factors != nullptr && (factors->type != GGML_TYPE_F32 || !ggml_is_contiguous(factors) ||
                               factors->ne[0] < (neox ? n_dims : op->ne[0]) / 2 || factors->ne[1] != 1 ||
                               factors->ne[2] != 1 || factors->ne[3] != 1)) {
        return false;
    }
    // Writing the output may not clobber positions or factors; in place over
    // x is fine when x is contiguous (each program reads its row first).
    if (partially_overlaps(op, x) || (op->data != nullptr && op->data == x->data && !ggml_is_contiguous(x)) ||
        partially_overlaps(op, pos) || (op->data != nullptr && op->data == pos->data) ||
        (factors != nullptr && (partially_overlaps(op, factors) || (op->data != nullptr && op->data == factors->data)))) {
        return false;
    }
    const int64_t rows = ggml_nrows(op);
    const char * name = neox ? "flagos_rope_ggml_neox_recurrence_f32" : "flagos_rope_ggml_recurrence_f32";
    const flagos_kernel_info * kernel = find_kernel(package, name, 20, rows);
    if (kernel == nullptr || kernel->block_size * 2 < op->ne[0]) {
        return false;
    }
    if (call != nullptr) {
        const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
        set_call(call, kernel, rows, { ptr(x), ptr(pos), ptr(factors != nullptr ? factors : x), ptr(op),
                 i32(op->ne[0]), i32(op->ne[1]), i32(op->ne[2]), i32(n_dims), i32(mode), i32(factors != nullptr),
                 i32(x->nb[1] / sizeof(float)), i32(x->nb[2] / sizeof(float)), i32(x->nb[3] / sizeof(float)),
                 i32(p[11]), i32(p[12]), i32(p[13]), i32(p[14]), f32(theta_scale), f32(freq_scale), f32(attn_factor) });
    }
    return true;
}

// FLASH_ATTN_EXT with F32 Q, F16 K/V/mask, head size 128 or 256, GQA by
// head-group division, online softmax over 16-key tiles. No sinks, ALiBi
// or logit softcap.
// - decode (one query): flagos_flash_attn_decode_f32_f16_generic_h{D}, 16
//   arguments, grid (Hq); needs the KV-cache layout (head stride D) and a
//   mask of exactly [n_kv, 1, 1, 1].
// - prefill: flagos_flash_attn_prefill_f32_f16_generic_h{D}, 19 arguments,
//   grid (Hq, ceil(n_q / BLOCK)); one mask plane shared by all heads.
bool plan_flash_attn(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    float params[3];
    std::memcpy(params, op->op_params, sizeof(params));
    const float scale = params[0], max_bias = params[1], softcap = params[2];
    if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr || op->src[4] != nullptr ||
        op->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || !std::isfinite(scale) || max_bias != 0.0f ||
        softcap != 0.0f || !ggml_is_contiguous(op) || !ggml_is_contiguous(mask) ||
        q->nb[0] != sizeof(float) || k->nb[0] != 2 || v->nb[0] != 2) {
        return false;
    }
    const int64_t D = q->ne[0], n_q = q->ne[1], Hq = q->ne[2], n_kv = k->ne[1], Hkv = k->ne[2];
    if ((D != 128 && D != 256) || k->ne[0] != D || v->ne[0] != D || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        !ggml_are_same_shape(k, v) || Hkv <= 0 || Hq % Hkv != 0 || n_kv <= 0 || n_q <= 0 ||
        op->ne[0] != D || op->ne[1] != Hq || op->ne[2] != n_q || op->ne[3] != 1 || v->nb[1] != k->nb[1] || v->nb[2] != k->nb[2] ||
        !element_offsets_fit_i32(q, sizeof(float)) || !element_offsets_fit_i32(k, 2) || !element_offsets_fit_i32(v, 2) ||
        !fits_i32(ggml_nelements(op)) || Hq > INT32_MAX) {
        return false;
    }
    for (const ggml_tensor * t : { q, k, v, mask }) {
        if (partially_overlaps(op, t) || (op->data != nullptr && op->data == t->data)) {
            return false;
        }
    }
    const bool decode = n_q == 1;
    if (decode) {
        if (mask->ne[0] != n_kv || mask->ne[1] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
            k->nb[2] != size_t(D) * 2 || k->nb[1] < size_t(D) * 2) {
            return false;
        }
    } else if (mask->ne[0] < n_kv || mask->ne[1] < n_q || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    const std::string name = std::string("flagos_flash_attn_") + (decode ? "decode" : "prefill") +
        "_f32_f16_generic_h" + std::to_string(D);
    const flagos_kernel_info * kernel = package.find(name);
    const int expected_args = decode ? 16 : 19;
    if (kernel == nullptr || (kernel->argument_count >= 0 && kernel->argument_count != expected_args) ||
        kernel->tile_k != D || kernel->tile_m != 16 || kernel->tile_n < 16 || kernel->tile_n > 64 ||
        (kernel->tile_n & (kernel->tile_n - 1)) != 0 || kernel->block_size != (decode ? kernel->tile_n : kernel->tile_m)) {
        return false;
    }
    const int64_t q_programs = decode ? 1 : (n_q + kernel->block_size - 1) / kernel->block_size;
    if (q_programs > 65535) {
        return false;
    }
    if (call != nullptr) {
        const int64_t gqa = Hq / Hkv;
        const int32_t sqt = q->nb[1] / 4, sqh = q->nb[2] / 4, skt = k->nb[1] / 2, skh = k->nb[2] / 2;
        const int32_t svt = v->nb[1] / 2, svh = v->nb[2] / 2, sot = op->nb[2] / 4, soh = op->nb[1] / 4;
        if (decode) {
            set_call(call, kernel, Hq, { ptr(q), ptr(k), ptr(v), ptr(mask), ptr(op), i32(n_kv), i32(gqa),
                     i32(sqt), i32(sqh), i32(skt), i32(skh), i32(svt), i32(svh), i32(sot), i32(soh), f32(scale) });
        } else {
            set_call(call, kernel, Hq, { ptr(q), ptr(k), ptr(v), ptr(mask), ptr(op), i32(n_q), i32(n_kv), i32(gqa),
                     i32(sqt), i32(sqh), i32(skt), i32(skh), i32(svt), i32(svh), i32(mask->nb[0] / 2),
                     i32(mask->nb[1] / 2), i32(sot), i32(soh), f32(scale) });
            call->grid[1] = static_cast<uint32_t>(q_programs);
        }
    }
    return true;
}

// SOFT_MAX over rows of a contiguous F32 tensor, without a mask or with an
// F16 mask of exactly the op's shape (the MoE router uses the unmasked
// form). The row (a power of two) must fit in one BLOCK; no sinks or ALiBi:
// flagos_soft_max_unmasked_f32 / flagos_soft_max_masked_f32_f16
// (x, mask|null, out, n_cols, scale), one program per row.
bool plan_soft_max(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * mask = op->src[1];
    float params[2];
    std::memcpy(params, op->op_params, sizeof(params));
    const float scale = params[0], max_bias = params[1];
    const int64_t n = op->ne[0];
    if (x == nullptr || op->src[2] != nullptr || !is_contiguous_f32(op) || !is_contiguous_f32(x) ||
        !ggml_are_same_shape(op, x) || !fits_i32(ggml_nelements(op)) || n <= 0 || (n & (n - 1)) != 0 ||
        !std::isfinite(scale) || scale == 0.0f || max_bias != 0.0f || partially_overlaps(op, x)) {
        return false;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || !ggml_are_same_shape(mask, op) ||
                            partially_overlaps(op, mask) || (op->data != nullptr && op->data == mask->data))) {
        return false;
    }
    const int64_t rows = ggml_nrows(op);
    const char * name = mask == nullptr ? "flagos_soft_max_unmasked_f32" : "flagos_soft_max_masked_f32_f16";
    const flagos_kernel_info * kernel = find_kernel(package, name, 5, rows);
    if (kernel == nullptr || kernel->block_size < n) {
        return false;
    }
    if (call != nullptr) {
        flagos_rt_arg mask_arg {};
        mask_arg.kind = FLAGOS_RT_ARG_PTR;
        mask_arg.value.ptr = mask != nullptr ? mask->data : nullptr;
        set_call(call, kernel, rows, { ptr(x), mask_arg, ptr(op), i32(n), f32(scale) });
    }
    return true;
}

// GET_ROWS from a GGUF-quantized table (token embeddings): one program per
// (row, quant block). flagos_get_rows_<type>_f32(w_u8, w_f16, idx, out, n_cols),
// grid (rows, n_cols / QK).
bool plan_get_rows_quant(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * table = op->src[0];
    const ggml_tensor * index = op->src[1];
    if (table == nullptr || index == nullptr) {
        return false;
    }
    const char * type = nullptr;
    int64_t qk = 0;
    switch (table->type) {
        case GGML_TYPE_Q4_0: type = "q4_0"; qk = 32; break;
        case GGML_TYPE_Q4_1: type = "q4_1"; qk = 32; break;
        case GGML_TYPE_Q8_0: type = "q8_0"; qk = 32; break;
        case GGML_TYPE_Q4_K: type = "q4_k"; qk = 256; break;
        case GGML_TYPE_Q5_K: type = "q5_k"; qk = 256; break;
        case GGML_TYPE_Q6_K: type = "q6_k"; qk = 256; break;
        default: return false;
    }
    if (op->type != GGML_TYPE_F32 || index->type != GGML_TYPE_I32 || op->ne[0] <= 0 || op->ne[0] % qk != 0 ||
        op->ne[1] <= 0 || table->ne[0] != op->ne[0] || op->ne[2] != 1 || op->ne[3] != 1 || table->ne[1] <= 0 ||
        table->ne[2] != 1 || table->ne[3] != 1 || ggml_nelements(index) != op->ne[1] || !ggml_is_contiguous(op) ||
        !ggml_is_contiguous(table) || !ggml_is_contiguous(index) || !fits_i32(ggml_nelements(op)) ||
        ggml_nbytes(table) > size_t(INT32_MAX) || op->ne[0] / qk > 65535 ||
        partially_overlaps(op, table) || partially_overlaps(op, index)) {
        return false;
    }
    const std::string name = std::string("flagos_get_rows_") + type + "_f32";
    const flagos_kernel_info * kernel = find_kernel(package, name.c_str(), 5, op->ne[1]);
    if (kernel == nullptr || kernel->block_size != qk) {
        return false;
    }
    set_call(call, kernel, op->ne[1], { ptr(table), ptr(table), ptr(index), ptr(op), i32(op->ne[0]) });
    if (call != nullptr) {
        call->grid[1] = static_cast<uint32_t>(op->ne[0] / qk);
    }
    return true;
}

// ----- MoE routing and expert ops (Qwen3.6-35B-A3B) -----

// ARGSORT of F32 rows into I32 indices (router top-k). Kernel capacity is
// the smallest of 64/256/1024/4096 that holds a row; ties keep index order.
// flagos_argsort_f32_i32_b<C>(x, out, columns, row_stride, descending).
bool plan_argsort(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * x = op->src[0];
    int32_t params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    std::memcpy(params, op->op_params, sizeof(params));
    const int32_t order = params[0];
    for (size_t i = 1; i < sizeof(params) / sizeof(params[0]); ++i) {
        if (params[i] != 0) {
            return false;
        }
    }
    if (x == nullptr || op->type != GGML_TYPE_I32 || !ggml_is_contiguous(op) || x->type != GGML_TYPE_F32 ||
        x->nb[0] != sizeof(float) || !ggml_are_same_shape(op, x) || (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC) ||
        !element_offsets_fit_i32(x, sizeof(float)) || !fits_i32(ggml_nelements(op)) || op->ne[0] > 4096 ||
        x->nb[1] < size_t(x->ne[0]) * sizeof(float) || (x->ne[2] > 1 && x->nb[2] != x->nb[1] * x->ne[1]) ||
        (x->ne[3] > 1 && x->nb[3] != x->nb[2] * x->ne[2]) || partially_overlaps(op, x) ||
        (op->data != nullptr && op->data == x->data)) {
        return false;
    }
    int capacity = 64;
    while (capacity < op->ne[0]) {
        capacity *= 4;   // 64, 256, 1024, 4096
    }
    const std::string name = "flagos_argsort_f32_i32_b" + std::to_string(capacity);
    const int64_t rows = ggml_nrows(op);
    const flagos_kernel_info * kernel = find_kernel(package, name.c_str(), 5, rows);
    if (kernel == nullptr || kernel->block_size != capacity) {
        return false;
    }
    set_call(call, kernel, rows, { ptr(x), ptr(op), i32(op->ne[0]), i32(x->nb[1] / sizeof(float)), i32(order) });
    return true;
}

// SUM_ROWS of contiguous F32 rows (one row per program, f64 accumulation).
bool plan_sum_rows(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * x = op->src[0];
    if (x == nullptr || !is_contiguous_f32(op) || !is_contiguous_f32(x) || op->ne[0] != 1 || x->ne[0] <= 0 ||
        op->ne[1] != x->ne[1] || op->ne[2] != x->ne[2] || op->ne[3] != x->ne[3] || !fits_i32(ggml_nelements(x)) ||
        partially_overlaps(op, x) || (op->data != nullptr && op->data == x->data)) {
        return false;
    }
    const int64_t rows = ggml_nrows(x);
    const flagos_kernel_info * kernel = find_kernel(package, "flagos_sum_rows_f32_acc_f64", 3, rows);
    if (kernel == nullptr || kernel->block_size < x->ne[0]) {
        return false;
    }
    set_call(call, kernel, rows, { ptr(x), ptr(op), i32(x->ne[0]) });
    return true;
}

// CLAMP of contiguous F32, in place or into a disjoint output.
bool plan_clamp(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * x = op->src[0];
    float bounds[2];
    std::memcpy(bounds, op->op_params, sizeof(bounds));
    if (x == nullptr || !is_contiguous_f32(op) || !is_contiguous_f32(x) || !ggml_are_same_shape(op, x) ||
        !fits_i32(ggml_nelements(op)) || std::isnan(bounds[0]) || std::isnan(bounds[1]) || bounds[0] > bounds[1] ||
        partially_overlaps(op, x)) {
        return false;
    }
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_clamp_compare_f32", 5, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n), { ptr(x), ptr(op), i32(n), f32(bounds[0]), f32(bounds[1]) });
    return true;
}

// ADD / MUL / DIV with GGML's per-dimension repeat of src1 and arbitrary
// input strides, into a dense output:
// flagos_{add,mul,div}_repeat_strided_f32(x, y, out, n[4], y_ne[4], sx[4], sy[4], n_elements).
// Used when the flat-broadcast kernels do not apply (MoE expert weighting,
// router normalization, strided expert views). The output may reuse a
// contiguous src0 exactly (each element is read before it is written).
bool plan_binary_strided(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (a == nullptr || b == nullptr || !is_contiguous_f32(op) || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
        a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || !ggml_are_same_shape(op, a) ||
        !fits_i32(ggml_nelements(op)) || !element_offsets_fit_i32(a, sizeof(float)) ||
        !element_offsets_fit_i32(b, sizeof(float))) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (b->ne[d] <= 0 || op->ne[d] % b->ne[d] != 0) {
            return false;
        }
    }
    // In place over an input is safe when the kernel reads that input at the
    // output's own index: a contiguous src0, or a src1 of the output's shape.
    if (partially_overlaps(op, a) || partially_overlaps(op, b) ||
        (op->data != nullptr && op->data == b->data && !(ggml_are_same_shape(op, b) && ggml_is_contiguous(b))) ||
        (op->data != nullptr && op->data == a->data && !ggml_is_contiguous(a))) {
        return false;
    }
    const char * name = op->op == GGML_OP_ADD ? "flagos_add_repeat_strided_f32"
                      : op->op == GGML_OP_MUL ? "flagos_mul_repeat_strided_f32" : "flagos_div_repeat_strided_f32";
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, name, 20, n);
    if (kernel == nullptr) {
        return false;
    }
    static_assert(3 + 4 * GGML_MAX_DIMS + 1 <= FLAGOS_MAX_KERNEL_ARGS, "strided binary arguments exceed the call capacity");
    if (call != nullptr) {
        set_call(call, kernel, programs_for(kernel, n), { ptr(a), ptr(b), ptr(op) });
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(op->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(b->ne[d]);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(a->nb[d] / sizeof(float));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) call->args[call->n_args++] = i32(b->nb[d] / sizeof(float));
        call->args[call->n_args++] = i32(n);
    }
    return true;
}

// GET_ROWS from a strided F32 table with 2-D I32 indices (router weights for
// several tokens): flagos_get_rows_batched_f32 (16 arguments); an index out
// of range yields 0.
bool plan_get_rows_batched(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * table = op->src[0];
    const ggml_tensor * index = op->src[1];
    if (table == nullptr || index == nullptr || op->type != GGML_TYPE_F32 || table->type != GGML_TYPE_F32 ||
        index->type != GGML_TYPE_I32 || !ggml_is_contiguous(op) || table->nb[0] != sizeof(float) ||
        table->ne[2] != index->ne[1] || table->ne[3] != index->ne[2] || index->ne[3] != 1 ||
        op->ne[0] != table->ne[0] || op->ne[1] != index->ne[0] || op->ne[2] != table->ne[2] || op->ne[3] != table->ne[3] ||
        !fits_i32(ggml_nelements(op)) || !element_offsets_fit_i32(table, sizeof(float)) ||
        !element_offsets_fit_i32(index, sizeof(int32_t)) || partially_overlaps(op, table) || partially_overlaps(op, index) ||
        (op->data != nullptr && (op->data == table->data || op->data == index->data))) {
        return false;
    }
    const int64_t n = ggml_nelements(op);
    const flagos_kernel_info * kernel = find_elementwise(package, "flagos_get_rows_batched_f32", 16, n);
    if (kernel == nullptr) {
        return false;
    }
    set_call(call, kernel, programs_for(kernel, n), { ptr(table), ptr(index), ptr(op),
             i32(table->ne[0]), i32(table->ne[1]), i32(index->ne[0]), i32(table->ne[2]), i32(table->ne[3]),
             i32(table->nb[0] / 4), i32(table->nb[1] / 4), i32(table->nb[2] / 4), i32(table->nb[3] / 4),
             i32(index->nb[0] / 4), i32(index->nb[1] / 4), i32(index->nb[2] / 4), i32(n) });
    return true;
}

// Dense F32 x F32 MUL_MAT with strided, broadcast batches (MoE router
// logits, shared-expert gate): flagos_mul_mat_f32_f32_strided_batched_ieee
// (15 arguments), grid (ceil(rows / tile_m), ceil(columns / tile_n), ne2 * ne3).
bool plan_mul_mat_f32(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    if (w == nullptr || x == nullptr || w->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || !is_contiguous_f32(op) ||
        w->nb[0] != sizeof(float) || x->nb[0] != sizeof(float) || w->ne[0] != x->ne[0] || op->ne[0] != w->ne[1] ||
        op->ne[1] != x->ne[1] || op->ne[2] != x->ne[2] || op->ne[3] != x->ne[3] ||
        x->ne[2] % w->ne[2] != 0 || x->ne[3] % w->ne[3] != 0 || !fits_i32(ggml_nelements(op)) ||
        !element_offsets_fit_i32(w, sizeof(float)) || !element_offsets_fit_i32(x, sizeof(float)) ||
        op->ne[2] * op->ne[3] > 65535) {
        return false;
    }
    for (const ggml_tensor * t : { w, x }) {
        if (partially_overlaps(op, t) || (op->data != nullptr && op->data == t->data)) {
            return false;
        }
    }
    const flagos_kernel_info * kernel = package.find("flagos_mul_mat_f32_f32_strided_batched_ieee");
    if (kernel == nullptr || (kernel->argument_count >= 0 && kernel->argument_count != 15) ||
        kernel->tile_m <= 0 || kernel->tile_n <= 0) {
        return false;
    }
    const int64_t gx = (op->ne[0] + kernel->tile_m - 1) / kernel->tile_m;
    const int64_t gy = (op->ne[1] + kernel->tile_n - 1) / kernel->tile_n;
    if (gx > INT32_MAX || gy > 65535) {
        return false;
    }
    if (call != nullptr) {
        set_call(call, kernel, gx, { ptr(w), ptr(x), ptr(op), i32(w->ne[0]), i32(op->ne[0]), i32(op->ne[1]),
                 i32(op->ne[2]), i32(x->ne[2] / w->ne[2]), i32(x->ne[3] / w->ne[3]),
                 i32(w->nb[1] / 4), i32(w->nb[2] / 4), i32(w->nb[3] / 4), i32(x->nb[1] / 4), i32(x->nb[2] / 4), i32(x->nb[3] / 4) });
        call->grid[1] = static_cast<uint32_t>(gy);
        call->grid[2] = static_cast<uint32_t>(op->ne[2] * op->ne[3]);
    }
    return true;
}

// MUL_MAT_ID (routed experts): each program reads its expert id, so no
// expert sort or workspace is needed. Quantized weights:
// flagos_mul_mat_id_<type>(w_u8, w_f16, x, ids, out, k, rows, picks, tokens,
//   input_picks, experts, sw1_bytes, sw2_bytes, sx1, sx2, si0, si1),
// grid (rows, picks * tokens). F32 weights: flagos_mul_mat_id_f32_f32 with
// element strides, grid (ceil(rows / tile_m), picks * tokens).
bool plan_mul_mat_id(const flagos_kernel_package & package, const ggml_tensor * op, flagos_kernel_call * call) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    const ggml_tensor * ids = op->src[2];
    if (w == nullptr || x == nullptr || ids == nullptr || !is_contiguous_f32(op) || x->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || w->ne[3] != 1 || x->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        op->ne[3] != 1 || w->ne[0] != x->ne[0] || w->ne[0] > 16384 || ids->ne[1] != x->ne[2] ||
        x->ne[1] <= 0 || ids->ne[0] % x->ne[1] != 0 || op->ne[0] != w->ne[1] || op->ne[1] != ids->ne[0] ||
        op->ne[2] != ids->ne[1] || !fits_i32(ggml_nelements(op)) || x->nb[0] != sizeof(float) ||
        !element_offsets_fit_i32(x, sizeof(float)) || !element_offsets_fit_i32(ids, sizeof(int32_t)) ||
        ids->nb[0] < sizeof(int32_t) || ids->ne[0] * ids->ne[1] > 65535) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++i) {
        int32_t v = 0;
        std::memcpy(&v, reinterpret_cast<const char *>(op->op_params) + i * sizeof(int32_t), sizeof(v));
        if (v != 0 && !((i == 0 || i == 3) && v == GGML_PREC_F32)) {
            return false;
        }
    }
    for (const ggml_tensor * t : { w, x, ids }) {
        if (partially_overlaps(op, t) || (op->data != nullptr && op->data == t->data)) {
            return false;
        }
    }
    const char * type = nullptr;
    int64_t qk = 0;
    switch (w->type) {
        case GGML_TYPE_Q4_0: type = "q4_0_f32_decoded_compensation"; qk = 32; break;
        case GGML_TYPE_Q4_1: type = "q4_1_f32"; qk = 32; break;
        case GGML_TYPE_Q8_0: type = "q8_0_f32_finite_compensation"; qk = 32; break;
        case GGML_TYPE_Q4_K: type = "q4_k_f32"; qk = 256; break;
        case GGML_TYPE_Q5_K: type = "q5_k_f32"; qk = 256; break;   // proposed FlagGems kernel
        case GGML_TYPE_Q6_K: type = "q6_k_f32"; qk = 256; break;   // proposed FlagGems kernel
        case GGML_TYPE_F32:  type = "f32_f32"; break;
        default: return false;
    }
    const int64_t picks = ids->ne[0], tokens = ids->ne[1];
    const std::string name = std::string("flagos_mul_mat_id_") + type;
    const flagos_kernel_info * kernel = package.find(name);
    if (kernel == nullptr) {
        return false;
    }
    if (qk != 0) {
        const size_t row_bytes = size_t(w->ne[0] / qk) * ggml_type_size(w->type);
        if (w->ne[0] % qk != 0 || w->nb[0] != ggml_type_size(w->type) || w->nb[1] < row_bytes ||
            w->nb[1] > size_t(INT32_MAX) || w->nb[2] > size_t(INT32_MAX) ||
            size_t(w->ne[2] - 1) * w->nb[2] + size_t(w->ne[1] - 1) * w->nb[1] + row_bytes > size_t(INT32_MAX) ||
            (kernel->argument_count >= 0 && kernel->argument_count != 17) || kernel->block_size != qk) {
            return false;
        }
        if (call != nullptr) {
            set_call(call, kernel, w->ne[1], { ptr(w), ptr(w), ptr(x), ptr(ids), ptr(op), i32(w->ne[0]), i32(w->ne[1]),
                     i32(picks), i32(tokens), i32(x->ne[1]), i32(w->ne[2]), i32(w->nb[1]), i32(w->nb[2]),
                     i32(x->nb[1] / 4), i32(x->nb[2] / 4), i32(ids->nb[0] / 4), i32(ids->nb[1] / 4) });
            call->grid[1] = static_cast<uint32_t>(picks * tokens);
        }
        return true;
    }
    if (w->nb[0] != sizeof(float) || !element_offsets_fit_i32(w, sizeof(float)) ||
        (kernel->argument_count >= 0 && kernel->argument_count != 16) || kernel->tile_m <= 0) {
        return false;
    }
    if (call != nullptr) {
        set_call(call, kernel, (w->ne[1] + kernel->tile_m - 1) / kernel->tile_m, { ptr(w), ptr(x), ptr(ids), ptr(op),
                 i32(w->ne[0]), i32(w->ne[1]), i32(picks), i32(tokens), i32(x->ne[1]), i32(w->ne[2]),
                 i32(w->nb[1] / 4), i32(w->nb[2] / 4), i32(x->nb[1] / 4), i32(x->nb[2] / 4), i32(ids->nb[0] / 4), i32(ids->nb[1] / 4) });
        call->grid[1] = static_cast<uint32_t>(picks * tokens);
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
            return plan_binary(package, op, call) || plan_binary_strided(package, op, call);
        case GGML_OP_DIV:
            return plan_binary_strided(package, op, call);
        case GGML_OP_ARGSORT:
            return plan_argsort(package, op, call);
        case GGML_OP_SUM_ROWS:
            return plan_sum_rows(package, op, call);
        case GGML_OP_CLAMP:
            return plan_clamp(package, op, call);
        case GGML_OP_MUL_MAT_ID:
            return plan_mul_mat_id(package, op, call);
        case GGML_OP_SCALE:
            return plan_scale(package, op, call);
        case GGML_OP_RMS_NORM:
            return plan_rms_norm(package, op, call) || plan_rms_norm_strided(package, op, call);
        case GGML_OP_GLU:
            return plan_glu(package, op, call);
        case GGML_OP_CONCAT:
            return plan_concat(package, op, call);
        case GGML_OP_L2_NORM:
            return plan_l2_norm(package, op, call);
        case GGML_OP_SSM_CONV:
            return plan_ssm_conv(package, op, call);
        case GGML_OP_GATED_DELTA_NET:
            return plan_gated_delta_net(package, op, call);
        case GGML_OP_ROPE:
            return plan_rope(package, op, call);
        case GGML_OP_FLASH_ATTN_EXT:
            return plan_flash_attn(package, op, call);
        case GGML_OP_UNARY:
            return plan_unary(package, op, call);
        case GGML_OP_MUL_MAT:
            return plan_mul_mat(package, op, call) || plan_mul_mat_f32(package, op, call);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return plan_copy(package, op, call);
        case GGML_OP_SET_ROWS:
            return plan_set_rows(package, op, call);
        case GGML_OP_GET_ROWS:
            return plan_get_rows(package, op, call) || plan_get_rows_quant(package, op, call) ||
                   plan_get_rows_batched(package, op, call);
        case GGML_OP_SOFT_MAX:
            return plan_soft_max(package, op, call);
        default:
            return false;
    }
}

bool flagos_fusion_plan(const flagos_kernel_package & package, const ggml_cgraph * graph,
                        const flagos_pattern_candidate & candidate, flagos_kernel_call * call) {
    if (graph == nullptr || !package.is_open()) {
        return false;
    }
    switch (candidate.id) {
        case flagos_pattern_id::rms_norm_mul:
            return plan_rms_norm_mul(package, graph, candidate, call);
        case flagos_pattern_id::add_rms_norm_mul:
            return plan_add_rms_norm_mul(package, graph, candidate, call);
        default:
            return false;
    }
}
