// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

struct amd_quantized_gemv_launch {
    const char * kernel = nullptr;
    unsigned int row_tile = 1;
    unsigned int grid_x = 0;
};

static bool amd_describe_quantized_gemv(const amd_device_context * device,
        const ggml_tensor * op, flagos_quantized_matmul_kind kind, int k, int rows,
        amd_quantized_gemv_launch * launch) {
    if (launch == nullptr || device == nullptr || device->aot == nullptr || op == nullptr || rows <= 0) {
        return false;
    }
    *launch = {};
    auto select = [&](const char * name, unsigned int row_tile, bool narrow) {
        const auto * metadata = name == nullptr ? nullptr : device->aot->find(name);
        const uint64_t grid = (static_cast<uint64_t>(rows) + row_tile - 1) / row_tile;
        if (metadata == nullptr || metadata->block_size <= 0 ||
            (narrow && metadata->block_size != static_cast<int>(row_tile)) ||
            !amd_kernel_launch_contract_fits(device, metadata, 6, 4, grid, 1, 1)) {
            return false;
        }
        *launch = {name, row_tile, static_cast<unsigned int>(grid)};
        return true;
    };
    // An optional variant can be declined before submission. It must satisfy
    // the same known resource and ABI checks as the ordinary GEMV.
    const unsigned int q40 = amd_q40_gemv_narrow_row_tile(device);
    const unsigned int q5 = amd_q5_gemv_narrow_row_tile(device);
    const unsigned int q4 = amd_q4_gemv_narrow_row_tile(device);
    if (kind == flagos_quantized_matmul_kind::q4_0 && q40 > 1 && rows % q40 == 0 &&
        select("flagos_mul_mat_q4_0_f32_narrow", q40, true)) {
        return true;
    }
    if (kind == flagos_quantized_matmul_kind::q5_k && q5 == 16 && rows % 16 == 0 &&
        select("flagos_mul_mat_q5_k_f32_narrow16", 16, true)) {
        return true;
    }
    if (kind == flagos_quantized_matmul_kind::q4_k && q4 > 1) {
        if (q4 >= 8 && rows % 8 == 0 && select("flagos_mul_mat_q4_k_f32_narrow8", 8, true)) {
            return true;
        }
        if (rows % 4 == 0 && select("flagos_mul_mat_q4_k_f32_narrow", 4, true)) {
            return true;
        }
    }
    const char * aligned = amd_aligned_quantized_gemv_kernel(device, op, kind, k, rows);
    return (aligned != nullptr && select(aligned, 1, false)) ||
        select(amd_quantized_gemv_kernel(kind), 1, false);
}
