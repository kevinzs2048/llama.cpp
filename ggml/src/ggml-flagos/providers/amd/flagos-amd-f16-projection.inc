// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

static bool amd_describe_f16_projection(const amd_device_context * device,
        const void * weights, const void * input, const void * output,
        int k, int rows, int columns, amd_f16_projection_launch * launch) {
    if (device == nullptr || device->aot == nullptr || launch == nullptr || k <= 0 || rows <= 0 || columns <= 1 ||
        (weights != nullptr && reinterpret_cast<uintptr_t>(weights) % sizeof(uint16_t) != 0) ||
        (input != nullptr && reinterpret_cast<uintptr_t>(input) % sizeof(float) != 0) ||
        (output != nullptr && reinterpret_cast<uintptr_t>(output) % sizeof(float) != 0)) {
        return false;
    }
    *launch = {};
    auto select = [&](const char * name, bool grouped, bool aligned) {
        const auto * metadata = device->aot->find(name);
        if (metadata == nullptr || metadata->tile_m <= 0 || metadata->tile_n <= 0 || metadata->tile_k <= 0) {
            return false;
        }
        const uint64_t blocks_m = (static_cast<uint64_t>(rows) + metadata->tile_m - 1) / metadata->tile_m;
        const uint64_t blocks_n = (static_cast<uint64_t>(columns) + metadata->tile_n - 1) / metadata->tile_n;
        const uint64_t gx = grouped ? blocks_m * blocks_n : blocks_m;
        const uint64_t gy = grouped ? 1 : blocks_n;
        if (!amd_kernel_launch_contract_fits(device, metadata, 6, 3, gx, gy, 1)) {
            return false;
        }
        *launch = {name, static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), grouped, aligned};
        return true;
    };
    if (amd_grouped_f16_gemm_enabled(device, columns) && select("flagos_mul_mat_f16_f32_grouped", true, false)) {
        return true;
    }
    const char * setting = std::getenv("FLAGOS_AMD_ALIGNED_F16_GEMM");
    if ((setting == nullptr || std::strcmp(setting, "0") != 0) &&
        flagos_amd::supports_aligned_f16_gemm_launch(weights, input, output, k, rows) &&
        select("flagos_mul_mat_f16_f32_batched_aligned", false, true)) {
        return true;
    }
    return select("flagos_mul_mat_f16_f32_batched", false, false);
}
