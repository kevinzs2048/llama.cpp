// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

struct amd_f16_ffn_launch {
    const char * kernel = nullptr;
    unsigned int grid_x = 0;
    unsigned int grid_y = 0;
    unsigned int pack_grid = 0;
    bool grouped = false;
    bool tiled = false;
    bool packed = false;
};

static bool amd_describe_f16_ffn(const amd_device_context * device,
        const void * gate_weights, const void * up_weights, const void * input, const void * output,
        int k, int rows, int columns, amd_f16_ffn_launch * launch) {
    if (device == nullptr || device->aot == nullptr || launch == nullptr || k <= 0 || rows <= 0 || columns <= 1 ||
        static_cast<uint64_t>(k) * rows > INT32_MAX || static_cast<uint64_t>(k) * columns > INT32_MAX ||
        static_cast<uint64_t>(rows) * columns > INT32_MAX ||
        reinterpret_cast<uintptr_t>(gate_weights) % sizeof(uint16_t) != 0 ||
        reinterpret_cast<uintptr_t>(up_weights) % sizeof(uint16_t) != 0 ||
        reinterpret_cast<uintptr_t>(input) % sizeof(float) != 0 ||
        reinterpret_cast<uintptr_t>(output) % sizeof(float) != 0) {
        return false;
    }
    *launch = {};
    auto select = [&](const char * name, bool grouped, bool tiled) {
        const auto * metadata = device->aot->find(name);
        if (metadata == nullptr || metadata->tile_m <= 0 || metadata->tile_n <= 0 || metadata->tile_k <= 0) {
            return false;
        }
        const uint64_t blocks_m = (static_cast<uint64_t>(rows) + metadata->tile_m - 1) / metadata->tile_m;
        const uint64_t blocks_n = (static_cast<uint64_t>(columns) + metadata->tile_n - 1) / metadata->tile_n;
        const uint64_t gx = grouped ? blocks_m * blocks_n : blocks_m;
        const uint64_t gy = grouped ? 1 : blocks_n;
        if (!amd_kernel_launch_contract_fits(device, metadata, 7, 4, gx, gy, 1)) {
            return false;
        }
        *launch = {name, static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), 0, grouped, tiled, false};
        return true;
    };
    if (amd_grouped_f16_ffn_enabled(device, columns) &&
        select("flagos_ffn_swiglu_f16_f32_grouped", true, false)) {
        return true;
    }
    const auto * tiled = device->aot->find("flagos_ffn_swiglu_f16_f32_batched_tiled");
    // Private dequant-cache bindings are naturally arena-aligned. Unknown graph
    // bindings may be considered early, but actual addresses are checked again
    // immediately before execution and cannot bypass the AOT divisibility ABI.
    const bool aligned = k % 16 == 0 && rows % 16 == 0 &&
        reinterpret_cast<uintptr_t>(gate_weights) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(up_weights) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(input) % 16 == 0 && reinterpret_cast<uintptr_t>(output) % 16 == 0;
    if (amd_env_enabled("FLAGOS_AMD_TILED_F16_FFN") && aligned && tiled != nullptr &&
        tiled->tile_m == 32 && tiled->tile_n == 128 && tiled->tile_k == 32 &&
        tiled->block_size == 128 && tiled->num_warps == 4 && tiled->warp_size == 32 &&
        select("flagos_ffn_swiglu_f16_f32_batched_tiled", false, true)) {
        return true;
    }
    return select("flagos_ffn_swiglu_f16_f32_batched", false, false);
}

static bool amd_describe_packed_f16_ffn(const amd_device_context * device,
        const void * gate_weights, const void * up_weights, const void * input,
        const void * output, const void * workspace, int k, int rows, int columns,
        const amd_f16_ffn_launch & regular, amd_f16_ffn_launch * launch) {
    if (!regular.tiled || launch == nullptr || device == nullptr || device->aot == nullptr ||
        !amd_env_enabled("FLAGOS_AMD_PACKED_F16_FFN") || amd_graph_capture_enabled() || columns != 512 ||
        !((k == 2048 && rows == 6144) || (k == 2560 && rows == 9216)) ||
        !flagos_amd::supports_tiled_f16_ffn_launch(gate_weights, up_weights, input, output, k, rows) ||
        reinterpret_cast<uintptr_t>(workspace) % 16 != 0) {
        return false;
    }
    const auto * packed = device->aot->find("flagos_ffn_swiglu_f16_packed_activation");
    const auto * pack = device->aot->find("flagos_pack_f32_f16");
    if (packed == nullptr || pack == nullptr || packed->tile_m != 32 || packed->tile_n != 128 ||
        packed->tile_k != 32 || packed->block_size != 32 || packed->num_warps != 4 || packed->warp_size != 32 ||
        pack->block_size != 256 || pack->num_warps != 4 || pack->warp_size != 32) {
        return false;
    }
    const uint64_t count = static_cast<uint64_t>(k) * columns;
    const uint64_t pack_grid = (count + pack->block_size - 1) / pack->block_size;
    const uint64_t gx = (static_cast<uint64_t>(rows) + packed->tile_m - 1) / packed->tile_m;
    const uint64_t gy = (static_cast<uint64_t>(columns) + packed->tile_n - 1) / packed->tile_n;
    // Check both kernels before allocating private workspace or packing input.
    if (count > INT32_MAX || !amd_kernel_launch_contract_fits(device, pack, 3, 2, pack_grid, 1, 1) ||
        !amd_kernel_launch_contract_fits(device, packed, 7, 4, gx, gy, 1)) {
        return false;
    }
    *launch = {"flagos_ffn_swiglu_f16_packed_activation", static_cast<unsigned int>(gx),
        static_cast<unsigned int>(gy), static_cast<unsigned int>(pack_grid), false, true, true};
    return true;
}
