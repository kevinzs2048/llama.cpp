#pragma once

// Included after the pinned native backend in native-backend.cpp.
#include <map>
#include <new>

namespace {

struct spacemit_buffer_context {
    void * base;
    std::mutex mutex;
    // Descriptors are copied because a GGML context may die before its buffer.
    std::map<uintptr_t, ggml_tensor> packed;
};

static bool spacemit_needs_pack(const ggml_tensor * t) {
    // The native packed MUL_MAT trait only handles 2D weights, while 3D
    // MUL_MAT_ID has its own packed trait. 4D MUL_MAT uses the native scalar
    // fallback, which reads GGML's canonical layout directly.
    // Also keep only canonical bytes for formats without a usable trait.
    return !flagos_spacemit_cpu_only_weight(t, global_spine_env_info.use_ime2) &&
           !t->view_src && !ggml_is_empty(t) && ggml_n_dims(t) != 4 &&
           ggml_is_contiguous(t) && ggml_spacemit_get_optimal_repack_type(t);
}

static size_t spacemit_canonical_offset(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    const size_t size = ggml_spacemit_nbytes(buft, t);
    GGML_ASSERT(size <= SIZE_MAX - 63);
    return (size + 63) & ~size_t(63);
}

static size_t spacemit_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
    if (!spacemit_needs_pack(t)) {
        return ggml_nbytes(t);
    }
    const size_t offset = spacemit_canonical_offset(buft, t);
    GGML_ASSERT(ggml_nbytes(t) <= SIZE_MAX - offset);
    return offset + ggml_nbytes(t);
}

static uint8_t * spacemit_canonical_data(ggml_backend_buffer_t buffer, const ggml_tensor * t,
                                        const ggml_tensor ** packed_root = nullptr) {
    const ggml_tensor * root = t;
    size_t offset = 0;
    while (root->view_src) {
        offset += root->view_offs;
        root = root->view_src;
    }
    if (packed_root) {
        *packed_root = root->extra ? root : nullptr;
    }
    if (!root->extra) {
        return static_cast<uint8_t *>(t->data);
    }
    GGML_ASSERT(offset <= ggml_nbytes(root) && ggml_nbytes(t) <= ggml_nbytes(root) - offset);
    return static_cast<uint8_t *>(root->data) + spacemit_canonical_offset(buffer->buft, root) + offset;
}

static void spacemit_pack(ggml_backend_buffer_t buffer, const ggml_tensor * root) {
    ggml_tensor layout = *root;
    GGML_ASSERT(ggml_riscv64_spacemit_repack_tensor(&layout, spacemit_canonical_data(buffer, root), ggml_nbytes(root)) == 0);
}

static void spacemit_free_buffer(ggml_backend_buffer_t buffer) {
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    spine_mem_pool_free(ctx->base);
    delete ctx;
}

static void * spacemit_get_base(ggml_backend_buffer_t buffer) {
    return static_cast<spacemit_buffer_context *>(buffer->context)->base;
}

static ggml_status spacemit_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t) {
    t->extra = nullptr;
    if (t->view_src) {
        return GGML_STATUS_SUCCESS;
    }
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(t->data);
    const size_t size = spacemit_alloc_size(buffer->buft, t);
    const uintptr_t base = reinterpret_cast<uintptr_t>(ctx->base);
    GGML_ASSERT(begin >= base && begin - base <= buffer->size && size <= buffer->size - (begin - base));
    auto it = ctx->packed.lower_bound(begin);
    if (it != ctx->packed.begin()) {
        --it;
    }
    while (it != ctx->packed.end() && it->first < begin + size) {
        const size_t old_size = spacemit_alloc_size(buffer->buft, &it->second);
        if (it->first < begin + size && begin < it->first + old_size) {
            it = ctx->packed.erase(it);
        } else {
            ++it;
        }
    }
    if (spacemit_needs_pack(t)) {
        t->extra = const_cast<ggml::spacemit::tensor_traits_base *>(ggml_spacemit_get_optimal_repack_type(t));
        try {
            ctx->packed.emplace(begin, *t);
        } catch (const std::bad_alloc &) {
            t->extra = nullptr;
            return GGML_STATUS_ALLOC_FAILED;
        }
        memset(spacemit_canonical_data(buffer, t), 0, ggml_nbytes(t));
    }
    return GGML_STATUS_SUCCESS;
}

static void spacemit_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * t,
                                void * data, size_t offset, size_t size) {
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    GGML_ASSERT(offset <= ggml_nbytes(t) && size <= ggml_nbytes(t) - offset);
    memcpy(data, spacemit_canonical_data(buffer, t) + offset, size);
}

static void spacemit_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t,
                                const void * data, size_t offset, size_t size) {
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    GGML_ASSERT(offset <= ggml_nbytes(t) && size <= ggml_nbytes(t) - offset);
    const ggml_tensor * root = nullptr;
    memmove(spacemit_canonical_data(buffer, t, &root) + offset, data, size);
    if (root && size) {
        spacemit_pack(buffer, root);
    }
}

static void spacemit_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * t,
                                   uint8_t value, size_t offset, size_t size) {
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    GGML_ASSERT(offset <= ggml_nbytes(t) && size <= ggml_nbytes(t) - offset);
    const ggml_tensor * root = nullptr;
    memset(spacemit_canonical_data(buffer, t, &root) + offset, value, size);
    if (root && size) {
        spacemit_pack(buffer, root);
    }
}

static void spacemit_clear_buffer(ggml_backend_buffer_t buffer, uint8_t value) {
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    memset(ctx->base, value, buffer->size);
    for (const auto & entry : ctx->packed) {
        spacemit_pack(buffer, &entry.second);
    }
}

static void spacemit_reset_buffer(ggml_backend_buffer_t buffer) {
    auto ctx = static_cast<spacemit_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->packed.clear();
}

static ggml_backend_buffer_t spacemit_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto ctx = new (std::nothrow) spacemit_buffer_context {};
    if (!ctx) {
        return nullptr;
    }
    ctx->base = spine_mem_pool_alloc(size, 64);
    if (!ctx->base) {
        delete ctx;
        return nullptr;
    }
    auto iface = ggml_backend_spacemit_buffer_i;
    iface.free_buffer = spacemit_free_buffer;
    iface.get_base = spacemit_get_base;
    iface.init_tensor = spacemit_init_tensor;
    iface.get_tensor = spacemit_get_tensor;
    iface.set_tensor = spacemit_set_tensor;
    iface.memset_tensor = spacemit_memset_tensor;
    iface.clear = spacemit_clear_buffer;
    iface.reset = spacemit_reset_buffer;
    return ggml_backend_buffer_init(buft, iface, ctx, size);
}

} // namespace
