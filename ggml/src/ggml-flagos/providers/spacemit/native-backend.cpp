#include "native.h"
#include "ggml-backend-impl.h"
#include "ime.h"

#include <cstdio>

static bool flagos_spacemit_get_work_size(int n_threads, const ggml_tensor * op, size_t * size);

static bool flagos_spacemit_compute_forward_checked(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const bool dispatched = ggml_spacemit_compute_forward(ctx, op);
    if (!dispatched && ctx.ith == 0) {
        const auto * src0 = op->src[0];
        const auto * src1 = op->src[1];
        std::fprintf(stderr,
                     "FlagOS SpaceMiT: native dispatch rejected %s '%s': src0='%s' type=%s ne=[%lld,%lld] "
                     "nb=[%zu,%zu] extra=%p; src1='%s' type=%s ne=[%lld,%lld] nb=[%zu,%zu]\n",
                     ggml_op_name(op->op), op->name,
                     src0 ? src0->name : "", src0 ? ggml_type_name(src0->type) : "none",
                     src0 ? (long long) src0->ne[0] : 0, src0 ? (long long) src0->ne[1] : 0,
                     src0 ? src0->nb[0] : 0, src0 ? src0->nb[1] : 0, src0 ? src0->extra : nullptr,
                     src1 ? src1->name : "", src1 ? ggml_type_name(src1->type) : "none",
                     src1 ? (long long) src1->ne[0] : 0, src1 ? (long long) src1->ne[1] : 0,
                     src1 ? src1->nb[0] : 0, src1 ? src1->nb[1] : 0);
        std::fflush(stderr);
    }
    return dispatched;
}

// FlagOS is the only public backend registration entry point.
#undef GGML_BACKEND_DL_IMPL
#define GGML_BACKEND_DL_IMPL(reg_fn)
#define ggml_spacemit_get_work_size flagos_spacemit_get_work_size
#define ggml_spacemit_compute_forward flagos_spacemit_compute_forward_checked
#include "ggml-spacemit.cpp"
#undef ggml_spacemit_compute_forward
#undef ggml_spacemit_get_work_size
#include "native-buffer.h"

#include <limits>

static bool flagos_spacemit_get_work_size(int n_threads, const ggml_tensor * op, size_t * size) {
    const bool found = ggml_spacemit_get_work_size(n_threads, op, size);
    if (op->op != GGML_OP_ROPE) {
        return found;
    }

    const size_t width = static_cast<size_t>(op->ne[0]) + ggml::spacemit::cache_line_size_f32;
    GGML_ASSERT(n_threads > 0 && width <= std::numeric_limits<size_t>::max() / sizeof(float) / n_threads);
    *size = std::max(*size, width * sizeof(float) * n_threads);
    return true;
}

bool flagos_spacemit_native_query(flagos_spacemit_native_info * out) {
    if (!out) {
        return false;
    }
    const auto info = spert::backend_info();
    const auto & env = global_spine_env_info;
    if (!info.num_cores || !info.vlen || info.vlen > std::numeric_limits<uint32_t>::max() / 8 ||
        env.num_cores <= 0 || static_cast<size_t>(env.num_cores) > info.num_cores) {
        GGML_LOG_WARN("FlagOS SpaceMiT: no usable runtime cores, or invalid GGML_SPACEMIT_WORKERS\n");
        return false;
    }
    const char * target = nullptr;
#ifdef RISCV64_SPACEMIT_IME1
    if (env.use_ime1) {
        target = env.perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ? "a60-ime1" : "x100-ime1";
    }
#endif
#ifdef RISCV64_SPACEMIT_IME2
    if (env.use_ime2) {
        target = "a100-ime2";
    }
#endif
    if (!target) {
        GGML_LOG_WARN("FlagOS SpaceMiT: runtime core architecture has no compiled IME implementation\n");
        return false;
    }
    *out = { static_cast<uint32_t>(info.vlen * 8), static_cast<uint32_t>(env.num_cores), target };
    return true;
}

ggml_backend_reg_t flagos_spacemit_native_registry() {
    auto reg = ggml_backend_spacemit_reg();
    auto buft = ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, 0));
    buft->iface.alloc_buffer = spacemit_alloc_buffer;
    buft->iface.get_alloc_size = spacemit_alloc_size;
    return reg;
}

bool flagos_spacemit_native_is_backend(ggml_backend_t backend) {
    return ggml_backend_is_spacemit(backend);
}
