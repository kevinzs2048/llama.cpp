#include "../../flagos-provider.h"
#include "native.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>
#include <cstdlib>
#include <mutex>

const flagos_provider_v1 * flagos_spacemit_provider();

namespace {

constexpr uint64_t SPACEMIT_ID = 0x53504143454d4954ull;
ggml_backend_reg_t native_reg = nullptr;
flagos_spacemit_native_info native_info;
bool (*native_supports_op)(ggml_backend_dev_t, const ggml_tensor *) = nullptr;

bool disabled_by_environment() {
    const char * value = std::getenv("GGML_FLAGOS_SPACEMIT_DISABLE");
    return value && std::strcmp(value, "1") == 0;
}

bool supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // The native quantized MUL_MAT flattens src1's batch dimensions but uses
    // only the first packed src0 matrix. Expert selection is a separate
    // MUL_MAT_ID implementation and remains eligible.
    if (op->op == GGML_OP_MUL_MAT && op->src[0] && ggml_is_quantized(op->src[0]->type) &&
        (op->src[0]->ne[2] != 1 || op->src[0]->ne[3] != 1)) {
        return false;
    }
    // The native F16 GEMM path fails almost all selected GGML CPU numerical
    // comparisons, including dense (non-KV) matrices.
    if (op->op == GGML_OP_MUL_MAT && op->src[0] && op->src[0]->type == GGML_TYPE_F16) {
        return false;
    }
    // Native F16 QK/QKV matmul gives incorrect results for KV-cache views,
    // including the contiguous V view. Keep these layouts on the CPU until
    // the kernel supports them and passes a numerical comparison.
    if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) &&
        op->src[0] && (!ggml_is_contiguous(op->src[0]) ||
                       (op->src[0]->type == GGML_TYPE_F16 && op->src[0]->view_src))) {
        return false;
    }
    // Reject lossy quantization and native traits that cannot dispatch.
    if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) &&
        op->src[0] && flagos_spacemit_cpu_only_weight(op->src[0],
            native_info.target && std::strcmp(native_info.target, "a100-ime2") == 0)) {
        return false;
    }
    if (!ggml_op_is_empty(op->op)) {
        // Packed storage is read-only to kernels and is valid only for whole weights.
        if (ggml_is_quantized(op->type)) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const auto src = op->src[i];
            if (src && ggml_is_quantized(src->type) &&
                (i != 0 || (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) ||
                 src->view_src || !ggml_is_contiguous(src))) {
                return false;
            }
        }
    }
    // The native RVV norm kernels assert for epsilon <= 0.
    if (op->op == GGML_OP_RMS_NORM || op->op == GGML_OP_NORM) {
        float epsilon;
        std::memcpy(&epsilon, op->op_params, sizeof(epsilon));
        if (!(epsilon > 0.0f) || !op->src[0] ||
            op->src[0]->nb[0] != ggml_type_size(op->src[0]->type)) {
            return false;
        }
    }
    // Native SCALE reads only the multiplier, not ggml_scale_bias's bias.
    if (op->op == GGML_OP_SCALE && ggml_get_op_params_f32(op, 1) != 0.0f) {
        return false;
    }
    // Native CONT's F16/BF16 paths fail the GGML CPU comparison on K3.
    if (op->op == GGML_OP_CONT && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16)) {
        return false;
    }
    // This native CPY dispatch branch treats a stride coincidence as a
    // transpose/permute, and fails GGML's strided-copy comparisons.
    if (op->op == GGML_OP_CPY && op->src[0] && op->type == op->src[0]->type &&
        op->nb[0] == op->src[0]->nb[1] && op->src[0]->nb[0] != op->src[0]->nb[1] &&
        ggml_nelements(op) == ggml_nelements(op->src[0])) {
        return false;
    }
    // The scalar L2 normalization kernel assumes contiguous F32 elements.
    if (op->op == GGML_OP_L2_NORM &&
        (!op->src[0] || op->src[0]->type != GGML_TYPE_F32 ||
         op->src[0]->nb[0] != sizeof(float) || op->type != GGML_TYPE_F32 ||
         op->nb[0] != sizeof(float))) {
        return false;
    }
    // Native PAD implements zero-valued right padding only; it ignores all
    // left offsets and the circular flag in GGML's extended contract.
    if (op->op == GGML_OP_PAD) {
        if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32 ||
            op->src[0]->nb[0] != sizeof(float) || op->type != GGML_TYPE_F32 ||
            op->nb[0] != sizeof(float) || ggml_get_op_params_i32(op, 8) != 0) {
            return false;
        }
        for (int i = 0; i < 8; i += 2) {
            if (ggml_get_op_params_i32(op, i) != 0) {
                return false;
            }
        }
    }
    // Native DIAG ignores leading batch dimensions.
    if (op->op == GGML_OP_DIAG) {
        return false;
    }
    // Every selected native triangular solve failed the GGML CPU comparison.
    if (op->op == GGML_OP_SOLVE_TRI) {
        return false;
    }
    // The native softmax kernel does not read mask or attention sinks.
    if (op->op == GGML_OP_SOFT_MAX && (op->src[1] || op->src[2])) {
        return false;
    }
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        const float max_bias = ggml_get_op_params_f32(op, 1);
        // The native RVV path requires a mask and fails the bias comparison.
        if (!op->src[3] || max_bias != 0.0f) {
            return false;
        }
    }
    return native_supports_op(dev, op);
}

bool probe(ggml_backend_reg_t) {
    // An intentional opt-out is a healthy provider with zero devices. It
    // avoids reporting a runtime probe failure for CPU-only measurements.
    if (disabled_by_environment()) {
        return true;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        if (!flagos_spacemit_native_query(&native_info)) {
            return;
        }
        auto reg = flagos_spacemit_native_registry();
        if (reg && ggml_backend_reg_dev_count(reg) == 1) {
            auto dev = ggml_backend_reg_dev_get(reg, 0);
            if (dev && dev->iface.supports_op) {
                native_supports_op = dev->iface.supports_op;
                dev->iface.supports_op = supports_op;
                native_reg = reg;
            }
        }
    });
    return native_reg != nullptr;
}

size_t device_count() {
    return !disabled_by_environment() && probe(nullptr) ? 1 : 0;
}

ggml_backend_dev_t device_get(size_t index) {
    return index == 0 && device_count() == 1 ? ggml_backend_reg_dev_get(native_reg, 0) : nullptr;
}

bool device_identity(size_t index, flagos_device_identity * identity) {
    if (!identity || !device_get(index)) {
        return false;
    }
    // One runtime execution domain; this is not a hardware serial number.
    *identity = { SPACEMIT_ID, SPACEMIT_ID, 1, 0, 0 };
    return true;
}

bool device_caps(size_t index, flagos_device_caps * caps) {
    if (!caps || !device_get(index)) {
        return false;
    }
    // Host-visible storage can still contain provider-specific packed weights.
    *caps = { flagos_provider_kind::cpu_accelerator, FLAGOS_MEMORY_HOST_VISIBLE, 0, nullptr };
    return true;
}

bool device_profile(size_t index, flagos_device_profile * profile) {
    if (!profile || !device_get(index)) {
        return false;
    }
    *profile = {};
    profile->struct_size = sizeof(*profile);
    profile->engine = flagos_engine_kind::cpu;
    profile->features = FLAGOS_FEATURE_SIMD | FLAGOS_FEATURE_MATRIX;
    profile->vector_bits = native_info.vector_bits;
    profile->concurrency = native_info.workers;
    profile->vendor = "SpaceMiT";
    profile->architecture = "riscv64";
    profile->target = native_info.target;
    profile->runtime = "spine-runtime";
    return true;
}

bool is_backend(ggml_backend_t backend) {
    return backend && flagos_spacemit_native_is_backend(backend);
}

bool set_device(size_t index) {
    return device_get(index) != nullptr;
}

int get_device() {
    return device_count() == 1 ? 0 : -1;
}

int score() {
    return 0;
}

} // namespace

const flagos_provider_v1 * flagos_spacemit_provider() {
    static const flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { SPACEMIT_ID, "spacemit", 1 },
        probe,
        device_count,
        device_get,
        device_identity,
        device_caps,
        is_backend,
        set_device,
        get_device,
        score,
        device_profile,
    };
    return &provider;
}
