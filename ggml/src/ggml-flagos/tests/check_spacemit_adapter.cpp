#include "flagos-provider.h"
#include "flagos-registry.h"
#include "ggml-flagos.h"
#include "providers/spacemit/native.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); std::abort(); } } while (0)

const flagos_provider_v1 * flagos_spacemit_provider();

static bool available = true;
static bool invalid_registry = false;
static int query_calls = 0;
static int compute_calls = 0;
static ggml_backend_device native_device {};
static ggml_backend_buffer_type native_buft {};
static ggml_cgraph * expected_graph = nullptr;

bool flagos_spacemit_native_query(flagos_spacemit_native_info * info) {
    ++query_calls;
    *info = { 1024, 4, "a100-ime2" };
    return available;
}

static const char * name(ggml_backend_dev_t) { return "SPACEMIT0"; }
static const char * backend_name(ggml_backend_t) { return "SPACEMIT0"; }
static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
static ggml_backend_buffer_type_t buffer_type(ggml_backend_dev_t) { return &native_buft; }
static bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_ADD || op->op == GGML_OP_RMS_NORM ||
           op->op == GGML_OP_SOFT_MAX || op->op == GGML_OP_FLASH_ATTN_EXT ||
           op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID ||
           op->op == GGML_OP_SCALE || op->op == GGML_OP_NORM ||
           op->op == GGML_OP_CONT || op->op == GGML_OP_L2_NORM ||
           op->op == GGML_OP_PAD || op->op == GGML_OP_DIAG ||
           op->op == GGML_OP_CPY || op->op == GGML_OP_SOLVE_TRI;
}
static bool supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &native_buft; }
static void free_backend(ggml_backend_t backend) { delete backend; }
static ggml_status compute(ggml_backend_t, ggml_cgraph * graph) {
    CHECK(graph == expected_graph);
    ++compute_calls;
    return GGML_STATUS_FAILED;
}

static ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    auto backend = new ggml_backend {};
    backend->device = dev;
    backend->iface.get_name = backend_name;
    backend->iface.free = free_backend;
    backend->iface.graph_compute = compute;
    return backend;
}

bool flagos_spacemit_native_is_backend(ggml_backend_t backend) {
    return backend && backend->iface.graph_compute == compute;
}

ggml_backend_reg_t flagos_spacemit_native_registry() {
    static ggml_backend_reg reg {};
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface.get_device_count = [](ggml_backend_reg_t) -> size_t { return invalid_registry ? 2 : 1; };
    reg.iface.get_device = [](ggml_backend_reg_t, size_t index) { return index == 0 ? &native_device : nullptr; };
    native_device.reg = &reg;
    native_device.iface.get_name = name;
    native_device.iface.get_type = device_type;
    native_device.iface.get_buffer_type = buffer_type;
    native_device.iface.supports_op = supports_op;
    native_device.iface.supports_buft = supports_buft;
    native_device.iface.init_backend = init_backend;
    native_buft.device = &native_device;
    return &reg;
}

int main(int argc, char ** argv) {
    available = argc < 2 || std::strcmp(argv[1], "unavailable") != 0;
    invalid_registry = argc >= 2 && std::strcmp(argv[1], "invalid-registry") == 0;
    const bool disabled = argc >= 2 && std::strcmp(argv[1], "disabled") == 0;
    if (disabled) {
#ifdef _WIN32
        CHECK(_putenv_s("GGML_FLAGOS_SPACEMIT_DISABLE", "1") == 0);
#else
        CHECK(setenv("GGML_FLAGOS_SPACEMIT_DISABLE", "1", 1) == 0);
#endif
    }
    const auto provider = flagos_spacemit_provider();
    CHECK(flagos_provider_is_valid(provider));
    CHECK(provider->score() == 0);
    CHECK(query_calls == 0);
    auto reg = ggml_backend_flagos_reg();
    if (!available || invalid_registry || disabled) {
        CHECK(ggml_backend_reg_dev_count(reg) == 0);
        CHECK(provider->device_count() == 0);
        CHECK(provider->get_device() == -1);
        CHECK(!provider->set_device(0));
        CHECK(!provider->device_get(0));
        CHECK(!ggml_backend_flagos_init(0));
    } else {
        CHECK(ggml_backend_reg_dev_count(reg) == 1);
        CHECK(ggml_backend_reg_dev_get(reg, 0) == &native_device);
        CHECK(ggml_backend_dev_backend_reg(&native_device) == reg);
        CHECK(ggml_backend_dev_type(&native_device) == GGML_BACKEND_DEVICE_TYPE_ACCEL);
        CHECK(ggml_backend_dev_buffer_type(&native_device) == &native_buft);
        CHECK(native_buft.device == &native_device);
        flagos_device_info info {};
        CHECK(flagos_registry_get_device_info(0, &info));
        CHECK(std::strcmp(info.provider.name, "spacemit") == 0);
        CHECK(info.caps.kind == flagos_provider_kind::cpu_accelerator);
        CHECK(info.caps.execution == 0);
        flagos_device_profile profile {};
        CHECK(flagos_registry_get_device_profile(0, &profile));
        CHECK(profile.engine == flagos_engine_kind::cpu);
        CHECK(profile.vector_bits == 1024 && profile.concurrency == 4);
        CHECK(std::strcmp(profile.target, "a100-ime2") == 0);
        flagos_provider_selector_v1 selector {};
        selector.struct_size = sizeof(selector);
        selector.provider_name = "spacemit";
        flagos_provider_selection_v1 selected {};
        selected.struct_size = sizeof(selected);
        CHECK(flagos_select_provider_v1(&selector, &selected) == FLAGOS_STATUS_SUCCESS);
        CHECK(selected.global_device_index == 0);
        ggml_tensor op {};
        op.op = GGML_OP_ADD;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_MUL_MAT;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        ggml_tensor weight {};
        weight.type = GGML_TYPE_Q4_K;
        op.src[0] = &weight;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_MUL_MAT_ID;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_Q4_1;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_MUL_MAT;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_Q5_0;
        weight.ne[0] = 256;
        weight.ne[2] = 1;
        weight.ne[3] = 1;
        weight.nb[0] = ggml_type_size(weight.type);
        weight.nb[1] = ggml_row_size(weight.type, weight.ne[0]);
        weight.ne[1] = 32;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.ne[1] = 16;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_Q5_1;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_Q5_K;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_IQ2_S;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_IQ3_XXS;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_Q6_K;
        weight.nb[0] = ggml_type_size(weight.type);
        weight.nb[1] = ggml_row_size(weight.type, weight.ne[0]);
        weight.ne[1] = 17;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.ne[1] = 32;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_Q8_0;
        weight.nb[0] = ggml_type_size(weight.type);
        weight.nb[1] = ggml_row_size(weight.type, weight.ne[0]);
        weight.ne[1] = 17;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.ne[1] = 32;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        weight.ne[2] = 4;
        weight.ne[3] = 2;
        weight.nb[2] = weight.nb[1] * weight.ne[1];
        weight.nb[3] = weight.nb[2] * weight.ne[2];
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_MUL_MAT_ID;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_MUL_MAT;
        weight.ne[3] = 1;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.ne[2] = 1;
        weight.ne[3] = 1;
        weight.type = GGML_TYPE_F16;
        weight.ne[0] = 48;
        weight.ne[1] = 256;
        weight.ne[2] = 6;
        weight.ne[3] = 1;
        weight.nb[0] = sizeof(ggml_fp16_t);
        weight.nb[1] = 48 * 6 * sizeof(ggml_fp16_t);
        weight.nb[2] = 48 * sizeof(ggml_fp16_t);
        weight.nb[3] = 48 * 256 * 6 * sizeof(ggml_fp16_t);
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.nb[1] = 48 * sizeof(ggml_fp16_t);
        weight.nb[2] = 48 * 256 * sizeof(ggml_fp16_t);
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.type = GGML_TYPE_F32;
        weight.nb[0] = sizeof(float);
        weight.nb[1] = 48 * sizeof(float);
        weight.nb[2] = 48 * 256 * sizeof(float);
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        ggml_tensor view_root {};
        weight.view_src = &view_root;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        weight.view_src = nullptr;
        op.src[0] = nullptr;
        op.op = GGML_OP_RMS_NORM;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        float epsilon = 1e-6f;
        std::memcpy(op.op_params, &epsilon, sizeof(epsilon));
        op.src[0] = &weight;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_NORM;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.src[0] = &weight;
        weight.type = GGML_TYPE_F32;
        weight.nb[0] = sizeof(float) * 2;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.nb[0] = sizeof(float);
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.src[0] = nullptr;
        epsilon = 0.0f;
        std::memcpy(op.op_params, &epsilon, sizeof(epsilon));
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op = {};
        op.op = GGML_OP_SCALE;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        const float bias = 1.0f;
        std::memcpy(&op.op_params[1], &bias, sizeof(bias));
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op = {};
        op.op = GGML_OP_CONT;
        op.type = GGML_TYPE_F16;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.type = GGML_TYPE_BF16;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.type = GGML_TYPE_F32;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op = {};
        op.op = GGML_OP_L2_NORM;
        op.nb[0] = sizeof(float);
        op.src[0] = &weight;
        weight.type = GGML_TYPE_F32;
        weight.nb[0] = sizeof(float) * 2;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        weight.nb[0] = sizeof(float);
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_PAD;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        int left_pad = 1;
        std::memcpy(op.op_params, &left_pad, sizeof(left_pad));
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.op_params[0] = 0;
        int circular = 1;
        std::memcpy(&op.op_params[8], &circular, sizeof(circular));
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_DIAG;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.op = GGML_OP_SOLVE_TRI;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op = {};
        op.op = GGML_OP_CPY;
        op.type = GGML_TYPE_F32;
        op.src[0] = &weight;
        weight.type = GGML_TYPE_F32;
        weight.ne[0] = 4;
        weight.ne[1] = 8;
        weight.ne[2] = weight.ne[3] = 1;
        op.ne[0] = 4;
        op.ne[1] = 8;
        op.ne[2] = op.ne[3] = 1;
        weight.nb[0] = sizeof(float) * 8;
        weight.nb[1] = sizeof(float) * 4;
        op.nb[0] = weight.nb[1];
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.nb[0] = sizeof(float);
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        ggml_tensor mask {};
        op = {};
        op.op = GGML_OP_SOFT_MAX;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        op.src[1] = &mask;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.src[1] = nullptr;
        op.src[2] = &mask;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op = {};
        op.op = GGML_OP_FLASH_ATTN_EXT;
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        op.src[3] = &mask;
        CHECK(ggml_backend_dev_supports_op(&native_device, &op));
        const float max_bias = 8.0f;
        std::memcpy(&op.op_params[1], &max_bias, sizeof(max_bias));
        CHECK(!ggml_backend_dev_supports_op(&native_device, &op));
        CHECK(ggml_backend_dev_supports_buft(&native_device, &native_buft));
        CHECK(!ggml_backend_dev_supports_buft(&native_device, nullptr));
        auto backend = ggml_backend_flagos_init(0);
        CHECK(backend && backend->device == &native_device);
        CHECK(ggml_backend_is_flagos(backend));
        CHECK(!ggml_backend_is_flagos(nullptr));
        auto ctx = ggml_init({ 1024 * 1024, nullptr, true });
        CHECK(ctx);
        expected_graph = ggml_new_graph(ctx);
        CHECK(ggml_backend_graph_compute(backend, expected_graph) == GGML_STATUS_FAILED);
        CHECK(compute_calls == 1);
        ggml_free(ctx);
        ggml_backend_free(backend);
        CHECK(provider->set_device(0));
        CHECK(provider->get_device() == 0);
    }
    CHECK(!provider->device_get(1));
    CHECK(!provider->set_device(1));
    CHECK(!provider->device_identity(0, nullptr));
    CHECK(!provider->device_caps(0, nullptr));
    CHECK(!provider->get_device_profile(0, nullptr));
    CHECK(query_calls == (disabled ? 0 : 1));
    std::puts("SpaceMiT adapter checks passed (test double; no hardware kernels executed)");
}
