// DEVICE-kind provider plugin used by check_loader. It exposes one GGML
// device with a complete (inert) interface.
#include "flagos-provider.h"
#include "flagos-plugin.h"

#include <cstring>

namespace {

ggml_backend_device mock_device {};
ggml_backend_buffer_type mock_buft {};

const char * dev_name(ggml_backend_dev_t) { return "MOCKDEV"; }
const char * dev_description(ggml_backend_dev_t) { return "mock device plugin"; }
void dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 0; }
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }
void dev_props(ggml_backend_dev_t, ggml_backend_dev_props * props) {
    *props = {};
    props->name = "MOCKDEV";
    props->description = "mock device plugin";
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
}
ggml_backend_t dev_init(ggml_backend_dev_t, const char *) { return nullptr; }
ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t) { return &mock_buft; }
bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return false; }
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t) { return false; }

bool probe(ggml_backend_reg_t) { return true; }
size_t device_count() { return 1; }
ggml_backend_dev_t device_get(size_t index) {
    if (index != 0) return nullptr;
    auto & i = mock_device.iface;
    i.get_name = dev_name; i.get_description = dev_description; i.get_memory = dev_memory;
    i.get_type = dev_type; i.get_props = dev_props; i.init_backend = dev_init;
    i.get_buffer_type = dev_buft; i.supports_op = dev_supports_op; i.supports_buft = dev_supports_buft;
    return &mock_device;
}
bool device_identity(size_t index, flagos_device_identity * identity) {
    if (index != 0 || identity == nullptr) return false;
    *identity = { 0x4d4f434b, 0x4d4f434b, 1, 0x4d4f434b, 0 };
    return true;
}
bool device_caps(size_t index, flagos_device_caps * caps) {
    if (index != 0 || caps == nullptr) return false;
    *caps = { flagos_provider_kind::other, FLAGOS_MEMORY_DEVICE_LOCAL, 0, nullptr };
    return true;
}
bool is_backend(ggml_backend_t) { return false; }
int score() { return 1; }
bool device_profile(size_t index, flagos_device_profile * profile) {
    if (index != 0 || profile == nullptr) return false;
    *profile = {};
    profile->struct_size = sizeof(flagos_device_profile);
    profile->vendor = "mock";
    profile->architecture = "mock";
    profile->target = "mock";
    profile->engine = flagos_engine_kind::other;
    profile->lane_count = 1;
    return true;
}

} // namespace

const flagos_provider_v1 * flagos_mock_device_provider() {
    static const flagos_provider_v1 provider = {
        FLAGOS_PROVIDER_API_VERSION,
        sizeof(flagos_provider_v1),
        { 0x4d4f434b, "mock-device", 1 },
        probe, device_count, device_get, device_identity, device_caps, is_backend,
        nullptr, nullptr, score, device_profile,
    };
    return &provider;
}

FLAGOS_DEFINE_DEVICE_PROVIDER_ENTRY("mock-device", flagos_mock_device_provider)
