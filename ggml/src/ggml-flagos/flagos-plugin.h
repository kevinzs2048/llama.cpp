#pragma once

// Helpers for FlagOS provider plugins built against GGML (DEVICE kind).

#include "flagos-runtime.h"
#include "../ggml-backend-impl.h"

#include <string>

// Identifies the GGML ABI a DEVICE provider was compiled against. Device
// providers hand GGML objects across the plugin boundary, so the core only
// accepts one whose tag matches its own.
inline const char * flagos_ggml_abi_tag() {
    static const std::string tag =
        "ggml-backend-api=" + std::to_string(GGML_BACKEND_API_VERSION) +
        ";tensor=" + std::to_string(sizeof(ggml_tensor)) +
        ";dev-iface=" + std::to_string(sizeof(ggml_backend_device_i)) +
        ";backend-iface=" + std::to_string(sizeof(ggml_backend_i)) +
        ";buft-iface=" + std::to_string(sizeof(ggml_backend_buffer_type_i)) +
        ";buffer-iface=" + std::to_string(sizeof(ggml_backend_buffer_i));
    return tag.c_str();
}

// Defines the plugin entry point for a DEVICE provider whose descriptor is
// returned by `descriptor_fn`. Only expands in a provider module build.
#ifdef GGML_FLAGOS_PROVIDER_MODULE
#    define FLAGOS_DEFINE_DEVICE_PROVIDER_ENTRY(provider_name, descriptor_fn)                       \
        extern "C" FLAGOS_PROVIDER_EXPORT const flagos_provider_entry_v1 *                           \
        flagos_provider_get_entry_v1(const flagos_host_v1 * host) {                                  \
            (void) host;                                                                             \
            static const flagos_provider_entry_v1 entry = {                                          \
                sizeof(flagos_provider_entry_v1), FLAGOS_PROVIDER_ABI_VERSION,                       \
                FLAGOS_PROVIDER_ENTRY_DEVICE, provider_name, flagos_ggml_abi_tag(), nullptr,         \
                descriptor_fn(),                                                                     \
            };                                                                                       \
            return &entry;                                                                           \
        }
#else
#    define FLAGOS_DEFINE_DEVICE_PROVIDER_ENTRY(provider_name, descriptor_fn)
#endif
