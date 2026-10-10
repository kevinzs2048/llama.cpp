#pragma once

// Core-owned GGML devices for RUNTIME-kind providers.
//
// The core builds one GGML device per runtime device and implements the GGML
// buffer, backend and event objects on top of flagos_runtime_v1. Kernel
// selection and argument binding come from the shared L2 op adapter
// (flagos-ops.h); the provider only allocates, copies, loads modules and
// launches.
//
// All runtime devices are exposed to the registry through one internal
// flagos_provider_v1 descriptor (the "runtime adapter"), so the registry,
// device selection and profiles work the same way for both provider kinds.

#include "flagos-provider.h"
#include "flagos-runtime.h"

#include <vector>

struct flagos_runtime_source {
    const flagos_runtime_v1 * runtime;
    const char * plugin_dir;   // directory of the plugin, may be NULL
};

static constexpr uint64_t FLAGOS_RUNTIME_ADAPTER_ID = 0x464c41474f535254ull; // "FLAGOSRT"

// Open every device of every runtime and return the adapter descriptor, or
// NULL when no runtime exposes a usable device. Safe to call once; later
// calls return the same descriptor.
const flagos_provider_v1 * flagos_runtime_adapter(const std::vector<flagos_runtime_source> & runtimes);
