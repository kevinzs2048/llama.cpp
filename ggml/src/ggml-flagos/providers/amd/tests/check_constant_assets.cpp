// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "../flagos-amd-aot.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static int loaded = 0, unloaded = 0;
static void check(bool value, const char * name) {
    if (!value) { std::fprintf(stderr, "CHECK_FAILED %s\n", name); std::exit(2); }
}
extern "C" hipError_t hipSetDevice(int) { return hipSuccess; }
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t * p, int) {
    *p = {}; p->warpSize = 32; p->maxThreadsPerBlock = 1024; p->sharedMemPerBlock = 65536;
    p->maxGridSize[0] = std::numeric_limits<int>::max(); p->maxGridSize[1] = p->maxGridSize[2] = 65535;
    return hipSuccess;
}
extern "C" hipError_t hipModuleLoadData(hipModule_t * m, const void *) {
    *m = reinterpret_cast<hipModule_t>(uintptr_t(++loaded)); return hipSuccess;
}
extern "C" hipError_t hipModuleUnload(hipModule_t) { ++unloaded; return hipSuccess; }
extern "C" hipError_t hipModuleGetFunction(hipFunction_t * f, hipModule_t, const char *) {
    *f = reinterpret_cast<hipFunction_t>(uintptr_t(1)); return hipSuccess;
}
int main(int argc, char ** argv) {
    check(argc >= 2 && argc <= 4, "package [asset-name|-] [valid|invalid]");
    const bool valid = argc < 4 || std::strcmp(argv[3], "invalid") != 0;
    const char * name = argc >= 3 ? argv[2] : "-";
    for (bool modules : {false, true}) {
        flagos_amd::kernel_registry registry;
        const int before = loaded;
        check(registry.initialize(argv[1], 0, "gfx1150", modules) == valid, "package acceptance");
        if (valid) {
            check(registry.size() > 0, "kernel registry");
            if (std::strcmp(name, "-") != 0) {
                const auto * asset = registry.find_asset(name);
                check(asset != nullptr && !asset->bytes.empty() && asset->sha256.size() == 64, "constant asset");
            } else {
                check(registry.find_asset("ggml-gelu-fp16-v1") == nullptr, "legacy package");
            }
            check(registry.find_asset("missing-asset") == nullptr, "unknown constant");
        } else {
            check(registry.size() == 0 && registry.find_asset("ggml-gelu-fp16-v1") == nullptr, "transaction rollback");
            check(loaded == before, "invalid asset rejected before module loading");
        }
        registry.reset();
        check(registry.size() == 0 && registry.find_asset(name) == nullptr, "reset clears assets");
        check(loaded == unloaded, "module lifecycle");
    }
    std::printf("CONSTANT_ASSETS_HOST_COMPLETE valid=%d modules=%d actual_GPU_calls=0\n", valid, loaded);
}
