// Host emulation of the HIP runtime subset the FlagOS AMD runtime uses.
//
// Loaded with LD_PRELOAD in front of a process that uses the amd-rt plugin,
// it replaces the HIP driver: memory is host memory, streams are
// synchronous, modules are identified by matching the loaded HSACO bytes to
// the AMD package manifest (SHA-256), and each launch runs the host
// reference of that FlagGems kernel once per program id. This lets a whole
// model run through the real AMD runtime code and the real AMD package
// without touching a GPU.
//
// Environment: FLAGOS_HIPEMU_PACKAGE=<AMD kernel package directory>.
// Never opens /dev/kfd; the host regression checks that with strace.

#include "flagos-package.h"
#include "reference_kernels.h"

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::mutex g_mutex;

const flagos_kernel_package & package() {
    static flagos_kernel_package pkg;
    static std::once_flag once;
    std::call_once(once, [] {
        const char * dir = std::getenv("FLAGOS_HIPEMU_PACKAGE");
        std::string error;
        if (dir == nullptr || !pkg.open(dir, error)) {
            std::fprintf(stderr, "hipemu: set FLAGOS_HIPEMU_PACKAGE to an AMD package (%s)\n", error.c_str());
            std::abort();
        }
    });
    return pkg;
}

struct emu_module {
    std::string name;   // manifest kernel whose artifact this image is
};

struct emu_function {
    std::string name;   // manifest kernel name (selects the host reference)
    const flagos_kernel_info * info;
};

struct emu_launch {
    emu_function * function;
    unsigned gx, gy, gz;
    std::vector<std::array<unsigned char, 8>> values;
};

struct emu_graph {
    std::vector<emu_launch> launches;
};

thread_local emu_graph * t_capturing = nullptr;

uint64_t g_launches = 0;

void report() {
    if (std::getenv("FLAGOS_HIPEMU_QUIET") == nullptr) {
        std::fprintf(stderr, "hipemu: %llu kernel launches emulated\n", (unsigned long long) g_launches);
    }
}

hipError_t run(emu_function * f, unsigned gx, unsigned gy, unsigned gz, void ** args) {
    for (unsigned z = 0; z < gz; ++z) {
        for (unsigned y = 0; y < gy; ++y) {
            for (unsigned x = 0; x < gx; ++x) {
                if (!flagos_ref::run(f->name, { x, y, z, uint32_t(f->info->block_size), uint32_t(f->info->tile_n), args })) {
                    std::fprintf(stderr, "hipemu: no host reference for %s\n", f->name.c_str());
                    return hipErrorInvalidValue;
                }
            }
        }
    }
    ++g_launches;
    return hipSuccess;
}

} // namespace

extern "C" {

hipError_t hipGetDeviceCount(int * count) { *count = 1; return hipSuccess; }
hipError_t hipSetDevice(int) { return hipSuccess; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t * p, int) {
    *p = {};
    std::strcpy(p->name, "FlagOS HIP host emulator");
    std::strcpy(p->gcnArchName, "gfx1150");
    p->integrated = 1;
    p->totalGlobalMem = size_t(16) << 30;
    p->warpSize = 32;
    p->maxThreadsPerBlock = 1024;
    p->sharedMemPerBlock = 65536;
    p->multiProcessorCount = 16;
    p->maxGridSize[0] = INT32_MAX;
    p->maxGridSize[1] = p->maxGridSize[2] = 65535;
    static bool registered = false;
    if (!registered) { registered = true; std::atexit(report); }
    return hipSuccess;
}
hipError_t hipMemGetInfo(size_t * free, size_t * total) { *free = *total = size_t(16) << 30; return hipSuccess; }
hipError_t hipMalloc(void ** ptr, size_t size) {
    *ptr = std::aligned_alloc(256, (size + 255) / 256 * 256);
    return *ptr ? hipSuccess : hipErrorOutOfMemory;
}
hipError_t hipFree(void * ptr) { std::free(ptr); return hipSuccess; }
hipError_t hipMemcpy(void * dst, const void * src, size_t size, hipMemcpyKind) { std::memmove(dst, src, size); return hipSuccess; }
hipError_t hipMemcpyAsync(void * dst, const void * src, size_t size, hipMemcpyKind, hipStream_t) { std::memmove(dst, src, size); return hipSuccess; }
hipError_t hipMemset(void * dst, int value, size_t size) { std::memset(dst, value, size); return hipSuccess; }
hipError_t hipMemsetAsync(void * dst, int value, size_t size, hipStream_t) { std::memset(dst, value, size); return hipSuccess; }
hipError_t hipStreamCreateWithFlags(hipStream_t * stream, unsigned int) { *stream = reinterpret_cast<hipStream_t>(new char); return hipSuccess; }
hipError_t hipStreamDestroy(hipStream_t stream) { delete reinterpret_cast<char *>(stream); return hipSuccess; }
hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
hipError_t hipEventCreateWithFlags(hipEvent_t * event, unsigned int) { *event = reinterpret_cast<hipEvent_t>(new char); return hipSuccess; }
hipError_t hipEventDestroy(hipEvent_t event) { delete reinterpret_cast<char *>(event); return hipSuccess; }
hipError_t hipEventRecord(hipEvent_t, hipStream_t) { return hipSuccess; }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int) { return hipSuccess; }
hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }

hipError_t hipModuleLoadData(hipModule_t * module, const void * image) {
    // hipModuleLoadData gets no size; amd-rt only loads package artifacts,
    // so identify the image by comparing it with each artifact's bytes.
    std::lock_guard<std::mutex> lock(g_mutex);
    static std::vector<std::pair<std::vector<uint8_t>, std::string>> artifacts;
    if (artifacts.empty()) {
        for (const flagos_kernel_info & info : package().kernels()) {
            std::vector<uint8_t> bytes;
            std::string error;
            if (package().read_artifact(info, bytes, error)) {
                artifacts.emplace_back(std::move(bytes), info.name);
            }
        }
    }
    for (const auto & [bytes, name] : artifacts) {
        if (std::memcmp(bytes.data(), image, bytes.size()) == 0) {
            *module = reinterpret_cast<hipModule_t>(new emu_module { name });
            return hipSuccess;
        }
    }
    std::fprintf(stderr, "hipemu: loaded image does not match any package artifact\n");
    return hipErrorInvalidImage;
}
hipError_t hipModuleUnload(hipModule_t module) { delete reinterpret_cast<emu_module *>(module); return hipSuccess; }
hipError_t hipModuleGetFunction(hipFunction_t * function, hipModule_t module, const char *) {
    // The module was matched to a manifest kernel name; the symbol inside
    // the artifact may differ from that name.
    const std::string & name = reinterpret_cast<emu_module *>(module)->name;
    *function = reinterpret_cast<hipFunction_t>(new emu_function { name, package().find(name) });
    return hipSuccess;
}

hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned gx, unsigned gy, unsigned gz,
                                 unsigned, unsigned, unsigned, unsigned, hipStream_t, void ** args, void **) {
    auto * f = reinterpret_cast<emu_function *>(function);
    if (t_capturing != nullptr) {
        emu_launch l { f, gx, gy, gz, {} };
        for (int i = 0; i < f->info->argument_count + 2; ++i) {
            std::array<unsigned char, 8> v {};
            std::memcpy(v.data(), args[i], v.size());
            l.values.push_back(v);
        }
        t_capturing->launches.push_back(std::move(l));
        return hipSuccess;
    }
    return run(f, gx, gy, gz, args);
}

hipError_t hipStreamBeginCapture(hipStream_t, hipStreamCaptureMode) { t_capturing = new emu_graph; return hipSuccess; }
hipError_t hipStreamEndCapture(hipStream_t, hipGraph_t * graph) {
    *graph = reinterpret_cast<hipGraph_t>(t_capturing);
    t_capturing = nullptr;
    return hipSuccess;
}
hipError_t hipGraphInstantiate(hipGraphExec_t * exec, hipGraph_t graph, hipGraphNode_t *, char *, size_t) {
    *exec = reinterpret_cast<hipGraphExec_t>(new emu_graph(*reinterpret_cast<emu_graph *>(graph)));
    return hipSuccess;
}
hipError_t hipGraphDestroy(hipGraph_t graph) { delete reinterpret_cast<emu_graph *>(graph); return hipSuccess; }
hipError_t hipGraphExecDestroy(hipGraphExec_t exec) { delete reinterpret_cast<emu_graph *>(exec); return hipSuccess; }
hipError_t hipGraphLaunch(hipGraphExec_t exec, hipStream_t) {
    for (emu_launch & l : reinterpret_cast<emu_graph *>(exec)->launches) {
        std::vector<void *> args;
        for (auto & v : l.values) args.push_back(v.data());
        const hipError_t e = run(l.function, l.gx, l.gy, l.gz, args.data());
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}

} // extern "C"
