// Loader and registry behaviour with provider plugins. The plugins are mock
// modules built next to this test; no device runtime is touched.
#include "flagos-loader.h"
#include "flagos-package.h"
#include "flagos-provider.h"
#include "flagos-registry.h"
#include "ggml-flagos.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static const flagos_loaded_provider * find(const flagos_loader_result & result, const char * name) {
    for (const auto & provider : result.providers) {
        if (provider.name == name) {
            return &provider;
        }
    }
    return nullptr;
}

static bool rejected_with(const flagos_loader_result & result, const char * file, const char * reason) {
    for (const std::string & line : result.rejected) {
        if (line.find(file) != std::string::npos && line.find(reason) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static void check_entry_validation() {
    std::string reason;
    CHECK(!flagos_loader_entry_is_valid(nullptr, reason));

    flagos_provider_entry_v1 entry {};
    entry.struct_size = sizeof(entry);
    entry.abi_version = FLAGOS_PROVIDER_ABI_VERSION;
    entry.kind = FLAGOS_PROVIDER_ENTRY_DEVICE;
    entry.name = "x";
    entry.ggml_abi_tag = "ggml-backend-api=0";
    CHECK(!flagos_loader_entry_is_valid(&entry, reason));
    CHECK(reason.find("GGML ABI tag mismatch") != std::string::npos);

    entry.ggml_abi_tag = flagos_ggml_abi_tag();
    entry.device = nullptr;
    CHECK(!flagos_loader_entry_is_valid(&entry, reason));

    flagos_runtime_v1 runtime {};
    entry.kind = FLAGOS_PROVIDER_ENTRY_RUNTIME;
    entry.runtime = &runtime;
    CHECK(!flagos_loader_entry_is_valid(&entry, reason));
    runtime.struct_size = sizeof(runtime);
    runtime.abi_version = FLAGOS_PROVIDER_ABI_VERSION;
    CHECK(!flagos_loader_entry_is_valid(&entry, reason));
    CHECK(reason.find("no name") != std::string::npos);
    runtime.name = "x";
    CHECK(!flagos_loader_entry_is_valid(&entry, reason));
    CHECK(reason.find("tier 1") != std::string::npos);


    entry.kind = static_cast<flagos_provider_entry_kind>(7);
    CHECK(!flagos_loader_entry_is_valid(&entry, reason));
}

static bool open_manifest(const std::string & kernels, flagos_kernel_package & pkg, std::string & error) {
    const auto dir = std::filesystem::temp_directory_path() / ("flagos-variants-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "k.bin") << "x";
    std::ofstream(dir / "manifest.json") << "{\"kernels\": [" << kernels << "]}";
    const bool ok = pkg.open(dir.string(), error);
    std::filesystem::remove_all(dir);
    return ok;
}

// Manifest variants: selection by constraints and priority, fallback to the
// base kernel, pruning, and rejection of malformed variants.
static void check_package_variants() {
    const std::string base = R"({"name": "gemv", "file": "k.bin", "argument_count": 6})";
    const std::string aligned = R"({"name": "gemv_a16", "file": "k.bin", "argument_count": 6, "contract": "gemv",
        "priority": 5, "when": {"k": {"multiple_of": 16}, "align": {"min": 16}}})";
    const std::string narrow = R"({"name": "gemv_narrow", "file": "k.bin", "contract": "gemv", "priority": 9,
        "when": {"rows": {"max": 64}}})";
    flagos_kernel_package pkg;
    std::string error;
    CHECK(open_manifest(base + "," + aligned + "," + narrow, pkg, error));
    CHECK(pkg.variant_count() == 2);
    CHECK(pkg.select("gemv", { { "k", 4096 }, { "align", 256 }, { "rows", 4096 } })->name == "gemv_a16");
    CHECK(pkg.select("gemv", { { "k", 4096 }, { "align", 256 }, { "rows", 32 } })->name == "gemv_narrow");
    CHECK(pkg.select("gemv", { { "k", 4100 }, { "align", 256 }, { "rows", 4096 } })->name == "gemv");
    // A fact the caller does not know fails the constraint (unbound data).
    CHECK(pkg.select("gemv", { { "k", 4096 }, { "rows", 4096 } })->name == "gemv");
    CHECK(pkg.select("missing", {}) == nullptr);
    pkg.retain_variants([](const flagos_kernel_info & k) { return k.name != "gemv_narrow"; });
    CHECK(pkg.variant_count() == 1);
    CHECK(pkg.select("gemv", { { "k", 4096 }, { "align", 256 }, { "rows", 32 } })->name == "gemv_a16");
    CHECK(pkg.find("gemv_narrow") != nullptr);

    const std::string bad_args = R"({"name": "v", "file": "k.bin", "argument_count": 7, "contract": "gemv"})";
    const std::string no_base = R"({"name": "v", "file": "k.bin", "contract": "nothing"})";
    const std::string chained = R"({"name": "v2", "file": "k.bin", "contract": "gemv_a16"})";
    const std::string bad_rule = R"({"name": "v", "file": "k.bin", "contract": "gemv", "when": {"k": {"multiple_of": 0}}})";
    const std::string when_alone = R"({"name": "v", "file": "k.bin", "when": {"k": {"min": 1}}})";
    for (const std::string & extra : { bad_args, no_base, chained, bad_rule, when_alone }) {
        flagos_kernel_package bad;
        CHECK(!open_manifest(base + "," + aligned + "," + extra, bad, error));
        CHECK(!bad.is_open());
    }
}

int main() {
    check_entry_validation();
    check_package_variants();

    CHECK(setenv("GGML_FLAGOS_PROVIDER_PATH", "/nonexistent/flagos:" FLAGOS_TEST_PLUGIN_DIR, 1) == 0);
    const auto paths = flagos_loader_search_paths();
    CHECK(paths.size() == 2);
    CHECK(paths[1] == FLAGOS_TEST_PLUGIN_DIR);

    const flagos_loader_result result = flagos_loader_load_all();
    CHECK(result.providers.size() == 2);
    const flagos_loaded_provider * device = find(result, "mock-device");
    const flagos_loaded_provider * runtime = find(result, "mock-runtime");
    CHECK(device != nullptr && runtime != nullptr);
    CHECK(device->entry->kind == FLAGOS_PROVIDER_ENTRY_DEVICE);
    CHECK(runtime->entry->kind == FLAGOS_PROVIDER_ENTRY_RUNTIME);
    CHECK(rejected_with(result, "libflagos-provider-mock-bad.so", "ABI version 99"));
    // Files without the plugin prefix are ignored rather than rejected.
    for (const std::string & line : result.rejected) {
        CHECK(line.find("not-a-provider") == std::string::npos);
    }

    uint32_t count = 0;
    CHECK(runtime->entry->runtime->device_count(&count) == FLAGOS_RT_OK && count == 1);

    // A runtime table from an older build that ends right after the Tier-1
    // functions is accepted; the optional tail reads as absent.
    {
        flagos_runtime_v1 older = *runtime->entry->runtime;
        older.struct_size = FLAGOS_RUNTIME_V1_MIN_SIZE;
        flagos_provider_entry_v1 older_entry = *runtime->entry;
        older_entry.runtime = &older;
        std::string why;
        CHECK(flagos_loader_entry_is_valid(&older_entry, why));
        CHECK(!FLAGOS_RT_HAS(flagos_runtime_v1, &older, event_create));
        older.struct_size = FLAGOS_RUNTIME_V1_MIN_SIZE - 1;
        CHECK(!flagos_loader_entry_is_valid(&older_entry, why));
    }

    // The registry exposes DEVICE-kind plugin devices directly and gives each
    // RUNTIME-kind device a core-owned GGML device. The mock runtime has no
    // kernel package, so its device supports only view operations.
    ggml_backend_reg_t reg = ggml_backend_flagos_reg();
    CHECK(ggml_backend_reg_dev_count(reg) == 2);
    ggml_backend_dev_t mock_dev = nullptr;
    ggml_backend_dev_t runtime_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        CHECK(ggml_backend_dev_backend_reg(dev) == reg);
        flagos_device_info info {};
        info.struct_size = sizeof(info);
        CHECK(flagos_registry_get_device_info(i, &info));
        if (std::strcmp(ggml_backend_dev_name(dev), "MOCKDEV") == 0) {
            CHECK(std::strcmp(info.provider.name, "mock-device") == 0);
            mock_dev = dev;
        } else if (std::strcmp(ggml_backend_dev_name(dev), "FlagOS-mock-runtime0") == 0) {
            CHECK(std::strcmp(info.provider.name, "flagos-runtime") == 0);
            CHECK(info.profile.target != nullptr && std::strcmp(info.profile.target, "mock-target") == 0);
            runtime_dev = dev;
        }
    }
    CHECK(mock_dev != nullptr && runtime_dev != nullptr);

    // Core-owned buffer round trip through the C runtime.
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(runtime_dev);
    CHECK(buft != nullptr && ggml_backend_dev_supports_buft(runtime_dev, buft));
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 256);
    CHECK(buffer != nullptr);
    ggml_init_params params = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    CHECK(ggml_backend_tensor_alloc(buffer, t, ggml_backend_buffer_get_base(buffer)) == GGML_STATUS_SUCCESS);
    float in[16];
    float out[16] = {};
    for (int i = 0; i < 16; ++i) in[i] = 0.5f * i;
    ggml_backend_tensor_set(t, in, 0, sizeof(in));
    ggml_backend_tensor_get(t, out, 0, sizeof(out));
    CHECK(std::memcmp(in, out, sizeof(in)) == 0);
    // Without a kernel package only view operations are accepted.
    ggml_tensor * sum = ggml_add(ctx, t, t);
    CHECK(!ggml_backend_dev_supports_op(runtime_dev, sum));
    CHECK(ggml_backend_dev_supports_op(runtime_dev, ggml_reshape_2d(ctx, t, 4, 4)));
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);

    std::puts("flagos loader checks passed");
    return 0;
}
