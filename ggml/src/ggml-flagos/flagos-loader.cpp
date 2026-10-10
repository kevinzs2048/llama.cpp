#include "flagos-loader.h"

#include "flagos-provider.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <set>

#if defined(_WIN32)
#    error "FlagOS provider plugins are not implemented for Windows yet"
#else
#    include <dlfcn.h>
#endif

namespace fs = std::filesystem;

static const char * const FLAGOS_PLUGIN_PREFIX = "libflagos-provider-";
static const char * const FLAGOS_PLUGIN_SUFFIX = ".so";

static void flagos_host_log(flagos_log_level level, const char * provider, const char * message) {
    const char * who = provider ? provider : "provider";
    switch (level) {
        case FLAGOS_LOG_DEBUG: GGML_LOG_DEBUG("FlagOS %s: %s\n", who, message); break;
        case FLAGOS_LOG_INFO:  GGML_LOG_INFO("FlagOS %s: %s\n", who, message); break;
        case FLAGOS_LOG_WARN:  GGML_LOG_WARN("FlagOS %s: %s\n", who, message); break;
        default:               GGML_LOG_ERROR("FlagOS %s: %s\n", who, message); break;
    }
}

static std::string flagos_core_module_dir() {
    Dl_info info {};
    if (dladdr(reinterpret_cast<void *>(&flagos_loader_search_paths), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    std::error_code ec;
    fs::path path = fs::weakly_canonical(fs::path(info.dli_fname), ec);
    if (ec) {
        path = fs::path(info.dli_fname);
    }
    return path.parent_path().string();
}

std::vector<std::string> flagos_loader_search_paths() {
    std::vector<std::string> paths;
    if (const char * env = std::getenv("GGML_FLAGOS_PROVIDER_PATH")) {
        std::string list(env);
        size_t start = 0;
        while (start <= list.size()) {
            const size_t end = std::min(list.find(':', start), list.size());
            if (end > start) {
                paths.push_back(list.substr(start, end - start));
            }
            start = end + 1;
        }
        return paths;
    }
    const std::string dir = flagos_core_module_dir();
    if (!dir.empty()) {
        paths.push_back((fs::path(dir) / "flagos-providers").string());
    }
    return paths;
}

static bool flagos_is_plugin_file(const fs::path & path) {
    const std::string name = path.filename().string();
    const size_t prefix = std::strlen(FLAGOS_PLUGIN_PREFIX);
    const size_t suffix = std::strlen(FLAGOS_PLUGIN_SUFFIX);
    return name.size() > prefix + suffix &&
        name.compare(0, prefix, FLAGOS_PLUGIN_PREFIX) == 0 &&
        name.compare(name.size() - suffix, suffix, FLAGOS_PLUGIN_SUFFIX) == 0;
}

bool flagos_loader_entry_is_valid(const flagos_provider_entry_v1 * entry, std::string & reason) {
    if (entry == nullptr) {
        reason = "entry point returned NULL";
        return false;
    }
    if (entry->struct_size < sizeof(flagos_provider_entry_v1)) {
        reason = "entry struct_size " + std::to_string(entry->struct_size) + " is too small";
        return false;
    }
    if (entry->abi_version != FLAGOS_PROVIDER_ABI_VERSION) {
        reason = "ABI version " + std::to_string(entry->abi_version) + " is not supported";
        return false;
    }
    if (entry->name == nullptr || entry->name[0] == '\0') {
        reason = "entry has no name";
        return false;
    }
    switch (entry->kind) {
        case FLAGOS_PROVIDER_ENTRY_DEVICE:
            if (entry->ggml_abi_tag == nullptr || std::strcmp(entry->ggml_abi_tag, flagos_ggml_abi_tag()) != 0) {
                reason = std::string("GGML ABI tag mismatch: provider '") +
                    (entry->ggml_abi_tag ? entry->ggml_abi_tag : "(none)") + "', core '" + flagos_ggml_abi_tag() + "'";
                return false;
            }
            if (!flagos_provider_is_valid(entry->device)) {
                reason = "invalid flagos_provider_v1 descriptor";
                return false;
            }
            return true;
        case FLAGOS_PROVIDER_ENTRY_RUNTIME: {
            const flagos_runtime_v1 * rt = entry->runtime;
            if (rt == nullptr || rt->struct_size < sizeof(flagos_runtime_v1) ||
                rt->abi_version != FLAGOS_PROVIDER_ABI_VERSION) {
                reason = "invalid or incompatible flagos_runtime_v1 table";
                return false;
            }
            const bool tier1 = rt->device_count && rt->device_open && rt->device_close && rt->device_props &&
                rt->device_memory && rt->buffer_alloc && rt->buffer_free && rt->buffer_base &&
                rt->stream_create && rt->stream_destroy && rt->stream_synchronize && rt->memcpy &&
                rt->memset && rt->module_load && rt->module_unload && rt->kernel_get && rt->launch;
            if (!tier1) {
                reason = "flagos_runtime_v1 is missing a required (tier 1) function";
                return false;
            }
            return true;
        }
        default:
            reason = "unknown provider kind " + std::to_string(static_cast<int>(entry->kind));
            return false;
    }
}

static void flagos_try_load(const fs::path & path, flagos_loader_result & result, std::set<std::string> & names) {
    const std::string file = path.string();
    // RTLD_LOCAL keeps each provider's SDK symbols private to it.
    void * handle = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char * error = dlerror();
        result.rejected.push_back(file + ": " + (error ? error : "dlopen failed"));
        return;
    }
    auto get_entry = reinterpret_cast<flagos_provider_get_entry_v1_fn>(dlsym(handle, FLAGOS_PROVIDER_ENTRY_SYMBOL));
    if (get_entry == nullptr) {
        result.rejected.push_back(file + ": missing " FLAGOS_PROVIDER_ENTRY_SYMBOL);
        dlclose(handle);
        return;
    }
    // The host table and its strings must outlive the provider; deque keeps
    // element addresses stable as plugins are added.
    static std::deque<std::string> provider_dirs;
    static std::deque<flagos_host_v1> hosts;
    provider_dirs.push_back(path.parent_path().string());
    hosts.push_back({ sizeof(flagos_host_v1), FLAGOS_PROVIDER_ABI_VERSION, provider_dirs.back().c_str(), flagos_host_log });
    const flagos_provider_entry_v1 * entry = get_entry(&hosts.back());
    std::string reason;
    if (!flagos_loader_entry_is_valid(entry, reason)) {
        result.rejected.push_back(file + ": " + reason);
        dlclose(handle);
        return;
    }
    if (!names.insert(entry->name).second) {
        result.rejected.push_back(file + ": provider '" + entry->name + "' is already loaded");
        dlclose(handle);
        return;
    }
    result.providers.push_back({ file, entry->name, handle, entry });
}

flagos_loader_result flagos_loader_load_all() {
    flagos_loader_result result;
    std::set<std::string> names;
    for (const std::string & entry : flagos_loader_search_paths()) {
        std::error_code ec;
        const fs::path path(entry);
        if (fs::is_regular_file(path, ec)) {
            flagos_try_load(path, result, names);
            continue;
        }
        if (!fs::is_directory(path, ec)) {
            continue;
        }
        std::vector<fs::path> files;
        for (const auto & item : fs::directory_iterator(path, fs::directory_options::skip_permission_denied, ec)) {
            if (item.is_regular_file(ec) && flagos_is_plugin_file(item.path())) {
                files.push_back(item.path());
            }
        }
        std::sort(files.begin(), files.end());  // deterministic device order
        for (const fs::path & file : files) {
            flagos_try_load(file, result, names);
        }
    }
    return result;
}
