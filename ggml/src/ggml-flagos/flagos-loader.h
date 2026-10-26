#pragma once

// Discovery and loading of FlagOS provider plugins (libflagos-provider-*.so).

#include "flagos-plugin.h"

#include <string>
#include <vector>

struct flagos_provider_v1;

struct flagos_loaded_provider {
    std::string path;   // plugin file, empty for built-ins
    std::string dir;    // directory holding the plugin
    std::string name;
    void * handle = nullptr;
    const flagos_provider_entry_v1 * entry = nullptr;
};

struct flagos_loader_result {
    std::vector<flagos_loaded_provider> providers;
    // One line per plugin that was found but rejected.
    std::vector<std::string> rejected;
};

// Directories searched, in order: each entry of GGML_FLAGOS_PROVIDER_PATH
// (':'-separated; a file entry loads that file), otherwise the
// "flagos-providers" directory next to the module that contains the core.
std::vector<std::string> flagos_loader_search_paths();

// Load every plugin found on the search paths. Called once by the registry;
// handles stay open for the life of the process.
flagos_loader_result flagos_loader_load_all();

// Validate one entry. Exposed for tests. On failure `reason` explains why.
bool flagos_loader_entry_is_valid(const flagos_provider_entry_v1 * entry, std::string & reason);
