# Shared settings for the FlagOS test scripts (sourced, not run).
# Every path can be overridden from the environment; see README.md.
FLAGOS_REPO=${FLAGOS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
FLAGOS_WORK=${FLAGOS_WORK:-$FLAGOS_REPO/build-flagos-tests}
BUILD_HOST=${BUILD_HOST:-$FLAGOS_WORK/build-host}          # Common + AMD contract checks, HIP mocked
BUILD_FW=${BUILD_FW:-$FLAGOS_WORK/build-fw}                # FlagOS core + hostsim runtime, no AMD
BUILD_MODULES=${BUILD_MODULES:-$FLAGOS_WORK/build-modules} # provider modules: amd-rt plugin, HIP host emulator
LOGS=${LOGS:-$FLAGOS_WORK/logs}
JOBS=${JOBS:-16}

require() {  # require VAR...: stop with a hint when a required variable is unset
    local v
    for v; do
        if [ -z "${!v:-}" ]; then
            echo "error: set $v (see $(dirname "${BASH_SOURCE[0]}")/README.md)" >&2
            exit 2
        fi
    done
}

# configure <build-dir> <log> <cmake -D options...>: configure once.
configure() {
    local dir=$1 log=$2
    shift 2
    [ -f "$dir/build.ninja" ] && return 0
    mkdir -p "$(dirname "$log")"
    cmake -S "$FLAGOS_REPO" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_FLAGOS=ON \
        -DGGML_FLAGOS_SPACEMIT=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF "$@" >"$log" 2>&1 ||
        { echo "CONFIGURE_FAIL $dir (log: $log)"; tail -20 "$log"; exit 2; }
}

configure_fw()      { configure "$BUILD_FW" "$LOGS/configure-fw.log" -DGGML_FLAGOS_AMD=OFF -DLLAMA_BUILD_EXAMPLES=OFF; }
configure_modules() { configure "$BUILD_MODULES" "$LOGS/configure-modules.log" -DGGML_FLAGOS_AMD=ON -DGGML_FLAGOS_PROVIDER_MODULES=ON -DLLAMA_BUILD_EXAMPLES=ON; }
configure_host()    { configure "$BUILD_HOST" "$LOGS/configure-host.log" -DGGML_FLAGOS_AMD=ON -DLLAMA_BUILD_EXAMPLES=OFF; }

# build <build-dir> <log> <targets...>
build() {
    local dir=$1 log=$2
    shift 2
    mkdir -p "$(dirname "$log")"
    cmake --build "$dir" --target "$@" -j "$JOBS" >"$log" 2>&1 ||
        { echo "BUILD_FAIL $dir (log: $log)"; grep -m20 -E "error|Error" "$log"; exit 3; }
}
