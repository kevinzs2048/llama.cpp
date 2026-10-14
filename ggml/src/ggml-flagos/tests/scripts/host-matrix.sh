#!/usr/bin/env bash
# Host-only regression: Common + AMD contract checks (HIP linked but mocked)
# and the SpaceMiT adapter host double. Every check runs under strace and
# fails if it opens /dev/kfd or /dev/dri.
#
#   FLAGOS_FIXTURE_PKG=<contract fixture package> host-matrix.sh
set -u
. "$(dirname "$0")/common.sh"
require FLAGOS_FIXTURE_PKG
PKG=$FLAGOS_FIXTURE_PKG
L=$LOGS/host
mkdir -p "$L"
configure_host
COMMON="flagos-check-public-header flagos-check-provider flagos-check-target flagos-check-registry flagos-check-graph-plan flagos-check-spacemit-adapter flagos-check-loader flagos-check-amd-runtime"
AMD=$(sed -n "s/.*foreach(FLAGOS_AMD_CHECK_NAME IN ITEMS \(.*\))/\1/p" "$FLAGOS_REPO/ggml/src/ggml-flagos/providers/amd/provider.cmake")
TARGETS="ggml-flagos $COMMON"
for n in $AMD; do TARGETS="$TARGETS flagos-check-amd-$n"; done
build "$BUILD_HOST" "$L/build.log" $TARGETS

pass=0; fail=0; failed=""
run() {
    local name=$1
    shift
    local trace="${L:?}/${name:?}.strace"
    if timeout 300 strace -f -qq -e trace=openat -o "$trace" "$BUILD_HOST/bin/$name" "$@" >"$L/$name.log" 2>&1; then
        if grep -qE "/dev/kfd|/dev/dri" "$trace"; then
            fail=$((fail + 1)); failed="$failed $name(touched-GPU)"
        else
            pass=$((pass + 1))
        fi
    else
        fail=$((fail + 1)); failed="$failed $name"
    fi
    rm -f "$trace"
}
for t in flagos-check-provider flagos-check-target flagos-check-registry flagos-check-graph-plan flagos-check-spacemit-adapter flagos-check-loader; do
    run "$t"
done
run flagos-check-amd-runtime "$PKG"
for n in $AMD; do
    if [ "$n" = constant-assets ]; then run "flagos-check-amd-$n" "$PKG" ggml-gelu-fp16-v1; else run "flagos-check-amd-$n" "$PKG"; fi
done
echo "HOST_MATRIX pass=$pass fail=$fail${failed:+ failed:$failed}"
[ $fail -eq 0 ]
