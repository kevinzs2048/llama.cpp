#!/usr/bin/env bash
# gpu-run.sh <log> <cmd...>: run one GPU command with the amd-rt plugin of
# BUILD_MODULES and package PKG under a timeout (TMO, default 1800 s), then
# report new amdgpu error lines in dmesg (fault, MES, timeout, reset, hang).
# Run one GPU job at a time; a wedged MES ring needs a reboot.
set -u
. "$(dirname "$0")/common.sh"
require PKG
export GGML_FLAGOS_PROVIDER_PATH=$BUILD_MODULES/flagos-providers-optional
export GGML_FLAGOS_KERNEL_DIR=$PKG
pat='amdgpu.*(MES failed|fault|timeout|reset|hang|ring .* test failed)'
b=$(sudo -n dmesg 2>/dev/null | grep -ciE "$pat")
log=$1
shift
timeout "${TMO:-1800}" "$@" >"$log" 2>&1
rc=$?
a=$(sudo -n dmesg 2>/dev/null | grep -ciE "$pat")
echo "rc=$rc dmesg_new=$((a - b))"
[ "$a" -gt "$b" ] && sudo -n dmesg | grep -iE "$pat" | tail -3
exit $rc
