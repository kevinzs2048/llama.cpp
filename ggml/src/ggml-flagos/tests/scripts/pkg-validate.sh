#!/usr/bin/env bash
# Validate a FlagOS kernel package on the real device before release.
#
#   PKG=<package dir> [DEV=FlagOS-amd-rt0] [VOPS=MUL_MAT,...] pkg-validate.sh
#
# 1. base:    test-backend-ops with variants off, under GGML_FLAGOS_GUARD
#             (redzones after every output and the workspace tail).
# 2. default: the same with variants selected by priority, as in production.
# 3. one run per variant in the manifest, with only that variant allowed,
#    over the ops in VOPS (MUL_MAT-family by default).
# Every run reports test results, GUARD violations and launched kernels
# (GGML_FLAGOS_STATS "kernel:" counts). The summary lists kernels the tests
# never launched: their correctness is not established by this run.
# New amdgpu error lines in dmesg are reported at the end (needs sudo -n).
set -u
. "$(dirname "$0")/common.sh"
require PKG
DEV=${DEV:-FlagOS-amd-rt0}
VOPS=${VOPS:-MUL_MAT,MUL_MAT_ID}
OUT=${OUT:-$LOGS/pkg-validate/$(basename "$PKG")}
mkdir -p "$OUT"
configure_modules
build "$BUILD_MODULES" "$OUT/build.log" test-backend-ops flagos-provider-amd-rt
BIN=$BUILD_MODULES/bin/test-backend-ops
ENV=(GGML_FLAGOS_PROVIDER_PATH="$BUILD_MODULES/flagos-providers-optional" GGML_FLAGOS_KERNEL_DIR="$PKG"
     GGML_FLAGOS_GUARD=1 GGML_FLAGOS_STATS=1)
m0=$(sudo -n dmesg 2>/dev/null | wc -l)

run() {  # name, variant policy, [op filter]
    local name=$1 policy=$2 ops=${3:-}
    local args=(-b "$DEV")
    [ -n "$ops" ] && args+=(-o "$ops")
    env "${ENV[@]}" GGML_FLAGOS_KERNEL_VARIANTS="$policy" timeout 3600 "$BIN" test "${args[@]}" >"$OUT/$name.log" 2>&1
    local rc=$?
    local passed guard
    passed=$(grep -oE '[0-9]+/[0-9]+ tests passed' "$OUT/$name.log" | tail -1)
    guard=$(grep -c 'GUARD' "$OUT/$name.log")
    grep -oE 'kernel:[A-Za-z0-9_]+=[0-9]+' "$OUT/$name.log" | sed 's/^kernel://' | sort -u >"$OUT/$name.kernels"
    printf 'RUN %-48s rc=%d %-22s guard=%d kernels=%d\n' "$name" $rc "${passed:-no-summary}" "$guard" "$(wc -l <"$OUT/$name.kernels")"
    grep 'GUARD' "$OUT/$name.log" | sort | uniq -c | head -20
}

run base 0
run default 1
python3 - "$PKG/manifest.json" >"$OUT/variants.txt" <<'PY'
import json, sys
for k in json.load(open(sys.argv[1]))["kernels"]:
    if k.get("contract"):
        print(k["name"])
PY
while read -r v; do
    run "variant-$v" "$v" "$VOPS"
    if ! grep -q "^$v=" "$OUT/variant-$v.kernels"; then
        echo "  variant $v was not launched: no test shape satisfies its 'when' constraints"
    fi
done <"$OUT/variants.txt"

python3 - "$PKG/manifest.json" "$OUT" <<'PY'
import glob, json, os, sys
names = [k["name"] for k in json.load(open(sys.argv[1]))["kernels"]]
launched = set()
for f in glob.glob(os.path.join(sys.argv[2], "*.kernels")):
    launched |= {line.split("=")[0] for line in open(f) if line.strip()}
missing = [n for n in names if n not in launched]
print(f"COVERAGE {len(names) - len(missing)}/{len(names)} kernels launched by the tests")
with open(os.path.join(sys.argv[2], "unlaunched.txt"), "w") as f:
    f.write("\n".join(missing) + "\n")
print(f"  never launched: {len(missing)} (list in {sys.argv[2]}/unlaunched.txt)")
PY
echo "dmesg_new=$(sudo -n dmesg 2>/dev/null | tail -n +$((m0 + 1)) | grep -ciE 'amdgpu.*(fault|MES|timeout|reset|fail)')"
