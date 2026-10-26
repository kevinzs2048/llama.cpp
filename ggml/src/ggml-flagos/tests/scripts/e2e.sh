#!/usr/bin/env bash
# End-to-end model checks of the FlagOS core runtime device without a GPU:
#  1. hostsim reference runtime (BUILD_FW);
#  2. the real AMD runtime plugin with a real package, HIP replaced by the
#     host emulator (BUILD_MODULES, LD_PRELOAD libflagos-hip-hostemu.so).
#     Without LD_PRELOAD the same command runs on hardware.
# Text runs must reproduce the CPU text. The MoE runs use tiny random weights
# (flat logits, so greedy text is not usable) and compare KL divergence with
# the CPU. No run may open the GPU device nodes.
#
#   FLAGOS_FIXTURE_PKG=<package>  FLAGOS_E2E_MODEL=<Qwen3.5-4B Q4_0 gguf>
#   FLAGOS_TINY_DIR=<dir with tiny-qwen35moe-{Q4_0,Q4_K_M}.gguf and kld-text.txt>
#   [FLAGOS_RELEASE_PKG=<package to release>]  e2e.sh
#
# The optional release lane runs the tiny MoE model with the release package
# through the emulator (ubatch 1 and 512) and fails if a launched kernel has
# no host reference (tests/plugins/reference_kernels.h): every kernel name a
# package can launch needs one.
set -u
. "$(dirname "$0")/common.sh"
require FLAGOS_FIXTURE_PKG FLAGOS_E2E_MODEL FLAGOS_TINY_DIR
PKG=$FLAGOS_FIXTURE_PKG
L=$LOGS/e2e
mkdir -p "$L"
configure_fw
configure_modules
build "$BUILD_FW" "$L/build-fw.log" llama-completion llama-perplexity flagos-provider-hostsim
build "$BUILD_MODULES" "$L/build-modules.log" llama-completion llama-perplexity flagos-provider-amd-rt flagos-hip-hostemu
EMU=(LD_PRELOAD="$BUILD_MODULES/bin/libflagos-hip-hostemu.so" GGML_FLAGOS_PROVIDER_PATH="$BUILD_MODULES/flagos-providers-optional")
PROMPT=${PROMPT:-"The capital of France is"}
common=(-m "$FLAGOS_E2E_MODEL" -p "$PROMPT" -n 8 --temp 0 -t 8 --no-warmup -no-cnv)
fail=0
"$BUILD_FW/bin/llama-completion" "${common[@]}" -dev none -ngl 0 2>"$L/cpu.err" >"$L/cpu.out"

run() {  # name device env...: text must match the CPU
    local name=$1 dev=$2
    shift 2
    local trace="${L:?}/${name:?}.strace"
    env "$@" GGML_FLAGOS_STATS=1 timeout 2400 strace -f -qq -e trace=openat -o "$trace" \
        "$BIN" "${common[@]}" -dev "$dev" -ngl 99 2>"$L/$name.err" >"$L/$name.out"
    local same=no
    cmp -s "$L/cpu.out" "$L/$name.out" && same=yes
    local gpu graphs
    gpu=$(grep -cE "/dev/kfd|/dev/dri" "$trace")
    graphs=$(grep "kernel launches" "$L/$name.err" | grep -v " 0 kernel launches" | head -1 |
             sed -E "s/.*: ([0-9]+) graphs, ([0-9]+) kernel launches.*/\1 graphs \2 launches/")
    echo "E2E $name: text_matches_cpu=$same gpu_opens=$gpu ($graphs)"
    [ "$same" = yes ] && [ "$gpu" = 0 ] || fail=1
    rm -f "$trace"
}
BIN=$BUILD_FW/bin/llama-completion
run hostsim FlagOS-hostsim0 GGML_FLAGOS_PROVIDER_PATH="$BUILD_FW/flagos-hostsim"
BIN=$BUILD_MODULES/bin/llama-completion
run amd-rt-hipemu FlagOS-amd-rt0 "${EMU[@]}" FLAGOS_HIPEMU_PACKAGE="$PKG" GGML_FLAGOS_KERNEL_DIR="$PKG"

T=$FLAGOS_TINY_DIR/kld-text.txt
# Random weights give nearly flat logits, which amplify the CPU's own
# activation quantization: even the hostsim runtime (F32 host references)
# differs from the CPU by KLD ~0.02 here, while real models stay below 0.01.
KLD_MAX=${FLAGOS_TINY_KLD_MAX:-0.05}
declare -A BASE_DONE
kld() {  # name device quant env...: KLD vs CPU < FLAGOS_TINY_KLD_MAX; KLD_UB selects the ubatch (default 512),
          # PPL the llama-perplexity binary (default BUILD_MODULES)
    local name=$1 dev=$2 q=$3
    shift 3
    local M=$FLAGOS_TINY_DIR/tiny-qwen35moe-$q.gguf B=$L/tiny-$q-base.kld
    local trace="${L:?}/${name:?}.strace"
    # The CPU baseline is regenerated on every run so it always matches the current CPU backend.
    [ -n "${BASE_DONE[$q]:-}" ] || "$BUILD_FW/bin/llama-perplexity" -m "$M" -f "$T" -c 128 --chunks 4 -t 8 -dev none -ngl 0 \
        --kl-divergence-base "$B" >/dev/null 2>&1
    BASE_DONE[$q]=1
    env "$@" timeout 2400 strace -f -qq -e trace=openat -o "$trace" \
        "${PPL:-$BUILD_MODULES/bin/llama-perplexity}" -m "$M" -f "$T" -c 128 --chunks 4 -t 8 -dev "$dev" -ngl 99 \
        --kl-divergence-base "$B" --kl-divergence -ub "${KLD_UB:-512}" >"$L/$name.out" 2>&1
    local k top gpu missing
    k=$(grep "Mean    KLD" "$L/$name.out" | awk '{print $3}')
    top=$(grep "^Same top p:" "$L/$name.out" | awk '{print $4}')
    gpu=$(grep -cE "/dev/kfd|/dev/dri" "$trace")
    missing=$(grep -c "no host reference" "$L/$name.out")
    echo "E2E $name: mean_kld=$k same_top1=$top% gpu_opens=$gpu missing_refs=$missing"
    awk -v k="$k" -v m="$KLD_MAX" 'BEGIN { exit !(k != "" && k < m) }' && [ "$gpu" = 0 ] && [ "$missing" = 0 ] || fail=1
    rm -f "$trace"
}
PPL=$BUILD_FW/bin/llama-perplexity kld moe-hostsim-Q4_K_M FlagOS-hostsim0 Q4_K_M GGML_FLAGOS_PROVIDER_PATH="$BUILD_FW/flagos-hostsim"
kld moe-amd-rt-hipemu-Q4_0 FlagOS-amd-rt0 Q4_0 "${EMU[@]}" FLAGOS_HIPEMU_PACKAGE="$PKG" FLAGOS_HIPEMU_QUIET=1 GGML_FLAGOS_KERNEL_DIR="$PKG"
if [ -n "${FLAGOS_RELEASE_PKG:-}" ]; then
    for ub in 1 512; do
        KLD_UB=$ub kld "moe-amd-rt-hipemu-release-ub$ub" FlagOS-amd-rt0 Q4_0 "${EMU[@]}" \
            FLAGOS_HIPEMU_PACKAGE="$FLAGOS_RELEASE_PKG" GGML_FLAGOS_KERNEL_DIR="$FLAGOS_RELEASE_PKG"
    done
fi
exit $fail
