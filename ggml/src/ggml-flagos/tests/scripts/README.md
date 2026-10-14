# FlagOS test scripts

Regression and release checks for the FlagOS backend. All scripts source
`common.sh`, configure and build what they need, and keep logs under
`$LOGS`.

| Script | Needs a GPU | What it checks |
|---|---|---|
| `host-matrix.sh` | no | Common and AMD contract checks (HIP mocked) and the SpaceMiT adapter host double; each check runs under strace and fails if it opens `/dev/kfd` or `/dev/dri`. |
| `e2e.sh` | no | Whole models through the core runtime device: the hostsim runtime, and the real amd-rt plugin with HIP replaced by the host emulator. Text must match the CPU; tiny MoE models compare KL divergence with a CPU baseline regenerated on each run. Optional release lane (see below). |
| `pkg-validate.sh` | yes | A kernel package on the device: `test-backend-ops` with variants off, with the default variant selection, and once per variant, all under `GGML_FLAGOS_GUARD`; reports tests, guard hits and kernel coverage. |
| `gpu-run.sh` | yes | Runs one command with the amd-rt plugin and a package under a timeout and reports new amdgpu errors in dmesg. |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `FLAGOS_REPO` | repository containing the scripts | source tree |
| `FLAGOS_WORK` | `$FLAGOS_REPO/build-flagos-tests` | build and log root |
| `BUILD_HOST` / `BUILD_FW` / `BUILD_MODULES` | `$FLAGOS_WORK/build-{host,fw,modules}` | build trees (configured on first use) |
| `LOGS` | `$FLAGOS_WORK/logs` | logs |
| `FLAGOS_FIXTURE_PKG` | (required by host-matrix, e2e) | contract fixture package the checks are written against |
| `FLAGOS_E2E_MODEL` | (required by e2e) | Qwen3.5-4B Q4_0 GGUF |
| `FLAGOS_TINY_DIR` | (required by e2e) | `tiny-qwen35moe-{Q4_0,Q4_K_M}.gguf` and `kld-text.txt` |
| `FLAGOS_RELEASE_PKG` | (optional, e2e) | package to release: its kernels must all have host references |
| `FLAGOS_TINY_KLD_MAX` | 0.05 | KLD limit for the tiny random-weight MoE runs (flat logits amplify the CPU's own activation quantization; hostsim with F32 references shows ~0.02) |
| `PKG` | (required by pkg-validate, gpu-run) | package under test |

Build trees: `build-host` (`GGML_FLAGOS_AMD=ON`, HIP mocked by the checks),
`build-fw` (core and hostsim runtime, `GGML_FLAGOS_AMD=OFF`) and
`build-modules` (`GGML_FLAGOS_PROVIDER_MODULES=ON`: the amd-rt plugin in
`flagos-providers-optional/` and `libflagos-hip-hostemu.so`).

## Host references

The host emulator (`providers/amd/tests/hip_hostemu.cpp`) executes kernels
with the C++ references in `tests/plugins/reference_kernels.h`. Every kernel
name that the core can launch from a package needs a reference there,
including new fused kernels and variants with a different geometry
(`tile_m` / `tile_n` / `block_size` are passed to the reference). Run `e2e.sh`
with `FLAGOS_RELEASE_PKG` before a release: it fails on "no host reference".

## Before a release

```
FLAGOS_FIXTURE_PKG=... host-matrix.sh
FLAGOS_FIXTURE_PKG=... FLAGOS_E2E_MODEL=... FLAGOS_TINY_DIR=... FLAGOS_RELEASE_PKG=<pkg> e2e.sh
PKG=<pkg> pkg-validate.sh          # on the device
```
