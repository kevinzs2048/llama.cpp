# SpaceMiT native provider (experimental)

This provider embeds the independent `ggml-spacemit` execution implementation in the FlagOS registry. It does not port the older `ggml-cpu/spacemit` path or translate SpaceMiT operators into AMD/Denglin operators.

```text
llama.cpp / GGML scheduler
          |
     FlagOS registry
          |
     spacemit provider             CPU backend (fallback)
          |                              ^
     native SPACEMIT0 device              |
     supports_op / buffer / graph_compute |
          |                              |
     SpaceMiT kernels                    |
     RVV + IME1/IME2 + scalar helpers ----+
          |
     spine-runtime (spert)
          |
     RISC-V AI compute cores
```

The scheduler decides fallback before execution. The horizontal connection above represents the native implementation's link dependency on CPU helper functions, not a fallback after failed execution.

llama.cpp's model loader also probes `supports_op` for each weight when choosing its buffer type. A Q4_K weight rejected by this provider is normally loaded into a CPU buffer immediately; it is not copied in full on every token. Mixed execution can still incur activation copies and short split launches, so Q4_K_M performance must be measured separately from Q4_0 performance.

## Ownership and interface

| Layer | Responsibilities |
| --- | --- |
| GGML | Graph semantics, backend placement, split boundaries, allocation and copies |
| FlagOS Common | Provider registration, identity, device selection and capability metadata |
| SpaceMiT adapter | Runtime probe, one native device, `cpu_accelerator` identity, RISC-V profile and canonical buffer transfers |
| Native SpaceMiT implementation | `supports_op`, buffer layout/repack, workspace, graph execution and kernel selection |
| spine-runtime | Compute-core allocation, launch, barriers and completion |

`flagos_spacemit_provider()` implements the existing `flagos_provider_v1`; no Common ABI extension is needed. The native device pointer is retained, with a small support guard described below. Common changes its owning registry to FlagOS, as for other providers. The buffer type continues to refer to that same device. The embedded native registry is not registered separately with GGML.

- Provider name: `spacemit`; native device name: `SPACEMIT0`.
- GGML device type: `ACCEL`; FlagOS kind: `cpu_accelerator`; profile engine: `cpu` with SIMD and matrix features.
- Runtime vector length is reported in bytes by spert and converted to bits for FlagOS.
- The current native runtime exposes one execution domain. Its identity is a provider-local identifier, not a chip serial number; no cross-provider shared-memory domain is claimed.
- Capability metadata does not replace native per-tensor `supports_op` checks.
- Provider-local legality guards reject nonpositive/NaN NORM/RMSNorm epsilon and strided elements, SCALE with bias, SOFT_MAX with mask or sinks, FLASH_ATTN_EXT without a mask or with nonzero max bias, Q4_1/Q4_K/Q5_1/Q5_K/IQ2_S/IQ3_XXS matrix multiplication, non-32-row Q6_K/Q8_0 matrix multiplication, batched quantized MUL_MAT weights, dense F16 MUL_MAT, and the IME2 Q5_0 packed path whose native trait has no dispatch branch. Native F16 KV-cache matrix multiplication is rejected for views, including contiguous V views, and matrix multiplication with any strided left operand is rejected. Additional guards reject F16/BF16 CONT, a faulty CPY transpose branch, unsupported PAD variants, DIAG and SOLVE_TRI. These guards run before GGML placement; other supported variants delegate to the native implementation.
- Although memory is host-visible, the buffer is not a general GGML host buffer: `tensor->data` for a packed weight points to kernel storage. Transfer callbacks expose its separate canonical GGML payload.
- Execution is synchronous from GGML's perspective: the native `graph_compute` waits on its runtime future. Async/event/native-graph capabilities are not advertised.
- Multiple backend handles for this device share one registry-owned native session. Its stream mutex serializes `graph_compute` calls, including separate GGML graphs launched from different host threads; this is safe but does not provide concurrent hardware execution. Concurrent input-buffer mutation during a graph is outside the contract.
- Runtime unavailability or a runtime architecture without a compiled IME implementation yields zero provider devices. A compute failure propagates without rerouting partially written tensors.
- The native implementation currently initializes its environment at library load. Adapter probing does not make that initialization lazy.

## Canonical transfers and packed storage

`native-buffer.h` replaces the native buffer callbacks inside the source adapter. It retains the vendor memory pool, repack routine and kernels. Common is unchanged.

```text
One GGML allocation for a packed root tensor:
[ native packed bytes ][ alignment padding ][ original GGML bytes ]
^ tensor->data                              ^ transfer callbacks
```

`get_alloc_size` accounts for both payloads. There is no unaccounted host weight cache. Plain tensors and CPU-only Q4_1/Q4_K/Q5_1/Q5_K/IQ2_S/IQ3_XXS weights, non-32-row Q6_K/Q8_0 weights, plus IME2 Q5_0 weights with row counts divisible by 32, use only canonical storage. Some native repacks convert quantization parameters (Q4_1 converts its floating offset to an integer zero point); reversing the packed byte order cannot recover the original tensor exactly.

- `set_tensor` / `memset_tensor`: update canonical bytes, then repack the root. Partial writes are supported, including writes through views.
- `get_tensor`: return canonical bytes, including partial reads and accumulated offsets of nested views.
- CPU/provider and provider/provider copies use GGML's existing get/set path.
- `clear`: clear storage and refresh registered packed roots. Buffer metadata owns descriptor copies; it does not dereference destroyed GGML contexts. Reinitialization removes overlapping records; `reset` discards metadata.
- Native compute treats packed weights as read-only, whole contiguous inputs to `MUL_MAT` / `MUL_MAT_ID`. Quantized outputs and quantized input views are rejected before placement. CPU fallback for a quantized view therefore receives canonical bytes, not a physical slice of the packed tensor.
- Q4_1, Q4_K, Q5_1 and Q5_K matrix multiplication currently falls back to CPU. The native repacks approximate floating offsets/minima with integer zero points. K3 large decode and prefill tests exceeded the GGML numerical threshold for these formats. These weights are not repacked or duplicated in provider storage while no native kernel may consume them. The scheduler fallback is covered by byte-exact canonical transfer and numerical reference checks, including `2048 × 2048` weights.
- Buffer callbacks serialize metadata and host transfers. As with GGML's synchronous transfer API, callers must finish computation before changing its inputs; the adapter does not promise concurrent mutation during kernel execution.

Cost: each packed weight retains an additional `ggml_nbytes(weight)` bytes plus alignment padding. Partial writes repack the entire root. This correctness baseline has not been optimized for memory use or weight-update throughput. Future lossless unpacking or selective retention must preserve exact transfers for all supported formats and aliases.

## Source and build contract

The adapter compiles vendor sources against this tree's GGML headers. It does not load an arbitrary vendor GGML binary: matching GGML API version numbers alone do not establish binary compatibility.

Pinned source: [spacemit-com/llama.cpp](https://github.com/spacemit-com/llama.cpp/tree/4e782bcb7e43bc73cf67ab0210a5490369e06a40/ggml/src/ggml-spacemit), commit `4e782bcb7e43bc73cf67ab0210a5490369e06a40`, from `agent/ggml-spacemit-backend`. `source-lock.cmake` checks the compiled vendor sources and local headers. Updating this dependency requires reviewing and updating that lock. The vendor checkout stays unmodified; its kernels are not duplicated in this repository.

Required:

- RISC-V Linux; a SpaceMiT compiler accepting the vendor RVV/IME instruction set.
- A full checkout of the pinned vendor source, supplied as `FLAGOS_SPACEMIT_SOURCE_DIR`.
- Matching spert headers and library in `SPERT_DIR`. Cross-build verification uses the official spine-runtime 0.6.3 SDK and SpaceMiT GCC toolchain v1.2.4 (GCC 15.2).
- `GGML_CPU=ON`, `GGML_BACKEND_DL=OFF`. The native scalar path uses `ggml_get_type_traits_cpu` and its memory query uses `ggml_backend_cpu_reg`. Dynamic CPU plugins require separate dependency work.
- `GGML_CPU_RISCV64_SPACEMIT=OFF` and `GGML_SPACEMIT=OFF`: those implementations define overlapping native symbols.

SpaceMiT-only compiler options, includes and IME definitions are scoped to `flagos-spacemit-native`, an object target embedded in `ggml-flagos`. Other providers do not inherit them. Normal shared-library builds are supported; `GGML_BACKEND_DL=OFF` means built-in backend registration, not necessarily static libraries.

Example cross-build, from this repository root:

```sh
export RISCV_ROOT_PATH=/path/to/spacemit-toolchain-linux-glibc-x86_64-v1.2.4
cmake -S . -B build-spacemit \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/spacemit-llama/cmake/riscv64-spacemit-linux-gnu-gcc.cmake \
  -DGGML_FLAGOS=ON -DGGML_FLAGOS_DENGLIN=OFF -DGGML_FLAGOS_SPACEMIT=ON \
  -DGGML_CPU=ON -DGGML_CPU_RISCV64_SPACEMIT=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_NATIVE=OFF \
  -DFLAGOS_SPACEMIT_SOURCE_DIR=/path/to/spacemit-llama \
  -DSPERT_DIR=/path/to/spine-runtime.riscv64.0.6.3 \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_APP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-spacemit --target llama-bench llama-completion test-backend-ops -j8
```

Install the matching spert runtime on the board or provide its `lib` directory through the target dynamic loader's search path. Build-time linker search paths do not deploy the runtime.

Use `--device SPACEMIT0` in llama tools. The existing native variables `GGML_SPACEMIT_WORKERS` and `SPACEMIT_MEM_BACKEND` retain their upstream meaning. Runtime configuration must be set before process startup.

For a CPU-only run from the same binary, set `GGML_FLAGOS_SPACEMIT_DISABLE=1` before startup and use `-dev none -ngl 0`. This provider then registers zero devices. The explicit opt-out matters because llama.cpp also considers ACCEL buffer types while selecting CPU weight buffers, even when `-ngl 0` is requested. The host adapter test covers the disabled state; K3 device enumeration confirms that no SpaceMiT device remains visible.

## Validation and outstanding native issues

`flagos-check-spacemit-adapter` runs on the host with a native-interface test double. It checks device adoption, provider selection, capability metadata, buffer/support delegation, support guards, compute-error propagation, no-device handling and invalid native device counts. It does not execute RISC-V kernels.

```sh
cmake --build build-host --target flagos-check-spacemit-adapter
build-host/bin/flagos-check-spacemit-adapter
build-host/bin/flagos-check-spacemit-adapter unavailable
build-host/bin/flagos-check-spacemit-adapter invalid-registry
build-host/bin/flagos-check-spacemit-adapter disabled
```

This is an opt-in integration, not a claim of complete board correctness or performance parity. The following boundaries remain before production use:

1. **Packed-weight transfers:** the previous Q4_0 CPU-reference failure is fixed by the canonical storage described above. Q4_0 numerical comparisons now pass under QEMU. Byte-exact transfers for other quantized formats do not establish their kernel numerical accuracy; board/model validation remains necessary.
2. **Fusion:** this revision contains `try_fuse_node`, but its graph executor does not call that helper. Do not report the unused RMSNorm/QKV/FFN pattern helpers as active graph fusion. Operator-internal specializations remain native implementation details.
3. **Scheduling and numerics:** K3 checks now cover selected operators, quantized CPU-fallback formats, dense F16 and F16 KV-cache matmul fallback, a 15M Q4_0 model and a diagnostic 15M Q4_K requantization. On the latter, `-fa off` logits match CPU exactly after the F16 view guard; with FA enabled, accepted per-op numerical differences amplify through the small model and can flip a close greedy choice. A broader operator scan found additional native support misreports: SCALE bias, NORM epsilon/strides, F16/BF16 CONT, strided L2_NORM, left/circular PAD, batched DIAG, CPY's transpose branch and SOLVE_TRI. The full MUL_MAT suite initially failed 114/545 accepted cases, concentrated in F16, IQ2_S/IQ3_XXS and non-32-row Q6_K/Q8_0; after the guards, all 414 still accepted cases pass. The first all-operator K3 run exposed 10 failed batched quantized MUL_MAT graphs: native compute flattens activation batches but reads only one packed weight matrix. After rejecting that shape, all 3145 still accepted all-operator graphs and all 66 focused fusion graphs pass; a separate scheduler test confirms canonical CPU fallback for a 4D Q4_0 weight. Two backend handles concurrently launched separate Q4_0 graphs on K3 and matched CPU references. A 1B+ model, perplexity and more batch/KV/cache paths remain to be checked. Restoring rejected acceleration needs an accurate native implementation.
4. **Hardware coverage:** runtime architecture labels describe compiled paths, not validation of all K1/K3 configurations. K3 is the first physical board checked; QEMU checks cannot establish throughput or device-memory behavior. Build SDK 0.6.3 and K3 runtime 0.6.2 should be aligned for a release configuration.

The adapter leaves vendor operator implementations in vendor code. Buffer corrections are explicit local changes in `native-buffer.h`, covered by `flagos-check-spacemit-buffer`; they do not modify the pinned external checkout or add vendor layout handling to Common.

Validation performed for this integration:

| Check | Result |
| --- | --- |
| SpaceMiT GCC 15.2 + spert 0.6.3 cross-build | Native IME1/IME2 objects, `llama-bench`, `llama-completion`, `test-backend-ops` built and linked |
| Host adapter tests | Available, unavailable and invalid-registry cases passed |
| Existing Common tests | Provider ABI, registry and target checks passed |
| Invalid host architecture / changed source lock | Configuration rejected as intended |
| Vendor QEMU, K3 simulation, VLEN 1024, one worker | `SPACEMIT0` enumerated; 5 ADD and 9 positive-epsilon RMSNorm cases passed against CPU; 2 zero-epsilon cases rejected by support guard |
| QEMU Q4_0 matrix multiplication | Previous `m=2880,n=32,k=2880` failure passes with one and two workers; small k=256 and k=288 cases pass after partial writes |
| QEMU canonical buffer tests | 10 quantization types, 13 shape/type cases: exact readback, partial writes, CPU/provider and provider/provider copies, view mutation, clear, descriptor lifetime and address reuse |
| QEMU mixed scheduling | Quantized weight views rejected by provider, placed on CPU, and equal to independently constructed CPU reference |
| Physical K3 board | ADD 75/75, RMS_NORM 42/42, ROPE 401/401, supported SOFT_MAX 21/21, full supported FLASH_ATTN_EXT 1187/1187; Q6_K large matrix 1/1 |
| K3 canonical transfer and fallback | 18 quantized shape/type cases pass, including nested views and Q4_1/Q4_K/Q5_1/Q5_K `2048 × 2048` weights stored without repack; their matrix multiplication is placed on CPU and exactly matches a CPU reference built from canonical weights |
| K3 shared session | Two backend handles launched independent Q4_0 graphs from separate host threads three times each; both outputs met the GGML CPU numerical threshold. Native compute is serialized by the registry-owned session mutex. |
| K3 all-operator suite and batched quantized MUL_MAT | First run: 3277/3287 accepted graphs passed; 10 failures were 4D Q4_0/Q8_0 weights in `MUL_MAT_VEC_FUSION`. Rejecting quantized `MUL_MAT` when either weight batch dimension exceeds one yields 3145/3145 accepted graphs across the full suite and 66/66 focused fusion graphs. The guard declines 142 cases, including 132 previously passing ones, because the native code does not read the corresponding batched weights. `MUL_MAT_ID` remains a distinct accepted path. A 4D Q4_0 weight also produces an exact CPU reference through the scheduler fallback. |
| K3 precision boundary | Q4_1 and Q4_K large decode matrices each fail 0/2 when directly executed through the provider; an independent pinned native build confirms the same failure class. Q5_1 and Q5_K each fail 0/2 large decode/prefill cases before the guard. The four formats are now rejected before placement. Q2_K/Q3_K/Q4_0/Q6_K/Q8_0 corresponding large cases pass, 10/10 after guarding. Full community MUL_MAT suite: 431/545 accepted cases passed before the later F16/IQ/partial-tile guards; 414/414 accepted cases now pass. |
| K3 extended operator scan | After support guards: SCALE 1/1, NORM 40/40, CONT 16/16, L2_NORM 20/20, PAD 16/16, CPY 59/59, SOLVE_TRI 0/0 and DIAG 0/0 (declined), MUL_MAT_ID 158/158, seven supported unary operations 14/14. SUB/MUL/DIV/GLU/GET_ROWS/CONCAT/REPEAT/SUM_ROWS/FILL/CUMSUM/TRI/SET/SET_ROWS/SSM_CONV/GATED_DELTA_NET also pass their supported selected cases. This is a test-suite coverage statement, not proof for untested shapes. |
| K3 whole model and performance | 15M Q4_0 completion matches pure CPU for a fixed 32-token prompt; layers 0–6 assigned to SPACEMIT0. Toy-model pp64/tg32 baseline recorded; 1B+ validation remains pending |

Run the native buffer regression on a board (or the vendor QEMU configured for K3 with VLEN 1024):

```sh
cmake --build build-spacemit --target flagos-check-spacemit-buffer
build-spacemit/bin/flagos-check-spacemit-buffer
```

It covers Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K and Q8_0. These test shapes target the K3/IME2 implementation; they are not a K1 acceptance suite.
