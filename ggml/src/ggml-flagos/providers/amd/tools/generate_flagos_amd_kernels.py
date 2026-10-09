#!/usr/bin/env python3
"""Build the AMD/HIP FlagOS AOT package for one installed Triton target.

The llama.cpp launcher consumes ordinary Triton AMD HSACO files plus a small
manifest.  This script intentionally keeps compilation in the FlagTree/FlagGems
Triton environment and does not make the runtime depend on Python or PyTorch.
The existing kernel definitions are reused from the common FlagOS generator;
the two attention/softmax variants below add the AMD prefill path and explicit
masked/unmasked softmax symbols so the C++ selector never passes a null mask to
a kernel specialized with ``HAS_MASK=True``.  An explicit
``FLAGOS_AMD_EMIT_FFN_FUSION=1`` opt-in also emits the Qwen-style dual-projection
prefill SwiGLU kernel and its conformance check. The independent
``FLAGOS_AMD_EMIT_Q4_FFN_DECODE=1`` experiment emits the packed-Q4_K decode
lowering for the same provider-neutral graph pattern, while
``FLAGOS_AMD_EMIT_Q40_FFN_DECODE=1`` emits the corresponding Q4_0 lowering.
Independent staged variants for both formats keep only one projection
accumulator live at a time so gfx1150 can trade a second cached activation
read for lower VGPR pressure.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
from triton.language.extra import libdevice

# Keep the validated one-warp GEMV as the AMD default.  The common generator
# accepts FLAGOS_GEMV_NUM_WARPS for controlled target experiments, but a wider
# setting was measured slower on gfx1150 and must not silently enter packages.
os.environ.setdefault("FLAGOS_GEMV_NUM_WARPS", "1")
# A 7x1000-run gfx1150 screen favored 64 channels and two wave32 warps for
# both single-token decode and long-token prefill. Keep the values overridable
# so another AMD target can establish its own package contract.
os.environ.setdefault("FLAGOS_SSM_CONV_SILU_BLOCK_SIZE", "64")
os.environ.setdefault("FLAGOS_SSM_CONV_SILU_NUM_WARPS", "2")
# The common generator is shared with the CUDA-compatible Denglin provider.
# Enable the wave32 Q4 experiments only for AMD packages so another provider
# does not accidentally compile or advertise these target-specific symbols.
os.environ.setdefault("FLAGOS_Q4_GEMV_NARROW_ENABLE", "1")
if os.environ.get("FLAGOS_AMD_EMIT_Q4_GEMV_NARROW8", "0") == "1":
    os.environ["FLAGOS_Q4_GEMV_NARROW8_ENABLE"] = "1"
if os.environ.get("FLAGOS_AMD_EMIT_Q5_GEMV_NARROW16", "0") == "1":
    os.environ["FLAGOS_Q5_GEMV_NARROW16_ENABLE"] = "1"
if os.environ.get("FLAGOS_AMD_EMIT_Q40_GEMV_NARROW", "0") == "1":
    os.environ["FLAGOS_Q40_GEMV_NARROW_ENABLE"] = "1"
# The common generator is also used by the Denglin provider.  Keep the
# experimental quantized tiled kernels AMD-only unless this tool explicitly
# opts into their compilation and manifest emission.
if os.environ.get("FLAGOS_AMD_EMIT_QUANT_TILED", "0") == "1":
    os.environ["FLAGOS_QUANT_TILE_ENABLE"] = "1"
else:
    os.environ.setdefault("FLAGOS_QUANT_TILE_ENABLE", "0")

HERE = Path(__file__).resolve().parents[3] / "kernels"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_flagos_kernels as common  # noqa: E402
from flagos_amd_profiles import (  # noqa: E402
    TUNING_PROFILE_KERNEL_LISTS,
    TUNING_PROFILES,
    tuning_profile_contracts,
)
from flagos_amd_hsaco import validate_hsaco  # noqa: E402

# AMD libdevice functions and wave reductions are allowed a small numerical
# delta from the CUDA-compatible provider used to establish the shared tests.
# These are tolerance floors, not a validation bypass: tests with an already
# wider, operation-specific contract retain that contract.
AMD_COMMON_CONFORMANCE_RTOL = 5e-4
AMD_COMMON_CONFORMANCE_ATOL = 5e-4
FLAGGEMS_AOT_EXPORT = None
FLAGGEMS_AOT_SOURCE_ROOT = None
FLAGGEMS_AOT_COMMIT = ""
FLAGGEMS_AOT_TREE_STATE = "unknown"


@contextmanager
def amd_hint_context():
    try:
        from triton.compiler import hint_manager
    except ImportError:
        yield
        return
    if not hasattr(hint_manager, "HintManager") or not hasattr(hint_manager, "_global_hint_manager"):
        raise RuntimeError("unsupported FlagTree hint API for explicit HIP compilation")
    previous = hint_manager._global_hint_manager
    # The CLI compiles for HIP; driver discovery can select CUDA or require a GPU.
    hint_manager._global_hint_manager = hint_manager.HintManager("hip")
    try:
        yield
    finally:
        hint_manager._global_hint_manager = previous


def assert_common_kernel_close(actual, expected, *args, **kwargs) -> None:
    rtol = kwargs.get("rtol")
    atol = kwargs.get("atol")
    kwargs["rtol"] = max(AMD_COMMON_CONFORMANCE_RTOL,
                         0.0 if rtol is None else float(rtol))
    kwargs["atol"] = max(AMD_COMMON_CONFORMANCE_ATOL,
                         0.0 if atol is None else float(atol))
    torch.testing.assert_close(actual, expected, *args, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_flaggems_aot_export(
        name="flagos_add_rms_norm_mul_residual_f32",
        semantic="transformer.add_rms_norm_scale_residual.v1",
        runtime_arguments=("residual_output", "mul_output", "x", "bias", "weight", "n_cols", "eps"),
        constexpr_arguments=("BLOCK",)):
    source = os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE")
    if not source:
        return None, None, "", "unknown"
    requested = Path(source).expanduser().resolve()
    package_root = requested / "src" if (requested / "src" / "flag_gems_aot").is_dir() else requested
    package_dir = package_root / "flag_gems_aot"
    if not (package_dir / "__init__.py").is_file():
        raise RuntimeError(
            "FLAGOS_FLAGGEMS_AOT_SOURCE must contain flag_gems_aot or src/flag_gems_aot")
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.import_module("flag_gems_aot")
    finally:
        sys.path.remove(str(package_root))
    module_file = Path(module.__file__).resolve()
    if not module_file.is_relative_to(package_dir.resolve()):
        raise RuntimeError(f"flag_gems_aot was imported from an unexpected path: {module_file}")
    export = module.get_kernel_export(name)
    if (export.abi != "flag-gems-aot-v1" or
            export.semantic != semantic or
            export.runtime_arguments != runtime_arguments or
            export.constexpr_arguments != constexpr_arguments):
        raise RuntimeError(f"incompatible FlagGems AOT export contract: {name}")
    commit = os.environ.get("FLAGOS_FLAGGEMS_COMMIT", "")
    if not commit:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True)
        candidate = result.stdout.strip()
        if result.returncode == 0 and len(candidate) == 40 and all(
                character in "0123456789abcdef" for character in candidate):
            commit = candidate
    status = subprocess.run(
        ["git", "-C", str(package_root), "status", "--porcelain", "--untracked-files=all"],
        check=False, capture_output=True, text=True)
    tree_state = "unknown" if status.returncode != 0 else "dirty" if status.stdout else "clean"
    return export, package_root, commit, tree_state


def package_metadata() -> dict:
    sources = [
        (Path(__file__).name, Path(__file__).resolve()),
        (Path(common.__file__).name, Path(common.__file__).resolve()),
        ("flagos_amd_profiles.py", Path(__file__).resolve().parent / "flagos_amd_profiles.py"),
        ("flagos_amd_hsaco.py", Path(__file__).resolve().parent / "flagos_amd_hsaco.py"),
    ]
    sources.extend((path.name, path) for path in TUNING_PROFILE_KERNEL_LISTS)
    for export in FLAGGEMS_AOT_EXPORTS:
        for source in export.source_files:
            source_path = Path(source).resolve()
            if FLAGGEMS_AOT_SOURCE_ROOT is None or not source_path.is_file() or not source_path.is_relative_to(
                    FLAGGEMS_AOT_SOURCE_ROOT):
                raise RuntimeError(f"invalid FlagGems AOT provenance source: {source_path}")
            sources.append((f"FlagGems/{source_path.relative_to(FLAGGEMS_AOT_SOURCE_ROOT)}", source_path))
    source_entries = [
        {"name": name, "sha256": sha256_file(source)}
        for name, source in dict(sources).items()
    ]
    combined = hashlib.sha256()
    for source in source_entries:
        combined.update(source["name"].encode())
        combined.update(b"\0")
        combined.update(source["sha256"].encode())
        combined.update(b"\0")
    metadata = {
        "abi": "flagos-amd-aot-v2",
        "backend": "hip",
        "binary_format": "hsaco",
        "provenance": "generated-with-manifest",
        "generator": Path(__file__).name,
        "compiler": {"name": "Triton", "version": getattr(triton, "__version__", "unknown")},
        "python_version": platform.python_version(),
        "source_sha256": combined.hexdigest(),
        "sources": source_entries,
    }
    if FLAGGEMS_AOT_EXPORTS:
        metadata["source_dependencies"] = [{
            "name": "FlagGems",
            "commit": FLAGGEMS_AOT_COMMIT or "unknown",
            "tree_state": FLAGGEMS_AOT_TREE_STATE,
            "abi": export.abi,
            "kernel": export.name,
            "semantic": export.semantic,
        } for export in FLAGGEMS_AOT_EXPORTS]
    tuning_profile = os.environ.get("FLAGOS_AMD_TUNING_PROFILE_NAME", "")
    if tuning_profile:
        tuning_profile_contracts(tuning_profile)
        metadata["tuning_profile"] = tuning_profile
    return metadata


def validate_tuning_profile_manifest(
        profile: str, arch: str, kernels: list[dict]) -> None:
    profile_arch, contracts = tuning_profile_contracts(profile)
    expected_names = frozenset(contracts)
    names = {kernel.get("name") for kernel in kernels}
    missing = expected_names - names
    extra = names - expected_names
    duplicate = len(names) != len(kernels)
    if arch != profile_arch or missing or extra or duplicate:
        detail = "wrong architecture"
        if missing:
            detail = "missing " + ", ".join(sorted(missing))
        elif extra:
            detail = "extra " + ", ".join(sorted(extra))
        elif duplicate:
            detail = "duplicate kernel name"
        raise RuntimeError(f"incomplete {profile} package: {detail}")
    for kernel in kernels:
        contract = contracts[kernel["name"]]
        actual = (
            kernel.get("argument_count", contract.argument_count),
            kernel["block_size"], kernel.get("exact_block_size", False), kernel.get("tile_m", 0),
            kernel.get("tile_n", 0), kernel.get("tile_k", 0),
            kernel["num_warps"], kernel["warp_size"],
        )
        if actual != contract:
            raise RuntimeError(
                f"tuned kernel launch contract mismatch for {kernel['name']}: "
                f"contract={contract}, metadata={actual}")


def write_manifest(output_dir: Path, arch: str, kernels: list[dict]) -> None:
    package = package_metadata()
    if package.get("tuning_profile"):
        validate_tuning_profile_manifest(package["tuning_profile"], arch, kernels)
    for kernel in kernels:
        artifact = output_dir / kernel["file"]
        validate_hsaco(artifact, arch, kernel)
        kernel["size_bytes"] = artifact.stat().st_size
        kernel["sha256"] = sha256_file(artifact)
    manifest = {
        "format": 2,
        "arch": arch,
        "package": package,
        "kernels": kernels,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


AMD_ROW_BLOCK_SIZE = 4096
AMD_RESIDUAL_BLOCK_SIZE = int(os.environ.get(
    "FLAGOS_AMD_RESIDUAL_BLOCK_SIZE", str(AMD_ROW_BLOCK_SIZE)))
AMD_RESIDUAL_NARROW_BLOCK_SIZE = int(os.environ.get(
    "FLAGOS_AMD_RESIDUAL_NARROW_BLOCK_SIZE", "2048"))
AMD_RMS_NORM_NARROW_BLOCK_SIZE = int(os.environ.get(
    "FLAGOS_AMD_RMS_NORM_NARROW_BLOCK_SIZE", "128"))
AMD_RMS_NORM_NARROW_NUM_WARPS = int(os.environ.get(
    "FLAGOS_AMD_RMS_NORM_NARROW_NUM_WARPS", "1"))
AMD_SILU_MUL_BLOCK_SIZE = int(os.environ.get(
    "FLAGOS_AMD_SILU_MUL_BLOCK_SIZE", "256"))
AMD_SILU_MUL_NUM_WARPS = int(os.environ.get(
    "FLAGOS_AMD_SILU_MUL_NUM_WARPS", "4"))
AMD_GDN_ALPHA_GATE_WIDTH = int(os.environ.get(
    "FLAGOS_AMD_GDN_ALPHA_GATE_WIDTH", "32"))
AMD_GDN_ALPHA_GATE_BLOCK_SIZE = int(os.environ.get(
    "FLAGOS_AMD_GDN_ALPHA_GATE_BLOCK_SIZE", "256"))
AMD_GDN_ALPHA_GATE_NUM_WARPS = int(os.environ.get(
    "FLAGOS_AMD_GDN_ALPHA_GATE_NUM_WARPS", "4"))
AMD_GDN_Q8_GATE_ROWS = int(os.environ.get(
    "FLAGOS_AMD_GDN_Q8_GATE_ROWS", "32"))
AMD_GDN_Q8_GATE_K = int(os.environ.get(
    "FLAGOS_AMD_GDN_Q8_GATE_K", "2560"))
AMD_GDN_Q8_GATE_BLOCK_M = int(os.environ.get(
    "FLAGOS_AMD_GDN_Q8_GATE_BLOCK_M", "4"))
AMD_GDN_Q8_GATE_NUM_WARPS = int(os.environ.get(
    "FLAGOS_AMD_GDN_Q8_GATE_NUM_WARPS", "1"))
AMD_SOFTMAX_BLOCK_SIZE = 4096
ATTENTION_HEAD_DIM = 128
ATTENTION_BLOCK_M = 16
ATTENTION_BLOCK_N = 32
# Keep the validated defaults, but expose the tile shape as an explicit
# package-generation knob.  This makes target-specific tuning reproducible
# without changing the provider ABI or the C++ launcher.
# gfx1150 tuning: 64x128x32 with four warps is the best validated shape for
# the standalone Qwen3 pp512 projection GEMM on the Radeon 890M.  Keep the
# fused FFN kernel on a separate, lower-register shape: applying this wider
# tile to the sigmoid-bearing kernel regresses it badly.
F16_MATMUL_BLOCK_M = int(os.environ.get("FLAGOS_F16_MATMUL_BLOCK_M", "64"))
F16_MATMUL_BLOCK_N = int(os.environ.get("FLAGOS_F16_MATMUL_BLOCK_N", "128"))
F16_MATMUL_BLOCK_K = int(os.environ.get("FLAGOS_F16_MATMUL_BLOCK_K", "32"))
F16_MATMUL_GROUP_M = int(os.environ.get("FLAGOS_F16_MATMUL_GROUP_M", "4"))
NUM_WARPS = int(os.environ.get("FLAGOS_AMD_NUM_WARPS", "4"))
F16_MATMUL_NUM_WARPS = int(os.environ.get("FLAGOS_F16_MATMUL_NUM_WARPS", "4"))
F16_MATMUL_GROUPED_NUM_STAGES = int(os.environ.get("FLAGOS_F16_MATMUL_GROUPED_NUM_STAGES", "1"))
F16_MATMUL_GROUPED_WAVES_PER_EU = int(os.environ.get("FLAGOS_F16_MATMUL_GROUPED_WAVES_PER_EU", "2"))
FFN_MATMUL_BLOCK_M = int(os.environ.get("FLAGOS_FFN_MATMUL_BLOCK_M", "32"))
FFN_MATMUL_BLOCK_N = int(os.environ.get("FLAGOS_FFN_MATMUL_BLOCK_N", "64"))
FFN_MATMUL_BLOCK_K = int(os.environ.get("FLAGOS_FFN_MATMUL_BLOCK_K", "32"))
FFN_MATMUL_NUM_WARPS = int(os.environ.get("FLAGOS_FFN_MATMUL_NUM_WARPS", "2"))
FFN_MATMUL_GROUP_M = int(os.environ.get("FLAGOS_FFN_MATMUL_GROUP_M", "2"))
FFN_MATMUL_GROUPED_NUM_STAGES = int(os.environ.get("FLAGOS_FFN_MATMUL_GROUPED_NUM_STAGES", "2"))
FFN_MATMUL_GROUPED_WAVES_PER_EU = int(os.environ.get("FLAGOS_FFN_MATMUL_GROUPED_WAVES_PER_EU", "2"))
# The full prefill FFN lowering keeps the gate/up tile above, but writes its
# graph-private SwiGLU result as F16 and feeds a separately tuned down GEMM.
# These defaults are the best validated pipeline shape on gfx1150/Radeon 890M.
FFN_DOWN_MATMUL_BLOCK_M = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_BLOCK_M", "128"))
FFN_DOWN_MATMUL_BLOCK_N = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_BLOCK_N", "128"))
FFN_DOWN_MATMUL_BLOCK_K = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_BLOCK_K", "64"))
FFN_DOWN_MATMUL_GROUP_M = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_GROUP_M", "4"))
FFN_DOWN_MATMUL_NUM_WARPS = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_NUM_WARPS", "8"))
FFN_DOWN_MATMUL_NUM_STAGES = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_NUM_STAGES", "2"))
FFN_DOWN_MATMUL_WAVES_PER_EU = int(os.environ.get("FLAGOS_FFN_DOWN_MATMUL_WAVES_PER_EU", "1"))
Q4_FFN_DECODE_BLOCK_M = int(os.environ.get("FLAGOS_Q4_FFN_DECODE_BLOCK_M", "8"))
Q4_FFN_DECODE_NUM_WARPS = int(os.environ.get("FLAGOS_Q4_FFN_DECODE_NUM_WARPS", "2"))
Q4_FFN_DECODE_WAVES_PER_EU = int(os.environ.get(
    "FLAGOS_Q4_FFN_DECODE_WAVES_PER_EU", "4"))
Q4_FFN_DECODE_STAGED_NUM_WARPS = int(os.environ.get(
    "FLAGOS_Q4_FFN_DECODE_STAGED_NUM_WARPS", "4"))
Q4_FFN_DECODE_STAGED_WAVES_PER_EU = int(os.environ.get(
    "FLAGOS_Q4_FFN_DECODE_STAGED_WAVES_PER_EU", "4"))
Q40_FFN_DECODE_BLOCK_M = int(os.environ.get("FLAGOS_Q40_FFN_DECODE_BLOCK_M", "8"))
Q40_FFN_DECODE_NUM_WARPS = int(os.environ.get("FLAGOS_Q40_FFN_DECODE_NUM_WARPS", "1"))
Q40_FFN_DECODE_WAVES_PER_EU = int(os.environ.get(
    "FLAGOS_Q40_FFN_DECODE_WAVES_PER_EU", "4"))
Q40_FFN_DECODE_STAGED_NUM_WARPS = int(os.environ.get(
    "FLAGOS_Q40_FFN_DECODE_STAGED_NUM_WARPS", "1"))
Q40_FFN_DECODE_STAGED_WAVES_PER_EU = int(os.environ.get(
    "FLAGOS_Q40_FFN_DECODE_STAGED_WAVES_PER_EU", "4"))
Q4_GEMV_NARROW8_NUM_WARPS = int(os.environ.get(
    "FLAGOS_Q4_GEMV_NARROW8_NUM_WARPS", "1"))
Q5_GEMV_NARROW16_NUM_WARPS = int(os.environ.get(
    "FLAGOS_Q5_GEMV_NARROW16_NUM_WARPS", "1"))
Q40_GEMV_NARROW_BLOCK_M = int(os.environ.get(
    "FLAGOS_Q40_GEMV_NARROW_BLOCK_M", "8"))
Q40_GEMV_NARROW_NUM_WARPS = int(os.environ.get(
    "FLAGOS_Q40_GEMV_NARROW_NUM_WARPS", "1"))
Q40_GEMV_NARROW_WAVES_PER_EU = int(os.environ.get(
    "FLAGOS_Q40_GEMV_NARROW_WAVES_PER_EU", "4"))


@triton.jit
def flagos_silu_f32(x, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, values * tl.sigmoid(values), mask=mask)


@triton.jit
def flagos_silu_mul_f32(x, other, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    multipliers = tl.load(other + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, values * tl.sigmoid(values) * multipliers, mask=mask)


@triton.jit
def flagos_sigmoid_mul_f32(x, other, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    multipliers = tl.load(other + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, tl.sigmoid(values) * multipliers, mask=mask)


@triton.jit
def flagos_softplus_mul_f32(x, other, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    multipliers = tl.load(other + offsets, mask=mask, other=0.0)
    activated = tl.where(values > 20.0, values, tl.log(1.0 + tl.exp(values)))
    tl.store(output + offsets, activated * multipliers, mask=mask)


@triton.jit
def flagos_gdn_alpha_gate_f32(
    x, bias, scale, output, n_elements,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    columns = offsets % WIDTH
    values = tl.load(x + offsets, mask=mask, other=0.0)
    values += tl.load(bias + columns, mask=mask, other=0.0)
    factors = tl.load(scale + columns, mask=mask, other=0.0)
    activated = tl.where(values > 20.0, values, tl.log(1.0 + tl.exp(values)))
    tl.store(output + offsets, activated * factors, mask=mask)


@triton.jit
def flagos_gdn_q8_gate_projections_f32(
    alpha_weights_u8, alpha_weights_f16,
    beta_weights_u8, beta_weights_f16,
    x, alpha_bias, alpha_scale, alpha_output, beta_output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Two Q8_0 decode projections with Qwen GDN gate epilogues."""
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    alpha_accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    beta_accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks = k // 32
    for block in tl.range(0, blocks):
        packed_block = row_ids[:, None] * blocks + block
        block_byte = packed_block * 34
        alpha_d = tl.load(alpha_weights_f16 + packed_block * 17).to(tl.float32)
        beta_d = tl.load(beta_weights_f16 + packed_block * 17).to(tl.float32)
        alpha_q = tl.load(alpha_weights_u8 + block_byte + 2 + lanes[None, :])
        beta_q = tl.load(beta_weights_u8 + block_byte + 2 + lanes[None, :])
        activation = tl.load(x + block * 32 + lanes).to(tl.float32)
        alpha_accumulator += (
            alpha_d * alpha_q.to(tl.int8).to(tl.float32) * activation[None, :])
        beta_accumulator += (
            beta_d * beta_q.to(tl.int8).to(tl.float32) * activation[None, :])
    mask = row_ids < rows
    alpha = tl.sum(alpha_accumulator, axis=1)
    alpha += tl.load(alpha_bias + row_ids, mask=mask, other=0.0)
    alpha = tl.where(alpha > 20.0, alpha, tl.log(1.0 + tl.exp(alpha)))
    alpha *= tl.load(alpha_scale + row_ids, mask=mask, other=0.0)
    beta = tl.sigmoid(tl.sum(beta_accumulator, axis=1))
    tl.store(alpha_output + row_ids, alpha, mask=mask)
    tl.store(beta_output + row_ids, beta, mask=mask)


@triton.jit
def flagos_add_rms_norm_mul_f32(
    norm_output, mul_output, x, bias, weight, n_cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    values += tl.load(bias + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(norm_output + row * n_cols + cols, normalized, mask=mask)
    tl.store(mul_output + row * n_cols + cols, normalized * weights, mask=mask)


@triton.jit
def flagos_rms_norm_mul_inplace_f32(
    output, x, weight, n_cols, eps, BLOCK: tl.constexpr
):
    """RMSNorm+scale variant for a scheduler-aliasing output buffer.

    The ordinary two-output kernel has separate pointer arguments and Triton
    may assume they do not alias.  GGML is allowed to reuse the dead RMSNorm
    allocation for the terminal MUL result, so this ABI intentionally exposes
    one output pointer and never materializes the dead intermediate.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(output + row * n_cols + cols, normalized * weights, mask=mask)


@triton.jit
def flagos_rms_norm_mul_inplace_f32_narrow(
    output, x, weight, n_cols, eps, BLOCK: tl.constexpr
):
    """Exact-width RMSNorm+scale kernel for narrow rows."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(x + row * n_cols + cols)
    weights = tl.load(weight + cols)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(output + row * n_cols + cols, normalized * weights)


@triton.jit
def flagos_add_rms_norm_mul_inplace_f32(
    output, x, bias, weight, n_cols, eps, BLOCK: tl.constexpr
):
    """ADD+RMSNorm+scale variant with a single terminal output pointer."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    values += tl.load(bias + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(output + row * n_cols + cols, normalized * weights, mask=mask)


@triton.jit
def flagos_add_rms_norm_mul_residual_f32(
    residual_output, mul_output, x, bias, weight, n_cols, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    values += tl.load(bias + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(residual_output + row * n_cols + cols, values, mask=mask)
    tl.store(mul_output + row * n_cols + cols, normalized * weights, mask=mask)


@triton.jit
def flagos_add_rms_norm_mul_residual_f32_narrow(
    residual_output, mul_output, x, bias, weight, n_cols, eps, BLOCK: tl.constexpr
):
    """Smaller-block variant selected by manifest-described row capacity."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=mask, other=0.0)
    values += tl.load(bias + row * n_cols + cols, mask=mask, other=0.0)
    weights = tl.load(weight + cols, mask=mask, other=0.0)
    variance = tl.sum(values * values, axis=0) / n_cols
    normalized = values * tl.rsqrt(variance + eps)
    tl.store(residual_output + row * n_cols + cols, values, mask=mask)
    tl.store(mul_output + row * n_cols + cols, normalized * weights, mask=mask)


(
    FLAGGEMS_AOT_EXPORT,
    FLAGGEMS_AOT_SOURCE_ROOT,
    FLAGGEMS_AOT_COMMIT,
    FLAGGEMS_AOT_TREE_STATE,
) = load_flaggems_aot_export()
if FLAGGEMS_AOT_EXPORT is not None:
    flagos_add_rms_norm_mul_residual_f32 = FLAGGEMS_AOT_EXPORT.function

FLAGGEMS_DENSE_GATHER_EXPORT, _, _, _ = load_flaggems_aot_export(
    "flagos_get_rows_f32",
    "ggml.get_rows.contiguous_f32.i32_indices.in_range.bitwise.v1",
    ("table", "row_index", "output", "n_cols"),
)
if FLAGGEMS_DENSE_GATHER_EXPORT is not None:
    common.flagos_get_rows_f32 = FLAGGEMS_DENSE_GATHER_EXPORT.function
FLAGGEMS_AOT_EXPORTS = tuple(export for export in (
    FLAGGEMS_AOT_EXPORT, FLAGGEMS_DENSE_GATHER_EXPORT) if export is not None)


@triton.jit
def flagos_rms_norm_mul_rope_neox_f32(
    x, weight, positions, output,
    n_cols, n_heads, n_tokens, n_dims, eps, freq_base, freq_scale,
    BLOCK: tl.constexpr,
):
    """One-row RMSNorm, scale, and NEOX RoPE with no intermediates."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    values = tl.load(x + row * n_cols + cols, mask=valid, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / n_cols
    norm_scale = tl.rsqrt(variance + eps)

    half = n_dims // 2
    pair_dim = tl.where(cols < half, cols, cols - half)
    pair_valid = valid & (cols < n_dims) & (pair_dim < half)
    x0 = tl.load(x + row * n_cols + pair_dim,
                 mask=pair_valid, other=0.0).to(tl.float32)
    x1 = tl.load(x + row * n_cols + pair_dim + half,
                 mask=pair_valid, other=0.0).to(tl.float32)
    w0 = tl.load(weight + pair_dim, mask=pair_valid, other=0.0).to(tl.float32)
    w1 = tl.load(weight + pair_dim + half,
                 mask=pair_valid, other=0.0).to(tl.float32)
    scaled0 = x0 * norm_scale * w0
    scaled1 = x1 * norm_scale * w1
    token = (row // n_heads) % n_tokens
    position = tl.load(positions + token).to(tl.float32)
    exponent = -2.0 * pair_dim.to(tl.float32) / n_dims
    theta = position * freq_scale * libdevice.pow(freq_base, exponent)
    c = libdevice.cos(theta)
    s = libdevice.sin(theta)
    rotated = tl.where(cols < half, scaled0 * c - scaled1 * s,
                       scaled0 * s + scaled1 * c)
    unrotated = values * norm_scale * tl.load(weight + cols, mask=valid, other=0.0)
    result = tl.where(pair_valid, rotated, unrotated)
    tl.store(output + row * n_cols + cols, result, mask=valid)


@triton.jit
def flagos_rms_norm_mul_rope_kv_store_neox_f32_f16(
    x, weight, positions, row_index, output,
    n_cols, n_heads, n_tokens, n_dst_rows, n_dims, eps, freq_base, freq_scale,
    BLOCK: tl.constexpr,
):
    """RMSNorm, scale, NEOX RoPE, and the terminal F16 KV-cache store."""
    row = tl.program_id(0)
    head = row % n_heads
    token = (row // n_heads) % n_tokens
    cols = tl.arange(0, BLOCK)
    valid_col = cols < n_cols
    values = tl.load(x + row * n_cols + cols,
                     mask=valid_col, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / n_cols
    norm_scale = tl.rsqrt(variance + eps)

    half = n_dims // 2
    pair_dim = tl.where(cols < half, cols, cols - half)
    pair_valid = valid_col & (cols < n_dims) & (pair_dim < half)
    x0 = tl.load(x + row * n_cols + pair_dim,
                 mask=pair_valid, other=0.0).to(tl.float32)
    x1 = tl.load(x + row * n_cols + pair_dim + half,
                 mask=pair_valid, other=0.0).to(tl.float32)
    w0 = tl.load(weight + pair_dim, mask=pair_valid, other=0.0).to(tl.float32)
    w1 = tl.load(weight + pair_dim + half,
                 mask=pair_valid, other=0.0).to(tl.float32)
    scaled0 = x0 * norm_scale * w0
    scaled1 = x1 * norm_scale * w1
    position = tl.load(positions + token).to(tl.float32)
    exponent = -2.0 * pair_dim.to(tl.float32) / n_dims
    theta = position * freq_scale * libdevice.pow(freq_base, exponent)
    c = libdevice.cos(theta)
    s = libdevice.sin(theta)
    rotated = tl.where(cols < half, scaled0 * c - scaled1 * s,
                       scaled0 * s + scaled1 * c)
    unrotated = values * norm_scale * tl.load(weight + cols, mask=valid_col, other=0.0)
    result = tl.where(pair_valid, rotated, unrotated)

    dst_row = tl.load(row_index + token).to(tl.int32)
    valid = valid_col & (dst_row >= 0) & (dst_row < n_dst_rows)
    dst_offset = (dst_row * n_heads + head) * n_cols + cols
    tl.store(output + dst_offset, result.to(tl.float16), mask=valid)


@triton.jit
def flagos_ffn_swiglu_q4_k_f32_decode(
    gate_u8, gate_f16, up_u8, up_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Fuse the two Q4_K decode projections and terminal SwiGLU.

    One wave computes a small output-row tile for both gate and up weights.
    The activation vector is shared between the two quantized dot products,
    and only the terminal SwiGLU tensor is materialized.
    """
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 32)
    gate_acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_offset = (row_ids[:, None] * blocks + block) * 144
        block_half = block_offset // 2
        gate_d = tl.load(gate_f16 + block_half).to(tl.float32)
        gate_dmin = tl.load(gate_f16 + block_half + 1).to(tl.float32)
        up_d = tl.load(up_f16 + block_half).to(tl.float32)
        up_dmin = tl.load(up_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_index = low_group if low_group < 4 else low_group + 4
            low_scale_hi_index = low_group - 4 if low_group >= 4 else 0
            high_scale_index = high_group if high_group < 4 else high_group + 4
            high_scale_hi_index = high_group - 4 if high_group >= 4 else 0

            gate_low_scale_byte = tl.load(gate_u8 + block_offset + 4 + low_scale_index)
            gate_low_scale_hi = tl.load(gate_u8 + block_offset + 4 + low_scale_hi_index)
            gate_low_scale = (gate_low_scale_byte & 63 if low_group < 4 else
                              (gate_low_scale_byte & 15) | ((gate_low_scale_hi >> 6) << 4)).to(tl.float32)
            gate_low_min_byte = tl.load(gate_u8 + block_offset + 4 + low_group + 4)
            gate_low_min_hi = tl.load(gate_u8 + block_offset + 4 + low_group)
            gate_low_min = (gate_low_min_byte & 63 if low_group < 4 else
                            (gate_low_min_byte >> 4) | ((gate_low_min_hi >> 6) << 4)).to(tl.float32)
            gate_high_scale_byte = tl.load(gate_u8 + block_offset + 4 + high_scale_index)
            gate_high_scale_hi = tl.load(gate_u8 + block_offset + 4 + high_scale_hi_index)
            gate_high_scale = (gate_high_scale_byte & 63 if high_group < 4 else
                               (gate_high_scale_byte & 15) | ((gate_high_scale_hi >> 6) << 4)).to(tl.float32)
            gate_high_min_byte = tl.load(gate_u8 + block_offset + 4 + high_group + 4)
            gate_high_min_hi = tl.load(gate_u8 + block_offset + 4 + high_group)
            gate_high_min = (gate_high_min_byte & 63 if high_group < 4 else
                             (gate_high_min_byte >> 4) | ((gate_high_min_hi >> 6) << 4)).to(tl.float32)

            up_low_scale_byte = tl.load(up_u8 + block_offset + 4 + low_scale_index)
            up_low_scale_hi = tl.load(up_u8 + block_offset + 4 + low_scale_hi_index)
            up_low_scale = (up_low_scale_byte & 63 if low_group < 4 else
                            (up_low_scale_byte & 15) | ((up_low_scale_hi >> 6) << 4)).to(tl.float32)
            up_low_min_byte = tl.load(up_u8 + block_offset + 4 + low_group + 4)
            up_low_min_hi = tl.load(up_u8 + block_offset + 4 + low_group)
            up_low_min = (up_low_min_byte & 63 if low_group < 4 else
                          (up_low_min_byte >> 4) | ((up_low_min_hi >> 6) << 4)).to(tl.float32)
            up_high_scale_byte = tl.load(up_u8 + block_offset + 4 + high_scale_index)
            up_high_scale_hi = tl.load(up_u8 + block_offset + 4 + high_scale_hi_index)
            up_high_scale = (up_high_scale_byte & 63 if high_group < 4 else
                             (up_high_scale_byte & 15) | ((up_high_scale_hi >> 6) << 4)).to(tl.float32)
            up_high_min_byte = tl.load(up_u8 + block_offset + 4 + high_group + 4)
            up_high_min_hi = tl.load(up_u8 + block_offset + 4 + high_group)
            up_high_min = (up_high_min_byte & 63 if high_group < 4 else
                           (up_high_min_byte >> 4) | ((up_high_min_hi >> 6) << 4)).to(tl.float32)

            gate_q = tl.load(gate_u8 + block_offset + 16 + chunk * 32 + lanes[None, :])
            up_q = tl.load(up_u8 + block_offset + 16 + chunk * 32 + lanes[None, :])
            a0 = tl.load(x + block * 256 + low_index).to(tl.float32)
            a1 = tl.load(x + block * 256 + high_index).to(tl.float32)
            gate_low = gate_d * gate_low_scale * (gate_q & 15).to(tl.float32) - gate_dmin * gate_low_min
            gate_high = gate_d * gate_high_scale * (gate_q >> 4).to(tl.float32) - gate_dmin * gate_high_min
            up_low = up_d * up_low_scale * (up_q & 15).to(tl.float32) - up_dmin * up_low_min
            up_high = up_d * up_high_scale * (up_q >> 4).to(tl.float32) - up_dmin * up_high_min
            gate_acc += gate_low * a0[None, :] + gate_high * a1[None, :]
            up_acc += up_low * a0[None, :] + up_high * a1[None, :]
    gate = tl.sum(gate_acc, axis=1)
    up = tl.sum(up_acc, axis=1)
    tl.store(output + row_ids, gate * tl.sigmoid(gate) * up, mask=row_ids < rows)


@triton.jit
def flagos_ffn_swiglu_q4_0_f32_decode(
    gate_u8, gate_f16, up_u8, up_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Fuse Q4_0 gate/up GEMVs and the terminal SwiGLU for decode.

    Q4_0 stores one F16 scale followed by sixteen packed bytes per 32-value
    block (18 bytes total).  The ABI intentionally mirrors the Q4_K fusion:
    the second pointer view addresses the F16 scale without a conversion or
    temporary, while the first view decodes packed nibbles.
    """
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    lanes = tl.arange(0, 16)
    gate_acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    blocks = k // 32
    for block in tl.range(0, blocks):
        gate_block = row_ids[:, None] * blocks + block
        up_block = gate_block
        gate_byte = gate_block * 18
        up_byte = up_block * 18
        gate_d = tl.load(gate_f16 + gate_byte // 2).to(tl.float32)
        up_d = tl.load(up_f16 + up_byte // 2).to(tl.float32)
        gate_q = tl.load(gate_u8 + gate_byte + 2 + lanes[None, :])
        up_q = tl.load(up_u8 + up_byte + 2 + lanes[None, :])
        a0 = tl.load(x + block * 32 + lanes).to(tl.float32)
        a1 = tl.load(x + block * 32 + 16 + lanes).to(tl.float32)
        gate_acc += gate_d * ((gate_q & 15).to(tl.float32) - 8.0) * a0[None, :]
        gate_acc += gate_d * ((gate_q >> 4).to(tl.float32) - 8.0) * a1[None, :]
        up_acc += up_d * ((up_q & 15).to(tl.float32) - 8.0) * a0[None, :]
        up_acc += up_d * ((up_q >> 4).to(tl.float32) - 8.0) * a1[None, :]
    gate = tl.sum(gate_acc, axis=1)
    up = tl.sum(up_acc, axis=1)
    tl.store(output + row_ids, gate * tl.sigmoid(gate) * up, mask=row_ids < rows)


@triton.jit
def flagos_q4_0_f32_row_tile_dot(
    weights_u8, weights_f16, x, k, row_ids,
    BLOCK_M: tl.constexpr,
):
    """One Q4_0 row-tile dot product used by the staged decode FFN."""
    lanes = tl.arange(0, 16)
    accumulator = tl.zeros((BLOCK_M, 16), dtype=tl.float32)
    blocks = k // 32
    for block in tl.range(0, blocks):
        block_index = row_ids[:, None] * blocks + block
        block_byte = block_index * 18
        scale = tl.load(weights_f16 + block_byte // 2).to(tl.float32)
        quant = tl.load(weights_u8 + block_byte + 2 + lanes[None, :])
        a0 = tl.load(x + block * 32 + lanes).to(tl.float32)
        a1 = tl.load(x + block * 32 + 16 + lanes).to(tl.float32)
        accumulator += scale * ((quant & 15).to(tl.float32) - 8.0) * a0[None, :]
        accumulator += scale * ((quant >> 4).to(tl.float32) - 8.0) * a1[None, :]
    return tl.sum(accumulator, axis=1)


@triton.jit
def flagos_ffn_swiglu_q4_0_f32_decode_staged(
    gate_u8, gate_f16, up_u8, up_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q4_0 decode FFN with sequential gate/up accumulator lifetimes."""
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    gate = flagos_q4_0_f32_row_tile_dot(
        gate_u8, gate_f16, x, k, row_ids, BLOCK_M=BLOCK_M)
    silu_gate = gate * tl.sigmoid(gate)
    up = flagos_q4_0_f32_row_tile_dot(
        up_u8, up_f16, x, k, row_ids, BLOCK_M=BLOCK_M)
    tl.store(output + row_ids, silu_gate * up, mask=row_ids < rows)


@triton.jit
def flagos_q4_k_f32_row_tile_dot(
    weights_u8, weights_f16, x, k, row_ids,
    BLOCK_M: tl.constexpr,
):
    """One Q4_K row-tile dot product used by the staged decode FFN."""
    lanes = tl.arange(0, 32)
    accumulator = tl.zeros((BLOCK_M, 32), dtype=tl.float32)
    blocks = k // 256
    for block in tl.range(0, blocks):
        block_offset = (row_ids[:, None] * blocks + block) * 144
        block_half = block_offset // 2
        d = tl.load(weights_f16 + block_half).to(tl.float32)
        dmin = tl.load(weights_f16 + block_half + 1).to(tl.float32)
        for chunk in tl.range(0, 4):
            low_group = chunk * 2
            high_group = low_group + 1
            low_index = chunk * 64 + lanes
            high_index = low_index + 32
            low_scale_index = low_group if low_group < 4 else low_group + 4
            low_scale_hi_index = low_group - 4 if low_group >= 4 else 0
            high_scale_index = high_group if high_group < 4 else high_group + 4
            high_scale_hi_index = high_group - 4 if high_group >= 4 else 0

            low_scale_byte = tl.load(weights_u8 + block_offset + 4 + low_scale_index)
            low_scale_hi = tl.load(weights_u8 + block_offset + 4 + low_scale_hi_index)
            low_scale = (low_scale_byte & 63 if low_group < 4 else
                         (low_scale_byte & 15) | ((low_scale_hi >> 6) << 4)).to(tl.float32)
            low_min_byte = tl.load(weights_u8 + block_offset + 4 + low_group + 4)
            low_min_hi = tl.load(weights_u8 + block_offset + 4 + low_group)
            low_min = (low_min_byte & 63 if low_group < 4 else
                       (low_min_byte >> 4) | ((low_min_hi >> 6) << 4)).to(tl.float32)
            high_scale_byte = tl.load(weights_u8 + block_offset + 4 + high_scale_index)
            high_scale_hi = tl.load(weights_u8 + block_offset + 4 + high_scale_hi_index)
            high_scale = (high_scale_byte & 63 if high_group < 4 else
                          (high_scale_byte & 15) | ((high_scale_hi >> 6) << 4)).to(tl.float32)
            high_min_byte = tl.load(weights_u8 + block_offset + 4 + high_group + 4)
            high_min_hi = tl.load(weights_u8 + block_offset + 4 + high_group)
            high_min = (high_min_byte & 63 if high_group < 4 else
                        (high_min_byte >> 4) | ((high_min_hi >> 6) << 4)).to(tl.float32)

            quant = tl.load(weights_u8 + block_offset + 16 + chunk * 32 + lanes[None, :])
            a0 = tl.load(x + block * 256 + low_index).to(tl.float32)
            a1 = tl.load(x + block * 256 + high_index).to(tl.float32)
            low = d * low_scale * (quant & 15).to(tl.float32) - dmin * low_min
            high = d * high_scale * (quant >> 4).to(tl.float32) - dmin * high_min
            accumulator += low * a0[None, :] + high * a1[None, :]
    return tl.sum(accumulator, axis=1)


@triton.jit
def flagos_ffn_swiglu_q4_k_f32_decode_staged(
    gate_u8, gate_f16, up_u8, up_f16, x, output, k, rows,
    BLOCK_M: tl.constexpr,
):
    """Q4_K decode FFN with sequential gate/up accumulator lifetimes.

    The validated fused kernel keeps both [BLOCK_M, 32] accumulators live and
    uses 163 VGPRs at BLOCK_M=8 on gfx1150.  This variant reduces gate to one
    scalar per row before starting the up projection.  It rereads the small
    activation vector, but leaves quantized weight traffic and F32 arithmetic
    unchanged while giving the compiler a much smaller live range.
    """
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    gate = flagos_q4_k_f32_row_tile_dot(
        gate_u8, gate_f16, x, k, row_ids, BLOCK_M=BLOCK_M)
    silu_gate = gate * tl.sigmoid(gate)
    up = flagos_q4_k_f32_row_tile_dot(
        up_u8, up_f16, x, k, row_ids, BLOCK_M=BLOCK_M)
    tl.store(output + row_ids, silu_gate * up, mask=row_ids < rows)


@triton.jit
def flagos_rope_kv_store_f32_f16(
    x, positions, row_index, output,
    n_cols, n_rows, n_dst_rows, head_dim, n_heads, n_dims,
    freq_base, freq_scale, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_cols * n_rows
    row = offsets // n_cols
    col = offsets % n_cols
    head = col // head_dim
    dim = col % head_dim
    dst_row = tl.load(row_index + row, mask=mask, other=-1).to(tl.int32)
    valid = mask & (dst_row >= 0) & (dst_row < n_dst_rows)
    # The Qwen KV path uses NEOX rotation over the first n_dims channels.
    # Channels outside that rotary prefix are copied unchanged.
    half = n_dims // 2
    pair_dim = tl.where(dim < half, dim, dim - half)
    pair_valid = (dim < n_dims) & (pair_dim < half)
    pair_col = head * head_dim + pair_dim
    pair_offset = row * n_cols + pair_col
    other_offset = pair_offset + half
    x0 = tl.load(x + pair_offset, mask=valid & pair_valid, other=0.0).to(tl.float32)
    x1 = tl.load(x + other_offset, mask=valid & pair_valid, other=0.0).to(tl.float32)
    position = tl.load(positions + row, mask=valid, other=0).to(tl.float32)
    exponent = -2.0 * pair_dim.to(tl.float32) / n_dims
    theta = position * freq_scale * libdevice.pow(freq_base, exponent)
    c = libdevice.cos(theta)
    s = libdevice.sin(theta)
    rotated = tl.where(dim < half, x0 * c - x1 * s, x0 * s + x1 * c)
    value = tl.where(pair_valid, rotated, tl.load(x + row * n_cols + col, mask=valid, other=0.0).to(tl.float32))
    tl.store(output + dst_row * n_cols + col, value.to(tl.float16), mask=valid)


@triton.jit
def flagos_soft_max_unmasked_f32(x, mask_ptr, output, n_cols, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=float("-inf")) * scale
    values = tl.where(mask, values, float("-inf"))
    values = tl.exp(values - tl.max(values, axis=0))
    tl.store(output + row * n_cols + offsets,
             values / tl.sum(values, axis=0), mask=mask)


@triton.jit
def flagos_soft_max_masked_f32_f16(x, mask_ptr, output, n_cols, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x + row * n_cols + offsets, mask=mask, other=float("-inf")) * scale
    values += tl.load(mask_ptr + row * n_cols + offsets, mask=mask, other=0.0)
    values = tl.where(mask, values, float("-inf"))
    values = tl.exp(values - tl.max(values, axis=0))
    tl.store(output + row * n_cols + offsets,
             values / tl.sum(values, axis=0), mask=mask)


@triton.jit(do_not_specialize=["q_per_kv", "stride_mask_query"])
def flagos_flash_attn_prefill_f32_f16(
    q,
    k,
    v,
    attention_mask,
    output,
    query_length,
    key_length,
    q_per_kv,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_mask_token,
    stride_mask_query,
    stride_output_token,
    stride_output_head,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_head = tl.program_id(0)
    query_block = tl.program_id(1)
    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    valid_query = query_rows < query_length
    kv_head = query_head // q_per_kv

    q_block = tl.load(
        q + query_rows[:, None] * stride_q_token + query_head * stride_q_head + dims[None, :],
        mask=valid_query[:, None], other=0.0,
    ).to(tl.float16)
    running_max = tl.where(valid_query, -float("inf"), 0.0)
    running_sum = tl.where(valid_query, 0.0, 1.0)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for key_start in tl.range(0, key_length, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < key_length
        k_block = tl.load(
            k + dims[:, None] + kv_head * stride_k_head + key_offsets[None, :] * stride_k_token,
            mask=key_mask[None, :], other=0.0,
        )
        scores = tl.dot(q_block, k_block) * scale
        scores += tl.load(
            attention_mask + key_offsets[None, :] * stride_mask_token + query_rows[:, None] * stride_mask_query,
            mask=valid_query[:, None] & key_mask[None, :], other=-float("inf"),
        )
        valid = valid_query[:, None] & key_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        block_max = tl.where(valid_query, block_max, 0.0)
        new_max = tl.maximum(running_max, block_max)
        rescale = tl.exp(running_max - new_max)
        probabilities = tl.where(valid, tl.exp(scores - new_max[:, None]), 0.0)
        block_sum = tl.sum(probabilities, axis=1)
        accumulator *= rescale[:, None]
        v_block = tl.load(
            v + key_offsets[:, None] * stride_v_token + kv_head * stride_v_head + dims[None, :],
            mask=key_mask[:, None], other=0.0,
        )
        accumulator += tl.dot(probabilities.to(tl.float16), v_block)
        running_sum = running_sum * rescale + block_sum
        running_max = new_max

    normalized = accumulator / running_sum[:, None]
    tl.store(
        output + query_rows[:, None] * stride_output_token + query_head * stride_output_head + dims[None, :],
        normalized, mask=valid_query[:, None],
    )


@triton.jit
def flagos_mul_mat_f16_f32_batched(
    weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """F16 weight x F32 activation GEMM used after one-time quant decode.

    GGML stores activations as contiguous F32 columns (k x columns) and the
    output as contiguous F32 columns (rows x columns).  The tile therefore
    computes a row-major weight tile times a column-major activation tile and
    stores with the column index as the outer stride.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, k, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            weights + rows_m[:, None] * k + ks[None, :],
            mask=(rows_m[:, None] < rows) & (ks[None, :] < k), other=0.0,
        ).to(tl.float16)
        b = tl.load(
            x + cols_n[None, :] * k + ks[:, None],
            mask=(cols_n[None, :] < columns) & (ks[:, None] < k), other=0.0,
        ).to(tl.float16)
        accumulator += tl.dot(a, b)
    tl.store(
        output + cols_n[None, :] * rows + rows_m[:, None], accumulator,
        mask=(rows_m[:, None] < rows) & (cols_n[None, :] < columns),
    )


@triton.jit
def flagos_mul_mat_f16_f32_grouped(
    weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """F16 x F32 GEMM with a one-dimensional, L2-friendly grouped schedule.

    The arithmetic and pointer ABI deliberately match
    ``flagos_mul_mat_f16_f32_batched``.  Only the program-to-tile mapping is
    different, so the C++ runtime can retain the two-dimensional kernel as a
    compatibility and small-prefill fallback.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(columns, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, k, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            weights + rows_m[:, None] * k + ks[None, :],
            mask=(rows_m[:, None] < rows) & (ks[None, :] < k), other=0.0,
        ).to(tl.float16)
        b = tl.load(
            x + cols_n[None, :] * k + ks[:, None],
            mask=(cols_n[None, :] < columns) & (ks[:, None] < k), other=0.0,
        ).to(tl.float16)
        accumulator += tl.dot(a, b)
    tl.store(
        output + cols_n[None, :] * rows + rows_m[:, None], accumulator,
        mask=(rows_m[:, None] < rows) & (cols_n[None, :] < columns),
    )


@triton.jit
def flagos_ffn_swiglu_f16_f32_batched(
    gate_weights, up_weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fuse the two F16 FFN projections and SwiGLU activation.

    Both projection weights are row-major F16 caches produced by the existing
    Q4_K/Q6_K dequant path. Activations are the GGML contiguous F32 [K,
    columns] layout and output is F32 [rows, columns]. One program computes a
    tile of both projections, eliminating two intermediate tensors and the
    standalone SwiGLU launch.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, k, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        row_mask = rows_m[:, None] < rows
        k_mask = ks[None, :] < k
        col_mask = cols_n[None, :] < columns
        gate = tl.load(
            gate_weights + rows_m[:, None] * k + ks[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        up = tl.load(
            up_weights + rows_m[:, None] * k + ks[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        activation = tl.load(
            x + cols_n[None, :] * k + ks[:, None],
            mask=col_mask & (ks[:, None] < k), other=0.0,
        ).to(tl.float16)
        gate_acc += tl.dot(gate, activation)
        up_acc += tl.dot(up, activation)
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    tl.store(
        output + cols_n[None, :] * rows + rows_m[:, None], result,
        mask=(rows_m[:, None] < rows) & (cols_n[None, :] < columns),
    )


@triton.jit
def flagos_ffn_swiglu_f16_f32_grouped(
    gate_weights, up_weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Grouped-schedule dual F16 projection plus SwiGLU for wide prefill."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(columns, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, k, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        row_mask = rows_m[:, None] < rows
        k_mask = ks[None, :] < k
        col_mask = cols_n[None, :] < columns
        gate = tl.load(
            gate_weights + rows_m[:, None] * k + ks[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        up = tl.load(
            up_weights + rows_m[:, None] * k + ks[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        activation = tl.load(
            x + cols_n[None, :] * k + ks[:, None],
            mask=col_mask & (ks[:, None] < k), other=0.0,
        ).to(tl.float16)
        gate_acc += tl.dot(gate, activation)
        up_acc += tl.dot(up, activation)
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    tl.store(
        output + cols_n[None, :] * rows + rows_m[:, None], result,
        mask=(rows_m[:, None] < rows) & (cols_n[None, :] < columns),
    )


@triton.jit
def flagos_ffn_swiglu_f16_f16_grouped(
    gate_weights, up_weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Dual F16 projection plus SwiGLU with graph-private F16 output."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(columns, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, k, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        row_mask = rows_m[:, None] < rows
        k_mask = ks[None, :] < k
        col_mask = cols_n[None, :] < columns
        gate = tl.load(
            gate_weights + rows_m[:, None] * k + ks[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        up = tl.load(
            up_weights + rows_m[:, None] * k + ks[None, :],
            mask=row_mask & k_mask, other=0.0,
        ).to(tl.float16)
        activation = tl.load(
            x + cols_n[None, :] * k + ks[:, None],
            mask=col_mask & (ks[:, None] < k), other=0.0,
        ).to(tl.float16)
        gate_acc += tl.dot(gate, activation)
        up_acc += tl.dot(up, activation)
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    tl.store(
        output + cols_n[None, :] * rows + rows_m[:, None], result.to(tl.float16),
        mask=(rows_m[:, None] < rows) & (cols_n[None, :] < columns),
    )


@triton.jit
def flagos_mul_mat_f16_f16_grouped(
    weights, x, output, k, rows, columns,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """F16 weight x graph-private F16 activation, with F32 output."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(columns, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rows_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, k, BLOCK_K):
        ks = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            weights + rows_m[:, None] * k + ks[None, :],
            mask=(rows_m[:, None] < rows) & (ks[None, :] < k), other=0.0,
        )
        b = tl.load(
            x + cols_n[None, :] * k + ks[:, None],
            mask=(cols_n[None, :] < columns) & (ks[:, None] < k), other=0.0,
        )
        accumulator += tl.dot(a, b)
    tl.store(
        output + cols_n[None, :] * rows + rows_m[:, None], accumulator,
        mask=(rows_m[:, None] < rows) & (cols_n[None, :] < columns),
    )


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/HIP device is required to build the AMD package")
    return torch.device("cuda")


def compile_residual(device: torch.device) -> None:
    if AMD_RESIDUAL_BLOCK_SIZE <= 0 or AMD_RESIDUAL_BLOCK_SIZE & (AMD_RESIDUAL_BLOCK_SIZE - 1):
        raise RuntimeError("FLAGOS_AMD_RESIDUAL_BLOCK_SIZE must be a positive power of two")
    rows, cols = 5, AMD_RESIDUAL_BLOCK_SIZE
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    bias = torch.randn_like(x)
    weight = torch.randn((cols,), device=device, dtype=torch.float32)
    residual_output = torch.empty_like(x)
    mul_output = torch.empty_like(x)
    eps = 1e-6
    flagos_add_rms_norm_mul_residual_f32[(rows,)](
        residual_output, mul_output, x, bias, weight, cols, eps,
        BLOCK=AMD_RESIDUAL_BLOCK_SIZE, num_warps=NUM_WARPS)
    residual_reference = x + bias
    normalized_reference = residual_reference * torch.rsqrt(
        residual_reference.square().mean(dim=1, keepdim=True) + eps)
    torch.testing.assert_close(
        residual_output, residual_reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        mul_output, normalized_reference * weight, rtol=2e-5, atol=2e-5)


def compile_residual_narrow(device: torch.device) -> None:
    if (AMD_RESIDUAL_NARROW_BLOCK_SIZE <= 0 or
            AMD_RESIDUAL_NARROW_BLOCK_SIZE & (AMD_RESIDUAL_NARROW_BLOCK_SIZE - 1)):
        raise RuntimeError(
            "FLAGOS_AMD_RESIDUAL_NARROW_BLOCK_SIZE must be a positive power of two")
    if AMD_RESIDUAL_NARROW_BLOCK_SIZE >= AMD_RESIDUAL_BLOCK_SIZE:
        raise RuntimeError(
            "FLAGOS_AMD_RESIDUAL_NARROW_BLOCK_SIZE must be smaller than the generic residual block")
    rows, cols = 5, AMD_RESIDUAL_NARROW_BLOCK_SIZE
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    bias = torch.randn_like(x)
    weight = torch.randn((cols,), device=device, dtype=torch.float32)
    residual_output = torch.empty_like(x)
    mul_output = torch.empty_like(x)
    eps = 1e-6
    flagos_add_rms_norm_mul_residual_f32_narrow[(rows,)](
        residual_output, mul_output, x, bias, weight, cols, eps,
        BLOCK=AMD_RESIDUAL_NARROW_BLOCK_SIZE, num_warps=NUM_WARPS)
    residual_reference = x + bias
    normalized_reference = residual_reference * torch.rsqrt(
        residual_reference.square().mean(dim=1, keepdim=True) + eps)
    torch.testing.assert_close(
        residual_output, residual_reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        mul_output, normalized_reference * weight, rtol=2e-5, atol=2e-5)


def compile_q4_ffn_decode(device: torch.device) -> None:
    if (Q4_FFN_DECODE_BLOCK_M <= 0 or Q4_FFN_DECODE_BLOCK_M > 32 or
            Q4_FFN_DECODE_BLOCK_M & (Q4_FFN_DECODE_BLOCK_M - 1)):
        raise RuntimeError("FLAGOS_Q4_FFN_DECODE_BLOCK_M must be a power of two from 1 to 32")
    if Q4_FFN_DECODE_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_Q4_FFN_DECODE_NUM_WARPS must be 1, 2, 4, or 8")
    if Q4_FFN_DECODE_WAVES_PER_EU not in (1, 2, 3, 4):
        raise RuntimeError("FLAGOS_Q4_FFN_DECODE_WAVES_PER_EU must be 1, 2, 3, or 4")
    decode_rows = max(16, 2 * Q4_FFN_DECODE_BLOCK_M)
    decode_blocks = 3
    decode_k = decode_blocks * common.QK_K
    gate_packed, gate_dequantized = common.make_q4_k_weights(decode_rows, decode_blocks)
    up_packed, up_dequantized = common.make_q4_k_weights(decode_rows, decode_blocks)
    decode_x = torch.randn((decode_k,), device=device, dtype=torch.float32)
    decode_output = torch.empty((decode_rows,), device=device, dtype=torch.float32)
    flagos_ffn_swiglu_q4_k_f32_decode[
        (triton.cdiv(decode_rows, Q4_FFN_DECODE_BLOCK_M),)
    ](
        gate_packed, gate_packed.view(torch.float16),
        up_packed, up_packed.view(torch.float16),
        decode_x, decode_output, decode_k, decode_rows,
        BLOCK_M=Q4_FFN_DECODE_BLOCK_M,
        num_warps=Q4_FFN_DECODE_NUM_WARPS,
        waves_per_eu=Q4_FFN_DECODE_WAVES_PER_EU,
    )
    gate_reference = gate_dequantized @ decode_x
    up_reference = up_dequantized @ decode_x
    torch.testing.assert_close(
        decode_output, torch.nn.functional.silu(gate_reference) * up_reference,
        rtol=5e-4, atol=5e-2,
    )


def compile_q40_ffn_decode(device: torch.device) -> None:
    if (Q40_FFN_DECODE_BLOCK_M <= 0 or Q40_FFN_DECODE_BLOCK_M > 32 or
            Q40_FFN_DECODE_BLOCK_M & (Q40_FFN_DECODE_BLOCK_M - 1)):
        raise RuntimeError("FLAGOS_Q40_FFN_DECODE_BLOCK_M must be a power of two from 1 to 32")
    if Q40_FFN_DECODE_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_Q40_FFN_DECODE_NUM_WARPS must be 1, 2, 4, or 8")
    if Q40_FFN_DECODE_WAVES_PER_EU not in (1, 2, 3, 4):
        raise RuntimeError("FLAGOS_Q40_FFN_DECODE_WAVES_PER_EU must be 1, 2, 3, or 4")
    decode_rows = max(16, 2 * Q40_FFN_DECODE_BLOCK_M)
    decode_blocks = 5
    decode_k = decode_blocks * common.QK4_0
    gate_packed, gate_dequantized = common.make_q4_0_weights(decode_rows, decode_blocks)
    up_packed, up_dequantized = common.make_q4_0_weights(decode_rows, decode_blocks)
    decode_x = torch.randn((decode_k,), device=device, dtype=torch.float32)
    decode_output = torch.empty((decode_rows,), device=device, dtype=torch.float32)
    flagos_ffn_swiglu_q4_0_f32_decode[
        (triton.cdiv(decode_rows, Q40_FFN_DECODE_BLOCK_M),)
    ](
        gate_packed, gate_packed.view(torch.float16),
        up_packed, up_packed.view(torch.float16),
        decode_x, decode_output, decode_k, decode_rows,
        BLOCK_M=Q40_FFN_DECODE_BLOCK_M,
        num_warps=Q40_FFN_DECODE_NUM_WARPS,
        waves_per_eu=Q40_FFN_DECODE_WAVES_PER_EU,
    )
    gate_reference = gate_dequantized @ decode_x
    up_reference = up_dequantized @ decode_x
    torch.testing.assert_close(
        decode_output, torch.nn.functional.silu(gate_reference) * up_reference,
        rtol=5e-4, atol=5e-2,
    )


def compile_q40_ffn_decode_staged(device: torch.device) -> None:
    if (Q40_FFN_DECODE_BLOCK_M <= 0 or Q40_FFN_DECODE_BLOCK_M > 32 or
            Q40_FFN_DECODE_BLOCK_M & (Q40_FFN_DECODE_BLOCK_M - 1)):
        raise RuntimeError("FLAGOS_Q40_FFN_DECODE_BLOCK_M must be a power of two from 1 to 32")
    if Q40_FFN_DECODE_STAGED_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError(
            "FLAGOS_Q40_FFN_DECODE_STAGED_NUM_WARPS must be 1, 2, 4, or 8")
    if Q40_FFN_DECODE_STAGED_WAVES_PER_EU not in (1, 2, 3, 4):
        raise RuntimeError(
            "FLAGOS_Q40_FFN_DECODE_STAGED_WAVES_PER_EU must be 1, 2, 3, or 4")
    decode_rows = max(16, 2 * Q40_FFN_DECODE_BLOCK_M)
    decode_blocks = 5
    decode_k = decode_blocks * common.QK4_0
    gate_packed, gate_dequantized = common.make_q4_0_weights(decode_rows, decode_blocks)
    up_packed, up_dequantized = common.make_q4_0_weights(decode_rows, decode_blocks)
    decode_x = torch.randn((decode_k,), device=device, dtype=torch.float32)
    decode_output = torch.empty((decode_rows,), device=device, dtype=torch.float32)
    flagos_ffn_swiglu_q4_0_f32_decode_staged[
        (triton.cdiv(decode_rows, Q40_FFN_DECODE_BLOCK_M),)
    ](
        gate_packed, gate_packed.view(torch.float16),
        up_packed, up_packed.view(torch.float16),
        decode_x, decode_output, decode_k, decode_rows,
        BLOCK_M=Q40_FFN_DECODE_BLOCK_M,
        num_warps=Q40_FFN_DECODE_STAGED_NUM_WARPS,
        waves_per_eu=Q40_FFN_DECODE_STAGED_WAVES_PER_EU,
    )
    gate_reference = gate_dequantized @ decode_x
    up_reference = up_dequantized @ decode_x
    torch.testing.assert_close(
        decode_output, torch.nn.functional.silu(gate_reference) * up_reference,
        rtol=5e-4, atol=5e-2,
    )


def compile_q4_ffn_decode_staged(device: torch.device) -> None:
    if (Q4_FFN_DECODE_BLOCK_M <= 0 or Q4_FFN_DECODE_BLOCK_M > 32 or
            Q4_FFN_DECODE_BLOCK_M & (Q4_FFN_DECODE_BLOCK_M - 1)):
        raise RuntimeError("FLAGOS_Q4_FFN_DECODE_BLOCK_M must be a power of two from 1 to 32")
    if Q4_FFN_DECODE_STAGED_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError(
            "FLAGOS_Q4_FFN_DECODE_STAGED_NUM_WARPS must be 1, 2, 4, or 8")
    if Q4_FFN_DECODE_STAGED_WAVES_PER_EU not in (1, 2, 3, 4):
        raise RuntimeError(
            "FLAGOS_Q4_FFN_DECODE_STAGED_WAVES_PER_EU must be 1, 2, 3, or 4")
    decode_rows = max(16, 2 * Q4_FFN_DECODE_BLOCK_M)
    decode_blocks = 3
    decode_k = decode_blocks * common.QK_K
    gate_packed, gate_dequantized = common.make_q4_k_weights(decode_rows, decode_blocks)
    up_packed, up_dequantized = common.make_q4_k_weights(decode_rows, decode_blocks)
    decode_x = torch.randn((decode_k,), device=device, dtype=torch.float32)
    decode_output = torch.empty((decode_rows,), device=device, dtype=torch.float32)
    flagos_ffn_swiglu_q4_k_f32_decode_staged[
        (triton.cdiv(decode_rows, Q4_FFN_DECODE_BLOCK_M),)
    ](
        gate_packed, gate_packed.view(torch.float16),
        up_packed, up_packed.view(torch.float16),
        decode_x, decode_output, decode_k, decode_rows,
        BLOCK_M=Q4_FFN_DECODE_BLOCK_M,
        num_warps=Q4_FFN_DECODE_STAGED_NUM_WARPS,
        waves_per_eu=Q4_FFN_DECODE_STAGED_WAVES_PER_EU,
    )
    gate_reference = gate_dequantized @ decode_x
    up_reference = up_dequantized @ decode_x
    torch.testing.assert_close(
        decode_output, torch.nn.functional.silu(gate_reference) * up_reference,
        rtol=5e-4, atol=5e-2,
    )


def compile_ffn_fusion(device: torch.device) -> None:
    ffn_rows, ffn_columns, ffn_k = 64, 37, 256
    ffn_gate = torch.randn((ffn_rows, ffn_k), device=device, dtype=torch.float16)
    ffn_up = torch.randn((ffn_rows, ffn_k), device=device, dtype=torch.float16)
    ffn_x = torch.randn((ffn_columns, ffn_k), device=device, dtype=torch.float32)
    ffn_output = torch.empty((ffn_columns, ffn_rows), device=device, dtype=torch.float32)
    flagos_ffn_swiglu_f16_f32_batched[
        (triton.cdiv(ffn_rows, FFN_MATMUL_BLOCK_M),
         triton.cdiv(ffn_columns, FFN_MATMUL_BLOCK_N))
    ](
        ffn_gate, ffn_up, ffn_x, ffn_output,
        ffn_k, ffn_rows, ffn_columns,
        BLOCK_M=FFN_MATMUL_BLOCK_M, BLOCK_N=FFN_MATMUL_BLOCK_N,
        BLOCK_K=FFN_MATMUL_BLOCK_K, num_warps=FFN_MATMUL_NUM_WARPS,
    )
    ffn_activation_ref = ffn_x.half().float()
    gate_ref = ffn_activation_ref @ ffn_gate.T.float()
    up_ref = ffn_activation_ref @ ffn_up.T.float()
    torch.testing.assert_close(
        ffn_output, torch.nn.functional.silu(gate_ref) * up_ref,
        rtol=5e-2, atol=5e-2,
    )
    grouped_ffn_output = torch.empty_like(ffn_output)
    flagos_ffn_swiglu_f16_f32_grouped[
        (triton.cdiv(ffn_rows, FFN_MATMUL_BLOCK_M) *
         triton.cdiv(ffn_columns, FFN_MATMUL_BLOCK_N),)
    ](
        ffn_gate, ffn_up, ffn_x, grouped_ffn_output,
        ffn_k, ffn_rows, ffn_columns,
        BLOCK_M=FFN_MATMUL_BLOCK_M, BLOCK_N=FFN_MATMUL_BLOCK_N,
        BLOCK_K=FFN_MATMUL_BLOCK_K, GROUP_M=FFN_MATMUL_GROUP_M,
        num_warps=FFN_MATMUL_NUM_WARPS,
        num_stages=FFN_MATMUL_GROUPED_NUM_STAGES,
        matrix_instr_nonkdim=16,
        waves_per_eu=FFN_MATMUL_GROUPED_WAVES_PER_EU,
    )
    torch.testing.assert_close(
        grouped_ffn_output, torch.nn.functional.silu(gate_ref) * up_ref,
        rtol=5e-2, atol=5e-2,
    )


def compile_ffn_down_f16(device: torch.device) -> None:
    """Compile and validate the two-symbol graph-private F16 FFN pipeline."""
    ffn_rows, ffn_columns, ffn_k = 64, 37, 256
    down_rows = 48
    gate = torch.randn((ffn_rows, ffn_k), device=device, dtype=torch.float16) * 0.1
    up = torch.randn((ffn_rows, ffn_k), device=device, dtype=torch.float16) * 0.1
    x = torch.randn((ffn_columns, ffn_k), device=device, dtype=torch.float32) * 0.1
    scratch = torch.empty((ffn_columns, ffn_rows), device=device, dtype=torch.float16)
    flagos_ffn_swiglu_f16_f16_grouped[
        (triton.cdiv(ffn_rows, FFN_MATMUL_BLOCK_M) *
         triton.cdiv(ffn_columns, FFN_MATMUL_BLOCK_N),)
    ](
        gate, up, x, scratch, ffn_k, ffn_rows, ffn_columns,
        BLOCK_M=FFN_MATMUL_BLOCK_M, BLOCK_N=FFN_MATMUL_BLOCK_N,
        BLOCK_K=FFN_MATMUL_BLOCK_K, GROUP_M=FFN_MATMUL_GROUP_M,
        num_warps=FFN_MATMUL_NUM_WARPS,
        num_stages=FFN_MATMUL_GROUPED_NUM_STAGES,
        matrix_instr_nonkdim=16,
        waves_per_eu=FFN_MATMUL_GROUPED_WAVES_PER_EU,
    )
    activation_ref = x.half().float()
    gate_ref = activation_ref @ gate.T.float()
    up_ref = activation_ref @ up.T.float()
    scratch_ref = (torch.nn.functional.silu(gate_ref) * up_ref).half()
    torch.testing.assert_close(scratch, scratch_ref, rtol=3e-2, atol=3e-2)

    down_weight = torch.randn(
        (down_rows, ffn_rows), device=device, dtype=torch.float16) * 0.1
    output = torch.empty((ffn_columns, down_rows), device=device, dtype=torch.float32)
    flagos_mul_mat_f16_f16_grouped[
        (triton.cdiv(down_rows, FFN_DOWN_MATMUL_BLOCK_M) *
         triton.cdiv(ffn_columns, FFN_DOWN_MATMUL_BLOCK_N),)
    ](
        down_weight, scratch, output, ffn_rows, down_rows, ffn_columns,
        BLOCK_M=FFN_DOWN_MATMUL_BLOCK_M, BLOCK_N=FFN_DOWN_MATMUL_BLOCK_N,
        BLOCK_K=FFN_DOWN_MATMUL_BLOCK_K, GROUP_M=FFN_DOWN_MATMUL_GROUP_M,
        num_warps=FFN_DOWN_MATMUL_NUM_WARPS,
        num_stages=FFN_DOWN_MATMUL_NUM_STAGES,
        matrix_instr_nonkdim=16,
        waves_per_eu=FFN_DOWN_MATMUL_WAVES_PER_EU,
    )
    torch.testing.assert_close(
        output, scratch_ref.float() @ down_weight.T.float(),
        rtol=3e-2, atol=3e-2,
    )


def compile_additional() -> None:
    device = _device()
    if os.environ.get("FLAGOS_AMD_ONLY_RESIDUAL") == "1":
        compile_residual(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_RESIDUAL_NARROW") == "1":
        compile_residual_narrow(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_Q4_FFN_DECODE") == "1":
        compile_q4_ffn_decode(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_Q40_FFN_DECODE") == "1":
        compile_q40_ffn_decode(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_Q40_FFN_DECODE_STAGED") == "1":
        compile_q40_ffn_decode_staged(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_Q4_FFN_DECODE_STAGED") == "1":
        compile_q4_ffn_decode_staged(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_F16_GEMM") == "1":
        # Tuning mode: compile only the dense prefill kernel.  The normal
        # package generator intentionally exercises every AMD kernel, but
        # recompiling those kernels into an existing cache creates duplicate
        # artifacts and obscures tile-level experiments.
        f16_rows, f16_columns, f16_k = 64, 37, 256
        f16_weights = torch.randn((f16_rows, f16_k), device=device, dtype=torch.float16)
        f16_activation = torch.randn((f16_columns, f16_k), device=device, dtype=torch.float32)
        f16_output = torch.empty((f16_columns, f16_rows), device=device, dtype=torch.float32)
        flagos_mul_mat_f16_f32_batched[
            (triton.cdiv(f16_rows, F16_MATMUL_BLOCK_M),
             triton.cdiv(f16_columns, F16_MATMUL_BLOCK_N))
        ](
            f16_weights, f16_activation, f16_output,
            f16_k, f16_rows, f16_columns,
            BLOCK_M=F16_MATMUL_BLOCK_M, BLOCK_N=F16_MATMUL_BLOCK_N,
            BLOCK_K=F16_MATMUL_BLOCK_K, num_warps=F16_MATMUL_NUM_WARPS,
        )
        torch.testing.assert_close(
            f16_output, f16_activation @ f16_weights.T.float(),
            rtol=3e-2, atol=3e-2,
        )
        grouped_output = torch.empty_like(f16_output)
        flagos_mul_mat_f16_f32_grouped[
            (triton.cdiv(f16_rows, F16_MATMUL_BLOCK_M) *
             triton.cdiv(f16_columns, F16_MATMUL_BLOCK_N),)
        ](
            f16_weights, f16_activation, grouped_output,
            f16_k, f16_rows, f16_columns,
            BLOCK_M=F16_MATMUL_BLOCK_M, BLOCK_N=F16_MATMUL_BLOCK_N,
            BLOCK_K=F16_MATMUL_BLOCK_K, GROUP_M=F16_MATMUL_GROUP_M,
            num_warps=F16_MATMUL_NUM_WARPS,
            num_stages=F16_MATMUL_GROUPED_NUM_STAGES,
            matrix_instr_nonkdim=16,
            waves_per_eu=F16_MATMUL_GROUPED_WAVES_PER_EU,
        )
        torch.testing.assert_close(
            grouped_output, f16_activation @ f16_weights.T.float(),
            rtol=3e-2, atol=3e-2,
        )
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_FFN_FUSION") == "1":
        compile_ffn_fusion(device)
        torch.cuda.synchronize()
        return
    if os.environ.get("FLAGOS_AMD_ONLY_FFN_DOWN_F16") == "1":
        compile_ffn_down_f16(device)
        torch.cuda.synchronize()
        return
    rows, cols = 5, AMD_ROW_BLOCK_SIZE
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    grid = (rows,)
    flat_n = rows * cols
    flagos_silu_f32[(triton.cdiv(flat_n, common.BLOCK_SIZE),)](
        x, out, flat_n, BLOCK=common.BLOCK_SIZE, num_warps=NUM_WARPS)
    torch.testing.assert_close(out, torch.nn.functional.silu(x), rtol=2e-5, atol=2e-5)
    if (AMD_SILU_MUL_BLOCK_SIZE <= 0 or
            AMD_SILU_MUL_BLOCK_SIZE & (AMD_SILU_MUL_BLOCK_SIZE - 1)):
        raise RuntimeError("FLAGOS_AMD_SILU_MUL_BLOCK_SIZE must be a positive power of two")
    if AMD_SILU_MUL_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_SILU_MUL_NUM_WARPS must be 1, 2, 4, or 8")
    flagos_silu_mul_f32[(triton.cdiv(flat_n, AMD_SILU_MUL_BLOCK_SIZE),)](
        x, y, out, flat_n, BLOCK=AMD_SILU_MUL_BLOCK_SIZE,
        num_warps=AMD_SILU_MUL_NUM_WARPS)
    torch.testing.assert_close(
        out, torch.nn.functional.silu(x) * y, rtol=2e-5, atol=2e-5)
    flagos_sigmoid_mul_f32[(triton.cdiv(flat_n, AMD_SILU_MUL_BLOCK_SIZE),)](
        x, y, out, flat_n, BLOCK=AMD_SILU_MUL_BLOCK_SIZE,
        num_warps=AMD_SILU_MUL_NUM_WARPS)
    torch.testing.assert_close(out, torch.sigmoid(x) * y, rtol=2e-5, atol=2e-5)
    flagos_softplus_mul_f32[(triton.cdiv(flat_n, AMD_SILU_MUL_BLOCK_SIZE),)](
        x, y, out, flat_n, BLOCK=AMD_SILU_MUL_BLOCK_SIZE,
        num_warps=AMD_SILU_MUL_NUM_WARPS)
    torch.testing.assert_close(
        out, torch.nn.functional.softplus(x) * y, rtol=2e-5, atol=2e-5)
    if AMD_GDN_ALPHA_GATE_WIDTH <= 0:
        raise RuntimeError("FLAGOS_AMD_GDN_ALPHA_GATE_WIDTH must be positive")
    if (AMD_GDN_ALPHA_GATE_BLOCK_SIZE <= 0 or
            AMD_GDN_ALPHA_GATE_BLOCK_SIZE & (AMD_GDN_ALPHA_GATE_BLOCK_SIZE - 1)):
        raise RuntimeError(
            "FLAGOS_AMD_GDN_ALPHA_GATE_BLOCK_SIZE must be a positive power of two")
    if AMD_GDN_ALPHA_GATE_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_GDN_ALPHA_GATE_NUM_WARPS must be 1, 2, 4, or 8")
    alpha_n = 5 * AMD_GDN_ALPHA_GATE_WIDTH
    alpha = torch.randn((alpha_n,), device=device, dtype=torch.float32)
    alpha_bias = torch.randn(
        (AMD_GDN_ALPHA_GATE_WIDTH,), device=device, dtype=torch.float32)
    alpha_scale = torch.randn_like(alpha_bias)
    alpha_out = torch.empty_like(alpha)
    flagos_gdn_alpha_gate_f32[(triton.cdiv(alpha_n, AMD_GDN_ALPHA_GATE_BLOCK_SIZE),)](
        alpha, alpha_bias, alpha_scale, alpha_out, alpha_n,
        WIDTH=AMD_GDN_ALPHA_GATE_WIDTH, BLOCK=AMD_GDN_ALPHA_GATE_BLOCK_SIZE,
        num_warps=AMD_GDN_ALPHA_GATE_NUM_WARPS)
    alpha_expected = torch.nn.functional.softplus(
        alpha + alpha_bias.repeat(alpha_n // AMD_GDN_ALPHA_GATE_WIDTH))
    alpha_expected *= alpha_scale.repeat(alpha_n // AMD_GDN_ALPHA_GATE_WIDTH)
    torch.testing.assert_close(alpha_out, alpha_expected, rtol=2e-5, atol=2e-5)

    norm_out = torch.empty_like(x)
    mul_out = torch.empty_like(x)
    weight = torch.randn((cols,), device=device, dtype=torch.float32)
    eps = 1e-6
    flagos_add_rms_norm_mul_f32[grid](
        norm_out, mul_out, x, y, weight, cols, eps,
        BLOCK=AMD_ROW_BLOCK_SIZE, num_warps=NUM_WARPS)
    biased = x + y
    expected_norm = biased * torch.rsqrt(biased.square().mean(dim=1, keepdim=True) + eps)
    torch.testing.assert_close(norm_out, expected_norm, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(mul_out, expected_norm * weight, rtol=2e-5, atol=2e-5)

    # Exercise the one-output aliases used when GGML reuses the dead RMSNorm
    # allocation for the terminal MUL result.
    inplace_out = torch.empty_like(x)
    flagos_rms_norm_mul_inplace_f32[grid](
        inplace_out, x, weight, cols, eps,
        BLOCK=AMD_ROW_BLOCK_SIZE, num_warps=NUM_WARPS)
    plain = x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + eps)
    torch.testing.assert_close(inplace_out, plain * weight, rtol=2e-5, atol=2e-5)
    if (AMD_RMS_NORM_NARROW_BLOCK_SIZE <= 0 or
            AMD_RMS_NORM_NARROW_BLOCK_SIZE & (AMD_RMS_NORM_NARROW_BLOCK_SIZE - 1)):
        raise RuntimeError(
            "FLAGOS_AMD_RMS_NORM_NARROW_BLOCK_SIZE must be a positive power of two")
    if AMD_RMS_NORM_NARROW_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_RMS_NORM_NARROW_NUM_WARPS must be 1, 2, 4, or 8")
    narrow_rows = 33
    narrow_cols = AMD_RMS_NORM_NARROW_BLOCK_SIZE
    narrow_x = torch.randn((narrow_rows, narrow_cols), device=device, dtype=torch.float32)
    narrow_weight = torch.randn((narrow_cols,), device=device, dtype=torch.float32)
    narrow_output = torch.empty_like(narrow_x)
    flagos_rms_norm_mul_inplace_f32_narrow[(narrow_rows,)](
        narrow_output, narrow_x, narrow_weight, narrow_cols, eps,
        BLOCK=AMD_RMS_NORM_NARROW_BLOCK_SIZE,
        num_warps=AMD_RMS_NORM_NARROW_NUM_WARPS)
    narrow_expected = narrow_x * torch.rsqrt(
        narrow_x.square().mean(dim=1, keepdim=True) + eps) * narrow_weight
    torch.testing.assert_close(
        narrow_output, narrow_expected, rtol=2e-5, atol=2e-5)
    flagos_add_rms_norm_mul_inplace_f32[grid](
        inplace_out, x, y, weight, cols, eps,
        BLOCK=AMD_ROW_BLOCK_SIZE, num_warps=NUM_WARPS)
    biased_expected = x + y
    biased_expected = biased_expected * torch.rsqrt(
        biased_expected.square().mean(dim=1, keepdim=True) + eps)
    torch.testing.assert_close(inplace_out, biased_expected * weight, rtol=2e-5, atol=2e-5)

    compile_residual(device)
    compile_residual_narrow(device)

    rope_rows, rope_heads, head_dim, n_dims = 3, 2, 128, 128
    rope_x = torch.randn((rope_rows, rope_heads, head_dim), device=device, dtype=torch.float32)
    rope_positions = torch.tensor([4, 7, 12], device=device, dtype=torch.int32)
    rope_indices = torch.tensor([9, 2, 6], device=device, dtype=torch.int64)
    rope_out = torch.zeros((16, rope_heads * head_dim), device=device, dtype=torch.float16)
    flagos_rope_kv_store_f32_f16[(triton.cdiv(rope_rows * rope_heads * head_dim, common.BLOCK_SIZE),)](
        rope_x, rope_positions, rope_indices, rope_out,
        rope_heads * head_dim, rope_rows, rope_out.shape[0], head_dim, rope_heads, n_dims,
        1_000_000.0, 1.0, BLOCK=common.BLOCK_SIZE, num_warps=NUM_WARPS)
    flat_rope = rope_x.reshape(rope_rows, rope_heads, head_dim)
    half = n_dims // 2
    dims = torch.arange(half, device=device, dtype=torch.float32)
    theta = rope_positions.float()[:, None, None] * torch.pow(
        torch.tensor(1_000_000.0, device=device), -2.0 * dims / n_dims)
    expected_rope = torch.empty_like(flat_rope)
    expected_rope[:, :, :half] = (
        flat_rope[:, :, :half] * torch.cos(theta) - flat_rope[:, :, half:] * torch.sin(theta))
    expected_rope[:, :, half:] = (
        flat_rope[:, :, :half] * torch.sin(theta) + flat_rope[:, :, half:] * torch.cos(theta))
    torch.testing.assert_close(
        rope_out[rope_indices].float().reshape_as(expected_rope), expected_rope,
        rtol=2e-3, atol=2e-3)

    norm_rope_weight = torch.randn((head_dim,), device=device, dtype=torch.float32)
    norm_rope_out = torch.empty_like(rope_x)
    flagos_rms_norm_mul_rope_neox_f32[(rope_rows * rope_heads,)](
        rope_x, norm_rope_weight, rope_positions, norm_rope_out,
        head_dim, rope_heads, rope_rows, n_dims, eps, 1_000_000.0, 1.0,
        BLOCK=head_dim, num_warps=NUM_WARPS)
    normalized_rope = rope_x * torch.rsqrt(
        rope_x.square().mean(dim=2, keepdim=True) + eps)
    scaled_rope = normalized_rope * norm_rope_weight
    expected_norm_rope = torch.empty_like(scaled_rope)
    expected_norm_rope[:, :, :half] = (
        scaled_rope[:, :, :half] * torch.cos(theta) -
        scaled_rope[:, :, half:] * torch.sin(theta))
    expected_norm_rope[:, :, half:] = (
        scaled_rope[:, :, :half] * torch.sin(theta) +
        scaled_rope[:, :, half:] * torch.cos(theta))
    torch.testing.assert_close(
        norm_rope_out, expected_norm_rope, rtol=2e-5, atol=2e-5)

    norm_rope_cache = torch.zeros_like(rope_out)
    flagos_rms_norm_mul_rope_kv_store_neox_f32_f16[(rope_rows * rope_heads,)](
        rope_x, norm_rope_weight, rope_positions, rope_indices, norm_rope_cache,
        head_dim, rope_heads, rope_rows, norm_rope_cache.shape[0], n_dims,
        eps, 1_000_000.0, 1.0,
        BLOCK=head_dim, num_warps=NUM_WARPS)
    torch.testing.assert_close(
        norm_rope_cache[rope_indices].float().reshape_as(expected_norm_rope),
        expected_norm_rope, rtol=2e-3, atol=2e-3)

    # The unmasked ABI retains the mask pointer slot for a stable C++ binding;
    # the kernel does not dereference it, so any valid device pointer is safe.
    flagos_soft_max_unmasked_f32[grid](x, x, out, cols, 0.125,
                              BLOCK=AMD_SOFTMAX_BLOCK_SIZE, num_warps=NUM_WARPS)
    torch.testing.assert_close(out, torch.softmax(x * 0.125, dim=1), rtol=2e-5, atol=2e-5)
    mask = torch.randn((rows, cols), device=device, dtype=torch.float16)
    flagos_soft_max_masked_f32_f16[grid](x, mask, out, cols, 0.125,
                                     BLOCK=AMD_SOFTMAX_BLOCK_SIZE, num_warps=NUM_WARPS)
    torch.testing.assert_close(out, torch.softmax(x * 0.125 + mask, dim=1), rtol=2e-5, atol=2e-5)

    query_length, key_length, q_heads, kv_heads = 37, 61, 12, 2
    q = torch.randn((query_length, q_heads, ATTENTION_HEAD_DIM), device=device, dtype=torch.float32)
    k = torch.randn((key_length, kv_heads, ATTENTION_HEAD_DIM), device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    # Keep the GGML physical layout: token is the contiguous dimension and
    # query is the row stride.  A transposed PyTorch view gives the same
    # strides and prevents Triton from baking the query stride into the ABI.
    attention_mask = torch.zeros((query_length, key_length), device=device,
                                 dtype=torch.float16).T
    q_positions = torch.arange(query_length, device=device) + key_length - query_length
    k_positions = torch.arange(key_length, device=device)
    attention_mask = torch.where(
        k_positions[:, None] > q_positions[None, :],
        torch.full_like(attention_mask, -float("inf")), attention_mask)
    attention_output = torch.empty((query_length, q_heads, ATTENTION_HEAD_DIM), device=device, dtype=torch.float32)
    flagos_flash_attn_prefill_f32_f16[
        (q_heads, triton.cdiv(query_length, ATTENTION_BLOCK_M))
    ](
        q, k, v, attention_mask, attention_output,
        query_length, key_length, q_heads // kv_heads,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), attention_mask.stride(0), attention_mask.stride(1),
        attention_output.stride(0), attention_output.stride(1),
        ATTENTION_HEAD_DIM ** -0.5,
        HEAD_DIM=ATTENTION_HEAD_DIM, BLOCK_M=ATTENTION_BLOCK_M,
        BLOCK_N=ATTENTION_BLOCK_N, num_warps=NUM_WARPS,
    )
    repeated_k = k.float().repeat_interleave(q_heads // kv_heads, dim=1)
    repeated_v = v.float().repeat_interleave(q_heads // kv_heads, dim=1)
    scores = torch.einsum("thd,shd->hts", q, repeated_k) * (ATTENTION_HEAD_DIM ** -0.5)
    scores += attention_mask.float().T[None, :, :]
    expected = torch.einsum("hts,shd->thd", torch.softmax(scores, dim=-1), repeated_v)
    torch.testing.assert_close(attention_output, expected, rtol=1e-2, atol=1e-2)

    if os.environ.get("FLAGOS_AMD_EMIT_Q4_FFN_DECODE", "0") == "1":
        compile_q4_ffn_decode(device)
    if os.environ.get("FLAGOS_AMD_EMIT_Q40_FFN_DECODE", "0") == "1":
        compile_q40_ffn_decode(device)
    if os.environ.get("FLAGOS_AMD_EMIT_Q40_FFN_DECODE_STAGED", "0") == "1":
        compile_q40_ffn_decode_staged(device)
    if os.environ.get("FLAGOS_AMD_EMIT_Q4_FFN_DECODE_STAGED", "0") == "1":
        compile_q4_ffn_decode_staged(device)

    # Validate the dense F16 GEMM used by the optional dequant-cache path.
    # Keep this ABI independent of the quantized source type: both Q4_K and
    # Q6_K weights are decoded to the same row-major F16 matrix.
    f16_rows, f16_columns, f16_k = 64, 37, 256
    f16_weights = torch.randn((f16_rows, f16_k), device=device, dtype=torch.float16)
    f16_activation = torch.randn((f16_columns, f16_k), device=device, dtype=torch.float32)
    f16_output = torch.empty((f16_columns, f16_rows), device=device, dtype=torch.float32)
    flagos_mul_mat_f16_f32_batched[
        (triton.cdiv(f16_rows, F16_MATMUL_BLOCK_M),
         triton.cdiv(f16_columns, F16_MATMUL_BLOCK_N))
    ](
        f16_weights, f16_activation, f16_output,
        f16_k, f16_rows, f16_columns,
        BLOCK_M=F16_MATMUL_BLOCK_M, BLOCK_N=F16_MATMUL_BLOCK_N,
        BLOCK_K=F16_MATMUL_BLOCK_K, num_warps=F16_MATMUL_NUM_WARPS,
    )
    torch.testing.assert_close(
        f16_output, f16_activation @ f16_weights.T.float(),
        rtol=3e-2, atol=3e-2,
    )
    grouped_output = torch.empty_like(f16_output)
    flagos_mul_mat_f16_f32_grouped[
        (triton.cdiv(f16_rows, F16_MATMUL_BLOCK_M) *
         triton.cdiv(f16_columns, F16_MATMUL_BLOCK_N),)
    ](
        f16_weights, f16_activation, grouped_output,
        f16_k, f16_rows, f16_columns,
        BLOCK_M=F16_MATMUL_BLOCK_M, BLOCK_N=F16_MATMUL_BLOCK_N,
        BLOCK_K=F16_MATMUL_BLOCK_K, GROUP_M=F16_MATMUL_GROUP_M,
        num_warps=F16_MATMUL_NUM_WARPS,
        num_stages=F16_MATMUL_GROUPED_NUM_STAGES,
        matrix_instr_nonkdim=16,
        waves_per_eu=F16_MATMUL_GROUPED_WAVES_PER_EU,
    )
    torch.testing.assert_close(
        grouped_output, f16_activation @ f16_weights.T.float(),
        rtol=3e-2, atol=3e-2,
    )
    if os.environ.get("FLAGOS_AMD_EMIT_FFN_FUSION", "0") == "1":
        ffn_rows, ffn_columns, ffn_k = 64, 37, 256
        ffn_gate = torch.randn((ffn_rows, ffn_k), device=device, dtype=torch.float16)
        ffn_up = torch.randn((ffn_rows, ffn_k), device=device, dtype=torch.float16)
        ffn_x = torch.randn((ffn_columns, ffn_k), device=device, dtype=torch.float32)
        ffn_output = torch.empty((ffn_columns, ffn_rows), device=device, dtype=torch.float32)
        flagos_ffn_swiglu_f16_f32_batched[
            (triton.cdiv(ffn_rows, FFN_MATMUL_BLOCK_M),
             triton.cdiv(ffn_columns, FFN_MATMUL_BLOCK_N))
        ](
            ffn_gate, ffn_up, ffn_x, ffn_output,
            ffn_k, ffn_rows, ffn_columns,
            BLOCK_M=FFN_MATMUL_BLOCK_M, BLOCK_N=FFN_MATMUL_BLOCK_N,
            BLOCK_K=FFN_MATMUL_BLOCK_K, num_warps=FFN_MATMUL_NUM_WARPS,
        )
        # The kernel deliberately casts the F32 activation tile to F16 before
        # the dot, matching the dense F16-cache GEMM ABI.  Compare against the
        # same arithmetic contract rather than a higher-precision F32 GEMM;
        # the latter magnifies harmless input-rounding differences after the
        # SiLU product, especially for near-zero outputs.
        ffn_activation_ref = ffn_x.half().float()
        gate_ref = ffn_activation_ref @ ffn_gate.T.float()
        up_ref = ffn_activation_ref @ ffn_up.T.float()
        torch.testing.assert_close(
            ffn_output, torch.nn.functional.silu(gate_ref) * up_ref,
            rtol=5e-2, atol=5e-2,
        )
        grouped_ffn_output = torch.empty_like(ffn_output)
        flagos_ffn_swiglu_f16_f32_grouped[
            (triton.cdiv(ffn_rows, FFN_MATMUL_BLOCK_M) *
             triton.cdiv(ffn_columns, FFN_MATMUL_BLOCK_N),)
        ](
            ffn_gate, ffn_up, ffn_x, grouped_ffn_output,
            ffn_k, ffn_rows, ffn_columns,
            BLOCK_M=FFN_MATMUL_BLOCK_M, BLOCK_N=FFN_MATMUL_BLOCK_N,
            BLOCK_K=FFN_MATMUL_BLOCK_K, GROUP_M=FFN_MATMUL_GROUP_M,
            num_warps=FFN_MATMUL_NUM_WARPS,
            num_stages=FFN_MATMUL_GROUPED_NUM_STAGES,
            matrix_instr_nonkdim=16,
            waves_per_eu=FFN_MATMUL_GROUPED_WAVES_PER_EU,
        )
        torch.testing.assert_close(
            grouped_ffn_output, torch.nn.functional.silu(gate_ref) * up_ref,
            rtol=5e-2, atol=5e-2,
        )
    if os.environ.get("FLAGOS_AMD_EMIT_FFN_DOWN_F16", "0") == "1":
        compile_ffn_down_f16(device)
    torch.cuda.synchronize()


def _find_artifact(cache_dir: Path, name: str) -> tuple[Path, dict]:
    matches = [p for p in cache_dir.rglob(f"{name}.hsaco") if p.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}.hsaco, found {len(matches)}")
    source = matches[0]
    metadata_path = source.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    return source, metadata


def copy_artifact(cache_dir: Path, output_dir: Path, name: str, block_size: int,
                  tile_m: int = 0, tile_n: int = 0, tile_k: int = 0) -> dict:
    source, metadata = _find_artifact(cache_dir, name)
    global_scratch_size = metadata.get("global_scratch_size", 0)
    profile_scratch_size = metadata.get("profile_scratch_size", 0)
    if global_scratch_size or profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={profile_scratch_size})")
    destination = output_dir / source.name
    shutil.copyfile(source, destination)
    target = metadata.get("target", {})
    function = globals().get(name)
    if function is None:
        function = getattr(common, name, None)
    arg_names = getattr(function, "arg_names", None)
    constexprs = getattr(function, "constexprs", None)
    if arg_names is None or constexprs is None:
        raise RuntimeError(f"kernel has no Triton signature: {name}")
    return {
        "name": metadata.get("name", name),
        "symbol": metadata.get("name", name),
        "file": destination.name,
        "shared": metadata.get("shared", 0),
        "num_warps": metadata["num_warps"],
        "warp_size": metadata.get("warp_size", target.get("warp_size", 32)),
        "block_size": block_size,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "argument_count": len(arg_names) - len(constexprs),
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": metadata.get("global_scratch_align", 1),
        "profile_scratch_size": profile_scratch_size,
        "profile_scratch_align": metadata.get("profile_scratch_align", 1),
    }


def amd_jit_specialization_attrs(pointer_count: int, scalar_indices: tuple[int, ...]) -> dict:
    """Reproduce the alignment attributes used by the model-shape HIP JIT."""
    attrs = {
        (index,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
        for index in range(pointer_count)
    }
    attrs.update({
        (index,): [["tt.divisibility", 16]]
        for index in scalar_indices
    })
    return attrs


def amd_q4_dequant_num_warps(arch: str, tuning_profile: str = "") -> int:
    """Choose the validated Q4_K cache-decode launch width for this target."""
    configured = os.environ.get("FLAGOS_Q4_DEQUANT_NUM_WARPS")
    if tuning_profile:
        _, contracts = tuning_profile_contracts(tuning_profile)
        num_warps = contracts["flagos_dequant_q4_k_f16"].num_warps
        if configured is not None and int(configured) != num_warps:
            raise RuntimeError(
                "FLAGOS_Q4_DEQUANT_NUM_WARPS conflicts with the tuning profile")
    else:
        num_warps = int(configured) if configured is not None else (
            2 if arch == "gfx1150" else NUM_WARPS
        )
    if num_warps not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_Q4_DEQUANT_NUM_WARPS must be 1, 2, 4, or 8")
    return num_warps


def compile_q4_dequant_kernel_without_launch(
        output_dir: Path, arch: str, tuning_profile: str = "") -> dict:
    """Compile the Q4_K-to-F16 cache fill with the target launch width."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    name = "flagos_dequant_q4_k_f16"
    signature = {
        "weights_u8": "*u8", "weights_f16": "*fp16", "output": "*fp16",
    }
    source = ASTSource(
        common.flagos_dequant_q4_k_f16,
        signature,
        {},
        attrs=amd_jit_specialization_attrs(3, ()),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": amd_q4_dequant_num_warps(arch, tuning_profile)},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={metadata.profile_scratch_size})"
        )
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    return {
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": common.QK_K,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 3,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }


def compile_q4_dequant_package_without_launch(output_dir: Path, arch: str) -> None:
    """Write a one-kernel Q4_K cache-decode tuning package."""
    kernel = compile_q4_dequant_kernel_without_launch(output_dir, arch)
    write_manifest(output_dir, arch, [kernel])


def compile_dense_gather_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile contiguous F32 gather without GPU initialization or launch."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("dense-gather requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_get_rows_f32"
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="ggml.get_rows.contiguous_f32.i32_indices.in_range.bitwise.v1",
        runtime_arguments=("table", "row_index", "output", "n_cols"),
        constexpr_arguments=("BLOCK",))
    if export.aot_divisibility:
        raise RuntimeError("generic F32 gather must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    block_size = 256
    source = ASTSource(
        export.function,
        {"table": "*fp32", "row_index": "*i32", "output": "*fp32",
         "n_cols": "i32", "BLOCK": "constexpr"},
        {"BLOCK": block_size},
        attrs=export.aot_attributes,
    )
    compiled = triton.compile(source, target=GPUTarget("hip", arch, 32),
                              options={"num_warps": NUM_WARPS})
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    artifact = output_dir / f"{name}.hsaco"
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name, "symbol": name, "file": artifact.name,
        "shared": metadata.shared, "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size, "block_size": block_size,
        "tile_m": 0, "tile_n": 0, "tile_k": 0, "argument_count": 4,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_aligned_gemm_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile generic and aligned GEMM from explicit FlagGems contracts."""
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("aligned-gemm requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    global FLAGGEMS_AOT_EXPORTS
    kernels = []
    for aligned in (False, True):
        name = "flagos_mul_mat_f16_f32_batched" + ("_aligned" if aligned else "")
        semantic = "ggml.gemm_f16_round_activation_f16_acc_f32" + (".ptr16_k_rows_multiple16" if aligned else "") + ".v1"
        export, _, _, _ = load_flaggems_aot_export(name=name, semantic=semantic,
            runtime_arguments=("weights", "x", "output", "k", "rows", "columns"),
            constexpr_arguments=("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        divisibility = tuple((argument, 16) for argument in ("weights", "x", "output", "k", "rows")) if aligned else ()
        if export.aot_divisibility != divisibility:
            raise RuntimeError(f"incompatible GEMM divisibility contract: {name}")
        FLAGGEMS_AOT_EXPORTS += (export,)
        signature = {"weights": "*fp16", "x": "*fp32", "output": "*fp32",
                     "k": "i32", "rows": "i32", "columns": "i32",
                     "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"}
        constants = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}
        compiled = triton.compile(ASTSource(export.function, signature, constants, attrs=export.aot_attributes),
            target=GPUTarget("hip", arch, 32), options={"num_warps": NUM_WARPS})
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
        artifact = output_dir / f"{name}.hsaco"
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": 128, "tile_m": 64, "tile_n": 128, "tile_k": 32,
            "argument_count": 6, "global_scratch_size": 0,
            "global_scratch_align": getattr(m, "global_scratch_align", 1),
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align,
            "aot_divisibility": divisibility})
    write_manifest(output_dir, arch, kernels)


def compile_tiled_ffn_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the checked tiled FFN export without GPU initialization."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("tiled-ffn requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_ffn_swiglu_f16_f32_batched_tiled"
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="transformer.ffn_f16_round_activation_f16_acc_f32_swiglu.ptr16_k_rows_multiple16.v1",
        runtime_arguments=("gate_weights", "up_weights", "x", "output", "k", "rows", "columns"),
        constexpr_arguments=("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    divisibility = tuple((argument, 16) for argument in ("gate_weights", "up_weights", "x", "output", "k", "rows"))
    if export.aot_divisibility != divisibility:
        raise RuntimeError("incompatible tiled FFN divisibility contract")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {"gate_weights": "*fp16", "up_weights": "*fp16", "x": "*fp32", "output": "*fp32",
        "k": "i32", "rows": "i32", "columns": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"}
    constants = {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}
    compiled = triton.compile(ASTSource(export.function, signature, constants, attrs=export.aot_attributes),
        target=GPUTarget("hip", arch, 32), options={"num_warps": 4})
    m = compiled.metadata
    if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
        raise RuntimeError("tiled FFN requires unsupported Triton scratch storage")
    artifact = output_dir / f"{name}.hsaco"
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{"name": name, "symbol": m.name, "file": artifact.name,
        "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
        "block_size": 128, "tile_m": 32, "tile_n": 128, "tile_k": 32,
        "argument_count": 7, "global_scratch_size": 0, "global_scratch_align": getattr(m, "global_scratch_align", 1),
        "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align, "aot_divisibility": divisibility}])


def compile_library_matrix_variants_without_launch(output_dir: Path, arch: str, variants) -> None:
    """Compile variants without changing the checked library contracts."""
    global FLAGGEMS_AOT_EXPORTS
    contracts = {
        "flagos_mul_mat_f16_f32_batched_aligned": (
         "ggml.gemm_f16_round_activation_f16_acc_f32.ptr16_k_rows_multiple16.v1",
         ("weights", "x", "output", "k", "rows", "columns"), 64),
        "flagos_ffn_swiglu_f16_f32_batched_tiled": (
         "transformer.ffn_f16_round_activation_f16_acc_f32_swiglu.ptr16_k_rows_multiple16.v1",
         ("gate_weights", "up_weights", "x", "output", "k", "rows", "columns"), 32),
    }
    kernels = []
    for source_name, suffix, options in variants:
        semantic, arguments, tile_m = contracts[source_name]
        export, _, _, _ = load_flaggems_aot_export(name=source_name, semantic=semantic,
            runtime_arguments=arguments, constexpr_arguments=("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        divisibility = tuple((argument, 16) for argument in arguments[:-1])
        if export.aot_divisibility != divisibility:
            raise RuntimeError(f"incompatible matrix divisibility contract: {source_name}")
        FLAGGEMS_AOT_EXPORTS += (export,)
        signature = {argument: "*fp16" if "weights" in argument else
                     "*fp32" if argument in ("x", "output") else "i32" for argument in arguments}
        signature.update({argument: "constexpr" for argument in export.constexpr_arguments})
        compiled = triton.compile(ASTSource(export.function, signature,
            {"BLOCK_M": tile_m, "BLOCK_N": 128, "BLOCK_K": 32}, attrs=export.aot_attributes),
            target=GPUTarget("hip", arch, 32), options=options)
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError(f"{source_name} requires unsupported Triton scratch storage")
        name = source_name + "_" + suffix
        artifact = output_dir / f"{name}.hsaco"
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": 128, "tile_m": tile_m, "tile_n": 128, "tile_k": 32,
            "argument_count": len(arguments), "global_scratch_size": 0,
            "global_scratch_align": getattr(m, "global_scratch_align", 1),
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align,
            "aot_divisibility": divisibility, "compile_options": options})
    write_manifest(output_dir, arch, kernels)


def compile_matrix_stage1_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile optional pipeline variants from the existing library exports."""
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("matrix-stage1 requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    compile_library_matrix_variants_without_launch(output_dir, arch, [
        (name, "stage1", {"num_warps": 4, "num_stages": 1}) for name in (
            "flagos_mul_mat_f16_f32_batched_aligned", "flagos_ffn_swiglu_f16_f32_batched_tiled")])


def compile_ffn_waves_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile FFN scheduler hints with the default two-stage pipeline."""
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("ffn-waves requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    compile_library_matrix_variants_without_launch(output_dir, arch, [
        ("flagos_ffn_swiglu_f16_f32_batched_tiled", f"waves{waves}",
         {"num_warps": 4, "num_stages": 2, "waves_per_eu": waves}) for waves in (2, 3)])


def compile_packed_ffn_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile both packing stages from versioned library contracts."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("packed-ffn requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    kernels = []
    for name, semantic, arguments, constexpr, signature, constants in (
        ("flagos_pack_f32_f16", "ggml.pack.contiguous_f32_to_f16.round_nearest.disjoint.v1",
         ("x", "output", "count"), ("BLOCK",),
         {"x": "*fp32", "output": "*fp16", "count": "i32", "BLOCK": "constexpr"}, {"BLOCK": 256}),
        ("flagos_ffn_swiglu_f16_packed_activation",
         "transformer.ffn_packed_f16_acc_f32_swiglu.ptr16_k_rows_multiple16.v1",
         ("gate_weights", "up_weights", "x", "output", "k", "rows", "columns"),
         ("BLOCK_M", "BLOCK_N", "BLOCK_K"),
         {"gate_weights": "*fp16", "up_weights": "*fp16", "x": "*fp16", "output": "*fp32",
          "k": "i32", "rows": "i32", "columns": "i32",
          "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"},
         {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}),
    ):
        export, _, _, _ = load_flaggems_aot_export(name=name, semantic=semantic,
            runtime_arguments=arguments, constexpr_arguments=constexpr)
        required = () if len(arguments) == 3 else tuple((argument, 16) for argument in arguments[:6])
        if export.aot_divisibility != required:
            raise RuntimeError("incompatible packed FFN divisibility contract")
        FLAGGEMS_AOT_EXPORTS += (export,)
        compiled = triton.compile(ASTSource(export.function, signature, constants, attrs=export.aot_attributes),
            target=GPUTarget("hip", arch, 32), options={"num_warps": 4})
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError("packed FFN requires unsupported Triton scratch storage")
        artifact = output_dir / f"{name}.hsaco"
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": constants.get("BLOCK", 32), "tile_m": constants.get("BLOCK_M", 0),
            "tile_n": constants.get("BLOCK_N", 0), "tile_k": constants.get("BLOCK_K", 0),
            "argument_count": len(arguments), "global_scratch_size": 0,
            "global_scratch_align": getattr(m, "global_scratch_align", 1),
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align,
            "aot_divisibility": required})
    write_manifest(output_dir, arch, kernels)


def compile_flash_attention_package_without_launch(output_dir: Path, arch: str, profile: str = "baseline") -> None:
    """Compile generic 128/256 attention from explicit library contracts."""
    global FLAGGEMS_AOT_EXPORTS
    if profile not in ("baseline", "gfx1150-head256"):
        raise RuntimeError(f"unknown flash-attention profile: {profile}")
    if profile != "baseline" and arch != "gfx1150":
        raise RuntimeError("gfx1150-head256 profile requires --arch gfx1150")
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("flash-attention requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    pointers = ("q", "k", "v", "attention_mask", "output")
    strides = ("stride_q_token", "stride_q_head", "stride_k_token", "stride_k_head",
               "stride_v_token", "stride_v_head", "stride_output_token", "stride_output_head")
    kernels = []
    for prefill in (False, True):
        kind = "prefill" if prefill else "decode"
        name = f"flagos_flash_attn_{kind}_f32_f16"
        lengths = ("query_length", "key_length", "q_per_kv") if prefill else ("key_length", "q_per_kv")
        mask_strides = ("stride_mask_token", "stride_mask_query") if prefill else ()
        runtime = pointers + lengths + strides[:6] + mask_strides + strides[6:] + ("scale",)
        signature = {argument: "i32" for argument in runtime}
        signature.update(q="*fp32", k="*fp16", v="*fp16", attention_mask="*fp16",
                         output="*fp32", scale="fp32", HEAD_DIM="constexpr", BLOCK_M="constexpr", BLOCK_N="constexpr")
        for head_dim in (128, 256):
            selected = profile == "gfx1150-head256" and head_dim == 256
            export_name = name
            semantic = f"ggml.flash_{kind}_q_round_f16_kv_f16_online_softmax.generic_strides.all_masked_zero.v1"
            warps = NUM_WARPS if profile == "baseline" else 4
            if selected:
                export_name += "_skip_masked_generic" if prefill else "_simt_generic"
                semantic = semantic[:-3] + (".skip_fully_masked_tiles.v1" if prefill else ".simt_f32_reductions.v1")
                warps = 8 if prefill else 4
            export, _, _, _ = load_flaggems_aot_export(name=export_name,
                semantic=semantic, runtime_arguments=runtime,
                constexpr_arguments=("HEAD_DIM", "BLOCK_M", "BLOCK_N"))
            if export.aot_divisibility:
                raise RuntimeError(f"generic attention must not require extra divisibility: {export_name}")
            FLAGGEMS_AOT_EXPORTS += (export,)
            entry = name + f"_generic_h{head_dim}"
            constants = {"HEAD_DIM": head_dim, "BLOCK_M": 16, "BLOCK_N": 16}
            compiled = triton.compile(ASTSource(export.function, signature, constants, attrs=export.aot_attributes),
                target=GPUTarget("hip", arch, 32), options={"num_warps": warps, "num_stages": 1})
            metadata = compiled.metadata
            if getattr(metadata, "global_scratch_size", 0) or metadata.profile_scratch_size:
                raise RuntimeError(f"{entry} requires unsupported Triton scratch storage")
            artifact = output_dir / f"{entry}.hsaco"
            artifact.write_bytes(compiled.asm["hsaco"])
            kernels.append({"name": entry, "symbol": metadata.name, "file": artifact.name,
                "shared": metadata.shared, "num_warps": metadata.num_warps, "warp_size": metadata.warp_size,
                "block_size": 16, "tile_m": 16, "tile_n": 16, "tile_k": head_dim,
                "argument_count": len(runtime), "global_scratch_size": 0,
                "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
                "profile_scratch_size": 0, "profile_scratch_align": metadata.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_splitkv_decode_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the explicit-workspace GGML decode pair without a GPU launch."""
    global FLAGGEMS_AOT_EXPORTS
    if arch != "gfx1150" or not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("flash-decode-splitkv requires gfx1150 and FLAGOS_FLAGGEMS_AOT_SOURCE")
    partial_runtime = ("q", "k", "v", "mask", "partial_m", "partial_l", "partial_a",
        "key_length", "q_per_kv", "split_count", "keys_per_split", "stride_q_token",
        "stride_q_head", "stride_k_token", "stride_k_head", "stride_v_token", "stride_v_head", "scale")
    merge_runtime = ("partial_m", "partial_l", "partial_a", "output",
        "split_count", "stride_output_token", "stride_output_head")
    pointers = {"q": "*fp32", "k": "*fp16", "v": "*fp16", "mask": "*fp16",
        "partial_m": "*fp32", "partial_l": "*fp32", "partial_a": "*fp32", "output": "*fp32"}
    stages = (
        (False, "flagos_flash_decode_splitk_stage1_f32_f16", partial_runtime,
         "ggml.flash_decode_splitkv.partial_m_l_a.q_round_f16.kv_f16.p_tile_f16.stats_f32.head128_256.explicit_workspace.single_query.all_masked_zero.v1",
         ("HEAD_DIM", "BLOCK_KV")),
        (True, "flagos_flash_decode_splitk_merge_f32", merge_runtime,
         "ggml.flash_decode_splitkv.merge_m_l_a.f32.head128_256.explicit_workspace.single_query.all_masked_zero.v1",
         ("HEAD_DIM", "MAX_SPLITS")),
    )
    kernels = []
    with amd_hint_context():
        for merge, export_name, runtime, semantic, constexpr in stages:
            export, _, _, _ = load_flaggems_aot_export(name=export_name, semantic=semantic,
                runtime_arguments=runtime, constexpr_arguments=constexpr)
            if export.aot_attributes:
                raise RuntimeError("split-KV requires generic pointer and stride attributes")
            FLAGGEMS_AOT_EXPORTS += (export,)
            for head in (128, 256):
                block = 16 if merge else 32
                constants = {"HEAD_DIM": head, "MAX_SPLITS" if merge else "BLOCK_KV": block}
                signature = {argument: pointers.get(argument, "fp32" if argument == "scale" else "i32")
                    for argument in runtime}
                signature.update({argument: "constexpr" for argument in constexpr})
                compiled = triton.compile(ASTSource(export.function, signature, constants, attrs={}),
                    target=GPUTarget("hip", arch, 32),
                    options={"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False})
                metadata = compiled.metadata
                if getattr(metadata, "global_scratch_size", 0) or metadata.profile_scratch_size:
                    raise RuntimeError("split-KV requires unsupported Triton scratch storage")
                name = ("flagos_flash_decode_splitkv_merge_f32" if merge else
                    "flagos_flash_decode_splitkv_partial_f32_f16") + f"_h{head}"
                artifact = output_dir / f"{name}.hsaco"
                artifact.write_bytes(compiled.asm["hsaco"])
                kernels.append({"name": name, "symbol": metadata.name, "file": artifact.name,
                    "shared": metadata.shared, "num_warps": metadata.num_warps,
                    "warp_size": metadata.warp_size, "block_size": block,
                    "tile_m": 0, "tile_n": block, "tile_k": head,
                    "argument_count": len(runtime), "global_scratch_size": 0,
                    "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
                    "profile_scratch_size": 0, "profile_scratch_align": metadata.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)



def compile_grouped_splitkv_decode_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the four independently qualified gfx1150 grouped profiles."""
    global FLAGGEMS_AOT_EXPORTS
    if arch != "gfx1150" or not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("flash-decode-grouped-splitkv requires gfx1150 and FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_flash_decode_splitkv_grouped_partial_f32_f16"
    runtime = ("q", "k", "v", "mask", "partial_m", "partial_l", "partial_a",
        "key_length", "q_per_kv", "split_count", "keys_per_split", "stride_q_token",
        "stride_q_head", "stride_k_token", "stride_k_head", "stride_v_token", "stride_v_head", "scale")
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="ggml.flash_decode_splitkv.grouped_partial_m_l_a.q_round_f16.kv_f16.p_tile_f16.dot_acc_f32.head128_256.gqa2_4_8.explicit_workspace.single_query.all_masked_zero.v1",
        runtime_arguments=runtime, constexpr_arguments=("HEAD_DIM", "BLOCK_KV", "Q_PER_KV", "BLOCK_Q"))
    if export.aot_divisibility:
        raise RuntimeError("generic grouped split-KV must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    pointer_types = {"q": "*fp32", "k": "*fp16", "v": "*fp16", "mask": "*fp16",
        "partial_m": "*fp32", "partial_l": "*fp32", "partial_a": "*fp32"}
    signature = {argument: pointer_types.get(argument, "fp32" if argument == "scale" else "i32")
        for argument in runtime}
    signature.update({argument: "constexpr" for argument in export.constexpr_arguments})
    kernels = []
    for head, group in ((128, 2), (128, 4), (128, 8), (256, 8)):
        constants = {"HEAD_DIM": head, "BLOCK_KV": 16, "Q_PER_KV": group, "BLOCK_Q": 16}
        compiled = triton.compile(ASTSource(export.function, signature, constants, attrs=export.aot_attributes),
            target=GPUTarget("hip", arch, 32),
            options={"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False})
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError("grouped split-KV requires unsupported Triton scratch storage")
        kernel_name = "flagos_flash_decode_splitkv_grouped_f32_f16_h" + str(head) + "_g" + str(group)
        artifact = output_dir / (kernel_name + ".hsaco")
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": kernel_name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": 16, "tile_m": 16, "tile_n": 16, "tile_k": head,
            "argument_count": 18, "global_scratch_size": 0, "global_scratch_align": 1,
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_ggml_rope_package_without_launch(output_dir: Path, arch: str, *, neox: bool = False) -> None:
    """Compile the no-YaRN GGML frequency-recurrence RoPE export."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("ggml-rope requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_rope_ggml_neox_recurrence_f32" if neox else "flagos_rope_ggml_recurrence_f32"
    runtime = ("x", "positions", "factors", "output", "ne0", "ne1", "ne2", "n_dims", "mode", "has_factors",
               "stride_x1", "stride_x2", "stride_x3", "section_t", "section_h", "section_w", "section_e",
               "theta_scale", "freq_scale", "attn_factor")
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="ggml.rope_neox_f32.cpu_frequency_recurrence.no_yarn.v2" if neox else
                 "ggml.rope_normal_imrope_f32.cpu_frequency_recurrence.no_yarn.v2",
        runtime_arguments=runtime, constexpr_arguments=("BLOCK",))
    if export.aot_divisibility:
        raise RuntimeError("GGML recurrence RoPE must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {argument: "i32" for argument in runtime}
    signature.update(x="*fp32", positions="*i32", factors="*fp32", output="*fp32",
                     theta_scale="fp32", freq_scale="fp32", attn_factor="fp32", BLOCK="constexpr")
    compiled = triton.compile(ASTSource(export.function, signature, {"BLOCK": 128}, attrs=export.aot_attributes),
        target=GPUTarget("hip", arch, 32), options={"num_warps": 4, "enable_fp_fusion": False})
    m = compiled.metadata
    if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
        raise RuntimeError("GGML recurrence RoPE requires unsupported Triton scratch storage")
    artifact = output_dir / f"{name}.hsaco"
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{"name": name, "symbol": m.name, "file": artifact.name,
        "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
        "block_size": 128, "tile_m": 0, "tile_n": 0, "tile_k": 0,
        "argument_count": len(runtime), "global_scratch_size": 0, "global_scratch_align": 1,
        "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align}])


def compile_q6_gemv_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the GGUF Q6_K GEMV export without device launches."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("q6-gemv requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_mul_mat_q6_k_f32"
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="gguf.q6_k.gemv_f32_activation_acc_f32.v1",
        runtime_arguments=("weights_u8", "weights_f16", "x", "output", "k", "rows"),
        constexpr_arguments=())
    if export.aot_divisibility:
        raise RuntimeError("generic Q6_K GEMV must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {"weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32",
        "output": "*fp32", "k": "i32", "rows": "i32"}
    compiled = triton.compile(ASTSource(export.function, signature, {}, attrs=export.aot_attributes),
        target=GPUTarget("hip", arch, 32), options={"num_warps": 1})
    m = compiled.metadata
    if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
        raise RuntimeError("Q6_K GEMV requires unsupported Triton scratch storage")
    artifact = output_dir / f"{name}.hsaco"
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{"name": name, "symbol": m.name, "file": artifact.name,
        "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
        "block_size": 256, "tile_m": 0, "tile_n": 0, "tile_k": 0,
        "argument_count": 6, "global_scratch_size": 0, "global_scratch_align": 1,
        "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align}])



def compile_core_ops_package_without_launch(
        output_dir: Path, arch: str, *, graph_ops: bool = False) -> None:
    """Compile selected generic F32 library exports as a separate package."""
    global FLAGGEMS_AOT_EXPORTS
    extension = "graph-ops" if graph_ops else "core-ops"
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError(f"{extension} requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    contracts = (
        ("add", "ggml.add.same_shape.contiguous_f32.tail.v1", ("x", "y", "output", "n_elements")),
        ("add_repeat", "ggml.add.leading_vector_repeat.contiguous_f32.tail.v1", ("x", "y", "output", "n_elements", "y_elements")),
        ("mul", "ggml.mul.flat_repeat.contiguous_f32.tail.v1", ("x", "y", "output", "n_elements", "y_elements")),
        ("scale", "ggml.scale.f32.multiply_add.profile_fp_fusion.tail.v1", ("x", "output", "scale", "bias", "n_elements")),
        ("sigmoid", "ggml.sigmoid.contiguous_f32.tail.v1", ("x", "output", "n_elements")),
        ("softplus", "ggml.softplus.f32.threshold20.tail.v1", ("x", "output", "n_elements")),
        ("swiglu_split", "ggml.swiglu.split_gate_silu_times_up.contiguous_f32.tail.v1", ("gate", "up", "output", "n_elements")),
        ("rms_norm", "ggml.rms_norm.f32.rowwise_eps_inside_sqrt.tail.v1", ("output", "x", "n_cols", "eps")),
        ("silu", "ggml.silu.contiguous_f32.tail.v1", ("x", "output", "n_elements")),
    )
    if graph_ops:
        contracts = (
            ("copy_strided", "ggml.copy.f32.flat_order.independent_shapes.element_strides.disjoint.v1",
                ("x", "output", "xe0", "xe1", "xe2", "xe3", "ye0", "ye1", "ye2", "ye3",
                 "sx0", "sx1", "sx2", "sx3", "sy0", "sy1", "sy2", "sy3", "n_elements")),
            ("concat", "ggml.concat.f32.axis0_3.strided_inputs.contiguous_output.disjoint.bitwise_select.v2",
                ("a", "b", "output", "dim", "ye0", "ye1", "ye2", "ye3", "ae0", "ae1", "ae2", "ae3",
                 "as0", "as1", "as2", "as3", "bs0", "bs1", "bs2", "bs3", "n_elements")),
            ("l2_norm_strided", "ggml.l2_norm.f32.inner_contiguous.outer_strides.sqrt_sum_clamp_eps.disjoint.v1",
                ("x", "output", "n_cols", "stride_row", "stride_channel", "stride_sample", "eps", "n_rows", "n_channels")),
            ("ssm_conv", "ggml.ssm_conv.f32.rolling_state.depthwise.explicit_strides.disjoint.v1",
                ("s", "c", "output", "d_conv", "d_inner", "n_tokens", "n_seqs", "ss1", "ss2", "sc1", "so0", "so1")),
        )
    exports = []
    for operation, semantic, runtime in contracts:
        name = "flagos_" + operation + "_f32"
        export, _, _, _ = load_flaggems_aot_export(name=name, semantic=semantic,
            runtime_arguments=runtime, constexpr_arguments=("BLOCK",))
        if export.aot_divisibility:
            raise RuntimeError(f"generic {extension} operations must not require extra divisibility")
        exports.append((operation, export))
    kernels = []
    with amd_hint_context():
        for operation, export in exports:
            FLAGGEMS_AOT_EXPORTS += (export,)
            # Preserve the legacy L2 row-width contract when replacing its
            # module. A 256-column export silently removed 512-column support.
            block = 4096 if operation == "rms_norm" else 512 if operation == "l2_norm_strided" else 256
            signature = {argument: "*fp32" if argument in ("x", "y", "output", "gate", "up", "a", "b", "s", "c")
                else "fp32" if argument in ("scale", "bias", "eps") else "i32"
                for argument in export.runtime_arguments}
            signature["BLOCK"] = "constexpr"
            compiled = triton.compile(ASTSource(export.function, signature, {"BLOCK": block}, attrs={}),
                target=GPUTarget("hip", arch, 32),
                options={"num_warps": 4, "enable_fp_fusion": not graph_ops and operation != "scale"})
            metadata = compiled.metadata
            if getattr(metadata, "global_scratch_size", 0) or metadata.profile_scratch_size:
                raise RuntimeError(f"{extension} operations require unsupported Triton scratch storage")
            artifact = output_dir / (export.name + ".hsaco")
            artifact.write_bytes(compiled.asm["hsaco"])
            kernels.append({"name": export.name, "symbol": metadata.name, "file": artifact.name,
                "shared": metadata.shared, "num_warps": metadata.num_warps, "warp_size": metadata.warp_size,
                "block_size": block, "tile_m": 0, "tile_n": 0, "tile_k": 0,
                "argument_count": len(export.runtime_arguments), "global_scratch_size": 0,
                "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
                "profile_scratch_size": 0, "profile_scratch_align": metadata.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_rows_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile GGUF gather, KV scatter, and bitwise F16 copy exports."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("row-ops requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    copy_arguments = ("x", "output", "xe0", "xe1", "xe2", "xe3", "ye0", "ye1", "ye2", "ye3",
        "sx0", "sx1", "sx2", "sx3", "sy0", "sy1", "sy2", "sy3", "n_elements")
    contracts = (
        ("flagos_get_rows_q4_k_f32", "ggml.get_rows.gguf_q4_k.contiguous.i32_indices.in_range.packed256.acc_f32.v1",
            ("weights_u8", "weights_f16", "row_index", "output", "n_cols"), ()),
        ("flagos_get_rows_q6_k_f32", "ggml.get_rows.gguf_q6_k.contiguous.i32_indices.in_range.packed256.acc_f32.v1",
            ("weights_u8", "weights_f16", "row_index", "output", "n_cols"), ()),
        ("flagos_set_rows_f32_f16", "ggml.set_rows.contiguous_f32_to_f16.i64_indices.in_range.unique_rows.state_write.v1",
            ("x", "row_index", "output", "n_cols", "n_rows", "n_dst_rows"), ("BLOCK",)),
        ("flagos_copy_strided_f16_bits", "ggml.copy.f16_bits.flat_order.independent_shapes.element_strides.disjoint.v1",
            copy_arguments, ("BLOCK",)),
    )
    exports = []
    for name, semantic, runtime, constexpr in contracts:
        export, _, _, _ = load_flaggems_aot_export(name=name, semantic=semantic,
            runtime_arguments=runtime, constexpr_arguments=constexpr)
        if export.aot_divisibility:
            raise RuntimeError("generic row operations must not require extra divisibility")
        exports.append(export)
    kernels = []
    with amd_hint_context():
        for export in exports:
            FLAGGEMS_AOT_EXPORTS += (export,)
            copy_bits = export.name == "flagos_copy_strided_f16_bits"
            scatter = export.name == "flagos_set_rows_f32_f16"
            pointer_types = {"weights_u8": "*u8", "weights_f16": "*fp16",
                "row_index": "*i64" if scatter else "*i32", "x": "*u16" if copy_bits else "*fp32",
                "output": "*u16" if copy_bits else "*fp16" if scatter else "*fp32"}
            signature = {argument: pointer_types.get(argument, "i32") for argument in export.runtime_arguments}
            constants = {argument: 256 for argument in export.constexpr_arguments}
            signature.update({argument: "constexpr" for argument in constants})
            compiled = triton.compile(ASTSource(export.function, signature, constants, attrs={}),
                target=GPUTarget("hip", arch, 32), options={"num_warps": 4, "enable_fp_fusion": False})
            m = compiled.metadata
            if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
                raise RuntimeError("row operations require unsupported Triton scratch storage")
            artifact = output_dir / (export.name + ".hsaco")
            artifact.write_bytes(compiled.asm["hsaco"])
            kernels.append({"name": export.name, "symbol": m.name, "file": artifact.name,
                "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
                "block_size": 256, "tile_m": 0, "tile_n": 0, "tile_k": 0,
                "argument_count": len(export.runtime_arguments), "global_scratch_size": 0,
                "global_scratch_align": getattr(m, "global_scratch_align", 1),
                "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_softmax_broadcast_package_without_launch(output_dir: Path, arch: str, wide: bool = False) -> None:
    """Compile a separate F32-mask softmax package for GGML head broadcasts."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("softmax-broadcast requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    runtime = ("x", "mask_ptr", "output", "n_cols", "n_rows", "n_heads", "mask_heads", "mask_samples",
        "mask_stride_row", "mask_stride_head", "mask_stride_sample", "scale")
    export, _, _, _ = load_flaggems_aot_export(name="flagos_soft_max_masked_broadcast_f32",
        semantic="ggml.soft_max.f32.mask_f32.head_sample_modulo.strides.scale.no_alibi_no_sinks.valid_row.v1",
        runtime_arguments=runtime, constexpr_arguments=("BLOCK",))
    if export.aot_divisibility:
        raise RuntimeError("generic softmax must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {argument: "*fp32" if argument in ("x", "mask_ptr", "output") else
        "fp32" if argument == "scale" else "i32" for argument in runtime}
    signature["BLOCK"] = "constexpr"
    profiles = ((1024, 4), (2048, 4), (4096, 4), (8192, 8)) if wide else ((512, 4),)
    kernels = []
    for block, warps in profiles:
        name = export.name + (f"_b{block}_w{warps}" if wide else "")
        with amd_hint_context():
            compiled = triton.compile(ASTSource(export.function, signature, {"BLOCK": block}, attrs={}),
                target=GPUTarget("hip", arch, 32), options={"num_warps": warps, "enable_fp_fusion": False})
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError("softmax requires unsupported Triton scratch storage")
        artifact = output_dir / (name + ".hsaco")
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": block, "tile_m": 0, "tile_n": 0, "tile_k": 0, "argument_count": len(runtime),
            "global_scratch_size": 0, "global_scratch_align": getattr(m, "global_scratch_align", 1),
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_strided_batched_gemm_package_without_launch(
        output_dir: Path, arch: str, preserve_src1: bool = False,
        f32_weights: bool = False) -> None:
    """Compile explicit GGML floating matrix multiplication with batch strides."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("strided-batched-gemm requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    runtime = ("weights", "x", "output", "k", "rows", "columns", "n_heads",
        "head_ratio", "sample_ratio", "sw1", "sw2", "sw3", "sx1", "sx2", "sx3")
    preserve_src1 = preserve_src1 or f32_weights
    weight_type = "f32" if f32_weights else "f16"
    name = "flagos_mul_mat_" + weight_type + "_f32_strided_batched" + ("_ieee" if preserve_src1 else "")
    source_semantic = "f32_src1_preserved.ieee_products" if preserve_src1 else "f32_src1_rne_f16"
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="ggml.mul_mat." + weight_type + "_weights." + source_semantic + ".acc_f32.outer_strides.batch_floor_broadcast.v1",
        runtime_arguments=runtime, constexpr_arguments=("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    if export.aot_divisibility:
        raise RuntimeError("generic strided batched GEMM must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {argument: "*fp16" if argument == "weights" and not f32_weights else
        "*fp32" if argument == "weights" else
        "*fp32" if argument in ("x", "output") else "i32" for argument in runtime}
    constants = {"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 32}
    signature.update({argument: "constexpr" for argument in constants})
    with amd_hint_context():
        compiled = triton.compile(ASTSource(export.function, signature, constants, attrs={}),
            target=GPUTarget("hip", arch, 32), options={"num_warps": 4, "enable_fp_fusion": False})
    m = compiled.metadata
    if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
        raise RuntimeError("strided batched GEMM requires unsupported Triton scratch storage")
    artifact = output_dir / (export.name + ".hsaco")
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{"name": export.name, "symbol": m.name, "file": artifact.name,
        "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
        "block_size": 32, "tile_m": 32, "tile_n": 16, "tile_k": 32,
        "argument_count": len(runtime), "global_scratch_size": 0,
        "global_scratch_align": getattr(m, "global_scratch_align", 1),
        "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align}])


def compile_half_gather_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile dense F16/BF16 embedding exports with explicit table bounds."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("half-gather requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    kernels = []
    for dtype, pointer in (("f16", "*fp16"), ("bf16_bits", "*u16")):
        table = "table" if dtype == "f16" else "table_bits"
        runtime = (table, "row_index", "output", "n_cols", "table_rows")
        export, _, _, _ = load_flaggems_aot_export(name="flagos_get_rows_" + dtype + "_f32",
            semantic="ggml.get_rows.contiguous_" + dtype + "_to_f32.i32_indices.in_range.mask_invalid_zero.v1",
            runtime_arguments=runtime, constexpr_arguments=("BLOCK",))
        if export.aot_divisibility:
            raise RuntimeError("generic half gather must not require extra divisibility")
        FLAGGEMS_AOT_EXPORTS += (export,)
        signature = {table: pointer, "row_index": "*i32", "output": "*fp32",
            "n_cols": "i32", "table_rows": "i32", "BLOCK": "constexpr"}
        with amd_hint_context():
            compiled = triton.compile(ASTSource(export.function, signature, {"BLOCK": 256}, attrs={}),
                target=GPUTarget("hip", arch, 32), options={"num_warps": 4, "enable_fp_fusion": False})
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError("half gather requires unsupported Triton scratch storage")
        artifact = output_dir / (export.name + ".hsaco")
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": export.name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": 256, "tile_m": 0, "tile_n": 0, "tile_k": 0,
            "argument_count": 5, "global_scratch_size": 0,
            "global_scratch_align": getattr(m, "global_scratch_align", 1),
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_gguf_gemv_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile generic and explicitly aligned GGUF GEMV library exports."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("gguf-gemv requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    kernels = []
    for kind, block, aligned in (("q4_0", 32, False), ("q4_1", 32, False), ("q8_0", 32, False),
            ("q4_k", 256, False), ("q5_k", 256, False),
            ("q4_1", 32, True), ("q8_0", 32, True), ("q4_k", 256, True)):
        name = "flagos_mul_mat_" + kind + "_f32" + ("_model_aligned" if aligned else "")
        semantic = "ggml.gguf_" + kind + "_gemv_dense_f32_acc_f32." + (
            "ptr16_k_rows16.byte_spans_i31.v1" if aligned else "ptr2_packed.v1")
        export, _, _, _ = load_flaggems_aot_export(name=name,
            semantic=semantic,
            runtime_arguments=("weights_u8", "weights_f16", "x", "output", "k", "rows"),
            constexpr_arguments=())
        required = tuple((argument, 16) for argument in export.runtime_arguments) if aligned else ()
        if export.aot_divisibility != required:
            raise RuntimeError("incompatible GGUF GEMV divisibility contract")
        FLAGGEMS_AOT_EXPORTS += (export,)
        signature = {"weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32",
            "output": "*fp32", "k": "i32", "rows": "i32"}
        attributes = export.aot_attributes
        if aligned:
            # The provider checks each operand byte span before this specialization.
            for index in range(4):
                attributes[(index,)].append(["tt.pointer_range", 32])
        compiled = triton.compile(ASTSource(export.function, signature, {}, attrs=attributes),
            target=GPUTarget("hip", arch, 32), options={"num_warps": 1})
        m = compiled.metadata
        if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
            raise RuntimeError("GGUF GEMV requires unsupported Triton scratch storage")
        artifact = output_dir / (name + ".hsaco")
        artifact.write_bytes(compiled.asm["hsaco"])
        kernels.append({"name": name, "symbol": m.name, "file": artifact.name,
            "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
            "block_size": block, "tile_m": 16 if aligned else 0,
            "tile_n": 16 if aligned else 0, "tile_k": 16 if aligned else 0,
            "argument_count": 6, "global_scratch_size": 0, "global_scratch_align": 1,
            "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align})
    write_manifest(output_dir, arch, kernels)


def compile_gdn_value_tile_package_without_launch(output_dir: Path, arch: str, *, baseline: bool = False) -> None:
    """Compile the GDN value-axis tile with the legacy packed output contract."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("gdn-value-tile requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_gated_delta_net_scalar_f32" if baseline else "flagos_gated_delta_net_scalar_f32_value_tiled"
    cols = 4 if baseline else 16
    arguments = ("q", "k", "v", "gate", "beta", "current_state", "output",
        "state_size", "n_heads", "n_tokens", "n_seqs", "sq1", "sq2", "sq3",
        "sv1", "sv2", "sv3", "sb1", "sb2", "sb3", "q_heads", "q_seq_ratio",
        "snapshot_count", "scale")
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="ggml.gated_delta_net_scalar_f32.legacy_math.v1" if baseline else
                 "ggml.gated_delta_net_scalar_f32.value_tiles.legacy_math.v1",
        runtime_arguments=arguments, constexpr_arguments=("BLOCK", "COLS"))
    if export.aot_divisibility:
        raise RuntimeError("generic GDN value tile must not require extra divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {argument: "*fp32" if i < 7 else "fp32" if argument == "scale" else "i32"
        for i, argument in enumerate(arguments)}
    signature.update(BLOCK="constexpr", COLS="constexpr")
    compiled = triton.compile(ASTSource(export.function, signature, {"BLOCK": 128, "COLS": cols},
        attrs=export.aot_attributes), target=GPUTarget("hip", arch, 32), options={"num_warps": 4})
    m = compiled.metadata
    if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
        raise RuntimeError("GDN value tile requires unsupported Triton scratch storage")
    artifact = output_dir / f"{name}.hsaco"
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{"name": name, "symbol": m.name, "file": artifact.name,
        "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size,
        "block_size": 128, "tile_m": 0, "tile_n": cols, "tile_k": 0,
        "argument_count": 24, "global_scratch_size": 0,
        "global_scratch_align": getattr(m, "global_scratch_align", 1),
        "profile_scratch_size": 0, "profile_scratch_align": m.profile_scratch_align}])


def compile_strided_rms_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the GGML outer-stride RMS export with its tested arithmetic."""
    global FLAGGEMS_AOT_EXPORTS
    if not os.environ.get("FLAGOS_FLAGGEMS_AOT_SOURCE"):
        raise RuntimeError("strided-rms requires FLAGOS_FLAGGEMS_AOT_SOURCE")
    name = "flagos_rms_norm_ggml_strided_f32"
    arguments = ("output", "x", "n_cols", "ne1", "ne2", "stride_x1", "stride_x2", "stride_x3", "eps")
    export, _, _, _ = load_flaggems_aot_export(name=name,
        semantic="ggml.rms_norm.inner_contiguous_outer_strided.square_f32_sum_f64.v1",
        runtime_arguments=arguments, constexpr_arguments=("BLOCK",))
    if export.aot_divisibility:
        raise RuntimeError("strided RMS must not add unproved divisibility")
    FLAGGEMS_AOT_EXPORTS += (export,)
    signature = {argument: "*fp32" if argument in ("output", "x") else "fp32" if argument == "eps" else "i32" for argument in arguments}
    signature["BLOCK"] = "constexpr"
    compiled = triton.compile(ASTSource(export.function, signature, {"BLOCK": 256}, attrs=export.aot_attributes),
        target=GPUTarget("hip", arch, 32), options={"num_warps": 4, "enable_fp_fusion": False})
    m = compiled.metadata
    if getattr(m, "global_scratch_size", 0) or m.profile_scratch_size:
        raise RuntimeError("strided RMS requires unsupported Triton scratch storage")
    artifact = output_dir / f"{name}.hsaco"
    artifact.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{"name": name, "symbol": m.name, "file": artifact.name,
        "shared": m.shared, "num_warps": m.num_warps, "warp_size": m.warp_size, "block_size": 256,
        "tile_m": 0, "tile_n": 0, "tile_k": 0, "argument_count": 9, "global_scratch_size": 0,
        "global_scratch_align": getattr(m, "global_scratch_align", 1), "profile_scratch_size": 0,
        "profile_scratch_align": m.profile_scratch_align}])


def compile_scale_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the common SCALE kernel for an AMD package without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    name = "flagos_scale_f32"
    signature = {
        "x": "*fp32", "output": "*fp32", "scale": "fp32", "bias": "fp32",
        "n_elements": "i32", "BLOCK": "constexpr",
    }
    source = ASTSource(
        common.flagos_scale_f32,
        signature,
        {"BLOCK": common.BLOCK_SIZE},
        attrs=amd_jit_specialization_attrs(2, ()),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": common.BLOCK_SIZE,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 5,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_ssm_conv_silu_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the fused SSM_CONV+SiLU kernel without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    common.validate_ssm_conv_silu_launch_contract()
    name = "flagos_ssm_conv_silu_f32"
    signature = {
        "s": "*fp32", "c": "*fp32", "output": "*fp32",
        "d_conv": "i32", "d_inner": "i32", "n_tokens": "i32", "n_seqs": "i32",
        "ss1": "i32", "ss2": "i32", "sc1": "i32", "so0": "i32", "so1": "i32",
        "BLOCK": "constexpr",
    }
    source = ASTSource(
        common.flagos_ssm_conv_silu_f32,
        signature,
        {"BLOCK": common.SSM_CONV_SILU_BLOCK_SIZE},
        attrs=amd_jit_specialization_attrs(3, ()),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": common.SSM_CONV_SILU_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": common.SSM_CONV_SILU_BLOCK_SIZE,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 12,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_silu_mul_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the fused SiLU+Mul kernel without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if (AMD_SILU_MUL_BLOCK_SIZE <= 0 or
            AMD_SILU_MUL_BLOCK_SIZE & (AMD_SILU_MUL_BLOCK_SIZE - 1)):
        raise RuntimeError("FLAGOS_AMD_SILU_MUL_BLOCK_SIZE must be a positive power of two")
    if AMD_SILU_MUL_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_SILU_MUL_NUM_WARPS must be 1, 2, 4, or 8")
    name = "flagos_silu_mul_f32"
    signature = {
        "x": "*fp32", "other": "*fp32", "output": "*fp32",
        "n_elements": "i32", "BLOCK": "constexpr",
    }
    source = ASTSource(
        flagos_silu_mul_f32,
        signature,
        {"BLOCK": AMD_SILU_MUL_BLOCK_SIZE},
        attrs=amd_jit_specialization_attrs(3, ()),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": AMD_SILU_MUL_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": AMD_SILU_MUL_BLOCK_SIZE,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 4,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_unary_mul_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile all supported unary+Mul kernels without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if (AMD_SILU_MUL_BLOCK_SIZE <= 0 or
            AMD_SILU_MUL_BLOCK_SIZE & (AMD_SILU_MUL_BLOCK_SIZE - 1)):
        raise RuntimeError("FLAGOS_AMD_SILU_MUL_BLOCK_SIZE must be a positive power of two")
    if AMD_SILU_MUL_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_SILU_MUL_NUM_WARPS must be 1, 2, 4, or 8")
    signature = {
        "x": "*fp32", "other": "*fp32", "output": "*fp32",
        "n_elements": "i32", "BLOCK": "constexpr",
    }
    kernels = []
    for name, kernel in (
            ("flagos_silu_mul_f32", flagos_silu_mul_f32),
            ("flagos_sigmoid_mul_f32", flagos_sigmoid_mul_f32),
            ("flagos_softplus_mul_f32", flagos_softplus_mul_f32)):
        source = ASTSource(
            kernel,
            signature,
            {"BLOCK": AMD_SILU_MUL_BLOCK_SIZE},
            attrs=amd_jit_specialization_attrs(3, ()),
        )
        compiled = triton.compile(
            source,
            target=GPUTarget("hip", arch, 32),
            options={"num_warps": AMD_SILU_MUL_NUM_WARPS},
        )
        metadata = compiled.metadata
        global_scratch_size = getattr(metadata, "global_scratch_size", 0)
        if global_scratch_size or metadata.profile_scratch_size:
            raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
        output = output_dir / f"{name}.hsaco"
        output.write_bytes(compiled.asm["hsaco"])
        kernels.append({
            "name": name,
            "symbol": name,
            "file": output.name,
            "shared": metadata.shared,
            "num_warps": metadata.num_warps,
            "warp_size": metadata.warp_size,
            "block_size": AMD_SILU_MUL_BLOCK_SIZE,
            "tile_m": 0,
            "tile_n": 0,
            "tile_k": 0,
            "argument_count": 4,
            "global_scratch_size": global_scratch_size,
            "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
            "profile_scratch_size": metadata.profile_scratch_size,
            "profile_scratch_align": metadata.profile_scratch_align,
        })
    write_manifest(output_dir, arch, kernels)


def compile_gdn_alpha_gate_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the fixed-width Qwen GDN alpha gate without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if AMD_GDN_ALPHA_GATE_WIDTH <= 0:
        raise RuntimeError("FLAGOS_AMD_GDN_ALPHA_GATE_WIDTH must be positive")
    if (AMD_GDN_ALPHA_GATE_BLOCK_SIZE <= 0 or
            AMD_GDN_ALPHA_GATE_BLOCK_SIZE & (AMD_GDN_ALPHA_GATE_BLOCK_SIZE - 1)):
        raise RuntimeError(
            "FLAGOS_AMD_GDN_ALPHA_GATE_BLOCK_SIZE must be a positive power of two")
    if AMD_GDN_ALPHA_GATE_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_GDN_ALPHA_GATE_NUM_WARPS must be 1, 2, 4, or 8")
    name = "flagos_gdn_alpha_gate_f32"
    signature = {
        "x": "*fp32", "bias": "*fp32", "scale": "*fp32", "output": "*fp32",
        "n_elements": "i32", "WIDTH": "constexpr", "BLOCK": "constexpr",
    }
    source = ASTSource(
        flagos_gdn_alpha_gate_f32,
        signature,
        {"WIDTH": AMD_GDN_ALPHA_GATE_WIDTH, "BLOCK": AMD_GDN_ALPHA_GATE_BLOCK_SIZE},
        attrs=amd_jit_specialization_attrs(4, ()),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": AMD_GDN_ALPHA_GATE_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": AMD_GDN_ALPHA_GATE_BLOCK_SIZE,
        "tile_m": 0,
        "tile_n": AMD_GDN_ALPHA_GATE_WIDTH,
        "tile_k": 0,
        "argument_count": 5,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_gdn_q8_gate_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the exact Qwen3.5 decode GDN projection fusion."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if AMD_GDN_Q8_GATE_ROWS != 32 or AMD_GDN_Q8_GATE_K != 2560:
        raise RuntimeError("the initial GDN Q8 gate contract requires rows=32 and k=2560")
    if AMD_GDN_Q8_GATE_BLOCK_M != 4 or AMD_GDN_Q8_GATE_NUM_WARPS != 1:
        raise RuntimeError("the validated gfx1150 GDN Q8 gate contract requires block_m=4 and one warp")
    name = "flagos_gdn_q8_gate_projections_f32"
    signature = {
        "alpha_weights_u8": "*u8", "alpha_weights_f16": "*fp16",
        "beta_weights_u8": "*u8", "beta_weights_f16": "*fp16",
        "x": "*fp32", "alpha_bias": "*fp32", "alpha_scale": "*fp32",
        "alpha_output": "*fp32", "beta_output": "*fp32",
        "k": "i32", "rows": "i32", "BLOCK_M": "constexpr",
    }
    source = ASTSource(
        flagos_gdn_q8_gate_projections_f32,
        signature,
        {"BLOCK_M": AMD_GDN_Q8_GATE_BLOCK_M},
        attrs=amd_jit_specialization_attrs(9, (9, 10)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": AMD_GDN_Q8_GATE_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": AMD_GDN_Q8_GATE_BLOCK_M,
        "tile_m": AMD_GDN_Q8_GATE_BLOCK_M,
        "tile_n": AMD_GDN_Q8_GATE_ROWS,
        "tile_k": AMD_GDN_Q8_GATE_K,
        "argument_count": 11,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_rms_norm_narrow_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the exact-width RMSNorm+scale kernel without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if (AMD_RMS_NORM_NARROW_BLOCK_SIZE <= 0 or
            AMD_RMS_NORM_NARROW_BLOCK_SIZE & (AMD_RMS_NORM_NARROW_BLOCK_SIZE - 1)):
        raise RuntimeError(
            "FLAGOS_AMD_RMS_NORM_NARROW_BLOCK_SIZE must be a positive power of two")
    if AMD_RMS_NORM_NARROW_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_AMD_RMS_NORM_NARROW_NUM_WARPS must be 1, 2, 4, or 8")
    name = "flagos_rms_norm_mul_inplace_f32_narrow"
    signature = {
        "output": "*fp32", "x": "*fp32", "weight": "*fp32",
        "n_cols": "i32", "eps": "fp32", "BLOCK": "constexpr",
    }
    source = ASTSource(
        flagos_rms_norm_mul_inplace_f32_narrow,
        signature,
        {"BLOCK": AMD_RMS_NORM_NARROW_BLOCK_SIZE},
        attrs=amd_jit_specialization_attrs(3, (3,)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": AMD_RMS_NORM_NARROW_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": AMD_RMS_NORM_NARROW_BLOCK_SIZE,
        "exact_block_size": True,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 5,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_residual_package_without_launch(output_dir: Path, arch: str, narrow: bool) -> None:
    """Compile one residual-aware ADD+RMSNorm export without dispatching it."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    block_size = AMD_RESIDUAL_NARROW_BLOCK_SIZE if narrow else AMD_RESIDUAL_BLOCK_SIZE
    if block_size <= 0 or block_size & (block_size - 1):
        raise RuntimeError("residual block size must be a positive power of two")
    if narrow and block_size >= AMD_RESIDUAL_BLOCK_SIZE:
        raise RuntimeError("narrow residual block size must be smaller than the generic block")
    name = ("flagos_add_rms_norm_mul_residual_f32_narrow" if narrow
            else "flagos_add_rms_norm_mul_residual_f32")
    function = (flagos_add_rms_norm_mul_residual_f32_narrow if narrow
                else flagos_add_rms_norm_mul_residual_f32)
    signature = {
        "residual_output": "*fp32", "mul_output": "*fp32",
        "x": "*fp32", "bias": "*fp32", "weight": "*fp32",
        "n_cols": "i32", "eps": "fp32", "BLOCK": "constexpr",
    }
    source = ASTSource(
        function,
        signature,
        {"BLOCK": block_size},
        attrs=amd_jit_specialization_attrs(5, (5, 6)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={metadata.profile_scratch_size})")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    kernels = [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": block_size,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 7,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }]
    write_manifest(output_dir, arch, kernels)


def compile_q4_ffn_package_without_launch(
        output_dir: Path, arch: str, staged: bool, quant_kind: str = "q4_k") -> None:
    """Compile one packed-Q4 FFN HSACO without allocating tensors or dispatching it."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if quant_kind not in ("q4_k", "q4_0"):
        raise RuntimeError(f"unsupported packed-Q4 FFN kind: {quant_kind}")
    block_m = Q40_FFN_DECODE_BLOCK_M if quant_kind == "q4_0" else Q4_FFN_DECODE_BLOCK_M
    if block_m <= 0 or block_m > 32 or block_m & (block_m - 1):
        variable = "FLAGOS_Q40_FFN_DECODE_BLOCK_M" if quant_kind == "q4_0" else "FLAGOS_Q4_FFN_DECODE_BLOCK_M"
        raise RuntimeError(f"{variable} must be a power of two from 1 to 32")
    if staged and quant_kind == "q4_0":
        num_warps = Q40_FFN_DECODE_STAGED_NUM_WARPS
        waves_per_eu = Q40_FFN_DECODE_STAGED_WAVES_PER_EU
    elif staged:
        num_warps = Q4_FFN_DECODE_STAGED_NUM_WARPS
        waves_per_eu = Q4_FFN_DECODE_STAGED_WAVES_PER_EU
    elif quant_kind == "q4_0":
        num_warps = Q40_FFN_DECODE_NUM_WARPS
        waves_per_eu = Q40_FFN_DECODE_WAVES_PER_EU
    else:
        num_warps = Q4_FFN_DECODE_NUM_WARPS
        waves_per_eu = Q4_FFN_DECODE_WAVES_PER_EU
    if num_warps not in (1, 2, 4, 8):
        variable = ("FLAGOS_Q40_FFN_DECODE_STAGED_NUM_WARPS"
                    if staged and quant_kind == "q4_0"
                    else "FLAGOS_Q4_FFN_DECODE_STAGED_NUM_WARPS" if staged
                    else "FLAGOS_Q40_FFN_DECODE_NUM_WARPS" if quant_kind == "q4_0"
                    else "FLAGOS_Q4_FFN_DECODE_NUM_WARPS")
        raise RuntimeError(f"{variable} must be 1, 2, 4, or 8")
    if waves_per_eu not in (1, 2, 3, 4):
        variable = ("FLAGOS_Q40_FFN_DECODE_STAGED_WAVES_PER_EU"
                    if staged and quant_kind == "q4_0"
                    else "FLAGOS_Q4_FFN_DECODE_STAGED_WAVES_PER_EU" if staged
                    else "FLAGOS_Q40_FFN_DECODE_WAVES_PER_EU" if quant_kind == "q4_0"
                    else "FLAGOS_Q4_FFN_DECODE_WAVES_PER_EU")
        raise RuntimeError(f"{variable} must be 1, 2, 3, or 4")

    if quant_kind == "q4_0":
        name = ("flagos_ffn_swiglu_q4_0_f32_decode_staged" if staged
                else "flagos_ffn_swiglu_q4_0_f32_decode")
        function = (flagos_ffn_swiglu_q4_0_f32_decode_staged if staged
                    else flagos_ffn_swiglu_q4_0_f32_decode)
    else:
        name = ("flagos_ffn_swiglu_q4_k_f32_decode_staged" if staged
                else "flagos_ffn_swiglu_q4_k_f32_decode")
        function = (flagos_ffn_swiglu_q4_k_f32_decode_staged if staged
                    else flagos_ffn_swiglu_q4_k_f32_decode)
    signature = {
        "gate_u8": "*u8", "gate_f16": "*fp16",
        "up_u8": "*u8", "up_f16": "*fp16",
        "x": "*fp32", "output": "*fp32", "k": "i32", "rows": "i32",
        "BLOCK_M": "constexpr",
    }
    source = ASTSource(
        function,
        signature,
        {"BLOCK_M": block_m},
        attrs=amd_jit_specialization_attrs(6, (6, 7)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": num_warps, "waves_per_eu": waves_per_eu},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={metadata.profile_scratch_size})")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    kernels = [{
            "name": name,
            "symbol": name,
            "file": output.name,
            "shared": metadata.shared,
            "num_warps": metadata.num_warps,
            "warp_size": metadata.warp_size,
            "block_size": block_m,
            "tile_m": 0,
            "tile_n": 0,
            "tile_k": 0,
            "argument_count": 8,
            "global_scratch_size": global_scratch_size,
            "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
            "profile_scratch_size": metadata.profile_scratch_size,
            "profile_scratch_align": metadata.profile_scratch_align,
        }]
    write_manifest(output_dir, arch, kernels)


def compile_q4_narrow8_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the AMD eight-row Q4 GEMV symbol without dispatching it."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if Q4_GEMV_NARROW8_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_Q4_GEMV_NARROW8_NUM_WARPS must be 1, 2, 4, or 8")
    name = "flagos_mul_mat_q4_k_f32_narrow8"
    signature = {
        "weights_u8": "*u8", "weights_f16": "*fp16",
        "x": "*fp32", "output": "*fp32", "k": "i32", "rows": "i32",
    }
    source = ASTSource(
        common.flagos_mul_mat_q4_k_f32_narrow8,
        signature,
        attrs=amd_jit_specialization_attrs(4, (4, 5)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": Q4_GEMV_NARROW8_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={metadata.profile_scratch_size})")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    kernels = [{
            "name": name,
            "symbol": name,
            "file": output.name,
            "shared": metadata.shared,
            "num_warps": metadata.num_warps,
            "warp_size": metadata.warp_size,
            "block_size": 8,
            "tile_m": 0,
            "tile_n": 0,
            "tile_k": 0,
            "argument_count": 6,
            "global_scratch_size": global_scratch_size,
            "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
            "profile_scratch_size": metadata.profile_scratch_size,
            "profile_scratch_align": metadata.profile_scratch_align,
        }]
    write_manifest(output_dir, arch, kernels)


def compile_q5_narrow16_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the gfx1150 sixteen-row Q5_K GEMV without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if Q5_GEMV_NARROW16_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError(
            "FLAGOS_Q5_GEMV_NARROW16_NUM_WARPS must be 1, 2, 4, or 8")
    name = "flagos_mul_mat_q5_k_f32_narrow16"
    signature = {
        "weights_u8": "*u8", "weights_f16": "*fp16",
        "x": "*fp32", "output": "*fp32", "k": "i32", "rows": "i32",
        "BLOCK_M": "constexpr",
    }
    source = ASTSource(
        common.flagos_mul_mat_q5_k_f32_narrow16,
        signature,
        {"BLOCK_M": 16},
        attrs=amd_jit_specialization_attrs(4, (4, 5)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={"num_warps": Q5_GEMV_NARROW16_NUM_WARPS},
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={metadata.profile_scratch_size})")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": 16,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 6,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_q40_narrow_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile one row-tiled Q4_0 GEMV without dispatching it."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    if (Q40_GEMV_NARROW_BLOCK_M <= 1 or Q40_GEMV_NARROW_BLOCK_M > 32 or
            Q40_GEMV_NARROW_BLOCK_M & (Q40_GEMV_NARROW_BLOCK_M - 1)):
        raise RuntimeError("FLAGOS_Q40_GEMV_NARROW_BLOCK_M must be a power of two from 2 to 32")
    if Q40_GEMV_NARROW_NUM_WARPS not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_Q40_GEMV_NARROW_NUM_WARPS must be 1, 2, 4, or 8")
    if Q40_GEMV_NARROW_WAVES_PER_EU not in (1, 2, 3, 4):
        raise RuntimeError("FLAGOS_Q40_GEMV_NARROW_WAVES_PER_EU must be 1, 2, 3, or 4")
    name = "flagos_mul_mat_q4_0_f32_narrow"
    signature = {
        "weights_u8": "*u8", "weights_f16": "*fp16",
        "x": "*fp32", "output": "*fp32", "k": "i32", "rows": "i32",
        "BLOCK_M": "constexpr",
    }
    source = ASTSource(
        common.flagos_mul_mat_q4_0_f32_narrow,
        signature,
        {"BLOCK_M": Q40_GEMV_NARROW_BLOCK_M},
        attrs=amd_jit_specialization_attrs(4, (4, 5)),
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options={
            "num_warps": Q40_GEMV_NARROW_NUM_WARPS,
            "waves_per_eu": Q40_GEMV_NARROW_WAVES_PER_EU,
        },
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(
            f"{name} requires unsupported Triton scratch storage "
            f"(global={global_scratch_size}, profile={metadata.profile_scratch_size})")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": Q40_GEMV_NARROW_BLOCK_M,
        "tile_m": 0,
        "tile_n": 0,
        "tile_k": 0,
        "argument_count": 6,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def compile_q41_q80_package_without_launch(output_dir: Path, arch: str) -> None:
    """Compile the small-block Q4_1/Q8_0 extension without device launches."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    definitions = [
        ("flagos_get_rows_q4_1_f32", common.flagos_get_rows_q4_1_f32,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "row_index": "*i32",
          "output": "*fp32", "n_cols": "i32"}, common.QK4_1),
        ("flagos_get_rows_q8_0_f32", common.flagos_get_rows_q8_0_f32,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "row_index": "*i32",
          "output": "*fp32", "n_cols": "i32"}, common.QK8_0),
        ("flagos_dequant_q4_1_f16", common.flagos_dequant_q4_1_f16,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "output": "*fp16"}, common.QK4_1),
        ("flagos_dequant_q8_0_f16", common.flagos_dequant_q8_0_f16,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "output": "*fp16"}, common.QK8_0),
        ("flagos_mul_mat_q4_1_f32", common.flagos_mul_mat_q4_1_f32,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32",
          "output": "*fp32", "k": "i32", "rows": "i32"}, common.QK4_1),
        ("flagos_mul_mat_q8_0_f32", common.flagos_mul_mat_q8_0_f32,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32",
          "output": "*fp32", "k": "i32", "rows": "i32"}, common.QK8_0),
        ("flagos_mul_mat_q4_1_f32_batched", common.flagos_mul_mat_q4_1_f32_batched,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32",
          "output": "*fp32", "k": "i32", "rows": "i32", "columns": "i32",
          "COLS_PER_BLOCK": "constexpr"}, common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_mul_mat_q8_0_f32_batched", common.flagos_mul_mat_q8_0_f32_batched,
         {"weights_u8": "*u8", "weights_f16": "*fp16", "x": "*fp32",
          "output": "*fp32", "k": "i32", "rows": "i32", "columns": "i32",
          "COLS_PER_BLOCK": "constexpr"}, common.MUL_MAT_COLS_PER_BLOCK),
    ]
    kernels = []
    for name, function, signature, block_size in definitions:
        constexprs = {key for key, value in signature.items() if value == "constexpr"}
        source = ASTSource(function, signature,
                           {key: block_size for key in constexprs},
                           attrs=amd_jit_specialization_attrs(
                               sum(value.startswith("*") for value in signature.values()),
                               tuple(index for index, (key, value) in enumerate(signature.items())
                                     if value != "constexpr" and not value.startswith("*"))))
        compile_warps = common.GEMV_NUM_WARPS if (
            name.startswith("flagos_mul_mat_q") and not name.endswith("_batched")
        ) else NUM_WARPS
        compiled = triton.compile(source, target=GPUTarget("hip", arch, 32),
                                  options={"num_warps": compile_warps})
        metadata = compiled.metadata
        if getattr(metadata, "global_scratch_size", 0) or metadata.profile_scratch_size:
            raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
        output = output_dir / f"{name}.hsaco"
        output.write_bytes(compiled.asm["hsaco"])
        kernels.append({
            "name": name, "symbol": name, "file": output.name,
            "shared": metadata.shared, "num_warps": metadata.num_warps,
            "warp_size": metadata.warp_size, "block_size": block_size,
            "tile_m": 0, "tile_n": 0, "tile_k": 0,
            "argument_count": len(signature) - len(constexprs),
            "global_scratch_size": getattr(metadata, "global_scratch_size", 0),
            "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
            "profile_scratch_size": metadata.profile_scratch_size,
            "profile_scratch_align": metadata.profile_scratch_align,
        })
    write_manifest(output_dir, arch, kernels)


def compile_gdn_cache_package_without_launch(
        output_dir: Path, arch: str, cache_only: bool = False,
        decode_only: bool = False) -> None:
    """Compile the GDN recurrent-cache fusion without a device launch."""
    if not arch:
        raise RuntimeError("--compile-only requires --arch")
    cols = int(os.environ.get("FLAGOS_GDN_CACHE_COLS", "4"))
    num_warps = int(os.environ.get("FLAGOS_GDN_CACHE_NUM_WARPS", "4"))
    waves_per_eu = int(os.environ.get("FLAGOS_GDN_CACHE_WAVES_PER_EU", "0"))
    if cols not in (1, 2, 4, 8, 16, 32, 64):
        raise RuntimeError("FLAGOS_GDN_CACHE_COLS must be 1, 2, 4, 8, 16, 32, or 64")
    if num_warps not in (1, 2, 4, 8):
        raise RuntimeError("FLAGOS_GDN_CACHE_NUM_WARPS must be 1, 2, 4, or 8")
    if waves_per_eu not in (0, 1, 2, 3, 4):
        raise RuntimeError("FLAGOS_GDN_CACHE_WAVES_PER_EU must be 0, 1, 2, 3, or 4")
    if decode_only and not cache_only:
        raise RuntimeError("decode-only GDN requires the cache-only contract")
    name = (
        "flagos_gated_delta_net_scalar_f32_cache_only_decode" if decode_only else
        "flagos_gated_delta_net_scalar_f32_cache_only" if cache_only else
        "flagos_gated_delta_net_scalar_f32_cache")
    signature = {
        "q": "*fp32", "k": "*fp32", "v": "*fp32", "gate": "*fp32",
        "beta": "*fp32", "current_state": "*fp32", "output": "*fp32",
        "cache": "*fp32", "state_size": "i32", "n_heads": "i32",
        "n_tokens": "i32", "n_seqs": "i32", "sq1": "i32", "sq2": "i32",
        "sq3": "i32", "sv1": "i32", "sv2": "i32", "sv3": "i32",
        "sb1": "i32", "sb2": "i32", "sb3": "i32", "q_heads": "i32",
        "q_seq_ratio": "i32", "snapshot_count": "i32", "scale": "fp32",
        "cache_slot_stride": "i32", "BLOCK": "constexpr", "COLS": "constexpr",
    }
    constexprs = {"BLOCK": 128, "COLS": cols}
    if cache_only:
        signature["EXACT_STATE_SIZE"] = "constexpr"
        signature["EXACT_SCALE"] = "constexpr"
        constexprs["EXACT_STATE_SIZE"] = 128
        constexprs["EXACT_SCALE"] = 128 ** -0.5
    function = (
        common.flagos_gated_delta_net_scalar_f32_cache_only_decode if decode_only else
        common.flagos_gated_delta_net_scalar_f32_cache_only if cache_only else
        common.flagos_gated_delta_net_scalar_f32_cache)
    argument_names = tuple(signature)
    divisible_arguments = tuple(argument_names.index(name) for name in (
        "state_size", "sq1", "sq2", "sq3", "sv1", "sv2", "sv3",
        "cache_slot_stride"))
    source = ASTSource(
        function,
        signature,
        constexprs,
        # Runtime dimensions include values such as n_tokens=1,
        # snapshot_count=1, q_seq_ratio=1, and an arbitrary F32 scale. A
        # blanket scalar divisibility attribute would be an invalid compiler
        # promise. State size, the Q/K/V row strides, and the optional cache
        # slot stride are the only values the provider proves 16-divisible.
        attrs=amd_jit_specialization_attrs(8, divisible_arguments),
    )
    compile_options = {"num_warps": num_warps}
    if waves_per_eu != 0:
        compile_options["waves_per_eu"] = waves_per_eu
    compiled = triton.compile(
        source,
        target=GPUTarget("hip", arch, 32),
        options=compile_options,
    )
    metadata = compiled.metadata
    global_scratch_size = getattr(metadata, "global_scratch_size", 0)
    if global_scratch_size or metadata.profile_scratch_size:
        raise RuntimeError(f"{name} requires unsupported Triton scratch storage")
    output = output_dir / f"{name}.hsaco"
    output.write_bytes(compiled.asm["hsaco"])
    write_manifest(output_dir, arch, [{
        "name": name,
        "symbol": name,
        "file": output.name,
        "shared": metadata.shared,
        "num_warps": metadata.num_warps,
        "warp_size": metadata.warp_size,
        "block_size": 128,
        "exact_block_size": cache_only,
        "tile_m": 0,
        "tile_n": cols,
        "tile_k": 0,
        "argument_count": 26,
        "global_scratch_size": global_scratch_size,
        "global_scratch_align": getattr(metadata, "global_scratch_align", 1),
        "profile_scratch_size": metadata.profile_scratch_size,
        "profile_scratch_align": metadata.profile_scratch_align,
    }])


def validate_tuning_profile_kernel_abis() -> None:
    """Keep the launcher argument contract tied to the Triton signatures."""
    known_contracts = {}
    for _, contracts in TUNING_PROFILES.values():
        for name, contract in contracts.items():
            previous = known_contracts.setdefault(name, contract)
            if previous.argument_count != contract.argument_count:
                raise RuntimeError(f"conflicting tuned kernel ABI contracts for {name}")
    for name, contract in known_contracts.items():
        function = globals().get(name)
        if function is None:
            function = getattr(common, name, None)
        arg_names = getattr(function, "arg_names", None)
        constexprs = getattr(function, "constexprs", None)
        if arg_names is None or constexprs is None:
            raise RuntimeError(f"tuned kernel has no Triton signature: {name}")
        actual = len(arg_names) - len(constexprs)
        if actual != contract.argument_count:
            raise RuntimeError(
                f"tuned kernel ABI mismatch for {name}: "
                f"contract={contract.argument_count}, signature={actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        default=(Path(os.environ["TRITON_CACHE_DIR"])
                                 if os.environ.get("TRITON_CACHE_DIR") else None))
    parser.add_argument("--arch", default=os.environ.get("FLAGOS_AMD_ARCH", ""))
    parser.add_argument(
        "--compile-only", action="store_true",
        help="compile one selected AMD tuning symbol without launching it")
    parser.add_argument("--extension", choices=("dense-gather", "aligned-gemm", "tiled-ffn", "packed-ffn", "strided-rms", "q6-gemv", "gguf-gemv", "core-ops", "graph-ops", "row-ops", "softmax-broadcast", "softmax-broadcast-wide", "strided-batched-gemm", "strided-batched-gemm-ieee", "dense-f32-gemm", "half-gather", "flash-attention", "flash-decode-splitkv", "flash-decode-grouped-splitkv", "ggml-rope", "ggml-neox-rope", "gdn-value-tile", "gdn-scalar", "matrix-stage1", "ffn-waves"),
                        help="compile an optional ABI extension as a separate package")
    parser.add_argument("--flash-attention-profile", choices=("baseline", "gfx1150-head256"),
                        help="optional offline attention profile; runtime dispatch and defaults are unchanged")
    args = parser.parse_args()
    if args.flash_attention_profile and args.extension != "flash-attention":
        raise RuntimeError("--flash-attention-profile requires --extension flash-attention")
    if args.flash_attention_profile == "gfx1150-head256" and args.arch != "gfx1150":
        raise RuntimeError("gfx1150-head256 profile requires --arch gfx1150")
    if args.extension and (not args.compile_only or not args.arch):
        raise RuntimeError("--extension requires --compile-only and --arch")
    if args.extension and (os.environ.get("FLAGOS_AMD_TUNING_PROFILE_NAME") or
            any(key.startswith("FLAGOS_AMD_ONLY_") and value == "1"
                for key, value in os.environ.items())):
        raise RuntimeError("--extension cannot be combined with a tuning profile or ONLY mode")
    validate_tuning_profile_kernel_abis()
    if args.cache_dir is None:
        raise RuntimeError("--cache-dir or TRITON_CACHE_DIR is required")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(args.cache_dir.resolve())
    final_output_dir = args.output_dir.resolve()
    if final_output_dir.exists() and (
            final_output_dir.is_symlink() or not final_output_dir.is_dir() or
            any(final_output_dir.iterdir())):
        raise RuntimeError(f"--output-dir must not exist or must be empty: {final_output_dir}")
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = tempfile.TemporaryDirectory(
        prefix=f".{final_output_dir.name}.", dir=final_output_dir.parent)
    staging_output_dir = Path(temporary_output.name) / "package"
    staging_output_dir.mkdir()
    args.output_dir = staging_output_dir

    def commit_output() -> None:
        if final_output_dir.exists():
            final_output_dir.rmdir()
        staging_output_dir.rename(final_output_dir)
        temporary_output.cleanup()
    if args.extension:
        if args.extension == "dense-gather":
            compile_dense_gather_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "aligned-gemm":
            compile_aligned_gemm_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "tiled-ffn":
            compile_tiled_ffn_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "packed-ffn":
            compile_packed_ffn_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "flash-attention":
            compile_flash_attention_package_without_launch(args.output_dir, args.arch, args.flash_attention_profile or "baseline")
        elif args.extension == "flash-decode-splitkv":
            compile_splitkv_decode_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "flash-decode-grouped-splitkv":
            compile_grouped_splitkv_decode_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "ggml-rope":
            compile_ggml_rope_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "ggml-neox-rope":
            compile_ggml_rope_package_without_launch(args.output_dir, args.arch, neox=True)
        elif args.extension == "q6-gemv":
            compile_q6_gemv_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "gguf-gemv":
            compile_gguf_gemv_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "core-ops":
            compile_core_ops_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "graph-ops":
            compile_core_ops_package_without_launch(args.output_dir, args.arch, graph_ops=True)
        elif args.extension == "row-ops":
            compile_rows_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "softmax-broadcast":
            compile_softmax_broadcast_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "softmax-broadcast-wide":
            compile_softmax_broadcast_package_without_launch(args.output_dir, args.arch, wide=True)
        elif args.extension == "strided-batched-gemm":
            compile_strided_batched_gemm_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "strided-batched-gemm-ieee":
            compile_strided_batched_gemm_package_without_launch(args.output_dir, args.arch, preserve_src1=True)
        elif args.extension == "dense-f32-gemm":
            compile_strided_batched_gemm_package_without_launch(args.output_dir, args.arch, f32_weights=True)
        elif args.extension == "half-gather":
            compile_half_gather_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "gdn-value-tile":
            compile_gdn_value_tile_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "gdn-scalar":
            compile_gdn_value_tile_package_without_launch(args.output_dir, args.arch, baseline=True)
        elif args.extension == "matrix-stage1":
            compile_matrix_stage1_package_without_launch(args.output_dir, args.arch)
        elif args.extension == "ffn-waves":
            compile_ffn_waves_package_without_launch(args.output_dir, args.arch)
        else:
            compile_strided_rms_package_without_launch(args.output_dir, args.arch)
        commit_output()
        print(f"wrote {args.extension} extension for {args.arch} to {final_output_dir}")
        return
    only_residual = os.environ.get("FLAGOS_AMD_ONLY_RESIDUAL") == "1"
    only_residual_narrow = os.environ.get("FLAGOS_AMD_ONLY_RESIDUAL_NARROW") == "1"
    only_q4_ffn_decode = os.environ.get("FLAGOS_AMD_ONLY_Q4_FFN_DECODE") == "1"
    only_q40_ffn_decode = os.environ.get("FLAGOS_AMD_ONLY_Q40_FFN_DECODE") == "1"
    only_q4_ffn_decode_staged = os.environ.get("FLAGOS_AMD_ONLY_Q4_FFN_DECODE_STAGED") == "1"
    only_q40_ffn_decode_staged = os.environ.get("FLAGOS_AMD_ONLY_Q40_FFN_DECODE_STAGED") == "1"
    only_q4_gemv_narrow8 = os.environ.get("FLAGOS_AMD_ONLY_Q4_GEMV_NARROW8") == "1"
    only_q5_gemv_narrow16 = os.environ.get("FLAGOS_AMD_ONLY_Q5_GEMV_NARROW16") == "1"
    only_q40_gemv_narrow = os.environ.get("FLAGOS_AMD_ONLY_Q40_GEMV_NARROW") == "1"
    only_q4_dequant = os.environ.get("FLAGOS_AMD_ONLY_Q4_DEQUANT") == "1"
    only_q41_q80 = os.environ.get("FLAGOS_AMD_ONLY_Q41_Q80") == "1"
    only_gdn_cache = os.environ.get("FLAGOS_AMD_ONLY_GDN_CACHE") == "1"
    only_gdn_cache_only = os.environ.get("FLAGOS_AMD_ONLY_GDN_CACHE_ONLY") == "1"
    only_gdn_cache_only_decode = (
        os.environ.get("FLAGOS_AMD_ONLY_GDN_CACHE_ONLY_DECODE") == "1")
    only_f16_gemm = os.environ.get("FLAGOS_AMD_ONLY_F16_GEMM") == "1"
    only_ffn_fusion = os.environ.get("FLAGOS_AMD_ONLY_FFN_FUSION") == "1"
    only_ffn_down_f16 = os.environ.get("FLAGOS_AMD_ONLY_FFN_DOWN_F16") == "1"
    only_scale = os.environ.get("FLAGOS_AMD_ONLY_SCALE") == "1"
    only_ssm_conv_silu = os.environ.get("FLAGOS_AMD_ONLY_SSM_CONV_SILU") == "1"
    only_silu_mul = os.environ.get("FLAGOS_AMD_ONLY_SILU_MUL") == "1"
    only_unary_mul = os.environ.get("FLAGOS_AMD_ONLY_UNARY_MUL") == "1"
    only_gdn_alpha_gate = os.environ.get("FLAGOS_AMD_ONLY_GDN_ALPHA_GATE") == "1"
    only_gdn_q8_gate = os.environ.get("FLAGOS_AMD_ONLY_GDN_Q8_GATE") == "1"
    only_rms_norm_narrow = os.environ.get("FLAGOS_AMD_ONLY_RMS_NORM_NARROW") == "1"
    if sum((only_residual, only_residual_narrow, only_q4_ffn_decode, only_q40_ffn_decode,
            only_q4_ffn_decode_staged, only_q40_ffn_decode_staged,
            only_q4_gemv_narrow8, only_q5_gemv_narrow16, only_q40_gemv_narrow,
            only_q4_dequant, only_q41_q80, only_gdn_cache, only_gdn_cache_only,
            only_gdn_cache_only_decode, only_f16_gemm, only_ffn_fusion,
            only_ffn_down_f16, only_scale, only_ssm_conv_silu, only_silu_mul,
            only_unary_mul, only_gdn_alpha_gate, only_gdn_q8_gate,
            only_rms_norm_narrow)) > 1:
        raise RuntimeError("select only one FLAGOS_AMD_ONLY_* tuning mode")
    if args.compile_only:
        if not (only_residual or only_residual_narrow or
                only_q4_ffn_decode or only_q40_ffn_decode or only_q4_ffn_decode_staged or
                only_q40_ffn_decode_staged or
                only_q4_gemv_narrow8 or only_q5_gemv_narrow16 or
                only_q40_gemv_narrow or only_q4_dequant or only_q41_q80 or
                only_gdn_cache or only_gdn_cache_only or only_gdn_cache_only_decode or
                only_scale or only_ssm_conv_silu or only_silu_mul or only_unary_mul or
                only_gdn_alpha_gate or only_gdn_q8_gate or
                only_rms_norm_narrow):
            raise RuntimeError(
                "--compile-only requires FLAGOS_AMD_ONLY_RESIDUAL=1, "
                "FLAGOS_AMD_ONLY_RESIDUAL_NARROW=1, "
                "FLAGOS_AMD_ONLY_Q4_FFN_DECODE=1, "
                "FLAGOS_AMD_ONLY_Q40_FFN_DECODE=1, "
                "FLAGOS_AMD_ONLY_Q4_FFN_DECODE_STAGED=1, "
                "FLAGOS_AMD_ONLY_Q40_FFN_DECODE_STAGED=1, "
                "FLAGOS_AMD_ONLY_Q4_GEMV_NARROW8=1, "
                "FLAGOS_AMD_ONLY_Q5_GEMV_NARROW16=1, "
                "FLAGOS_AMD_ONLY_Q40_GEMV_NARROW=1, "
                "FLAGOS_AMD_ONLY_Q4_DEQUANT=1, "
                "FLAGOS_AMD_ONLY_Q41_Q80=1, "
                "FLAGOS_AMD_ONLY_GDN_CACHE=1, "
                "FLAGOS_AMD_ONLY_GDN_CACHE_ONLY=1 or "
                "FLAGOS_AMD_ONLY_GDN_CACHE_ONLY_DECODE=1 or "
                "FLAGOS_AMD_ONLY_SCALE=1 or "
                "FLAGOS_AMD_ONLY_SSM_CONV_SILU=1 or "
                "FLAGOS_AMD_ONLY_SILU_MUL=1 or "
                "FLAGOS_AMD_ONLY_UNARY_MUL=1 or "
                "FLAGOS_AMD_ONLY_GDN_ALPHA_GATE=1 or "
                "FLAGOS_AMD_ONLY_GDN_Q8_GATE=1 or "
                "FLAGOS_AMD_ONLY_RMS_NORM_NARROW=1")
        if only_residual or only_residual_narrow:
            compile_residual_package_without_launch(
                args.output_dir, args.arch, only_residual_narrow)
        elif only_q4_gemv_narrow8:
            compile_q4_narrow8_package_without_launch(args.output_dir, args.arch)
        elif only_q5_gemv_narrow16:
            compile_q5_narrow16_package_without_launch(args.output_dir, args.arch)
        elif only_q40_gemv_narrow:
            compile_q40_narrow_package_without_launch(args.output_dir, args.arch)
        elif only_q4_dequant:
            compile_q4_dequant_package_without_launch(args.output_dir, args.arch)
        elif only_q41_q80:
            compile_q41_q80_package_without_launch(args.output_dir, args.arch)
        elif only_scale:
            compile_scale_package_without_launch(args.output_dir, args.arch)
        elif only_ssm_conv_silu:
            compile_ssm_conv_silu_package_without_launch(args.output_dir, args.arch)
        elif only_silu_mul:
            compile_silu_mul_package_without_launch(args.output_dir, args.arch)
        elif only_unary_mul:
            compile_unary_mul_package_without_launch(args.output_dir, args.arch)
        elif only_gdn_alpha_gate:
            compile_gdn_alpha_gate_package_without_launch(args.output_dir, args.arch)
        elif only_gdn_q8_gate:
            compile_gdn_q8_gate_package_without_launch(args.output_dir, args.arch)
        elif only_rms_norm_narrow:
            compile_rms_norm_narrow_package_without_launch(args.output_dir, args.arch)
        elif only_gdn_cache or only_gdn_cache_only or only_gdn_cache_only_decode:
            compile_gdn_cache_package_without_launch(
                args.output_dir, args.arch,
                only_gdn_cache_only or only_gdn_cache_only_decode,
                only_gdn_cache_only_decode)
        elif only_q40_ffn_decode or only_q40_ffn_decode_staged:
            compile_q4_ffn_package_without_launch(
                args.output_dir, args.arch, only_q40_ffn_decode_staged, "q4_0")
        else:
            compile_q4_ffn_package_without_launch(
                args.output_dir, args.arch, only_q4_ffn_decode_staged)
        commit_output()
        print(f"wrote compiler-only AMD package for {args.arch} to {final_output_dir}")
        return
    if only_q4_gemv_narrow8:
        raise RuntimeError("FLAGOS_AMD_ONLY_Q4_GEMV_NARROW8 requires --compile-only")
    if only_q5_gemv_narrow16:
        raise RuntimeError("FLAGOS_AMD_ONLY_Q5_GEMV_NARROW16 requires --compile-only")
    if only_q40_gemv_narrow:
        raise RuntimeError("FLAGOS_AMD_ONLY_Q40_GEMV_NARROW requires --compile-only")
    if only_q4_dequant:
        raise RuntimeError("FLAGOS_AMD_ONLY_Q4_DEQUANT requires --compile-only")
    if only_gdn_cache:
        raise RuntimeError("FLAGOS_AMD_ONLY_GDN_CACHE requires --compile-only")
    if only_gdn_cache_only:
        raise RuntimeError("FLAGOS_AMD_ONLY_GDN_CACHE_ONLY requires --compile-only")
    if only_gdn_cache_only_decode:
        raise RuntimeError(
            "FLAGOS_AMD_ONLY_GDN_CACHE_ONLY_DECODE requires --compile-only")
    if only_scale:
        raise RuntimeError("FLAGOS_AMD_ONLY_SCALE requires --compile-only")
    if only_ssm_conv_silu:
        raise RuntimeError("FLAGOS_AMD_ONLY_SSM_CONV_SILU requires --compile-only")
    if only_silu_mul:
        raise RuntimeError("FLAGOS_AMD_ONLY_SILU_MUL requires --compile-only")
    if only_unary_mul:
        raise RuntimeError("FLAGOS_AMD_ONLY_UNARY_MUL requires --compile-only")
    if only_gdn_alpha_gate:
        raise RuntimeError("FLAGOS_AMD_ONLY_GDN_ALPHA_GATE requires --compile-only")
    if only_gdn_q8_gate:
        raise RuntimeError("FLAGOS_AMD_ONLY_GDN_Q8_GATE requires --compile-only")
    if only_rms_norm_narrow:
        raise RuntimeError("FLAGOS_AMD_ONLY_RMS_NORM_NARROW requires --compile-only")
    compile_common = (os.environ.get("FLAGOS_AMD_SKIP_COMMON", "0") != "1" and
                      not only_residual and not only_residual_narrow and
                      not only_q4_ffn_decode and not only_q40_ffn_decode and
                      not only_q4_ffn_decode_staged and not only_q40_ffn_decode_staged and
                      not only_f16_gemm and not only_ffn_fusion and not only_ffn_down_f16)
    if compile_common:
        common.compile_kernels(assert_close=assert_common_kernel_close)
    compile_additional()

    names = [
        ("flagos_add_f32", common.BLOCK_SIZE),
        ("flagos_add_repeat_f32", common.BLOCK_SIZE),
        ("flagos_mul_f32", common.BLOCK_SIZE),
        ("flagos_scale_f32", common.BLOCK_SIZE),
        ("flagos_copy_strided_f32", common.BLOCK_SIZE),
        ("flagos_concat_f32", common.BLOCK_SIZE),
        ("flagos_rms_norm_f32", common.RMS_NORM_BLOCK_SIZE),
        ("flagos_l2_norm_strided_f32", common.ROW_BLOCK_SIZE),
        ("flagos_rms_norm_mul_f32", common.RMS_NORM_BLOCK_SIZE),
        ("flagos_add_rms_norm_mul_f32", common.RMS_NORM_BLOCK_SIZE),
        ("flagos_rms_norm_mul_inplace_f32", AMD_ROW_BLOCK_SIZE),
        ("flagos_rms_norm_mul_inplace_f32_narrow",
         AMD_RMS_NORM_NARROW_BLOCK_SIZE, 0, 0, 0, True),
        ("flagos_add_rms_norm_mul_inplace_f32", AMD_ROW_BLOCK_SIZE),
        ("flagos_add_rms_norm_mul_residual_f32", AMD_RESIDUAL_BLOCK_SIZE),
        ("flagos_add_rms_norm_mul_residual_f32_narrow", AMD_RESIDUAL_NARROW_BLOCK_SIZE),
        ("flagos_swiglu_split_f32", common.BLOCK_SIZE),
        ("flagos_flash_attn_decode_f32_f16", common.ATTENTION_BLOCK_N),
        ("flagos_flash_attn_prefill_f32_f16", ATTENTION_BLOCK_M),
        ("flagos_rope_neox_f32", common.BLOCK_SIZE),
        ("flagos_mrope_f32", common.BLOCK_SIZE),
        ("flagos_rope_kv_store_f32_f16", common.BLOCK_SIZE),
        ("flagos_rms_norm_mul_rope_neox_f32", ATTENTION_HEAD_DIM),
        ("flagos_rms_norm_mul_rope_kv_store_neox_f32_f16", ATTENTION_HEAD_DIM),
        ("flagos_silu_f32", common.BLOCK_SIZE),
        ("flagos_silu_mul_f32", AMD_SILU_MUL_BLOCK_SIZE),
        ("flagos_sigmoid_mul_f32", AMD_SILU_MUL_BLOCK_SIZE),
        ("flagos_softplus_mul_f32", AMD_SILU_MUL_BLOCK_SIZE),
        ("flagos_gdn_alpha_gate_f32", AMD_GDN_ALPHA_GATE_BLOCK_SIZE,
         0, AMD_GDN_ALPHA_GATE_WIDTH, 0),
        ("flagos_sigmoid_f32", common.BLOCK_SIZE),
        ("flagos_softplus_f32", common.BLOCK_SIZE),
        ("flagos_ssm_conv_f32", common.BLOCK_SIZE),
        ("flagos_ssm_conv_silu_f32", common.SSM_CONV_SILU_BLOCK_SIZE),
        ("flagos_gated_delta_net_scalar_f32", 128, 0, 4, 0),
        ("flagos_gated_delta_net_scalar_f32_cache", 128, 0, 4, 0),
        ("flagos_set_rows_f32_f16", common.BLOCK_SIZE),
        ("flagos_get_rows_q4_0_f32", common.QK4_0),
        ("flagos_get_rows_q4_1_f32", common.QK4_1),
        ("flagos_get_rows_q5_k_f32", common.QK_K),
        ("flagos_get_rows_q4_k_f32", common.QK_K),
        ("flagos_get_rows_q6_k_f32", common.QK_K),
        ("flagos_dequant_q4_0_f16", common.QK4_0),
        ("flagos_dequant_q4_1_f16", common.QK4_1),
        ("flagos_dequant_q5_k_f16", common.QK_K),
        ("flagos_dequant_q4_k_f16", common.QK_K),
        ("flagos_dequant_q6_k_f16", common.QK_K),
        ("flagos_mul_mat_q4_0_f32", common.QK4_0),
        ("flagos_mul_mat_q4_1_f32", common.QK4_1),
        ("flagos_mul_mat_q5_k_f32", common.QK_K),
        ("flagos_mul_mat_q4_k_f32", common.QK_K),
        ("flagos_mul_mat_q4_k_f32_narrow", 4),
        ("flagos_mul_mat_q6_k_f32", common.QK_K),
        ("flagos_mul_mat_q4_0_f32_batched", common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_mul_mat_q4_1_f32_batched", common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_mul_mat_q5_k_f32_batched", common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_mul_mat_q4_k_f32_batched", common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_mul_mat_q6_k_f32_batched", common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_get_rows_q8_0_f32", common.QK8_0),
        ("flagos_dequant_q8_0_f16", common.QK8_0),
        ("flagos_mul_mat_q8_0_f32", common.QK8_0),
        ("flagos_mul_mat_q8_0_f32_batched", common.MUL_MAT_COLS_PER_BLOCK),
        ("flagos_mul_mat_f16_f32_batched", F16_MATMUL_BLOCK_N,
         F16_MATMUL_BLOCK_M, F16_MATMUL_BLOCK_N, F16_MATMUL_BLOCK_K),
        ("flagos_mul_mat_f16_f32_grouped", F16_MATMUL_BLOCK_N,
         F16_MATMUL_BLOCK_M, F16_MATMUL_BLOCK_N, F16_MATMUL_BLOCK_K),
        ("flagos_soft_max_unmasked_f32", AMD_SOFTMAX_BLOCK_SIZE),
        ("flagos_soft_max_masked_f32_f16", AMD_SOFTMAX_BLOCK_SIZE),
    ]
    if only_residual:
        names = [("flagos_add_rms_norm_mul_residual_f32", AMD_RESIDUAL_BLOCK_SIZE)]
    if only_residual_narrow:
        names = [(
            "flagos_add_rms_norm_mul_residual_f32_narrow",
            AMD_RESIDUAL_NARROW_BLOCK_SIZE)]
    if only_q4_ffn_decode:
        names = [("flagos_ffn_swiglu_q4_k_f32_decode", Q4_FFN_DECODE_BLOCK_M)]
    if only_q40_ffn_decode:
        names = [("flagos_ffn_swiglu_q4_0_f32_decode", Q40_FFN_DECODE_BLOCK_M)]
    if only_q4_ffn_decode_staged:
        names = [("flagos_ffn_swiglu_q4_k_f32_decode_staged", Q4_FFN_DECODE_BLOCK_M)]
    if only_q40_ffn_decode_staged:
        names = [("flagos_ffn_swiglu_q4_0_f32_decode_staged", Q40_FFN_DECODE_BLOCK_M)]
    if only_f16_gemm:
        names = [
            ("flagos_mul_mat_f16_f32_batched", F16_MATMUL_BLOCK_N,
             F16_MATMUL_BLOCK_M, F16_MATMUL_BLOCK_N, F16_MATMUL_BLOCK_K),
            ("flagos_mul_mat_f16_f32_grouped", F16_MATMUL_BLOCK_N,
             F16_MATMUL_BLOCK_M, F16_MATMUL_BLOCK_N, F16_MATMUL_BLOCK_K),
        ]
    if only_ffn_fusion:
        names = [
            ("flagos_ffn_swiglu_f16_f32_batched", FFN_MATMUL_BLOCK_N,
             FFN_MATMUL_BLOCK_M, FFN_MATMUL_BLOCK_N, FFN_MATMUL_BLOCK_K),
            ("flagos_ffn_swiglu_f16_f32_grouped", FFN_MATMUL_BLOCK_N,
             FFN_MATMUL_BLOCK_M, FFN_MATMUL_BLOCK_N, FFN_MATMUL_BLOCK_K),
        ]
    if only_ffn_down_f16:
        names = [
            ("flagos_ffn_swiglu_f16_f16_grouped", FFN_MATMUL_BLOCK_N,
             FFN_MATMUL_BLOCK_M, FFN_MATMUL_BLOCK_N, FFN_MATMUL_BLOCK_K),
            ("flagos_mul_mat_f16_f16_grouped", FFN_DOWN_MATMUL_BLOCK_N,
             FFN_DOWN_MATMUL_BLOCK_M, FFN_DOWN_MATMUL_BLOCK_N,
             FFN_DOWN_MATMUL_BLOCK_K),
        ]
    only_tuning = (only_residual or only_residual_narrow or only_q4_ffn_decode or
                   only_q40_ffn_decode or
                   only_q4_ffn_decode_staged or only_q40_ffn_decode_staged or
                   only_f16_gemm or only_ffn_fusion or only_ffn_down_f16)
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q4_GEMV_NARROW8", "0") == "1":
        # Eight output rows per physical wave is a gfx1150 experiment.  Keep
        # it out of the stable manifest unless explicitly requested.
        names.insert(20, ("flagos_mul_mat_q4_k_f32_narrow8", 8))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q5_GEMV_NARROW16", "0") == "1":
        names.insert(20, ("flagos_mul_mat_q5_k_f32_narrow16", 16))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q40_GEMV_NARROW", "0") == "1":
        names.insert(20, ("flagos_mul_mat_q4_0_f32_narrow", Q40_GEMV_NARROW_BLOCK_M))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q4_FFN_DECODE", "0") == "1":
        names.insert(23, ("flagos_ffn_swiglu_q4_k_f32_decode", Q4_FFN_DECODE_BLOCK_M))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q40_FFN_DECODE", "0") == "1":
        names.insert(24, ("flagos_ffn_swiglu_q4_0_f32_decode", Q40_FFN_DECODE_BLOCK_M))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q4_FFN_DECODE_STAGED", "0") == "1":
        names.insert(24, ("flagos_ffn_swiglu_q4_k_f32_decode_staged", Q4_FFN_DECODE_BLOCK_M))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_Q40_FFN_DECODE_STAGED", "0") == "1":
        names.insert(24, ("flagos_ffn_swiglu_q4_0_f32_decode_staged", Q40_FFN_DECODE_BLOCK_M))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_QUANT_TILED", "0") == "1":
        names[22:22] = [
            ("flagos_mul_mat_q4_k_f32_tiled", common.QUANT_TILE_BLOCK_N,
             common.QUANT_TILE_BLOCK_M, common.QUANT_TILE_BLOCK_N, 64),
            ("flagos_mul_mat_q6_k_f32_tiled", common.QUANT_TILE_BLOCK_N,
             common.QUANT_TILE_BLOCK_M, common.QUANT_TILE_BLOCK_N, 64),
        ]
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_FFN_FUSION", "0") == "1":
        names.insert(23, ("flagos_ffn_swiglu_f16_f32_batched", FFN_MATMUL_BLOCK_N,
                          FFN_MATMUL_BLOCK_M, FFN_MATMUL_BLOCK_N, FFN_MATMUL_BLOCK_K))
        names.insert(24, ("flagos_ffn_swiglu_f16_f32_grouped", FFN_MATMUL_BLOCK_N,
                          FFN_MATMUL_BLOCK_M, FFN_MATMUL_BLOCK_N, FFN_MATMUL_BLOCK_K))
    if not only_tuning and os.environ.get("FLAGOS_AMD_EMIT_FFN_DOWN_F16", "0") == "1":
        names.insert(25, ("flagos_ffn_swiglu_f16_f16_grouped", FFN_MATMUL_BLOCK_N,
                          FFN_MATMUL_BLOCK_M, FFN_MATMUL_BLOCK_N, FFN_MATMUL_BLOCK_K))
        names.insert(26, ("flagos_mul_mat_f16_f16_grouped", FFN_DOWN_MATMUL_BLOCK_N,
                          FFN_DOWN_MATMUL_BLOCK_M, FFN_DOWN_MATMUL_BLOCK_N,
                          FFN_DOWN_MATMUL_BLOCK_K))
    # A declared profile is an exact, versioned package contract. The broad
    # default package may carry optional mixed-quant symbols, but profiled
    # builds must contain exactly the canonical entries expected by the C++
    # loader.
    tuning_profile = os.environ.get("FLAGOS_AMD_TUNING_PROFILE_NAME", "")
    if tuning_profile:
        _, contracts = tuning_profile_contracts(tuning_profile)
        names = [
            (name, contract.block_size, contract.tile_m, contract.tile_n, contract.tile_k,
             contract.exact_block_size)
            for name, contract in contracts.items()
        ]
    arch = args.arch or triton.runtime.driver.active.get_current_target().arch
    manifest_kernels = []
    for entry in names:
        name, block = entry[:2]
        tile_m, tile_n, tile_k = (tuple(entry[2:]) + (0, 0, 0))[:3]
        if name == "flagos_dequant_q4_k_f16":
            kernel = compile_q4_dequant_kernel_without_launch(
                args.output_dir, arch, tuning_profile)
        else:
            kernel = copy_artifact(
                args.cache_dir, args.output_dir, name, block, tile_m, tile_n, tile_k)
        if len(entry) > 5 and entry[5]:
            kernel["exact_block_size"] = True
        manifest_kernels.append(kernel)
    write_manifest(args.output_dir, arch, manifest_kernels)
    commit_output()
    print(f"wrote {len(manifest_kernels)} AMD kernels for {arch} to {final_output_dir}")


if __name__ == "__main__":
    with amd_hint_context():
        main()
