#!/usr/bin/env python3
"""Compile explicit FlagGems exports to an AMD package without GPU execution."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from flagos_amd_hsaco import validate_hsaco
from merge_aot_packages import (copy_validated_artifact, file_integrity, load_manifest,
                               MAX_CONSTANT_ASSET_BYTES, read_manifest, reject_json_constant)

RUNTIME_TYPES = {"i32", "i64", "fp32", "fp64", "*u8", "*i8", "*i32", "*i64", "*fp16", "*bf16", "*fp32", "*fp64"}


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise RuntimeError("duplicate JSON field: " + name)
        result[name] = value
    return result


def positive_int(value, field):
    if type(value) is not int or not 0 < value <= 2147483647:
        raise RuntimeError("invalid positive integer: " + field)
    return value


def plain_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", value):
        raise RuntimeError("invalid artifact name")
    return value


def validate_profile(profile, export):
    if not isinstance(profile, dict) or profile.get("semantic") != export.semantic:
        raise RuntimeError("export semantic mismatch")
    name = plain_name(profile.get("name", export.name))
    signature = profile.get("signature")
    constants = profile.get("constants")
    if (not isinstance(signature, dict) or set(signature) != set(export.runtime_arguments) or
            any(not isinstance(value, str) or value not in RUNTIME_TYPES for value in signature.values()) or
            not isinstance(constants, dict) or set(constants) != set(export.constexpr_arguments)):
        raise RuntimeError("export signature or constants mismatch")
    expected_types = getattr(export, "runtime_types", ())
    if expected_types and tuple(signature[name] for name in export.runtime_arguments) != expected_types:
        raise RuntimeError("export runtime type mismatch")
    if not 0 < len(signature) <= 30:
        raise RuntimeError("invalid explicit argument count")
    for field, value in constants.items():
        if type(value) not in (bool, int) or (type(value) is int and not -2147483648 <= value <= 2147483647):
            raise RuntimeError("invalid constexpr value: " + field)
    geometry = profile.get("geometry")
    if not isinstance(geometry, dict) or "block_size" not in geometry or set(geometry) - {"block_size", "tile_m", "tile_n", "tile_k"}:
        raise RuntimeError("invalid logical geometry")
    resolved = {}
    for field, value in geometry.items():
        if isinstance(value, str):
            if value not in constants:
                raise RuntimeError("geometry references an unknown constexpr")
            value = constants[value]
        resolved[field] = positive_int(value, field)
    warps = profile.get("num_warps", 4)
    if type(warps) is not int or warps not in (1, 2, 4, 8, 16):
        raise RuntimeError("invalid number of warps")
    return name, signature, constants, resolved, warps


def source_proof(root, exports):
    result = {}
    for export in exports:
        for name in export.source_files:
            path = Path(name).resolve()
            try:
                relative = str(path.relative_to(root))
            except ValueError as error:
                raise RuntimeError("export source is outside its source root") from error
            result[relative] = file_integrity(path)[1]
    return dict(sorted(result.items()))


def git_state(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return git("rev-parse", "HEAD"), git("status", "--porcelain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    profile_path = args.profile_file.resolve()
    specification_bytes = read_manifest(profile_path)
    specification = json.loads(specification_bytes, object_pairs_hook=unique_object, parse_constant=reject_json_constant)
    if not isinstance(specification, dict) or specification.get("format") != 1 or type(specification.get("format")) is not int:
        raise RuntimeError("unsupported export profile format")
    arch = specification.get("arch")
    wave = specification.get("warp_size", 32)
    profiles = specification.get("profiles")
    if (not isinstance(arch, str) or not re.fullmatch(r"gfx[0-9a-f]+", arch) or
            type(wave) is not int or wave not in (32, 64) or
            not isinstance(profiles, list) or not 0 < len(profiles) <= 64):
        raise RuntimeError("invalid target or profile count")
    revision, status = git_state(root)
    if status and not args.allow_dirty_source:
        raise RuntimeError("source tree is dirty; use an explicit --allow-dirty-source override")
    if "TRITON_INTERPRET" in os.environ:
        raise RuntimeError("offline compilation requires TRITON_INTERPRET to be unset")
    source_root = root / "src" if (root / "src/flag_gems_aot").is_dir() else root
    sys.path.insert(0, str(source_root))
    library = importlib.import_module("flag_gems_aot")
    if Path(library.__file__).resolve().parent != source_root / "flag_gems_aot":
        raise RuntimeError("wrong FlagGems source was imported")
    exports = []
    prepared = []
    names = set()
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("export"), str):
            raise RuntimeError("invalid export profile")
        export = library.get_kernel_export(profile["export"])
        contract = validate_profile(profile, export)
        if contract[0] in names:
            raise RuntimeError("duplicate exported profile name")
        names.add(contract[0]); exports.append(export); prepared.append((export, contract))
    sources = source_proof(root, exports)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        os.environ["TRITON_CACHE_DIR"] = str(args.cache_dir.absolute())
    import triton
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    try:
        from triton.compiler import hint_manager
        hint_manager._global_hint_manager = hint_manager.HintManager("hip")
    except ImportError:
        pass
    with tempfile.TemporaryDirectory(prefix=".flaggems-aot-", dir=output.parent) as temporary:
        stage = Path(temporary); kernels = []; assets = []
        for export, (name, signature, constants, geometry, warps) in prepared:
            fn = export.function
            full_signature = dict(signature, **{key: "constexpr" for key in constants})
            kernel = triton.compile(ASTSource(fn, full_signature, constants, attrs=export.aot_attributes),
                target=GPUTarget("hip", arch, wave), options={"num_warps": warps, "enable_fp_fusion": False})
            metadata = kernel.metadata
            if getattr(metadata, "global_scratch_size", 0) or getattr(metadata, "profile_scratch_size", 0):
                raise RuntimeError("export requires unsupported Triton scratch")
            file = name + ".hsaco"; blob = kernel.asm["hsaco"]; (stage / file).write_bytes(blob)
            item = dict(name=name, symbol=metadata.name, file=file, **geometry, exact_block_size=True,
                num_warps=warps, warp_size=wave, shared=metadata.shared, argument_count=len(export.runtime_arguments),
                global_scratch_size=0, profile_scratch_size=0, size_bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                library_export={"abi": export.abi, "name": export.name, "semantic": export.semantic,
                    "runtime_arguments": export.runtime_arguments, "signature": signature, "constants": constants})
            validate_hsaco(stage / file, arch, item); kernels.append(item)
        declared_assets = specification.get("assets", [])
        if not isinstance(declared_assets, list) or len(declared_assets) > 16:
            raise RuntimeError("invalid asset list")
        asset_root = args.asset_dir.absolute() if args.asset_dir else profile_path.parent
        asset_names = set()
        for asset in declared_assets:
            if not isinstance(asset, dict):
                raise RuntimeError("invalid constant asset")
            name = plain_name(asset.get("name")); file = plain_name(asset.get("file")); source_file = plain_name(asset.get("source_file", file))
            if name in asset_names or (stage / file).exists() or file == "manifest.json":
                raise RuntimeError("duplicate or reserved asset")
            asset_names.add(name); size, digest = file_integrity(asset_root / source_file)
            if (size > MAX_CONSTANT_ASSET_BYTES or type(asset.get("binary_size")) is not int or
                    size != asset["binary_size"] or digest != asset.get("sha256")):
                raise RuntimeError("constant asset integrity mismatch")
            copy_validated_artifact(asset_root / source_file, stage / file, size, digest)
            assets.append(dict(name=name, file=file, binary_size=size, sha256=digest))
        if source_proof(root, exports) != sources or git_state(root) != (revision, status):
            raise RuntimeError("source changed during compilation")
        manifest = {"format": 2, "arch": arch, "kernels": kernels, "assets": assets,
            "package": {"abi": "flagos-amd-aot-v2", "backend": "hip", "binary_format": "hsaco",
                "compiler": {"name": "Triton", "version": triton.__version__}, "source_revision": revision,
                "source_tree_state": "dirty" if status else "clean", "source_files": sources,
                "source_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
                "profile_sha256": hashlib.sha256(specification_bytes).hexdigest(),
                "provenance": "explicit-FlagGems-AOT-exports", "GPU_execution": False}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_manifest(stage)
        if output.exists() or output.is_symlink():
            raise RuntimeError("output appeared during compilation")
        os.rename(stage, output)
    print(json.dumps({"output": str(output), "profiles": len(kernels), "assets": len(assets), "GPU_execution": False}))


if __name__ == "__main__":
    main()
