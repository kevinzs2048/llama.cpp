#!/usr/bin/env python3

import json
import hashlib
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
MERGE = HERE / "merge_aot_packages.py"


def _string(value: str) -> bytes:
    encoded = value.encode()
    if len(encoded) < 32:
        return bytes([0xA0 | len(encoded)]) + encoded
    return b"\xD9" + bytes([len(encoded)]) + encoded


def _uint(value: int) -> bytes:
    if value < 128:
        return bytes([value])
    if value < 256:
        return b"\xCC" + bytes([value])
    return b"\xCD" + value.to_bytes(2, "big")


def _array(values) -> bytes:
    values = list(values)
    prefix = bytes([0x90 | len(values)])
    return prefix + b"".join(values)


def _map(values) -> bytes:
    values = list(values)
    prefix = bytes([0x80 | len(values)])
    return prefix + b"".join(_string(key) + value for key, value in values)


def _argument(offset: int, size: int, pointer: bool) -> bytes:
    values = [
        (".offset", _uint(offset)),
        (".size", _uint(size)),
        (".value_kind", _string("global_buffer" if pointer else "by_value")),
    ]
    if pointer:
        values.insert(0, (".address_space", _string("global")))
    return _map(values)


def _hsaco(name: str) -> bytes:
    descriptor = _map([
        ("amdhsa.kernels", _array([_map([
            (".args", _array([
                _argument(0, 8, True), _argument(8, 4, False),
                _argument(16, 8, True), _argument(24, 8, True),
            ])),
            (".group_segment_fixed_size", _uint(0)),
            (".kernarg_segment_size", _uint(32)),
            (".max_flat_workgroup_size", _uint(64)),
            (".name", _string(name)),
            (".private_segment_fixed_size", _uint(0)),
            (".symbol", _string(f"{name}.kd")),
            (".wavefront_size", _uint(32)),
        ])])),
        ("amdhsa.target", _string("amdgcn-amd-amdhsa--gfx1150")),
        ("amdhsa.version", _array([_uint(1), _uint(2)])),
    ])
    note_name = b"AMDGPU\0"
    note = struct.pack("<III", len(note_name), len(descriptor), 0x20)
    note += note_name + b"\0" * ((-len(note_name)) & 3)
    note += descriptor + b"\0" * ((-len(descriptor)) & 3)
    note_offset = 0x100
    image = bytearray(note_offset + len(note))
    image[:16] = b"\x7fELF\x02\x01\x01\x40\x03" + b"\0" * 7
    struct.pack_into("<HHI", image, 16, 3, 224, 1)
    struct.pack_into("<Q", image, 32, 64)
    struct.pack_into("<HH", image, 52, 64, 56)
    struct.pack_into("<H", image, 56, 1)
    struct.pack_into("<IIQQQQQQ", image, 64, 4, 4, note_offset, 0, 0,
                     len(note), len(note), 4)
    image[note_offset:] = note
    return bytes(image)


def _write_package(root: Path, names: str | list[str], package: bool) -> None:
    root.mkdir()
    if isinstance(names, str):
        names = [names]
    kernels = []
    for name in names:
        artifact = root / f"{name}.hsaco"
        artifact.write_bytes(_hsaco(name))
        kernels.append({
            "name": name,
            "symbol": name,
            "file": artifact.name,
            "num_warps": 2,
            "warp_size": 32,
            "block_size": 8,
            "argument_count": 2,
        })
    manifest = {
        "format": 2,
        "arch": "gfx1150",
        "kernels": kernels,
    }
    if package:
        manifest["package"] = {
            "abi": "flagos-amd-aot-v2",
            "backend": "hip",
            "binary_format": "hsaco",
            "provenance": "generated",
        }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _write_asset(root: Path, name="table", filename="table.bin", data=b"reference-table"):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    (root / filename).write_bytes(data)
    manifest.setdefault("assets", []).append({
        "name": name, "file": filename, "binary_size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    })
    path.write_text(json.dumps(manifest))


class MergePackageTest(unittest.TestCase):
    def _merge(self, base: Path, extension: Path, output: Path, tuning="", add_missing=False, replace_existing=False):
        command = [sys.executable, str(MERGE), "--base-dir", str(base),
                   "--additional-dir", str(extension), "--output-dir", str(output)]
        if tuning:
            command += ["--tuning-profile", tuning]
        if add_missing:
            command += ["--add-missing"]
        if replace_existing:
            command += ["--replace-existing"]
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_legacy_base_plus_v2_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", False)
            _write_package(root / "extension", "extension", True)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["package"]["provenance"], "merged-package")
            self.assertEqual(manifest["package"]["source_scope"], "additional-package-only")
            self.assertEqual(manifest["components"][0]["kind"], "legacy-base")
            self.assertEqual(len(manifest["kernels"]), 2)

    def test_v2_base_plus_legacy_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", True)
            _write_package(root / "extension", "extension", False)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["package"]["provenance"], "merged-package")
            self.assertEqual(manifest["components"][0]["kind"], "legacy-extension")

    def test_legacy_plus_legacy_has_no_fabricated_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", False)
            _write_package(root / "extension", "extension", False)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertNotIn("package", manifest)

    def test_tuning_profile_rejects_legacy_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", False)
            _write_package(root / "extension", "extension", True)
            result = self._merge(
                root / "base", root / "extension", root / "out",
                "gfx1150-wave32-q4ffn-v1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires v2 metadata", result.stderr)

    def test_qwen35_profiles_are_known_and_require_exact_membership(self):
        for profile in ("gfx1150-qwen35-q4km-v2", "gfx1150-qwen35-q4km-v3"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_package(root / "base", "base", True)
                _write_package(root / "extension", "extension", True)
                result = self._merge(
                    root / "base", root / "extension", root / "out", profile)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"incomplete {profile} package", result.stderr)
                self.assertNotIn("unsupported tuning profile", result.stderr)

    def test_add_missing_preserves_base_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", ["shared", "base"], True)
            _write_package(root / "extension", ["shared", "extension"], True)
            extension_manifest_path = root / "extension" / "manifest.json"
            extension_manifest = json.loads(extension_manifest_path.read_text())
            extension_manifest["kernels"][0]["block_size"] = 16
            extension_manifest_path.write_text(json.dumps(extension_manifest))
            result = self._merge(
                root / "base", root / "extension", root / "out", add_missing=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(
                [kernel["name"] for kernel in manifest["kernels"]],
                ["shared", "base", "extension"],
            )
            self.assertEqual(manifest["kernels"][0]["block_size"], 8)

    def test_replace_existing_same_artifact_preserves_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", ["a", "keep"], package=True)
            _write_package(root / "extension", "a", package=True)
            _write_asset(root / "base")
            path = root / "extension/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["kernels"][0]["block_size"] = 16
            path.write_text(json.dumps(manifest))
            artifact = root / "extension/a.hsaco"
            artifact.write_bytes(artifact.read_bytes() + b"replacement")
            rejected = self._merge(root / "base", root / "extension", root / "rejected")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / "rejected").exists())
            result = self._merge(root / "base", root / "extension", root / "out", replace_existing=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            merged = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual([k["name"] for k in merged["kernels"]], ["a", "keep"])
            self.assertEqual(merged["kernels"][0]["block_size"], 16)
            self.assertEqual((root / "out/a.hsaco").read_bytes(), artifact.read_bytes())
            self.assertEqual((root / "out/keep.hsaco").read_bytes(), (root / "base/keep.hsaco").read_bytes())
            self.assertEqual((root / "out/table.bin").read_bytes(), b"reference-table")

    def test_replace_existing_rejects_constant_file_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "a", package=True)
            _write_package(root / "extension", "a", package=True)
            _write_asset(root / "base", filename="conflict.hsaco")
            path = root / "extension/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["kernels"][0]["file"] = "conflict.hsaco"
            path.write_text(json.dumps(manifest))
            (root / "extension/a.hsaco").rename(root / "extension/conflict.hsaco")
            result = self._merge(root / "base", root / "extension", root / "out", replace_existing=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("kernel and constant asset file collision", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_exact_block_contract_requires_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", True)
            _write_package(root / "extension", "extension", True)
            manifest_path = root / "extension" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["kernels"][0]["exact_block_size"] = 1
            manifest_path.write_text(json.dumps(manifest))
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exact_block_size", result.stderr)


class ConstantAssetMergeTest(unittest.TestCase):
    _merge = MergePackageTest._merge
    def test_constants_from_both_inputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_package(root / "base", "base", True)
            _write_package(root / "extension", "extension", True)
            _write_asset(root / "base", "first", "first.bin")
            _write_asset(root / "extension", "second", "second.bin")
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual([a["name"] for a in manifest["assets"]], ["first", "second"])
            for asset in manifest["assets"]:
                self.assertEqual((root / "out" / asset["file"]).read_bytes(), b"reference-table")

    def test_identical_constant_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("base", "extension"):
                _write_package(root / folder, folder, True)
                _write_asset(root / folder)
            result = self._merge(root / "base", root / "extension", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads((root / "out/manifest.json").read_text())["assets"]), 1)

    def test_conflicting_constants_do_not_publish(self):
        for collision in ("name", "file", "kernel"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_package(root / "base", "base", True)
                _write_package(root / "extension", "extension", True)
                _write_asset(root / "base")
                if collision == "name":
                    _write_asset(root / "extension", data=b"changed-table")
                elif collision == "file":
                    _write_asset(root / "extension", name="other")
                else:
                    _write_asset(root / "extension", filename="base.hsaco")
                result = self._merge(root / "base", root / "extension", root / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "out").exists())

    def test_invalid_constant_metadata_does_not_publish(self):
        mutations = [
            lambda a: a.update(binary_size=True), lambda a: a.update(binary_size=0),
            lambda a: a.update(binary_size=16*1024*1024+1),
            lambda a: a.update(sha256="a"*64), lambda a: a.update(file="../table.bin"),
            lambda a: a.update(name="a\0b"), lambda a: a.pop("sha256"),
        ]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_package(root / "base", "base", True)
                _write_package(root / "extension", "extension", True)
                _write_asset(root / "extension")
                path = root / "extension/manifest.json"
                manifest = json.loads(path.read_text())
                mutate(manifest["assets"][0])
                path.write_text(json.dumps(manifest))
                result = self._merge(root / "base", root / "extension", root / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
