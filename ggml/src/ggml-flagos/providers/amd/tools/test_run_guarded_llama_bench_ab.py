#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run_guarded_llama_bench_ab", HERE / "run_guarded_llama_bench_ab.py")
AB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AB)


def _row(model: str, prompt: int, generation: int, rate: float) -> dict:
    return {
        "model_filename": model,
        "model_type": "test model",
        "model_size": 1024,
        "model_n_params": 100,
        "n_batch": 512,
        "n_ubatch": 512,
        "n_threads": 12,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": 99,
        "n_cpu_moe": 0,
        "split_mode": "layer",
        "no_kv_offload": False,
        "flash_attn": -1,
        "embeddings": False,
        "no_op_offload": 0,
        "n_prompt": prompt,
        "n_gen": generation,
        "n_depth": 0,
        "avg_ts": rate,
    }


class GuardedLlamaBenchABTest(unittest.TestCase):
    def test_explicit_device_selection(self):
        for device in ("", "none", "ROCm0,ROCm1"):
            with self.assertRaises(ValueError):
                AB.validate_device_selection(device, "candidate")
        AB.validate_device_selection("ROCm0", "baseline")
        AB.validate_device_selection("FlagOS:AMD:0", "candidate")

    def test_executed_device_rejects_cpu_fallback_and_different_device(self):
        for row in ({}, {"devices": ""}, {"devices": "ROCm01"}):
            with self.assertRaises(ValueError):
                AB.validate_executed_device([row], "ROCm0", "baseline")
        AB.validate_executed_device([{"devices": "ROCm0"}], "ROCm0", "baseline")
        with self.assertRaises(ValueError):
            AB.validate_executed_device([
                {"devices": "FlagOS:AMD:0"}, {"devices": ""}], "FlagOS:AMD:0", "candidate")

    def test_run_preserves_explicit_guard_limit(self):
        for limit in (2, 4):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with mock.patch.object(AB.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                    self.assertEqual(AB.run_one(Path("guard.py"), 180, ["llama-bench"], {},
                        root / "out", root / "err", limit), 0)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--max-processes-per-boot") + 1], str(limit))
                self.assertEqual(command[command.index("--") + 1:], ["llama-bench"])

    def test_json_metadata_replace_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            AB.write_json(path, {"status": "ready"})
            AB.write_json(path, {"status": "complete", "return_codes": {"candidate": 0}})
            self.assertEqual(
                json.loads(path.read_text()),
                {"status": "complete", "return_codes": {"candidate": 0}},
            )
            self.assertFalse(path.with_name("run.json.tmp").exists())

    def test_environment_assignments(self):
        self.assertEqual(
            AB.parse_environment(["FLAG=1", "PATH_VALUE=/tmp/a=b"]),
            {"FLAG": "1", "PATH_VALUE": "/tmp/a=b"},
        )
        with self.assertRaisesRegex(ValueError, "invalid environment"):
            AB.parse_environment(["bad-name=1"])

    def test_compare_pairs_independent_of_row_order(self):
        candidate = [_row("a.gguf", 0, 128, 90), _row("a.gguf", 512, 0, 110)]
        baseline = [_row("a.gguf", 512, 0, 100), _row("a.gguf", 0, 128, 100)]
        result = AB.compare_rows(candidate, baseline, "candidate", "baseline")
        self.assertEqual([row["test"] for row in result], ["PP512", "TG128"])
        self.assertAlmostEqual(result[0]["relative_percent"], 10.0)
        self.assertAlmostEqual(result[1]["relative_percent"], -10.0)

    def test_compare_rejects_unpaired_rows(self):
        with self.assertRaisesRegex(ValueError, "do not pair"):
            AB.compare_rows(
                [_row("a.gguf", 512, 0, 100)],
                [_row("a.gguf", 0, 128, 100)],
                "candidate",
                "baseline",
            )

    def test_expected_rows_require_both_tests_for_each_model(self):
        rows = [
            _row("/models/a.gguf", 512, 0, 100),
            _row("/models/a.gguf", 0, 128, 90),
            _row("/models/b.gguf", 512, 0, 80),
            _row("/models/b.gguf", 0, 128, 70),
        ]
        AB.validate_expected_rows(
            rows, [Path("/models/a.gguf"), Path("/models/b.gguf")], 512, 128, "candidate")
        with self.assertRaisesRegex(ValueError, "every requested model"):
            AB.validate_expected_rows(
                rows[:-1], [Path("/models/a.gguf"), Path("/models/b.gguf")], 512, 128, "candidate")

    def test_benchmark_command_uses_separate_pp_and_tg(self):
        args = type("Args", (), {
            "model": [Path("a.gguf"), Path("b.gguf")],
            "prompt": 512,
            "generation": 128,
            "batch": 512,
            "ubatch": 512,
            "repetitions": 3,
            "gpu_layers": 99,
            "threads": 12,
        })()
        command = AB.benchmark_command(
            args, Path("llama-bench"), "ROCm0", "blk\\..*=FlagOS_AMD")
        self.assertIn("-p", command)
        self.assertIn("-n", command)
        self.assertNotIn("-pg", command)
        self.assertIn("a.gguf,b.gguf", command)
        self.assertEqual(command[-4:], ["-dev", "ROCm0", "-ot", "blk\\..*=FlagOS_AMD"])


if __name__ == "__main__":
    unittest.main()
