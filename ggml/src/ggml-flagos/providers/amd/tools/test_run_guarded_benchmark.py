#!/usr/bin/env python3

import importlib.util
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run_guarded_benchmark", HERE / "run_guarded_benchmark.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class GuardedBenchmarkTest(unittest.TestCase):
    def run_continuous_guard(self, state, errors, args):
        with mock.patch.object(GUARD, "boot_id", return_value="test-boot"), \
                mock.patch.object(GUARD, "state_path", return_value=state), \
                mock.patch.object(GUARD, "kernel_gpu_errors", return_value=errors), \
                mock.patch.object(sys, "argv", ["guard", *args]):
            return GUARD.main()

    def test_continuous_preserves_counters_and_historical_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            state.write_text(json.dumps({"boot_id": "test-boot", "commands_started": 13,
                "commands_succeeded": 10, "commands_failed": 3, "last_process_limit": 13}))
            self.assertEqual(self.run_continuous_guard(state, [],
                ["--continuous-health-guard", "--", sys.executable, "-c", "pass"]), 0)
            record = json.loads(state.read_text())
            self.assertEqual(record["commands_started"], 14)
            self.assertEqual(record["commands_succeeded"], 11)
            self.assertEqual(record["commands_failed"], 3)
            self.assertEqual(record["last_process_limit"], 13)
            self.assertEqual(record["last_count_limit_mode"], "continuous-health")
            self.assertFalse(record["launch_in_progress"])

    def test_bounded_mode_still_preserves_historical_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            initial = {"boot_id": "test-boot", "commands_started": 13, "last_process_limit": 13}
            state.write_text(json.dumps(initial))
            with mock.patch.object(GUARD.subprocess, "Popen") as popen:
                self.assertEqual(self.run_continuous_guard(state, [], ["--", "/bin/true"]), 4)
                popen.assert_not_called()
            self.assertEqual(json.loads(state.read_text()), initial)

    def test_continuous_check_reports_uncapped_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            state.write_text(json.dumps({"boot_id": "test-boot", "commands_started": 13,
                "last_process_limit": 13}))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), mock.patch.object(GUARD.subprocess, "Popen") as popen:
                self.assertEqual(self.run_continuous_guard(state, [],
                    ["--continuous-health-guard", "--check-only", "--json"]), 0)
                popen.assert_not_called()
            report = json.loads(stream.getvalue())
            self.assertFalse(report["count_limit_enforced"])
            self.assertIsNone(report["remaining"])
            self.assertEqual(report["commands_started"], 13)

    def test_continuous_fault_still_refuses_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            with mock.patch.object(GUARD.subprocess, "Popen") as popen:
                self.assertEqual(self.run_continuous_guard(state, ["MES ring buffer is full."],
                    ["--continuous-health-guard", "--", "/bin/true"]), 3)
                popen.assert_not_called()
            self.assertTrue(json.loads(state.read_text())["unhealthy"])

    def test_continuous_busy_lock_still_refuses_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            state.write_text("{}")
            with state.open("r+") as held, mock.patch.object(GUARD.subprocess, "Popen") as popen:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(self.run_continuous_guard(state, [],
                    ["--continuous-health-guard", "--", "/bin/true"]), 2)
                popen.assert_not_called()

    def test_continuous_latched_fault_cannot_be_cleared_by_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            state.write_text(json.dumps({"boot_id": "test-boot", "unhealthy": True,
                "commands_started": 13, "last_process_limit": 13}))
            with mock.patch.object(GUARD.subprocess, "Popen") as popen:
                self.assertEqual(self.run_continuous_guard(state, [],
                    ["--continuous-health-guard", "--", "/bin/true"]), 3)
                popen.assert_not_called()
            record = json.loads(state.read_text())
            self.assertTrue(record["unhealthy"])
            self.assertEqual(record["commands_started"], 13)

    def test_continuous_new_boot_starts_new_boot_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            state.write_text(json.dumps({"boot_id": "previous-boot", "unhealthy": True,
                "launch_in_progress": True, "commands_started": 13, "last_process_limit": 13}))
            self.assertEqual(self.run_continuous_guard(state, [],
                ["--continuous-health-guard", "--", sys.executable, "-c", "pass"]), 0)
            record = json.loads(state.read_text())
            self.assertEqual(record["boot_id"], "test-boot")
            self.assertEqual(record["commands_started"], 1)
            self.assertEqual(record["commands_succeeded"], 1)
            self.assertFalse(record["unhealthy"])
            self.assertFalse(record["launch_in_progress"])

    def test_continuous_unclean_previous_command_still_latches_fault(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            state.write_text(json.dumps({"boot_id": "test-boot", "launch_in_progress": True}))
            with mock.patch.object(GUARD.subprocess, "Popen") as popen:
                self.assertEqual(self.run_continuous_guard(state, [],
                    ["--continuous-health-guard", "--", "/bin/true"]), 3)
                popen.assert_not_called()
            self.assertTrue(json.loads(state.read_text())["unhealthy"])

    def test_continuous_and_bounded_limits_are_mutually_exclusive(self):
        with mock.patch.object(sys, "argv", ["guard", "--continuous-health-guard",
                "--max-processes-per-boot", "14", "--", "/bin/true"]):
            with self.assertRaises(SystemExit):
                GUARD.main()

    def test_successful_leader_cannot_leave_descendant(self):
        parent_code = """
import subprocess
import sys

subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
"""
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            argv = [
                "run_guarded_benchmark.py", "--timeout", "10",
                "--max-processes-per-boot", "2", "--",
                sys.executable, "-c", parent_code,
            ]
            with mock.patch.object(GUARD, "boot_id", return_value="test-boot"), \
                    mock.patch.object(GUARD, "state_path", return_value=state), \
                    mock.patch.object(GUARD, "kernel_gpu_errors", return_value=[]), \
                    mock.patch.object(sys, "argv", argv):
                self.assertEqual(GUARD.main(), 125)

            record = json.loads(state.read_text())
            self.assertFalse(record["launch_in_progress"])
            self.assertEqual(record["commands_started"], 1)
            self.assertEqual(record["commands_failed"], 1)
            self.assertEqual(record["last_return_code"], 125)

    def test_stop_process_group_kills_descendant_after_leader_exits(self):
        parent_code = """
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
subprocess.Popen([
    sys.executable, "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
print("ready", flush=True)
time.sleep(30)
"""
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(process.stdout.readline().strip(), "ready")
        # Give the descendant time to exec with SIGTERM ignored. The group
        # leader exits on SIGTERM, so checking only process.wait() would return
        # a false success and leave the descendant behind.
        time.sleep(0.05)
        self.assertTrue(GUARD.stop_process_group(process, grace_seconds=1))
        self.assertFalse(GUARD.process_group_exists(process.pid))
        process.stdout.close()

    def test_signal_during_popen_terminates_new_process_group(self):
        real_popen = subprocess.Popen
        child = None

        def interrupting_popen(command, start_new_session):
            nonlocal child
            child = real_popen(command, start_new_session=start_new_session)
            # Reproduce the former race: the child exists, but Popen has not
            # returned to main and therefore its local process variable has not
            # yet been assigned.
            os.kill(os.getpid(), signal.SIGHUP)
            return child

        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            argv = [
                "run_guarded_benchmark.py", "--timeout", "10",
                "--max-processes-per-boot", "2", "--", "/bin/sleep", "30",
            ]
            with mock.patch.object(GUARD, "boot_id", return_value="test-boot"), \
                    mock.patch.object(GUARD, "state_path", return_value=state), \
                    mock.patch.object(GUARD, "kernel_gpu_errors", return_value=[]), \
                    mock.patch.object(GUARD.subprocess, "Popen", side_effect=interrupting_popen), \
                    mock.patch.object(sys, "argv", argv):
                self.assertEqual(GUARD.main(), 128 + signal.SIGHUP)

            self.assertIsNotNone(child)
            self.assertIsNotNone(child.poll())
            record = json.loads(state.read_text())
            self.assertFalse(record["launch_in_progress"])
            self.assertEqual(record["commands_started"], 1)
            self.assertEqual(record["commands_failed"], 1)
            self.assertEqual(record["last_return_code"], 128 + signal.SIGHUP)

    def run_live_failure(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "guard.json"
            checks = [[], failure, []]
            def health():
                value = checks.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value
            argv = ["guard", "--timeout", "30", "--", sys.executable, "-c", "import time; time.sleep(30)"]
            before = time.monotonic()
            with mock.patch.object(GUARD, "boot_id", return_value="test-boot"), \
                    mock.patch.object(GUARD, "state_path", return_value=state), \
                    mock.patch.object(GUARD, "kernel_gpu_errors", side_effect=health), \
                    mock.patch.object(sys, "argv", argv):
                self.assertEqual(GUARD.main(), 125)
            self.assertLess(time.monotonic() - before, 10)
            record = json.loads(state.read_text())
            self.assertTrue(record["unhealthy"])
            self.assertFalse(record["launch_in_progress"])
            self.assertEqual(record["commands_failed"], 1)
            self.assertEqual(record.get("commands_succeeded", 0), 0)
            self.assertIn("last_live_health_error", record)

    def test_live_mes_fault_stops_command_before_wall_timeout(self):
        self.run_live_failure(["MES ring buffer is full."])

    def test_live_log_failure_still_blocks_boot_if_final_check_recovers(self):
        self.run_live_failure(GUARD.KernelLogUnavailable("temporary journal failure"))

    def test_health_poll_keeps_original_wall_timeout(self):
        with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True) as process:
            try:
                with mock.patch.object(GUARD, "kernel_gpu_errors", return_value=[]):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        GUARD.wait_with_health_checks(process, 0.1)
            finally:
                self.assertTrue(GUARD.stop_process_group(process, grace_seconds=1))


if __name__ == "__main__":
    unittest.main()
