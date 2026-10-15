"""Timeouts stop real compiler subprocess groups and invalidate success records."""

import json
import subprocess
import sys
from time import monotonic

import pytest

from benchmarks.attention.suite import bounded_process
from xdsl_npu import toolchain


def test_operator_watchdog_stops_and_records_cpu_compilation(tmp_path):
    start = monotonic()
    with pytest.raises(TimeoutError, match="stopped"):
        bounded_process(
            [sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, timeout_s=0.1
        )
    assert monotonic() - start < 2
    record = json.loads((tmp_path / "build-result.json").read_text())
    assert record["status"] == "timeout" and record["returncode"] != 0


def test_cann_watchdog_kills_the_process_group_and_invalidates_manifest(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    processes = []

    def compiler(_command, **kwargs):
        process = real_popen(
            [
                sys.executable,
                "-c",
                "import time;print('compiler-started',flush=True);time.sleep(30)",
            ],
            **kwargs,
        )
        processes.append(process)
        return process

    monkeypatch.setattr(toolchain.subprocess, "Popen", compiler)
    manifest = tmp_path / "test.compile.json"
    manifest.write_text('{"status":"old success"}')
    with pytest.raises(TimeoutError, match="process group killed"):
        toolchain.offline_compile("source", tmp_path, "test", timeout_s=0.1)
    assert processes[0].poll() is not None
    assert not manifest.exists()
    record = json.loads((tmp_path / "test.timeout.json").read_text())
    assert record["status"] == "timeout" and record["source_sha256"]
