"""Cold benchmark identity, timing boundaries, and real process-group cleanup."""

import json
import sys
from pathlib import Path
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from benchmarks import compile as benchmark


@pytest.fixture
def fake_trials(monkeypatch):
    launch = benchmark._run_trial
    commands = []

    def run(command, result_path, log_path, **options):
        commands.append(command)
        mode = command[command.index("--mode") + 1]
        script = (
            "import json,os,pathlib\n"
            f"result={{'status':'passed','pid':os.getpid(),'mode':{mode!r},"
            "'source_sha256':'source','package_tree_sha256':'package','asc_sha256':'asc',"
            "'source_to_asc_s':0.1,'peak_rss_kib':123}\n"
            f"pathlib.Path({str(result_path)!r}).write_text(json.dumps(result))\n"
        )
        if "--artifacts" in command:
            artifacts = command[command.index("--artifacts") + 1]
            script += (
                f"directory=pathlib.Path({artifacts!r});directory.mkdir()\n"
                "(directory/'kernel.asc').write_text('asc')\n"
            )
        return launch([sys.executable, "-c", script], result_path, log_path, **options)

    monkeypatch.setattr(benchmark, "_run_trial", run)
    return commands


def test_default_five_trials_are_fresh_processes_with_one_artifact(tmp_path, fake_trials):
    output = tmp_path / "cold"
    assert benchmark.main(["--input", "case.py", "--entry", "kernel", "--output", str(output)]) == 0
    report = json.loads((output / "benchmark.json").read_text())
    (case,) = report["cases"]
    assert report["measurement_kind"] == "cold_trials"
    assert report["completed_trials"] == case["completed_trials"] == 5
    assert len({trial["pid"] for trial in case["trials"]}) == 5
    assert sum("--artifacts" in command for command in fake_trials) == 1
    assert len(list(output.rglob("*.asc"))) == 1
    assert len(list(output.rglob("trial-*.json"))) == 5


def test_probe_modes_and_explicit_parameters_are_recorded(tmp_path, fake_trials):
    output = tmp_path / "probe"
    assert (
        benchmark.main(
            [
                "--input",
                "case.py",
                "--entry",
                "kernel",
                "--parameters",
                '{"B":64}',
                "--metadata",
                '{"policy":"fixed"}',
                "--mode",
                "both",
                "--trials",
                "1",
                "--timeout",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "benchmark.json").read_text())
    assert report["measurement_kind"] == "probe"
    assert report["parameters"] == {"B": 64}
    assert report["metadata"] == {"policy": "fixed"}
    assert [case["mode"] for case in report["cases"]] == ["normal", "deep"]
    assert report["normal_deep"][0]["same_asc"]
    assert len(fake_trials) == 2
    assert all(
        json.loads(command[command.index("--parameters") + 1]) == {"B": 64}
        for command in fake_trials
    )


@pytest.mark.parametrize("mode", ("normal", "deep"))
def test_worker_includes_input_construction_and_snapshot_work_before_artifact_io(
    tmp_path, monkeypatch, mode
):
    import xdsl_npu

    source = tmp_path / "kernel.py"
    source.write_text(
        "from xdsl_npu import language as N\n@N.prim_func\ndef kernel(B:N.constexpr=8):\n    pass\n"
    )
    calls = []

    def write(directory):
        sleep(0.01)
        directory.mkdir()
        (directory / "kernel.asc").write_text("asc")

    def compile_definition(definition, *, timeout_s, debug_verify):
        calls.append((dict(definition.specializations), timeout_s, debug_verify))
        sleep(0.01)
        return SimpleNamespace(
            source_ir="source",
            before_ir="before",
            plan_ir="plan",
            asc="asc",
            timings={"total_s": 0.01, "snapshot_s": 0.005},
            report={"native_templates": 1, "verification_mode": mode},
            write=write,
        )

    monkeypatch.setattr(xdsl_npu, "compile", compile_definition)
    monkeypatch.setattr(benchmark, "_package_sha", lambda _: "package")
    metrics = benchmark.worker(
        source,
        entry="kernel",
        parameters={"B": 16},
        mode=mode,
        metadata={"case": "tiny"},
        deadline=monotonic() + 5,
        artifacts=tmp_path / "artifact",
    )
    assert calls[0][0] == {"B": 16} and calls[0][2] == (mode == "deep")
    assert 0 < calls[0][1] <= 5
    assert metrics["source_to_asc_s"] >= 0.01
    assert metrics["artifact_write_s"] >= 0.01
    assert metrics["worker_s"] >= metrics["source_to_asc_s"] + metrics["artifact_write_s"]
    assert metrics["source_sha256"] == benchmark._sha(source.read_bytes())
    assert metrics["verification"]["native_templates"] == 1
    assert (tmp_path / "artifact" / "input_source.py").read_bytes() == source.read_bytes()


def test_failure_is_persisted_and_stops_later_inputs_modes_and_trials(tmp_path, monkeypatch):
    launched = []
    launch = benchmark._run_trial

    def fail(command, result_path, log_path, **options):
        launched.append(command)
        return launch(
            [sys.executable, "-c", "raise SystemExit(7)"], result_path, log_path, **options
        )

    monkeypatch.setattr(benchmark, "_run_trial", fail)
    assert (
        benchmark.main(
            [
                "--input",
                "first.py",
                "later.py",
                "--entry",
                "kernel",
                "--mode",
                "both",
                "--output",
                str(tmp_path),
            ]
        )
        == 1
    )
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert report["status"] == "failed" and report["stopped"]
    assert report["completed_trials"] == 0 and len(launched) == 1
    assert len(report["cases"]) == 1
    (result,) = list(tmp_path.rglob("trial-*.json"))
    assert json.loads(result.read_text())["returncode"] == 7


def _running(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("exit_code", (None, 0, 7))
def test_worker_descendants_are_stopped_after_timeout_success_or_failure(tmp_path, exit_code):
    child_pid = tmp_path / "child.pid"
    result_path = tmp_path / "trial.json"
    script = (
        "import json,pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
    )
    if exit_code is None:
        script += "time.sleep(30)\n"
    else:
        script += (
            f"pathlib.Path({str(result_path)!r}).write_text(json.dumps("
            "{'status':'passed','source_sha256':'s','package_tree_sha256':'p',"
            "'asc_sha256':'a','source_to_asc_s':0.1,'peak_rss_kib':1}))\n"
            f"raise SystemExit({exit_code})\n"
        )
    start = monotonic()
    result = benchmark._run_trial(
        [sys.executable, "-c", script], result_path, tmp_path / "trial.log", timeout_s=0.3
    )
    assert monotonic() - start < 2
    assert result["status"] == (
        "timeout" if exit_code is None else "passed" if exit_code == 0 else "failed"
    )
    pid = int(child_pid.read_text())
    end = monotonic() + 1
    while _running(pid) and monotonic() < end:
        sleep(0.005)
    assert not _running(pid)


@pytest.mark.parametrize("timeout", ("0", "-1", "301", "nan", "inf"))
def test_invalid_total_deadlines_are_rejected_before_spawning(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--input", "case.py", "--output", str(tmp_path), "--timeout", timeout])
    assert error.value.code == 2


def test_identity_change_stops_remaining_fresh_trials(tmp_path, monkeypatch, fake_trials):
    launch = benchmark._run_trial
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        result = launch(*args, **kwargs)
        calls += 1
        if calls == 2:
            result["asc_sha256"] = "changed"
        return result

    monkeypatch.setattr(benchmark, "_run_trial", changed)
    assert (
        benchmark.main(["--input", "case.py", "--entry", "kernel", "--output", str(tmp_path)]) == 1
    )
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert calls == 2 and report["status"] == "failed"
    assert report["cases"][0]["completed_trials"] == 1


def test_normal_deep_difference_stops_at_first_deep_trial(tmp_path, monkeypatch, fake_trials):
    launch = benchmark._run_trial

    def changed(command, *args, **kwargs):
        result = launch(command, *args, **kwargs)
        if command[command.index("--mode") + 1] == "deep":
            result["asc_sha256"] = "changed"
        return result

    monkeypatch.setattr(benchmark, "_run_trial", changed)
    assert (
        benchmark.main(
            [
                "--input",
                "first.py",
                "later.py",
                "--entry",
                "kernel",
                "--mode",
                "both",
                "--output",
                str(tmp_path),
            ]
        )
        == 1
    )
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert len(fake_trials) == 6 and len(report["cases"]) == 2
    assert report["status"] == "failed" and report["stopped"]
    assert not report["normal_deep"][0]["same_asc"]
    assert report["cases"][1]["completed_trials"] == 0


def test_worker_timeout_status_survives_early_process_exit(tmp_path):
    result_path = tmp_path / "trial.json"
    script = (
        "import json,pathlib\n"
        f"pathlib.Path({str(result_path)!r}).write_text(json.dumps("
        "{'status':'timeout','error':'compile deadline exceeded'}))\n"
        "raise SystemExit(1)\n"
    )
    result = benchmark._run_trial(
        [sys.executable, "-c", script], result_path, tmp_path / "trial.log", timeout_s=2
    )
    assert result["status"] == "timeout"
    assert result["error"] == "compile deadline exceeded"
