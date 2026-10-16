"""Timeouts stop real compiler subprocess groups and invalidate success records."""

import json
import subprocess
import sys
from pathlib import Path
from time import monotonic

import pytest

from xdsl_npu.backend import cann
from xdsl_npu.build import build_file, build_module
from xdsl_npu.runtime.process import bounded_process


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

    monkeypatch.setattr(cann.subprocess, "Popen", compiler)
    manifest = tmp_path / "test.compile.json"
    manifest.write_text('{"status":"old success"}')
    with pytest.raises(TimeoutError, match="process group killed"):
        cann.offline_compile("source", tmp_path, "test", timeout_s=0.1)
    assert processes[0].poll() is not None
    assert not manifest.exists()
    record = json.loads((tmp_path / "test.timeout.json").read_text())
    assert record["status"] == "timeout" and record["source_sha256"]


def fake_cann(tmp_path, monkeypatch, *, pause=0, background=False, exit_code=0):
    """A process fixture, never an invocation of CANN or a native compiler."""
    root = tmp_path / "fake-cann"
    binary = root / "tools/bisheng_compiler/bin"
    binary.mkdir(parents=True)
    script = f"#!{sys.executable}\nimport pathlib,subprocess,sys,time\ntime.sleep({pause!r})\n"
    if background:
        children = tmp_path / "children"
        children.mkdir()
        child = (
            "import os,pathlib,time;time.sleep(0.3);"
            f"pathlib.Path({str(children)!r},'survived-'+str(os.getpid())).write_text('alive')"
        )
        script += (
            f"child=subprocess.Popen([sys.executable,'-c',{child!r}],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            f"pathlib.Path({str(children)!r},'started-'+str(child.pid)).write_text('started')\n"
        )
    if exit_code:
        script += f"raise SystemExit({exit_code})\n"
    script += (
        "if '--version' in sys.argv:\n"
        " print('fake compiler version')\n"
        "else:\n"
        " pathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'fake artifact')\n"
    )
    for name in ("bisheng", "ld.lld"):
        executable = binary / name
        executable.write_text(script)
        executable.chmod(0o755)
    monkeypatch.setattr(cann, "CANN", root)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_standalone_cann_reaps_background_children_after_any_exit(tmp_path, monkeypatch, exit_code):
    import time

    fake_cann(tmp_path, monkeypatch, background=True, exit_code=exit_code)
    output = tmp_path / "output"
    if exit_code:
        with pytest.raises(RuntimeError, match="CANN failed"):
            cann.offline_compile("reference", output, "reference", shared=True, timeout_s=3)
        assert not list(output.glob("*.compile.json"))
    else:
        cann.offline_compile("reference", output, "reference", shared=True, timeout_s=3)
    # DEVNULL keeps the background process out of communicate()'s pipe lifetime.
    # Its delayed side effect detects escape without looking for unrelated PIDs.
    assert len(list((tmp_path / "children").glob("started-*"))) == (1 if exit_code else 4)
    time.sleep(0.4)
    assert not list((tmp_path / "children").glob("survived-*"))


def test_input_construction_and_all_cann_commands_share_one_deadline(tmp_path, monkeypatch):
    import time

    fake_cann(tmp_path, monkeypatch, pause=0.3)
    output = tmp_path / "output"

    def construct():
        time.sleep(0.35)
        return "handwritten reference"

    started = monotonic()
    with pytest.raises(TimeoutError, match="stopped"):
        build_module(construct, output, name="reference", shared=True, timeout_s=1)
    assert monotonic() - started < 2
    assert json.loads((output / "build-result.json").read_text())["status"] == "timeout"
    assert not list(output.glob("*.compile.json"))
    assert not list(output.glob("*.so"))
    assert not list(output.glob("*.aibin"))
    assert not list(output.glob(".building-*"))


def test_cann_phases_do_not_restart_the_standalone_deadline(tmp_path, monkeypatch):
    fake_cann(tmp_path, monkeypatch, pause=0.15)
    output = tmp_path / "output"
    with pytest.raises(TimeoutError, match="shared deadline"):
        cann.offline_compile("reference", output, "reference", shared=True, timeout_s=0.4)
    assert not list(output.glob("*.compile.json"))
    assert not list(output.glob("*.aibin"))
    assert not list(output.glob("*.so"))


def test_published_manifest_names_final_sources_and_has_matching_hashes(tmp_path, monkeypatch):
    from benchmarks.artifacts import verified_source

    fake_cann(tmp_path, monkeypatch)
    output = tmp_path / "output"
    report = build_module(
        lambda: "handwritten reference", output, name="reference", shared=True, timeout_s=5
    )
    manifest = json.loads((output / "reference.compile.json").read_text())
    assert report["kind"] == "asc_reference"
    assert report["cann"] == manifest
    assert verified_source(manifest) == output / "reference.asc"
    assert all(
        ".building-" not in argument for command in manifest["commands"] for argument in command
    )
    assert len(manifest["command_seconds"]) == 3
    assert json.loads((output / "build-result.json").read_text())["status"] == "passed"


def test_hard_timeout_reaps_builder_children_and_invalidates_old_success(tmp_path):
    import time

    output = tmp_path / "output"
    output.mkdir()
    for name in ("old.compile.json", "old.so", "old.aibin", "verification.json"):
        (output / name).write_text("old success")
    child_pid = tmp_path / "child.pid"

    def construct():
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
        child_pid.write_text(str(child.pid))
        time.sleep(30)

    with pytest.raises(TimeoutError, match="stopped"):
        build_module(construct, output, timeout_s=0.2)
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"
    assert not list(output.glob("*.compile.json"))
    assert not list(output.glob("*.so"))
    assert not (output / "verification.json").exists()


def test_file_input_is_loaded_inside_the_deadline_and_rejects_builder_entries(tmp_path):
    source = tmp_path / "definition.py"
    source.write_text("import time\ntime.sleep(30)\n")
    with pytest.raises(TimeoutError, match="stopped"):
        build_file(source, tmp_path / "slow", entry="unused", timeout_s=0.1)
    source.write_text("def old_builder():\n    return None\n")
    with pytest.raises(RuntimeError, match="captured AST prim_func"):
        build_file(source, tmp_path / "builder", entry="old_builder", timeout_s=2)


@pytest.mark.parametrize("kind", ["module", "file", "command"])
def test_caught_nested_watchdog_rejection_never_creates_a_subtree(tmp_path, kind):
    import time

    called, survived, caught = (tmp_path / name for name in ("called", "survived", "caught"))
    child = (
        f"import pathlib,time;time.sleep(0.1);pathlib.Path({str(survived)!r}).write_text('alive')"
    )
    source = tmp_path / "nested.py"
    source.write_text(
        "import pathlib,subprocess,sys,time\n"
        f"pathlib.Path({str(called)!r}).write_text('called')\n"
        f"subprocess.Popen([sys.executable,'-c',{child!r}],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "time.sleep(30)\n"
    )

    def inner():
        called.write_text("called")
        subprocess.Popen(
            [sys.executable, "-c", child], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(30)

    def construct():
        try:
            if kind == "module":
                build_module(inner, tmp_path / "inner", timeout_s=0.05)
            elif kind == "file":
                build_file(source, tmp_path / "inner", entry="unused", timeout_s=0.05)
            else:
                bounded_process([sys.executable, str(source)], tmp_path / "inner", timeout_s=0.05)
        except (ValueError, TimeoutError) as error:
            caught.write_text(str(error))
        time.sleep(0.2)
        return "outer reference"

    build_module(construct, tmp_path / "outer", name="outer", timeout_s=3)
    assert "nested build/watchdog is unsupported" in caught.read_text()
    assert not called.exists() and not survived.exists()
    assert not (tmp_path / "inner").exists()


def test_shared_command_group_rejects_an_independent_shorter_build_deadline(tmp_path):
    called, caught = (tmp_path / name for name in ("called", "caught"))
    script = tmp_path / "command.py"
    script.write_text(
        "from pathlib import Path\n"
        "from xdsl_npu.build import build_module\n"
        "def construct():\n"
        f" Path({str(called)!r}).write_text('called')\n"
        " return 'reference'\n"
        "try:\n"
        f" build_module(construct,{str(tmp_path / 'inner')!r},name='inner',timeout_s=0.05)\n"
        "except ValueError as error:\n"
        f" Path({str(caught)!r}).write_text(str(error))\n"
        "else:\n"
        " raise AssertionError('short shared-group deadline was accepted')\n"
    )
    bounded_process([sys.executable, str(script)], tmp_path / "command", timeout_s=3)
    assert "must inherit the outer deadline" in caught.read_text()
    assert not called.exists() and not (tmp_path / "inner").exists()


def test_shared_group_cann_cannot_start_an_independent_shorter_timeout(tmp_path, monkeypatch):
    def no_spawn(*args, **kwargs):
        raise AssertionError("a rejected shared-group deadline must not launch a compiler")

    monkeypatch.setenv("XDSL_NPU_BUILD_GROUP", "1")
    monkeypatch.setenv("XDSL_NPU_BUILD_DEADLINE", str(monotonic() + 3))
    monkeypatch.setattr(cann.subprocess, "Popen", no_spawn)
    with pytest.raises(ValueError, match="must inherit the outer deadline"):
        cann.offline_compile("reference", tmp_path / "inner", "inner", timeout_s=0.05)
    assert not (tmp_path / "inner").exists()


def test_command_watchdog_allows_one_cli_build_with_its_inherited_deadline(tmp_path):
    source = Path(__file__).parents[1] / "examples/vector/scale.py"
    output = tmp_path / "build"
    record = bounded_process(
        [
            sys.executable,
            "-m",
            "xdsl_npu.cli",
            "--input",
            str(source),
            "--entry",
            "scale",
            "--output",
            str(output),
        ],
        tmp_path / "command",
        timeout_s=3,
    )
    assert record["status"] == "passed"
    assert (output / "scale.asc").exists()
    assert json.loads((output / "build-result.json").read_text())["status"] == "passed"


def test_command_watchdog_cannot_leave_a_nested_cli_worker_or_its_child(tmp_path):
    output = tmp_path / "output"
    child_pid = tmp_path / "child.pid"
    source = tmp_path / "definition.py"
    source.write_text(
        "import pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(TimeoutError, match="stopped"):
        bounded_process(
            [
                sys.executable,
                "-m",
                "xdsl_npu.cli",
                "--input",
                str(source),
                "--entry",
                "unused",
                "--output",
                str(output),
            ],
            output,
            timeout_s=0.7,
        )
    status = Path(f"/proc/{int(child_pid.read_text())}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"
    assert not list(output.glob(".building-*"))


def test_file_ast_build_specializes_then_publishes_a_checked_plan(tmp_path):
    from benchmarks.artifacts import read_artifact
    from xdsl_npu.diagnostics import CompileError
    from xdsl_npu.dialects.physical import LoopChannelOp, LoopSyncOp
    from xdsl_npu.sync import verify_plan

    source = Path(__file__).parents[1] / "examples/vector/scale.py"
    report = build_file(source, tmp_path, entry="scale", parameters={"B": 32}, timeout_s=5)
    assert report["native_templates"] == 3
    assert "npu_size_tiles" in (tmp_path / "scale.asc").read_text()
    assert (tmp_path / "source.xdsl").exists()
    assert (tmp_path / "after.xdsl").exists()
    assert json.loads((tmp_path / "build-result.json").read_text())["status"] == "passed"
    artifact = read_artifact(tmp_path)
    first, second = artifact.plan, artifact.plan
    assert first is not second
    channel = next(op for op in first.walk() if isinstance(op, LoopChannelOp))
    for op in list(first.walk()):
        if isinstance(op, LoopSyncOp) and op.channel == channel.sym_name:
            op.parent.erase_op(op)
    channel.parent.erase_op(channel)
    with pytest.raises(CompileError, match="missing"):
        verify_plan(first)
    assert verify_plan(second)["channels"] == artifact.report["channels"]
    changed = artifact.report
    changed["channels"] = -1
    assert artifact.report["channels"] >= 0


def test_source_only_reference_artifact_does_not_import_stale_plan_ir(tmp_path):
    from benchmarks.artifacts import read_artifact
    from xdsl_npu.diagnostics import CompileError

    (tmp_path / "after.xdsl").write_text("stale compiler plan")
    report = build_module(lambda: "reference source", tmp_path, name="reference", timeout_s=2)
    artifact = read_artifact(tmp_path)
    assert report["kind"] == "asc_reference" and artifact.asc == "reference source"
    with pytest.raises(CompileError, match="no compiler PlanIR"):
        _ = artifact.plan
    assert not list(tmp_path.glob("*.compile.json"))


@pytest.mark.parametrize("timeout", [0, -1, 301, float("nan"), float("inf"), True])
def test_total_deadline_cannot_exceed_300_seconds(tmp_path, timeout):
    with pytest.raises(ValueError, match="in \\(0,300\\]"):
        build_module(lambda: None, tmp_path, timeout_s=timeout)
