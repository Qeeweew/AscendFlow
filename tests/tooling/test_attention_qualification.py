import json

import pytest

from ascendflow.diagnostics import CompileError
from benchmarks.attention import run
from benchmarks.attention.compare import compare, qualification
from tests.support.attention_runner import mock_cann, mock_execution, publish


def test_clear_slowdown_fails_and_overlap_is_not_called_success():
    assert qualification([200, 201, 202], [100, 101, 102])["status"] == "failed"
    assert qualification([100, 110, 120], [101, 102, 103])["status"] == "unresolved"
    assert qualification([97, 98, 99], [100, 101, 102])["status"] == "passed"
    assert qualification([102, 102, 102], [100, 100, 100])["status"] == "failed"


def test_single_batch_cannot_qualify_an_operator():
    with pytest.raises(CompileError, match="three paired"):
        qualification([90], [100])


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Actual runner JSON with CPU ACL doubles and independent timing CSVs."""
    mock_cann(tmp_path, monkeypatch)
    mock_execution(monkeypatch, run.DEFAULT_POLICY)
    monkeypatch.setattr(run, "health_snapshot", lambda *args: {"physical_device": 2})
    monkeypatch.setattr(run, "hardware_snapshot", lambda *args: {"platform_sha256": "CPU fixture"})
    artifact = publish(tmp_path / "generated")
    manifest_path = artifact / "attention_target.compile.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["commands"] = [["bisheng", str(artifact / "attention_target.asc"), "-o", "unused.o"]]
    manifest_path.write_text(json.dumps(manifest))
    verification_path = artifact / "verification.json"
    verification = json.loads(verification_path.read_text())
    verification["cann"] = manifest
    verification_path.write_text(json.dumps(verification))
    directory = tmp_path / "profiles"
    for variant in ("generated", "aclnn"):
        record = run.main(
            [
                "--output",
                str(artifact),
                "--backend",
                variant,
                "--physical",
                "2",
                "--cases",
                "256:1024",
                "--warmup",
                "1",
                "--repeats",
                "2",
            ]
        )
        for trial in range(3):
            destination = directory / f"trial-{trial}" / variant
            (destination / "device").mkdir(parents=True)
            (destination / "device/device.json").write_text(json.dumps(record))
            source = destination / "op_summary.csv"
            duration = (10 if variant == "generated" else 20) + trial
            source.write_text(
                "Device_id,Op Name,Task Type,Task Start Time(us),Task Duration(us),Block Num\n"
                + "".join(f"2,{variant},MIX_AIC,{i * 100},{duration},20\n" for i in range(3))
            )
            summary = {
                "csv": str(source),
                "csv_sha256": run.sha256(source.read_bytes()),
                "warmup": 1,
                "repeats": 2,
                "durations_us": [duration, duration],
                "operators": {variant: {"task_type": "MIX_AIC", "blocks": [20]}},
            }
            (destination / "summary.json").write_text(json.dumps(summary))
    return directory


def test_compare_consumes_current_runner_records_and_installed_aclnn_identity(profiles):
    result = compare(profiles)
    assert result["status"] == "passed"
    assert result["generated_over_aclnn"] == pytest.approx(11 / 21)
    assert result["generated_batch_p50_us"] == [10, 11, 12]
    assert all(trial["source_sha256"] is None for trial in result["trials"]["aclnn"])


@pytest.mark.parametrize("change", ("incomplete", "multiple_cases", "input", "aclnn_library"))
def test_compare_rejects_unmatched_or_incomplete_measurements(profiles, change):
    path = profiles / "trial-1/aclnn/device/device.json"
    record = json.loads(path.read_text())
    if change == "incomplete":
        record["status"] = "failed"
    elif change == "multiple_cases":
        record["cases"] *= 2
    elif change == "input":
        record["cases"][0]["argument_sha256"]["q"] = "different input"
    else:
        from pathlib import Path

        Path(record["artifact"]["libraries"]["libnnopbase.so"]["path"]).write_bytes(b"changed")
    path.write_text(json.dumps(record))
    with pytest.raises(
        CompileError, match="completed numerical case|contracts differ|ACLNN libraries"
    ):
        compare(profiles)
