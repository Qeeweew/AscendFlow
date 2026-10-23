"""Keep full ACLNN invocation costs and reject changed reference evidence."""

import json
import sys

import pytest

from ascendflow.diagnostics import CompileError
from benchmarks.kda import build_reference
from benchmarks.kda.profile_prefill import summarize
from benchmarks.references.kda_aclnn import file_digest, verify_reference


def test_prefill_counts_layout_and_auxiliary_kernels_inside_invocation(tmp_path):
    (tmp_path / "watchdog").mkdir()
    (tmp_path / "watchdog/build-result.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "command": [
                    "msprof",
                    "op",
                    "--aic-metrics=BasicInfo",
                    "--mstx=on",
                    "--mstx-include=kda",
                ],
            }
        )
    )
    (tmp_path / "device").mkdir()
    (tmp_path / "device/results.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "results": {"ascendc": {}},
                "health": {"physical_device": 6},
            }
        )
    )
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/OpBasicInfo.csv").write_text(
        "Device Id,Op Name,Task Duration(us),Block Dim,Current Freq\n"
        "6,chunk_kda_fwd_prepare,20,20,1800\n"
        "6,chunk_fwd_h,30,20,1800\n"
        "6,chunk_kda_fwd_finalize,40,20,1800\n"
        "6,TensorMove_123,5,40,1800\n"
        "6,Cast_456,7,40,1800\n"
    )
    result = summarize(tmp_path)
    assert result["kernel_us"] == 102
    assert len(result["kernels"]) == 5
    capture = tmp_path / "watchdog/build-result.json"
    data = json.loads(capture.read_text())
    data["command"] = data["command"][:-2]
    capture.write_text(json.dumps(data))
    with pytest.raises(CompileError, match="complete invocation"):
        summarize(tmp_path)


def test_reference_rejects_modified_runtime_or_manifest(tmp_path):
    library = tmp_path / "library.so"
    library.write_bytes(b"original reference artifact")
    manifest = tmp_path / "reference-build.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "passed",
                "soc": "ascend910b",
                "source_status": "",
                "checkout": str(tmp_path),
                "source_files_sha256": {},
                "runtime_files_sha256": {"library.so": file_digest(library)},
            }
        )
    )
    record = {"vendor": str(tmp_path), "build_sha256": file_digest(manifest)}
    verify_reference(record)
    library.write_bytes(b"changed artifact")
    with pytest.raises(CompileError, match="KDA reference changed"):
        verify_reference(record)
    manifest.write_text("{}")
    with pytest.raises(CompileError, match="build record changed"):
        verify_reference(record)


def test_reference_timeout_invalidates_a_late_publication(monkeypatch, tmp_path):
    output = tmp_path / "reference"
    manifest = output / "opp/vendors/fla_npu_transformer/reference-build.json"

    def timeout(command, directory, *, timeout_s):
        assert timeout_s == 300
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"status":"passed"}')
        raise TimeoutError("outer build deadline")

    monkeypatch.setattr(build_reference, "bounded_process", timeout)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_reference",
            "--checkout",
            str(tmp_path),
            "--dependencies",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(TimeoutError, match="outer build deadline"):
        build_reference.main()
    assert not manifest.exists()
