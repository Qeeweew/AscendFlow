"""Published-artifact reuse, shape safety and the independent mask oracle."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ascendflow.diagnostics import CompileError
from benchmarks.vector import run_masks


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_artifact(directory, shape=(8, 128), *, source=b"// test asc\n"):
    """Minimal published artifact: snapshot, ASC, library and manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, columns = shape
    (directory / "source.xdsl").write_text(
        "builtin.module {\n"
        '    %X = "npu_source.alloc"() {sym_name = "X", versions = #builtin.int<1>, '
        'access = "read", ring = #builtin.int<0>, dynamic_axes = []} : () -> '
        f"!npu_source.tensor<[#builtin.int<{rows}>, #builtin.int<{columns}>], "
        '"f32", "GM", "linear">\n'
        "}\n"
    )
    (directory / "masks.asc").write_bytes(source)
    (directory / "masks.so").write_bytes(b"\x7fELF-test-shared")
    (directory / "verification.json").write_text(
        json.dumps({"kernel": "masks", "kind": "compiled", "compilation": {}}) + "\n"
    )
    manifest = {
        "target": "dav-2201",
        "commands": [["cc", str(directory / "masks.asc"), "-o", str(directory / "masks.o")]],
        "source_sha256": _digest(source),
        "shared_sha256": _digest((directory / "masks.so").read_bytes()),
    }
    (directory / "masks.compile.json").write_text(json.dumps(manifest) + "\n")
    return directory


def test_shape_uses_published_dimensions_and_rejects_mismatch(tmp_path):
    artifact = _write_artifact(tmp_path / "artifact", (4, 256))
    assert run_masks.resolve_shape(artifact, None, None) == (4, 256)
    assert run_masks.resolve_shape(artifact, 4, 256) == (4, 256)
    with pytest.raises(CompileError, match="does not match published rows"):
        run_masks.resolve_shape(artifact, 8, 256)
    with pytest.raises(CompileError, match="does not match published columns"):
        run_masks.resolve_shape(artifact, 4, 128)
    # A mismatched CLI shape is rejected before any library load or health check.
    with pytest.raises(CompileError, match="does not match published rows"):
        run_masks.main(
            [
                "--output",
                str(tmp_path / "evidence"),
                "--artifact",
                str(artifact),
                "--library",
                str(artifact / "masks.so"),
                "--physical",
                "2",
                "--rows",
                "7",
            ]
        )


def test_artifact_requires_whole_64_lane_columns(tmp_path):
    artifact = _write_artifact(tmp_path / "artifact", (4, 96))
    with pytest.raises(CompileError, match="whole packed 64-lane blocks"):
        run_masks.resolve_shape(artifact, None, None)


def test_library_reuse_never_invokes_the_compiler(tmp_path, monkeypatch):
    artifact = _write_artifact(tmp_path / "artifact", (8, 128))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("published reuse must not invoke the compiler")

    monkeypatch.setattr(run_masks, "build_file", forbidden)
    evidence = tmp_path / "evidence"
    assert (
        run_masks.main(
            [
                "--output",
                str(evidence),
                "--artifact",
                str(artifact),
                "--library",
                str(artifact / "masks.so"),
            ]
        )
        == 0
    )
    assert calls == []
    summary = json.loads((evidence / "runner.json").read_text())
    assert summary["shape"] == [8, 128]
    assert summary["provenance"]["asc_sha256"] == _digest((artifact / "masks.asc").read_bytes())
    assert summary["provenance"]["shared_sha256"] == _digest((artifact / "masks.so").read_bytes())
    assert summary["provenance"]["example_path"] == "examples/vector/masks.py"
    assert summary["provenance"]["runner_path"] == "benchmarks/vector/run_masks.py"


def test_build_forwards_second_shape_parameters(tmp_path, monkeypatch):
    calls = []

    def fake_build(source, output, *, entry, parameters, compile_asc, shared, timeout_s):
        calls.append(
            {
                "source": Path(source),
                "output": Path(output),
                "entry": entry,
                "parameters": dict(parameters),
                "compile_asc": compile_asc,
                "shared": shared,
                "timeout_s": timeout_s,
            }
        )
        _write_artifact(Path(output), (parameters["ROWS"], parameters["COLUMNS"]))

    monkeypatch.setattr(run_masks, "build_file", fake_build)
    output = tmp_path / "build"
    assert (
        run_masks.main(
            [
                "--build",
                "--output",
                str(output),
                "--rows",
                "4",
                "--columns",
                "256",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["source"] == run_masks.SOURCE
    assert call["entry"] == "masks"
    assert call["parameters"] == {"ROWS": 4, "COLUMNS": 256}
    assert call["compile_asc"] is True and call["shared"] is True
    assert 0 < call["timeout_s"] <= 300
    assert json.loads((output / "runner.json").read_text())["shape"] == [4, 256]


def test_reuse_rejects_tampered_library(tmp_path):
    artifact = _write_artifact(tmp_path / "artifact")
    library = artifact / "masks.so"
    from benchmarks.artifacts import read_artifact

    result = read_artifact(artifact)
    library.write_bytes(library.read_bytes() + b"tampered")
    with pytest.raises(CompileError, match="does not match the published manifest"):
        run_masks.published_manifest(artifact, result, library)


def test_runtime_origin_extremes_are_exact_and_discriminating():
    x, z = run_masks.field_tensors(2, 128)
    empty_q0 = run_masks.oracle(x, z, 0, 0, 0)
    empty_q7 = run_masks.oracle(x, z, 7, 0, 0)
    empty_max = run_masks.oracle(x, z, run_masks.INT64_MAX, 0, 0)
    # rows=2, empty interval: chunk 0 is periodic. q=0 selects only cols > row;
    # q=7 (and INT64_MAX) satisfies cols+q > row on every chunk-0 lane.
    assert empty_q0[0][0, 0] == x[0, 0] + np.float32(1)
    assert empty_q0[0][0, 1] == np.float32(-3)
    assert empty_q0[0][1, 1] == x[1, 1] + np.float32(1)
    assert empty_q0[0][1, 2] == np.float32(-3)
    assert empty_q7[0][0, 0] == np.float32(-3)
    assert empty_q7[0][1, 0] == np.float32(-3)
    for actual, expected in zip(empty_q7[0], empty_max[0], strict=True):
        assert np.array_equal(actual, expected)
    # Nonnegative origins never change the data-predicate outputs.
    for left, right in zip(empty_q0[1:], empty_max[1:], strict=True):
        assert np.array_equal(left, right)


def test_empty_full_and_partial_intervals_preserve_visible_lanes():
    x, z = run_masks.field_tensors(1, 128)
    y0, _, _ = run_masks.oracle(x, z, 0, 32, 96)
    # q=0 origin is cols > 0, so col 0 exercises origin=false preservation.
    assert y0[0, 0] == x[0, 0] + np.float32(1)  # untouched outside interval
    assert y0[0, 10] == np.float32(-3)  # filled chunk-0 outside lane
    assert y0[0, 40] == x[0, 40] * np.float32(2) + np.float32(1)  # updated preserved
    assert y0[0, 80] == np.float32(-3)  # filled chunk-1 interval lane
    assert y0[0, 120] == x[0, 120] + np.float32(1)  # untouched chunk-1 lane
    empty, _, _ = run_masks.oracle(x, z, 0, 5, 5)
    assert empty[0, 64] == x[0, 64] + np.float32(1)
    full, _, _ = run_masks.oracle(x, z, 0, 0, 128)
    assert full[0, 0] == x[0, 0] * np.float32(2) + np.float32(1)
    assert full[0, 64] == np.float32(-3)


def test_data_compare_chunks_and_not_or_and_are_discriminating():
    for shape in ((8, 128), (4, 256)):
        x, z = run_masks.field_tensors(*shape)
        discrimination = run_masks.compare_discrimination(x, z)
        assert discrimination["chunks"] == shape[1] // 64
        assert discrimination["adjacent_chunks_distinct"] is True
        assert discrimination["p_true_lanes"] > 0
        assert discrimination["p_false_lanes"] > 0
        assert discrimination["q_true_lanes"] > 0
        assert discrimination["q_false_lanes"] > 0
        assert discrimination["p_true_q_false_lanes"] > 0
        assert discrimination["p_false_q_true_lanes"] > 0
        p = x > np.float32(0.25)
        data = x * np.float32(5)
        q_pred = data > z
        both = p & q_pred
        not_p = ~p
        either = both | not_p
        y1 = np.where(either, data, z)
        # Removing NOT/OR leaves false-p lanes on z; removing the AND leaves
        # true-p lanes on z. Both alternatives change observable output lanes.
        assert not np.array_equal(y1, np.where(both, data, z))
        assert not np.array_equal(y1, np.where(not_p, data, z))
        y2 = np.where(both, np.float32(-9), y1)
        assert np.any(y2 == np.float32(-9))
        assert np.any(y2 != np.float32(-9))
        # The materialized AND is separately observable in the masked fill:
        # using q_pred or p instead of p&q fills different lanes.
        assert not np.array_equal(y2, np.where(q_pred, np.float32(-9), y1))
        assert not np.array_equal(y2, np.where(p, np.float32(-9), y1))


def test_default_cases_cover_interval_classes_and_origin_values():
    x, z = run_masks.field_tensors(8, 128)
    coverages, origins = set(), set()
    for value in run_masks.DEFAULT_CASES:
        q, lo, hi = (int(part) for part in value.split(":"))
        mask = run_masks.coordinates(8, 128, q, lo, hi)
        coverage = run_masks.lane_coverage(mask, mask["fill"])
        coverages.add(coverage["interval"])
        origins.add(q)
        assert coverage["filled_lanes"] > 0
        interval = np.array(mask["interval"], dtype=bool)
        if interval.any():
            assert coverage["interval_true_preserved_lanes"] > 0
        if (~interval).any():
            assert coverage["interval_false_preserved_lanes"] > 0
    assert coverages == {"empty", "full", "partial"}
    assert {0, 7, run_masks.INT64_MAX} <= origins
    assert not np.array_equal(
        run_masks.oracle(x, z, 0, 32, 96)[0],
        run_masks.oracle(x, z, 7, 32, 96)[0],
    )
    assert run_masks._parse_cases([f"{run_masks.INT64_MAX}:0:0"])
    with pytest.raises(CompileError, match="nonnegative signed int64"):
        run_masks._parse_cases(["-1:0:0"])
    with pytest.raises(CompileError, match="nonnegative signed int64"):
        run_masks._parse_cases([f"{run_masks.INT64_MAX + 1}:0:0"])
