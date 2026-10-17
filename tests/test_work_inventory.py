"""Small IR reporting checks; no synchronization, CANN, or device execution."""

import importlib
import json
import sys

import pytest

from ascendflow import language as N
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import dump
from ascendflow.passes.lower_native import lower_native_structure
from benchmarks.work import read_work, structure_inventory, work_record


def inventory(count):
    source = N.from_source(f"""
def work(tiles:N.int64):
    with N.Kernel(3,kind="vector"):
        x=N.alloc_ub((64,),name="x")
        for i in N.Pipelined({count},stages=1):
            if tiles>1:
                N.fill(x,1.0)
            else:
                N.fill(x,2.0)
""")
    module = lower_native_structure(source).module
    before = dump(module)
    result = structure_inventory(module)
    assert dump(module) == before
    return result


def test_inventory_counts_both_branch_templates_without_loop_or_grid_weighting():
    empty, single, large = (inventory(count) for count in (0, 1, 1_000_000))
    assert empty == single == large
    assert empty["kernel"] == "work"
    assert empty["groups"] == 3
    assert empty["runtime_parameters"] == ["tiles"]
    assert empty["native_templates_by_native"] == {"fill": 2}
    assert empty["native_templates"] == 2
    assert empty["allocation_declarations_by_bank"] == {"UB": 1}
    assert empty["notification_templates"] == 0
    assert empty["control_templates_by_operation"]["scf.if"] == 1


def test_zero_algorithm_work_does_not_turn_unavailable_runtime_requests_into_zero():
    work = work_record(
        inventory(0),
        algorithm_counts={"fp32_multiply_operations": 0},
        algorithm_scope="empty scalar scale problem",
    )
    assert read_work({"work": work}, required_counts=("fp32_multiply_operations",)) is work
    assert work["algorithm_work"]["counts"] == {"fp32_multiply_operations": 0}
    assert work["runtime_requests"]["status"] == "unavailable"
    assert "native_invocations" not in work["runtime_requests"]
    assert "gm_requested_bytes" not in work["runtime_requests"]
    assert work["structure_inventory"]["native_templates"] == 2


def test_analysis_reports_missing_counts_instead_of_inventing_runtime_work():
    work = work_record(
        inventory(1), algorithm_counts={}, algorithm_scope="unspecified problem size"
    )
    with pytest.raises(CompileError, match="missing required counts: fma_flops"):
        read_work({"work": work}, required_counts=("fma_flops",))
    with pytest.raises(CompileError, match="source-work-v1"):
        read_work({"requested_work": {"native_invocations": {"mad": 10}}})


def test_unknown_algorithm_count_requires_an_explicit_status_instead_of_none():
    with pytest.raises(CompileError, match="nonnegative integer counts"):
        work_record(
            inventory(1),
            algorithm_counts={"fma_flops": None},
            algorithm_scope="unknown count cannot stand for zero",
        )


@pytest.mark.parametrize(
    "module_name",
    ("benchmarks.gemm.analyze", "benchmarks.state.analyze", "benchmarks.normalization.analyze"),
)
def test_analyzers_diagnose_missing_work_before_using_profile_counts(
    module_name, tmp_path, monkeypatch
):
    trial = tmp_path / "trial-0" / "generated"
    trial.mkdir(parents=True)
    (trial / "summary.json").write_text("{}")
    (trial / "device.json").write_text(json.dumps({"cases": [{}]}))
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(sys, "argv", [module_name, str(tmp_path), "--output", str(output)])
    analyzer = importlib.import_module(module_name)
    with pytest.raises(CompileError, match="source-work-v1"):
        analyzer.main()
    assert not output.exists()
