"""Keep the runnable example inventory complete and stop a failed qualification."""

import ast
import json
import sys

from benchmarks import check_examples


def test_smoke_inventory_covers_every_public_example_entry():
    declared = set()
    for path in (check_examples.ROOT / "examples").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "prim_func"
                for decorator in node.decorator_list
            ):
                declared.add((str(path.relative_to(check_examples.ROOT)), node.name))
    cases = check_examples.examples()
    assert len({case.name for case in cases}) == len(cases)
    assert {(case.source, case.entry) for case in cases} == declared


def test_failed_build_stops_suite_and_preserves_failure_record(monkeypatch, tmp_path):
    calls = []

    def fail(command, directory, **options):
        calls.append(command)
        raise TimeoutError("test build reached its hard deadline")

    monkeypatch.setattr(check_examples, "bounded_process", fail)
    monkeypatch.setattr(sys, "argv", ["check_examples", "--output", str(tmp_path / "suite")])
    assert check_examples.main() == 1
    assert len(calls) == 1
    report = json.loads((tmp_path / "suite/results.json").read_text())
    assert report["status"] == "failed"
    assert report["cases"][0]["name"] == "scale"
    assert report["cases"][0]["status"] == "failed"
    assert "hard deadline" in report["error"]
