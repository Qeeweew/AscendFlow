"""Normal/deep comparisons must retain the entire checked program contract."""

from copy import deepcopy

import pytest

from tests.report_contract import semantic_report


@pytest.mark.parametrize(
    "field",
    (
        "native_templates",
        "native_accesses",
        "storage_residue_accesses",
        "physical_dependence_templates",
        "parent_memory_obligations",
        "channels",
        "allocated_ids",
        "maximum_static_queries",
        "required_checks",
        "verified",
    ),
)
def test_semantic_report_keeps_checked_program_counts_and_results(field):
    expected = {"children": [{field: 1}], "verification": "actual IR"}
    changed = deepcopy(expected)
    changed["children"][0][field] = 2
    assert semantic_report(expected) != semantic_report(changed)


def test_semantic_report_keeps_actual_endpoints_and_ids_with_distinct_analysis_origins():
    expected = {
        "verification": {"verified": True, "checks": [{"source": "a", "target": "b", "ids": [3]}]},
        "verification_origin": "constructed",
        "completion_statistics": {"fact_transfers": 100},
    }
    deep = deepcopy(expected)
    deep.update(verification_origin="actual IR", completion_statistics={"fact_transfers": 200})
    assert semantic_report(expected) == semantic_report(deep)
    assert expected["verification_origin"] != deep["verification_origin"]
    deep["verification"]["checks"][0]["ids"] = [4]
    assert semantic_report(expected) != semantic_report(deep)
