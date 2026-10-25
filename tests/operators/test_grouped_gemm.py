"""Cumulative expert boundaries, empty groups, padding and repeated retirement."""

import pytest

from ascendflow import compile
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse
from examples.moe.grouped_gemm import grouped_gemm
from tests.models.structured_events import _Interpret, check_structured_events


@pytest.fixture(
    scope="module",
    params=(
        ("linear", "normal", False),
        ("NZ", "auto", False),
        ("linear", "auto", True),
        ("NZ", "auto", True),
    ),
)
def compiled(request):
    layout, cache, unit_flag = request.param
    return compile(
        grouped_gemm.specialize(
            E=3,
            K=64,
            N=64,
            BM=16,
            BN=64,
            BK=64,
            CK=32 if unit_flag else 64,
            GROUPS=6,
            WEIGHT_LAYOUT=layout,
            WEIGHT_CACHE=cache,
            UNIT_FLAG=unit_flag,
        )
    )


@pytest.mark.parametrize(
    "ends,capacity",
    [
        ([0, 0, 0], 0),
        ([0, 0, 0], 7),
        ([1, 1, 19], 19),
        ([0, 1, 19], 19),
        ([0, 0, 19], 19),
        ([16, 16, 49], 56),
        ([17, 31, 65], 65),
        ([49, 49, 114], 121),
    ],
)
def test_direct_expert_ends_cover_output_once(compiled, monkeypatch, ends, capacity):
    original = _Interpret.native

    def metadata(self, operation, values, *args):
        original(self, operation, values, *args)
        if operation.native.data == "index_load":
            root, offset, _ = values[operation.refs[0]]
            assert root.sym_name.data == "GroupList"
            values[operation.value] = ends[offset // 8]

    monkeypatch.setattr(_Interpret, "native", metadata)
    memory = collect_native_memory(compiled.plan)
    before = collect_native_memory(parse(compiled.before_ir))
    checked = check_structured_events(
        memory,
        {"capacity": capacity},
        logical_reference=before,
        invocations=2,
        groups=range(compiled.report["abi"]["groups"]),
    )
    for invocation in (0, 1):
        writes = [
            span
            for command in checked.natives
            if command.occurrence.invocation == invocation
            for span in command.spans
            if span.root.sym_name.data == "Y"
        ]
        assert sum(span.end - span.start for span in writes) == capacity * 64 * 2
        assert all(0 <= span.start < span.end <= capacity * 64 * 2 for span in writes)


def test_changing_expert_ends_retires_each_workers_private_pipeline(compiled, monkeypatch):
    original = _Interpret.native
    ends_cases = ([1, 2, 4], [0, 0, 4], [0, 0, 0], [1, 1, 4])

    def metadata(self, operation, values, invocation, *args):
        original(self, operation, values, invocation, *args)
        if operation.native.data == "index_load":
            _, offset, _ = values[operation.refs[0]]
            values[operation.value] = ends_cases[invocation][offset // 8]

    monkeypatch.setattr(_Interpret, "native", metadata)
    memory = collect_native_memory(compiled.plan)
    before = collect_native_memory(parse(compiled.before_ir))
    # Stream ordering between distinct launches covers outputs whose group owner
    # changes. Check each worker's private event/storage reuse independently.
    for group in range(compiled.report["abi"]["groups"]):
        check_structured_events(
            memory, {"capacity": 4}, logical_reference=before, invocations=4, groups=(group,)
        )


def test_static_kn_tails_use_exact_gm_and_fix_geometry():
    result = compile(
        grouped_gemm.specialize(
            E=2,
            K=19,
            N=67,
            BM=16,
            BN=64,
            BK=32,
            CK=32,
            GROUPS=2,
        )
    )
    memory = collect_native_memory(result.plan)
    fixes = [t for t in memory.templates if t.spec.native == "fix_rows"]
    assert {t.spec.params[1] for t in fixes} == {3, 64}
    assert "group_list_type" not in result.asc


@pytest.mark.parametrize("columns", (64, 80))
@pytest.mark.parametrize("window", (2, 4))
def test_diagonal_windows_cover_full_and_partial_m_strips(monkeypatch, columns, window):
    result = compile(
        grouped_gemm.specialize(
            E=2,
            K=16,
            N=columns,
            BM=16,
            BN=16,
            BK=16,
            CK=16,
            GROUPS=4,
            WEIGHT_LAYOUT="NZ",
            SWIZZLE=window,
        )
    )
    original = _Interpret.native
    ends = [97, 225]  # Seven row tiles (a partial window), then eight full tiles.

    def metadata(self, operation, values, *args):
        original(self, operation, values, *args)
        if operation.native.data == "index_load":
            _, offset, _ = values[operation.refs[0]]
            values[operation.value] = ends[offset // 8]

    monkeypatch.setattr(_Interpret, "native", metadata)
    checked = check_structured_events(
        collect_native_memory(result.plan),
        {"capacity": 231},
        logical_reference=collect_native_memory(parse(result.before_ir)),
        invocations=2,
        groups=range(4),
    )
    for invocation in (0, 1):
        writes = [
            s
            for c in checked.natives
            if c.occurrence.invocation == invocation
            for s in c.spans
            if s.root.sym_name.data == "Y"
        ]
        assert sum(s.end - s.start for s in writes) == 231 * columns * 2
        assert all(0 <= s.start < s.end <= 231 * columns * 2 for s in writes)
