"""Physical carry/snapshot and event contracts, not a public GDN DSL API."""

import pytest
from xdsl.transforms.dead_code_elimination import dce

from benchmarks.validation.recurrent import gdn_reference, recurrent_inputs
from tests.fixtures.physical_state import recurrent_gdn
from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu._physical_builder import Kernel
from xdsl_npu.check import bounded_check
from xdsl_npu.ir import SyncOp, TaskOp, dump, parse
from xdsl_npu.model import CompileError, Instance, dependencies, extract
from xdsl_npu.sync import completion_graph, execution


@pytest.mark.parametrize(
    "steps,slots,snapshot_slots", [(0, 2, 3), (1, 1, 1), (1, 2, 3), (3, 1, 1), (3, 2, 1), (5, 2, 3)]
)
def test_static_state_streams_use_the_shared_physical_plan(steps, slots, snapshot_slots):
    source = recurrent_gdn(2, steps, value_dim=8, slots=slots, snapshot_slots=snapshot_slots)
    original = dump(source)
    plan = autosync(parse(original))
    assert dump(source) == original
    assert extract(source) == extract(plan)
    assert verify_plan(parse(dump(plan)))["independent_cores"] == 2
    code = emit_asc(plan)
    assert code.count("asc_sync_pipe(PIPE_ALL)") == 1
    if steps:
        assert "asc_fma(" in code and "asc_brcb(" in code
        assert "asc_copy(" in code and "asc_repeat_reduce_sum(" in code
    dce(plan)
    verify_plan(plan)


def test_state_carry_survives_snapshot_output_without_waiting_for_its_store():
    plan = autosync(recurrent_gdn(1, 3, value_dim=8))
    p = extract(plan)
    ids = {t.uid: i for i, t in enumerate(p.tasks)}
    graph, begin, end, _, _ = completion_graph(execution(plan))
    assert graph.precedes(
        end[Instance(0, ids["capture_snapshot"])], begin[Instance(1, ids["decay_state"])]
    )
    assert not graph.precedes(
        end[Instance(0, ids["store_snapshot"])], begin[Instance(1, ids["decay_state"])]
    )
    p = extract(plan)
    assert next(t for t in p.tasks if t.uid == "update_state").modes == ("R", "R", "RW")
    assert any(
        o.hazard == "WAR"
        and o.storage == "snapshot_slots"
        and p.tasks[o.source.task].uid == "store_snapshot"
        and p.tasks[o.destination.task].uid == "capture_snapshot"
        for o in dependencies(p)
    )
    assert len([i for i in p.instances() if p.tasks[i.task].uid == "load_initial"]) == 1
    assert next(t for t in p.tasks if t.uid == "decay_state").refs[0].storage.name == "carry"


def state_snapshot_example():
    k = Kernel("state_snapshot", disjoint_external=True)
    initial = k.buffer("initial", "GM", 256, external=True, initialized=True)
    result = k.buffer("result", "GM", 768, external=True)
    carry = k.buffer("carry", "UB", 256)
    snapshots = k.buffer("snapshot_ring", "UB", 512)
    k.loop(3)
    state = k.view(carry, 256)
    snapshot = k.view(snapshots, 256, stride=256, slots=2)
    k.task("load", "copy", k.view(initial, 256), state, active=(0, 1))
    k.scale("update", state, state, 2, stage=1)
    k.task("snapshot", "vector_copy", state, snapshot, stage=1)
    k.task("store", "copy", snapshot, k.view(result, 256, stride=256), stage=2, issue_order=0)
    return k.finish()


def test_minimal_recurrence_interleavings_and_missing_v_completion():
    plan = autosync(state_snapshot_example())
    report = bounded_check(plan)
    assert report.exhaustive and report.terminal_states == 1
    # V dispatch order alone does not complete the preceding in-place update
    # or the state reader. Both final proof and independent execution reject it.
    for op in list(plan.walk()):
        if isinstance(op, SyncOp) and op.action.data == "barrier":
            op.parent.erase_op(op)
    with pytest.raises(CompileError, match="missing (RAW|WAR|WAW)"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="counterexample"):
        bounded_check(plan)


@pytest.mark.parametrize("steps", [1, 2])
def test_short_full_gdn_has_a_finite_independent_check(steps):
    result = bounded_check(
        autosync(recurrent_gdn(1, steps, value_dim=8, slots=1, snapshot_slots=1)),
        max_states=200_000,
    )
    assert result.exhaustive and result.terminal_states == 1


def test_state_update_requires_the_actual_initial_writer():
    source = recurrent_gdn(1, 2, value_dim=8)
    initial = next(
        o for o in source.walk() if isinstance(o, TaskOp) and o.uid.data == "load_initial"
    )
    initial.parent.erase_op(initial)
    with pytest.raises(CompileError, match="undefined carry"):
        autosync(source)


@pytest.mark.parametrize(
    "options",
    [
        {"key_dim": 128},
        {"key_dim": 64.0},
        {"value_dim": 7},
        {"steps": 129},
        {"slots": 0},
        {"snapshot_slots": 0},
    ],
)
def test_fixture_rejects_unsupported_physical_state_contracts(options):
    kwargs = {"sequences": 1, "steps": 3, **options}
    with pytest.raises(CompileError):
        recurrent_gdn(**kwargs)


def test_fixed_snapshot_ring_cannot_be_shrunk_to_fit_ub():
    with pytest.raises(CompileError, match="UB capacity exceeded"):
        autosync(recurrent_gdn(1, 3, value_dim=128, snapshot_slots=8))


def test_event_pressure_cannot_silently_change_input_slots_or_issue_order():
    # Three independent input readiness streams already pressure MTE2->V.
    # Under the current completion contracts, their finite lifetimes need nine
    # distinct IDs; packing cannot invent a missing consumption path.
    source = recurrent_gdn(1, 5, value_dim=8, slots=3)
    before = dump(source)
    with pytest.raises(CompileError, match="event bank has 8 IDs"):
        autosync(source)
    assert dump(source) == before


def test_independent_reference_keeps_all_state_snapshots():
    # One nonzero key coordinate makes the recurrence exactly representable:
    # S0=0 -> S1=v/2 -> S2=3*v/4, and q selects the same coordinate.
    q = keys = [1.0] + [0.0] * 63
    result = gdn_reference(
        q * 2,
        keys * 2,
        [2.0] * 16,
        [0.0] * 8,
        [0.5] * 8,
        [0.0] * 512,
        sequences=1,
        steps=2,
        value_dim=8,
        scale=1.0,
    )
    assert list(result["y"]) == [1.0] * 8 + [1.5] * 8
    assert list(result["snapshots"][::64]) == [1.0] * 8 + [1.5] * 8
    data, expected, _, shape = recurrent_inputs(extract(recurrent_gdn(2, 3, value_dim=8)))
    assert shape == [2, 3, 8, 64]
    assert len(data[-1]) == len(expected["snapshots"]) * 4
    assert data[-1] == b"\xcd" * len(data[-1])
