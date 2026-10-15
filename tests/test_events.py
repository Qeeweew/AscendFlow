import itertools

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from examples.legacy.gdn import recurrent_gdn
from tests.fixtures.pipelines import phased_pipeline, vector_pipeline
from xdsl_npu import autosync, emit_asc, verify_plan
from xdsl_npu.check import bounded_check
from xdsl_npu.events import minimum_chains, pack_event_lifetimes, selection_expression
from xdsl_npu.ir import ChannelOp, dump, parse
from xdsl_npu.model import CompileError, extract
from xdsl_npu.sync import explain_plan


def test_matching_minimum_against_every_five_node_partial_order():
    n = 5
    candidates = list(itertools.combinations(range(n), 2))
    for mask in range(1 << len(candidates)):
        successors = [0] * n
        for bit, (a, b) in enumerate(candidates):
            if mask & (1 << bit):
                successors[a] |= 1 << b
        for a in reversed(range(n)):
            for b in range(a + 1, n):
                if successors[a] & (1 << b):
                    successors[a] |= successors[b]
        chains, witness = minimum_chains(successors)
        maximum = max(
            len(nodes)
            for size in range(n + 1)
            for nodes in itertools.combinations(range(n), size)
            if all(not successors[a] & (1 << b) for a, b in itertools.combinations(nodes, 2))
        )
        assert len(chains) == len(witness) == maximum
        assert sorted(itertools.chain.from_iterable(chains)) == list(range(n))
        assert all(successors[a] & (1 << b) for c in chains for a, b in itertools.pairwise(c))
        assert all(not successors[a] & (1 << b) for a, b in itertools.combinations(witness, 2))


@pytest.mark.parametrize("phases,length", [(9, 1), (5, 3), (16, 2)])
def test_physical_lifetimes_can_share_flags_across_logical_channels(phases, length):
    source = phased_pipeline(phases, length)
    original = dump(source)
    plan = autosync(source)
    assert dump(source) == original
    assert extract(plan) == extract(source)
    report = verify_plan(parse(dump(plan)))
    assert all(len(ids) <= 2 for ids in report["event_ids"].values())
    proof = explain_plan(plan)
    assert any(w["old_channel"] != w["next_channel"] for w in proof["binary_event_reuse"])
    assert all(
        w["consumption_path"][0].split(".")[1].startswith("wait")
        for w in proof["binary_event_reuse"]
    )
    assert emit_asc(plan).count("asc_sync_pipe(PIPE_ALL)") == 1


def test_nonperiodic_channel_encoding_and_corrupt_shared_flag_are_checked():
    plan = autosync(phased_pipeline(5, 3))
    channels = [o for o in plan.walk() if isinstance(o, ChannelOp)]
    assert any(o.event_ids is not None for o in channels)
    source = emit_asc(plan)
    assert " ? " in source and "constexpr uint8_t" not in source
    # Collapse two overlapping output-ready lifetimes into one flag. This is
    # unsafe even though each logical channel still has its own paired Wait.
    for op in channels:
        if op.destination.data.startswith("store_"):
            op.attributes.update(base=IntAttr(0), width=IntAttr(1))
            op.attributes.pop("event_ids", None)
    with pytest.raises(CompileError, match="re-set before old Wait"):
        verify_plan(plan)
    with pytest.raises(CompileError, match="re-Set|wrong epoch"):
        bounded_check(plan)


def test_recycled_phases_have_an_independent_finite_interleaving_check():
    report = bounded_check(autosync(phased_pipeline(9)), max_states=200_000)
    assert report.exhaustive and report.terminal_states == 1


@pytest.mark.parametrize(
    "ids,diagnostic", [([0, 1], "every notification"), ([0, 1, 2, 8], "[0,7]")]
)
def test_explicit_id_tables_must_be_complete_and_in_range(ids, diagnostic):
    plan = autosync(vector_pipeline(4))
    channel = next(o for o in plan.walk() if isinstance(o, ChannelOp))
    channel.attributes.update(
        base=IntAttr(0), width=IntAttr(1), event_ids=ArrayAttr(IntAttr(i) for i in ids)
    )
    with pytest.raises(CompileError, match=diagnostic):
        verify_plan(plan)


def test_explicit_ids_are_not_silently_overridden_by_rotating_fields():
    plan = autosync(vector_pipeline(4))
    channel = next(o for o in plan.walk() if isinstance(o, ChannelOp))
    channel.attributes["event_ids"] = ArrayAttr(IntAttr(i) for i in (0, 1, 0, 1))
    with pytest.raises(CompileError, match="neutral base"):
        verify_plan(plan)


def test_scalar_selection_matches_each_finite_ordinal():
    for ids in ((3,), (1, 0, 1, 0, 1), (0, 2, 4, 6, 2, 3, 1, 7, 0, 5)):
        # Check the generated condition tree structurally, including the
        # periodic shortened form, without reusing its construction logic.
        expression = selection_expression(ids, "index")
        import re

        def evaluate(value, index):
            value = value.strip()
            if re.fullmatch(r"\d+", value):
                return int(value)
            # The expression is a fully parenthesized ternary. Scan nesting
            # to split its condition, true arm and false arm.
            assert value[0] == "(" and value[-1] == ")"
            value = value[1:-1]
            depth = 0
            splits = {}
            for i, c in enumerate(value):
                depth += (c == "(") - (c == ")")
                if depth == 0 and c in "?:":
                    splits[c] = i
            a, b = splits["?"], splits[":"]
            condition = eval(value[:a], {"__builtins__": {}}, {"index": index})
            return evaluate(value[a + 1 : b] if condition else value[b + 1 :], index)

        assert [evaluate(expression, i) for i in range(len(ids))] == list(ids)


def test_finite_packing_does_not_invent_paths_to_hide_real_pressure():
    source = recurrent_gdn(1, 5, value_dim=8, slots=3)
    before = dump(source)
    with pytest.raises(CompileError, match="packing requires 9.*Incomparable notifications"):
        autosync(source)
    assert dump(source) == before


def test_manual_repacking_keeps_commands_and_has_the_same_final_checks():
    plan = autosync(vector_pipeline(5, slots=1))
    program = extract(plan)
    pack_event_lifetimes(plan)
    assert extract(plan) == program
    verify_plan(plan)
    assert bounded_check(plan).exhaustive


def test_explicit_epoch_maps_are_idempotent_under_repacking():
    plan = autosync(phased_pipeline(5, 3))
    text = dump(plan)
    pack_event_lifetimes(plan)
    assert dump(plan) == text
    verify_plan(plan)
