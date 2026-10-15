"""Actual query captures, common slot permutations, and rejected IR mutations."""

from collections import defaultdict
from dataclasses import replace
from itertools import combinations

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType, IntAttr
from xdsl.ir import Operation

from tests.fixtures.attention_target import attention_target
from tests.fixtures.scoped_queries import scoped_query_recurrence
from tests.test_index_frontend import evaluate as evaluate_ssa
from tests.test_scope_serial_source import SOURCE
from xdsl_npu import language as N
from xdsl_npu.indexing import arguments
from xdsl_npu.ir import (
    BufferOp,
    CaptureOp,
    CoreIdOp,
    StreamOp,
    TaskOp,
    ViewOp,
    dump,
)
from xdsl_npu.lowering import lower_source
from xdsl_npu.model import CompileError
from xdsl_npu.scalar import expression, literal
from xdsl_npu.scope_model import extract_scopes
from xdsl_npu.scope_query_access import normalize_query_access


@pytest.fixture(scope="module")
def recurrence():
    program = extract_scopes(lower_source(scoped_query_recurrence))
    return program, normalize_query_access(program)


@pytest.fixture(scope="module")
def attention():
    program = extract_scopes(lower_source(attention_target()))
    return program, normalize_query_access(program)


def _name(leaf, captured):
    name = captured.root.sym_name.data
    return name if captured.root.external.data else leaf.actor + "/" + name


def _capture(program, uid, name):
    return next(cap for cap in program.leaves[uid].captures if cap.root.sym_name.data == name)


def _clone(fixture):
    return extract_scopes(fixture[0].module.clone())


def _definitions(value, seen=None):
    seen = set() if seen is None else seen
    if value in seen or not isinstance(value.owner, Operation):
        return
    seen.add(value)
    yield value.owner
    for operand in value.owner.operands:
        yield from _definitions(operand, seen)


def _ring_rem(capture):
    return next(op for op in _definitions(capture.op.offset) if isinstance(op, arith.RemUIOp))


def _constant_before(op, value):
    constant = arith.ConstantOp.from_int_and_width(value, IndexType())
    op.parent.insert_op_before(constant, op)
    return constant.result


def _replace_operand(op, index, value):
    operands = list(op.operands)
    operands[index] = value
    op.operands = operands


def test_recurrence_contracts_are_tied_to_actual_allocations(recurrence):
    program, mapped = recurrence
    assert {
        name: (mapped.contracts[name].kind, mapped.contracts[name].frame_bytes)
        for name in ("Q", "K", "Score", "O")
    } == {
        "Q": ("readonly", 1024),
        "K": ("readonly", 1024),
        "Score": ("epoch_ring", 2048),
        "O": ("query_frame", 2048),
    }
    assert mapped.contracts["Score"].slots == 2
    assert mapped.contracts["Score"].actual_extent.constant == 8192
    assert mapped.storages["aiv0", "O"].nbytes == 4096
    assert mapped.storages["aiv1", "O"].nbytes == 4096
    assert mapped.contracts["O"].total == program.invocations["aic"].cyclic.total
    assert len(mapped.offsets) == sum(len(leaf.captures) for leaf in program.leaves.values())
    for uid, leaf in program.leaves.items():
        for captured in leaf.captures:
            name = _name(leaf, captured)
            assert mapped.contracts[name].actual_root is captured.root
            if name in {"Q", "K"}:
                assert mapped.offsets[uid, name] == literal(0)
            if not captured.root.external.data:
                assert mapped.contracts[name].kind == "reused"
                assert mapped.offsets[uid, name] is captured.offset
                assert (leaf.actor, name) not in mapped.storages


def test_real_attention_uses_common_gm_and_private_alpha_rings(attention):
    program, mapped = attention
    assert {name for name, root in mapped.contracts.items() if root.kind == "readonly"} == {
        "Q",
        "K",
        "V",
    }
    expected = {"Score": 524288, "P": 262144, "PV": 65536, "aiv0/ub_18": 256, "aiv1/ub_18": 256}
    assert {
        name: root.frame_bytes
        for name, root in mapped.contracts.items()
        if root.kind == "epoch_ring"
    } == expected
    assert mapped.contracts["O"].kind == "query_frame"
    assert mapped.contracts["O"].frame_bytes == 32768
    assert mapped.storages["aiv0", "O"].nbytes == program.groups * 32768
    assert mapped.storages["aic", "K"].nbytes == 262144
    assert mapped.storages["aic", "V"].nbytes == 262144
    for name in expected:
        root = mapped.contracts[name]
        assert root.slots == 2
        assert arguments(root.phase) == {"$parent1", "Sk"}
        assert root.actual_root.nbytes.data == 2 * root.frame_bytes * (
            program.groups if root.actual_root.external.data else 1
        )
        assert all(key[1] != name for key in mapped.storages)
    assert all(storage.external for storage in mapped.storages.values())
    assert all(
        storage.count_scale == 0 and storage.extent is None for storage in mapped.storages.values()
    )
    assert all(
        storage.initialized == bool(mapped.contracts[name].actual_root.initialized.data)
        for (_, name), storage in mapped.storages.items()
    )


@pytest.mark.parametrize("fixture_name", ("recurrence", "attention"))
def test_mapping_never_changes_ir_native_templates_or_local_storage(
    request, fixture_name, monkeypatch
):
    program, _ = request.getfixturevalue(fixture_name)
    before = dump(program.module)
    private = {
        op: tuple(op.attributes.items())
        for op in program.module.walk()
        if isinstance(op, BufferOp) and not op.external.data
    }

    def no_clone(*args, **kwargs):
        pytest.fail("query normalization cloned physical IR")

    for op in (TaskOp, ViewOp, BufferOp, StreamOp):
        monkeypatch.setattr(op, "clone", no_clone)
    normalize_query_access(program)
    assert dump(program.module) == before
    assert all(tuple(op.attributes.items()) == attrs for op, attrs in private.items())


def _relative_segments(program):
    """Interpret actual child ViewOps; no normalized Ref or access frontier."""
    result = defaultdict(list)
    for uid, leaf in program.leaves.items():
        for task in leaf.stream.walk():
            if not isinstance(task, TaskOp):
                continue
            for value in task.refs:
                if not isinstance(value.owner, ViewOp):
                    continue  # Scalar SSA operands are not memory references.
                final = value.owner
                rows = 1 if final.rows is None else final.rows.data
                pitch = 0 if final.row_stride is None else final.row_stride.data
                for inner in range(leaf.stream.count.data):
                    ref, start = value, 0
                    while isinstance(ref.owner, ViewOp):
                        view = ref.owner
                        divisor = 1 if view.divisor is None else view.divisor.data
                        coordinate = inner // divisor
                        if view.modulus.data:
                            coordinate %= view.modulus.data
                        start += view.offset.data + view.stride.data * coordinate
                        for term in view.index_terms or ():
                            stride, scale, modulus = (part.data for part in term)
                            position = inner // scale
                            start += stride * (position % modulus if modulus else position)
                        assert not view.core_stride.data and not view.core_terms
                        assert view.count_offset is None or view.count_offset.data == 0
                        assert view.count_core_stride is None or view.count_core_stride.data == 0
                        ref = view.source
                    assert isinstance(ref.owner, CaptureOp)
                    for row in range(rows):
                        lo = start + row * pitch
                        hi = lo + final.nbytes.data
                        assert 0 <= lo < hi <= ref.owner.nbytes.data
                        result[uid, ref.owner].append((lo, hi))
    return result


def _oracle(program, mapped, parameters):
    """Small-domain real-SSA address/overlap oracle, never a runtime expansion."""
    scalar_bindings = {op.value: parameters[op.sym_name.data] for op in program.parameters}
    invocation = next(iter(program.invocations.values()))
    total = evaluate_ssa(invocation.op.ub.owner.total, scalar_bindings)
    bound = (
        program.count
        if program.bound is None
        else evaluate_ssa(next(iter(program.parents.values())).limit, scalar_bindings)
    )
    counts = [len(range(core, total, program.groups)) for core in range(program.groups)]
    relative = _relative_segments(program)
    for query in range(max(counts, default=0)):
        events = defaultdict(list)
        for core, count in enumerate(counts):
            if query >= count:
                continue
            for kv in range(bound):
                values = parameters | {"$core": core, "$parent1": query, "$parent0": kv}
                for uid, leaf in program.leaves.items():
                    bindings = scalar_bindings | {
                        program.invocations[leaf.actor].op.body.block.args[0]: query,
                        program.parents[leaf.actor].body.block.args[0]: kv,
                        next(
                            op.value
                            for op in program.kernels[leaf.actor].body.block.ops
                            if isinstance(op, CoreIdOp)
                        ): core,
                    }
                    for captured in leaf.captures:
                        name = _name(leaf, captured)
                        contract = mapped.contracts[name]
                        actual = evaluate_ssa(captured.op.offset, bindings)
                        normalized = mapped.offsets[uid, name].evaluate(values)
                        if contract.kind == "readonly":
                            assert normalized == 0
                            continue
                        if contract.kind == "query_frame":
                            assert (
                                actual - normalized == query * program.groups * contract.frame_bytes
                            )
                        elif contract.kind == "epoch_ring":
                            tile = contract.frame_bytes
                            owner = (
                                core * contract.slots * tile if captured.root.external.data else 0
                            )
                            slot, bias = divmod(actual - owner, tile)
                            phase = contract.phase.evaluate(values)
                            assert phase == (query * bound) % contract.slots
                            assert (
                                normalized
                                == owner + ((slot - phase) % contract.slots) * tile + bias
                            )
                            # All real native row segments use the same slot
                            # permutation, including the last byte of a slot.
                            for lo, hi in relative[uid, captured.op]:
                                assert (actual + lo - owner) // tile == slot
                                assert (actual + hi - 1 - owner) // tile == slot
                                for byte in (lo, hi - 1):
                                    byte_slot, within = divmod(actual + byte - owner, tile)
                                    permuted = (
                                        owner
                                        + ((byte_slot - phase) % contract.slots) * tile
                                        + within
                                    )
                                    assert permuted == normalized + byte
                        else:
                            assert contract.kind == "reused"
                            assert actual == normalized
                        key = name, None if captured.root.external.data else core
                        events[key].append((actual, normalized, captured.op.nbytes.data))
        for records in events.values():
            for (left, mapped_left, left_span), (right, mapped_right, right_span) in combinations(
                records, 2
            ):
                actual_overlap = left < right + right_span and right < left + left_span
                normalized_overlap = (
                    mapped_left < mapped_right + right_span
                    and mapped_right < mapped_left + left_span
                )
                assert actual_overlap == normalized_overlap
    return counts


@pytest.mark.parametrize("total", (0, 1, 3, 5))
@pytest.mark.parametrize("kv_tiles", (1, 2, 3))
def test_cyclic_recurrence_actual_byte_overlap_oracle(recurrence, total, kv_tiles):
    program, mapped = recurrence
    counts = _oracle(program, mapped, {"total_queries": total, "kv_tiles": kv_tiles})
    assert sum(counts) == total
    if total < program.groups:
        assert 0 in counts
    if total % program.groups:
        assert max(counts) != min(counts)


@pytest.mark.parametrize("Sq", (128, 384))
@pytest.mark.parametrize("Sk", (1024, 2048, 3072))
def test_attention_actual_native_byte_overlap_oracle(attention, Sq, Sk):
    counts = _oracle(*attention, {"Sq": Sq, "Sk": Sk})
    assert sum(counts) == 8 * (Sq // 128)
    assert max(counts) != min(counts)


def test_actual_reader_with_epoch_plus_query_is_rejected_despite_identical_child_refs(attention):
    program = _clone(attention)
    captured = _capture(program, "aiv0_call1", "Score")
    rem = _ring_rem(captured)
    query = program.invocations["aiv0"].op.body.block.args[0]
    wrong = arith.AddiOp(rem.operands[0], query)
    rem.parent.insert_op_before(wrong, rem)
    _replace_operand(rem, 0, wrong.result)
    rebuilt = extract_scopes(program.module)
    original_tasks = next(iter(program.leaves["aiv0_call1"].loop.actors.values())).program.tasks
    changed_tasks = next(iter(rebuilt.leaves["aiv0_call1"].loop.actors.values())).program.tasks
    assert original_tasks == changed_tasks
    with pytest.raises(CompileError, match="Score:.*epoch phase"):
        normalize_query_access(rebuilt)


def test_actual_hidden_input_write_across_calls_removes_readonly_classification(attention):
    program = _clone(attention)
    output = _capture(program, "aiv0_call3", "O").op
    q_root = next(
        cap.root
        for leaf in program.leaves.values()
        for cap in leaf.captures
        if cap.root.sym_name.data == "Q"
    )
    _replace_operand(output, 0, q_root.ref)
    rebuilt = extract_scopes(program.module)
    mapped = normalize_query_access(rebuilt)
    assert mapped.contracts["Q"].kind == "query_frame"
    assert mapped.contracts["Q"].actual_root is q_root
    assert mapped.storages["aic", "Q"].initialized
    assert mapped.offsets["aic_call0", "Q"].evaluate({"$core": 1}) == 32768


def test_actual_last_case_write_to_k_cannot_hide_behind_other_readonly_calls(attention):
    program = _clone(attention)
    output = _capture(program, "aiv0_call3", "O").op
    k_root = next(
        cap.root
        for leaf in program.leaves.values()
        for cap in leaf.captures
        if cap.root.sym_name.data == "K"
    )
    _replace_operand(output, 0, k_root.ref)
    _replace_operand(output, 1, _constant_before(output, 0))
    rebuilt = extract_scopes(program.module)
    with pytest.raises(CompileError, match="K:.*query-dependent"):
        normalize_query_access(rebuilt)


def test_uninitialized_actual_all_read_input_does_not_gain_a_live_in(recurrence):
    program = _clone(recurrence)
    _capture(program, "aic_call0", "Q").root.attributes["initialized"] = IntAttr(0)
    rebuilt = extract_scopes(program.module)
    with pytest.raises(CompileError, match="Q: readonly GM needs actual initialized"):
        normalize_query_access(rebuilt)


@pytest.mark.parametrize("mutation", ("slots", "coefficient", "capacity"))
def test_actual_ring_alias_period_coefficient_and_capacity_are_proved(recurrence, mutation):
    program = _clone(recurrence)
    captured = _capture(program, "aiv0_call0", "Score")
    rem = _ring_rem(captured)
    if mutation == "slots":
        _replace_operand(rem, 1, _constant_before(rem, 1))
    elif mutation == "coefficient":
        product = next(
            op
            for op in _definitions(captured.op.offset)
            if isinstance(op, arith.MuliOp) and rem.result in op.operands
        )
        index = next(i for i, value in enumerate(product.operands) if value is not rem.result)
        _replace_operand(product, index, _constant_before(product, 1024))
    else:
        captured.root.attributes["nbytes"] = IntAttr(captured.root.nbytes.data + 2)
    rebuilt = extract_scopes(program.module)
    with pytest.raises(CompileError, match="Score: actual ring"):
        normalize_query_access(rebuilt)


def test_private_epoch_ring_capacity_is_not_silently_resized(attention):
    program = _clone(attention)
    root = next(
        cap.root
        for leaf in program.leaves.values()
        if leaf.actor == "aiv0"
        for cap in leaf.captures
        if cap.root.sym_name.data == "ub_18"
    )
    root.attributes["nbytes"] = IntAttr(root.nbytes.data + 32)
    rebuilt = extract_scopes(program.module)
    with pytest.raises(CompileError, match="aiv0/ub_18: actual ring aliases"):
        normalize_query_access(rebuilt)


def test_query_frame_uses_actual_extent_not_the_output_name(recurrence):
    program = _clone(recurrence)
    root = _capture(program, "aiv0_call0", "O").root
    root.attributes["count_scale"] = IntAttr(root.count_scale.data * 2)
    rebuilt = extract_scopes(program.module)
    with pytest.raises(CompileError, match="O: unsupported query-dependent root"):
        normalize_query_access(rebuilt)


def test_static_query_frame_bias_is_preserved_and_bounded(recurrence):
    program = _clone(recurrence)
    cap = _capture(program, "aiv0_call0", "O").op
    bias = _constant_before(cap, 32)
    shifted = arith.AddiOp(cap.offset, bias)
    cap.parent.insert_op_before(shifted, cap)
    _replace_operand(cap, 1, shifted.result)
    rebuilt = extract_scopes(program.module)
    mapped = normalize_query_access(rebuilt)
    assert mapped.contracts["O"].biases == {"aiv0_call0": 32, "aiv1_call0": 0}
    assert mapped.offsets["aiv0_call0", "O"].evaluate({"$core": 1}) == 2080


def test_stationary_written_storage_remains_uninitialized_reused_memory(recurrence):
    program = _clone(recurrence)
    total = next(op for op in program.parameters if op.sym_name.data == "total_queries")
    total.attributes["minimum"] = IntAttr(1)
    for leaf in program.leaves.values():
        for captured in leaf.captures:
            if captured.root.sym_name.data == "O":
                _replace_operand(captured.op, 1, _constant_before(captured.op, 0))
    rebuilt = extract_scopes(program.module)
    mapped = normalize_query_access(rebuilt)
    assert mapped.contracts["O"].kind == "reused"
    assert mapped.contracts["O"].frame_bytes is None
    assert not mapped.contracts["O"].actual_root.initialized.data
    assert all(name != "O" for _, name in mapped.storages)


def test_wrong_actual_call_clock_cannot_use_a_cached_extraction(recurrence):
    program = _clone(recurrence)
    leaf = program.leaves["aiv0_call0"]
    leaf.call.operands = (program.invocations[leaf.actor].op.body.block.args[0],)
    with pytest.raises(CompileError, match="different actual call, KV clock"):
        normalize_query_access(program)


def test_stale_capture_expression_is_rejected(recurrence):
    program = _clone(recurrence)
    leaf = program.leaves["aic_call0"]
    cap = leaf.captures[0]
    changed = replace(cap, offset=expression("add", cap.offset, literal(32)))
    leaves = dict(program.leaves)
    leaves[leaf.call.uid.data] = replace(leaf, captures=(changed, *leaf.captures[1:]))
    with pytest.raises(CompileError, match="stale actual capture"):
        normalize_query_access(replace(program, leaves=leaves))


def test_changed_actual_native_requires_reextraction(recurrence):
    program = _clone(recurrence)
    task = next(
        op
        for op in program.leaves["aiv0_call0"].stream.walk()
        if isinstance(op, TaskOp) and op.native.data == "add"
    )
    task.attributes["native"] = task.attributes["native"].__class__("mul")
    with pytest.raises(CompileError, match="stale native/view descriptor"):
        normalize_query_access(program)


def test_changed_actual_query_total_requires_reextraction(recurrence):
    program = _clone(recurrence)
    query = program.invocations["aiv0"].op
    kv_tiles = next(op.value for op in program.parameters if op.sym_name.data == "kv_tiles")
    query.ub.owner.operands = (kv_tiles,)
    with pytest.raises(CompileError, match="stale actual query/parent domain"):
        normalize_query_access(program)


def test_uniform_queries_fail_explicitly_until_their_frame_contract_is_supported():
    text = (
        SOURCE.replace("N.ceildiv(N.max(0,total_queries-group),GROUPS)", "total_queries")
        .replace("((total_queries,2,16,16)", "((GROUPS,total_queries,2,16,16)")
        .replace("((total_queries,4,64)", "((GROUPS,total_queries,4,64)")
        .replace("Q[flat,part,:,:]", "Q[group,query,part,:,:]")
        .replace("O[flat,lane*2+part,:]", "O[group,query,lane*2+part,:]")
    )
    program = extract_scopes(lower_source(N.from_source(text, disjoint_external=True)))
    with pytest.raises(CompileError, match="requires actual cyclic query ownership"):
        normalize_query_access(program)
