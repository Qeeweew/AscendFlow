"""Captured addresses from the real attention source, without native lowering."""

from collections import defaultdict
from dataclasses import dataclass

import pytest
from xdsl.dialects import scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue

from tests.fixtures.attention_target import attention_target
from tests.test_index_frontend import evaluate as evaluate_ssa
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.source import AllocationOp, DomainOp, PipelineOp, ProgramOp, SizeOp, ViewOp
from xdsl_npu.indexing import arguments, check_axis, equivalent
from xdsl_npu.lowering import _refs
from xdsl_npu.scalar import Expr, Fact, expression, facts_from_sizes, from_ssa, literal
from xdsl_npu.scope_address import split_offset
from xdsl_npu.source_scopes import Scope, Work, collect_source_scopes
from xdsl_npu.target.native import ITEMSIZE


def test_captured_multiclock_remainder_keeps_its_proved_slot_range():
    core, query, kv, child, count = (
        Expr("argument", (name,)) for name in ("$core", "$query", "$kv", "$child", "n")
    )
    epoch = expression("add", expression("mul", query, count), kv)
    slot = expression("rem", epoch, literal(2))
    base = expression(
        "add", expression("mul", core, literal(4096)), expression("mul", slot, literal(2048))
    )
    address = expression(
        "add", base, expression("add", expression("mul", child, literal(128)), literal(32))
    )
    facts = {"n": Fact(1)}
    parent, relative = split_offset(address, "$child", {"$core", "$query", "$kv"}, facts)
    domains = [
        (core, literal(0), literal(1)),
        (query, literal(0), literal(1 << 40)),
        (kv, literal(0), expression("sub", count, literal(1))),
    ]
    check_axis(parent, 2048, literal(8192), domains, facts)
    with pytest.raises(CompileError, match="bounded"):
        check_axis(parent, 2049, literal(8192), domains, facts)
    for n in (1, 2, 3, 1024):
        for q in (0, 1, 7, 1 << 40):
            for k in (0, n - 1):
                values = {"n": n, "$core": 1, "$query": q, "$kv": k, "$child": 3}
                assert parent.evaluate(values) == 4096 + ((q * n + k) % 2) * 2048
                assert parent.evaluate(values) + relative.evaluate(values) == address.evaluate(
                    values
                )


@dataclass
class AttentionAddresses:
    module: ModuleOp
    program: ProgramOp
    root: Scope
    outer: Scope
    children: tuple[Scope, ...]
    coordinates: dict[SSAValue, str]
    facts: dict[str, Fact]
    continuous: bool

    def address(self, ref):
        view = ref.owner
        assert isinstance(view, ViewOp)
        root = view.source.owner
        assert isinstance(root, AllocationOp)
        memo = {}

        def scalar(value):
            return from_ssa(value, coordinates=self.coordinates, memo=memo, facts=self.facts)

        dynamic = {
            axis.data: scalar(value)
            for axis, value in zip(root.dynamic_axes or (), root.dimensions, strict=True)
        }
        linear = literal(0)
        for axis, (extent, origin) in enumerate(
            zip(root.ref.type.shape, view.origins, strict=True)
        ):
            width = dynamic[axis] if extent.data == -1 else literal(extent.data)
            linear = expression("add", expression("mul", linear, width), scalar(origin))
        return root, expression("mul", linear, literal(ITEMSIZE[root.ref.type.dtype.data]))

    def split(self, work, ref, child):
        root, address = self.address(ref)
        frames = work.parent_clocks if child else work.context.clocks
        captured = {self.coordinates[frame.coordinate] for frame in frames} | {"$core"}
        if work.context.owner is not None:
            captured.add(self.coordinates[work.context.owner])
        clock = self.coordinates[work.clock.coordinate] if child else None
        parent, relative = split_offset(address, clock, captured, self.facts)
        return root, parent, relative

    def bindings(self, *, Sq, Sk, core, query, kv, child=0, lane=0):
        sizes = {op.sym_name.data: op.value for op in self.module.walk() if isinstance(op, SizeOp)}
        bindings = {sizes["Sq"]: Sq, sizes["Sk"]: Sk, self.program.body.block.args[0]: core}
        for scope in self.root.walk_scopes():
            if scope.clock is None:
                continue
            if isinstance(scope.op, scf.ForOp):
                value = query
            elif scope is self.outer:
                value = query * (Sk // 1024) + kv if self.continuous else kv
            else:
                value = child
            bindings[scope.clock.coordinate] = value
        for op in self.module.walk():
            if isinstance(op, DomainOp):
                bindings[op.body.block.args[0]] = lane if op.kind.data == "vector" else 0
        expressions = {name: bindings[value] for value, name in self.coordinates.items()}
        expressions.update(Sq=Sq, Sk=Sk)
        return bindings, expressions

    def oracle(self, ref, bindings):
        """Interpret original SSA axes independently of the index polynomial."""
        view = ref.owner
        root = view.source.owner
        dynamic = {
            axis.data: evaluate_ssa(value, bindings)
            for axis, value in zip(root.dynamic_axes or (), root.dimensions, strict=True)
        }
        linear = 0
        for axis, (extent, origin) in enumerate(
            zip(root.ref.type.shape, view.origins, strict=True)
        ):
            width = dynamic[axis] if extent.data == -1 else extent.data
            linear = linear * width + evaluate_ssa(origin, bindings)
        return linear * ITEMSIZE[root.ref.type.dtype.data]


@pytest.fixture(scope="module", params=[False, True], ids=["nested", "continuous"])
def attention(request):
    module = attention_target(continuous=request.param).to_ir()
    program = next(op for op in module.walk() if isinstance(op, ProgramOp))
    root = collect_source_scopes(program)
    pipelines = tuple(scope for scope in root.walk_scopes() if isinstance(scope.op, PipelineOp))
    coordinates = {program.body.block.args[0]: "$core"}
    coordinates.update(
        (scope.clock.coordinate, f"$clock{i}")
        for i, scope in enumerate(root.walk_scopes())
        if scope.clock is not None
    )
    # A lane is a captured source coordinate. Bind it to the physical actor
    # after the split, so it does not become a static relative view bias.
    coordinates.update(
        (op.body.block.args[0], "$lane" if op.kind.data == "vector" else "$cube")
        for op in module.walk()
        if isinstance(op, DomainOp)
    )
    return AttentionAddresses(
        module,
        program,
        root,
        pipelines[0],
        pipelines[1:],
        coordinates,
        facts_from_sizes(program.body.block.ops),
        request.param,
    )


SAMPLES = (
    {"Sq": 128, "Sk": 1024, "core": 0, "query": 0, "kv": 0},
    {"Sq": 384, "Sk": 2048, "core": 0, "query": 1, "kv": 1},
    {"Sq": 512, "Sk": 3072, "core": 5, "query": 0, "kv": 2},
)


def assert_common_parents(attention, records):
    parents = defaultdict(list)
    for root, parent, _ in records:
        parents[root].append(parent)
    assert parents
    for values in parents.values():
        assert all(equivalent(values[0], other, attention.facts) for other in values[1:])


def captures(attention, scope, root):
    return [
        attention.split(work, ref, child=True)
        for work in scope.walk_work()
        for ref in _refs(work.op)
        if ref.owner.source.owner is root
    ]


def allocation(attention, name):
    return next(
        op
        for op in attention.module.walk()
        if isinstance(op, AllocationOp) and op.sym_name.data == name
    )


def test_real_attention_four_children_share_one_parent_per_allocation_and_actor(attention):
    assert len(attention.children) == 4
    assert [scope.actors for scope in attention.children] == [
        {"cube"},
        {"vector"},
        {"cube"},
        {"vector"},
    ]
    for child in attention.children:
        assert child.clock.bound == 8
        child_name = attention.coordinates[child.clock.coordinate]
        for lane in (0, 1) if child.actors == {"vector"} else (0,):
            records = []
            for work in child.walk_work():
                assert work.clock.coordinate is child.clock.coordinate
                for ref in _refs(work.op):
                    record = attention.split(work, ref, child=True)
                    root, parent, relative = record
                    records.append(record)
                    assert ref.owner.source is root.ref
                    assert child_name not in arguments(parent)
                    assert arguments(relative) <= {child_name}
                    for sample in SAMPLES:
                        for index in (0, 7):
                            bindings, values = attention.bindings(**sample, child=index, lane=lane)
                            assert parent.evaluate(values) + relative.evaluate(
                                values
                            ) == attention.oracle(ref, bindings)
            assert_common_parents(attention, records)


def ordinary_activations(outer):
    """Keep source actor, stage and guard changes as separate calls."""
    result = []
    previous = None
    for item in outer.items:
        if isinstance(item, Scope):
            previous = None
            continue
        assert isinstance(item, Work)
        activation = (item.actor, item.context.domain, item.context.clocks, item.context.guards)
        if activation != previous:
            result.append([])
        result[-1].append(item)
        previous = activation
    return result


def test_real_attention_ordinary_segments_keep_common_bases_and_distinct_activations(attention):
    segments = ordinary_activations(attention.outer)
    assert [(segment[0].actor, segment[0].clock.offset, len(segment)) for segment in segments] == [
        ("cube", 0, 2),
        ("vector", 1, 2),
        ("cube", 1, 1),
        ("vector", 2, 1),
    ]
    assert [bool(segment[0].context.guards) for segment in segments] == [True, True, False, True]
    for segment in segments:
        for lane in (0, 1) if segment[0].actor == "vector" else (0,):
            records = []
            for work in segment:
                for ref in _refs(work.op):
                    record = attention.split(work, ref, child=False)
                    records.append(record)
                    _, parent, relative = record
                    assert relative.constant is not None
                    for sample in SAMPLES:
                        bindings, values = attention.bindings(**(sample | {"kv": 0}), lane=lane)
                        assert parent.evaluate(values) + relative.evaluate(
                            values
                        ) == attention.oracle(ref, bindings)
            assert_common_parents(attention, records)
    pv_export = segments[2][0]
    root, parent, relative = attention.split(pv_export, pv_export.op.source, child=False)
    assert root.ref.type.memory.data == "L0C"
    assert parent.constant == relative.constant == 0
    assert pv_export.clock.coordinate is attention.outer.clock.coordinate
    assert pv_export.clock.stage.label.data == "pv"
    preceding_child = attention.outer.items[attention.outer.items.index(pv_export) - 1]
    assert isinstance(preceding_child, Scope) and preceding_child is attention.children[2]


@pytest.mark.parametrize(
    "name,scopes,bias", [("Score", (0, 1), 64 * 1024 * 4), ("P", (2, 1), 64 * 1024 * 2)]
)
def test_score_and_probability_lane_capture_bases_depend_on_actual_scope(
    attention, name, scopes, bias
):
    root = allocation(attention, name)
    first, second = (attention.children[index] for index in scopes)
    first_records = captures(attention, first, root)
    second_records = captures(attention, second, root)
    assert first_records and second_records and first.op is not second.op
    first_parent, second_parent = first_records[0][1], second_records[0][1]
    lane = from_ssa(
        next(second.walk_work()).context.owner,
        coordinates=attention.coordinates,
        facts=attention.facts,
    )
    lane_bias = expression("mul", lane, literal(bias))
    assert equivalent(second_parent, expression("add", first_parent, lane_bias), attention.facts)
    assert not equivalent(first_parent, second_parent, attention.facts)
    # Lane zero can have equal addresses across distinct source calls. The
    # allocation and scope identities still belong to the caller's capture key.
    for sample in SAMPLES:
        for owner in (0, 1):
            _, values = attention.bindings(**sample, lane=owner)
            assert second_parent.evaluate(values) - first_parent.evaluate(values) == owner * bias
    assert (root, first.op) != (root, second.op)


def test_alpha_capture_retains_the_original_ring_and_aliases_later_parent_tasks(attention):
    alpha = next(
        op
        for op in attention.module.walk()
        if isinstance(op, AllocationOp)
        and op.ref.type.memory.data == "UB"
        and tuple(n.data for n in op.ref.type.shape) == (2, 64)
    )
    for lane in (0, 1):
        softmax = captures(attention, attention.children[1], alpha)
        update = captures(attention, attention.children[3], alpha)
        assert softmax and update
        assert_common_parents(attention, softmax + update)
        parent = softmax[0][1]
        offsets = []
        for task in range(3):
            _, values = attention.bindings(Sq=128, Sk=3072, core=0, query=0, kv=task, lane=lane)
            offsets.append(parent.evaluate(values))
        assert offsets == [0, 64 * 4, 0]
        for records in (softmax, update):
            for root, _, relative in records:
                assert root is alpha
                for row in (0, 7):
                    _, values = attention.bindings(
                        Sq=128, Sk=3072, core=0, query=0, kv=0, child=row, lane=lane
                    )
                    assert relative.evaluate(values) == row * 8 * 4


def actual_coordinates(attention):
    child = attention.children[0].clock.coordinate
    parent = attention.outer.clock.coordinate
    return (
        from_ssa(parent, coordinates=attention.coordinates, facts=attention.facts),
        from_ssa(child, coordinates=attention.coordinates, facts=attention.facts),
    )


def test_runtime_parent_quotient_and_static_bias_survive_split(attention):
    _, child = actual_coordinates(attention)
    sq = next(
        op for op in attention.module.walk() if isinstance(op, SizeOp) and op.sym_name.data == "Sq"
    )
    quotient = expression("floordiv", from_ssa(sq.value, facts=attention.facts), literal(128))
    address = expression(
        "add", quotient, expression("add", expression("mul", child, literal(8)), literal(32))
    )
    parent, relative = split_offset(address, child.args[0], (), attention.facts)
    assert equivalent(parent, quotient, attention.facts)
    assert "floordiv" in parent.text()
    assert arguments(parent) == {"Sq"}
    for length in (128, 384, 512):
        for row in (0, 7):
            values = {"Sq": length, child.args[0]: row}
            assert parent.evaluate(values) == length // 128
            assert relative.evaluate(values) == row * 8 + 32
            assert parent.evaluate(values) + relative.evaluate(values) == address.evaluate(values)


def test_static_view_biases_share_capture_and_arbitrary_child_name_uses_native_grammar(attention):
    parent_coordinate, _ = actual_coordinates(attention)
    child_ssa = attention.children[0].clock.coordinate
    child = from_ssa(child_ssa, coordinates={child_ssa: "chunk"}, facts=attention.facts)
    ring = expression("rem", expression("floordiv", child, literal(2)), literal(2))
    relative_terms = expression(
        "add", expression("mul", child, literal(8)), expression("mul", ring, literal(32))
    )
    base = expression("mul", parent_coordinate, literal(4096))
    results = [
        split_offset(
            expression("add", base, expression("add", relative_terms, literal(bias))),
            "chunk",
            {parent_coordinate.args[0]},
            attention.facts,
        )
        for bias in (0, 64)
    ]
    assert equivalent(results[0][0], results[1][0], attention.facts)
    for index in range(8):
        values = {"chunk": index}
        assert results[0][1].evaluate(values) == index * 8 + ((index // 2) % 2) * 32
        assert results[1][1].evaluate(values) - results[0][1].evaluate(values) == 64


def test_mixed_parent_child_atom_and_runtime_child_coefficient_fail_with_diagnostics(attention):
    parent, child = actual_coordinates(attention)
    mixed = expression("rem", expression("add", parent, child), literal(2))
    with pytest.raises(CompileError, match="mixed parent/child index atom floordiv"):
        split_offset(mixed, child.args[0], {parent.args[0]}, attention.facts)
    sk = next(
        op for op in attention.module.walk() if isinstance(op, SizeOp) and op.sym_name.data == "Sk"
    )
    runtime_coefficient = expression("mul", child, from_ssa(sk.value, facts=attention.facts))
    with pytest.raises(
        CompileError, match="child index coefficient depends on runtime captured variables: Sk"
    ):
        split_offset(runtime_coefficient, child.args[0], {parent.args[0]}, attention.facts)


def test_uncaptured_coordinate_is_rejected_before_algebraic_cancellation(attention):
    parent, child = actual_coordinates(attention)
    canceled = expression("add", child, expression("sub", parent, parent))
    with pytest.raises(CompileError, match="scope address uses uncaptured variables"):
        split_offset(canceled, child.args[0], (), attention.facts)
    with pytest.raises(CompileError, match="child clock cannot also be captured or uniform"):
        split_offset(child, child.args[0], {child.args[0]}, attention.facts)
