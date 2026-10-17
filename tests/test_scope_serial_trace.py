"""Source-coordinate and numeric oracles, without a hardware scheduling claim."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest
from xdsl.dialects import arith, scf

from ascendflow.dialects.io import dump, parse
from ascendflow.dialects.source import (
    AllocationOp,
    CopyOp,
    DomainOp,
    ElementOp,
    MatmulOp,
    PipelineOp,
    PredicateOp,
    ProgramOp,
    SizeOp,
    StageOp,
    ViewOp,
    YieldOp,
)
from ascendflow.target.native import ITEMSIZE
from tests.fixtures.scoped_queries import GROUPS, scoped_query_recurrence
from tests.test_index_frontend import evaluate as evaluate_index


def view(ref):
    assert isinstance(ref.owner, ViewOp)
    return ref.owner


def root(ref):
    allocation = view(ref).source.owner
    assert isinstance(allocation, AllocationOp)
    return allocation


def scalar(value, bindings):
    op = value.owner
    if isinstance(op, arith.CmpiOp):
        assert op.predicate.value.data == 0  # The source uses equality guards.
        return evaluate_index(op.lhs, bindings) == evaluate_index(op.rhs, bindings)
    if isinstance(op, PredicateOp):

        def side(terms, offset):
            result = offset.data
            for term in terms:
                index, coefficient, divisor, modulus = (part.data for part in term)
                coordinate = evaluate_index(op.indices[index], bindings) // divisor
                result += coefficient * (coordinate % modulus if modulus else coordinate)
            return result

        lhs = side(op.lhs_terms, op.lhs_offset)
        rhs = side(op.rhs_terms, op.rhs_offset)
        return {"eq": lhs == rhs, "lt": lhs < rhs, "le": lhs <= rhs}[op.comparison.data]
    return evaluate_index(value, bindings)


@dataclass
class QuerySource:
    program: ProgramOp
    serial: scf.ForOp
    kv: PipelineOp
    cube: PipelineOp
    vector: PipelineOp
    domains: dict[str, DomainOp]
    allocations: dict[str, AllocationOp]
    sizes: dict[str, SizeOp]
    copies: dict[tuple[str, str], CopyOp]

    def bindings(self, total, kv_tiles, group=0, query=0, kv=0, child=0, lane=0):
        return {
            self.sizes["total_queries"].value: total,
            self.sizes["kv_tiles"].value: kv_tiles,
            self.program.body.block.args[0]: group,
            self.serial.body.block.args[0]: query,
            self.kv.body.block.args[0]: kv,
            self.cube.body.block.args[0]: child,
            self.vector.body.block.args[0]: child,
            self.domains["cube"].body.block.args[0]: 0,
            self.domains["vector"].body.block.args[0]: lane,
        }

    def shape(self, allocation, bindings):
        dynamic = {
            axis.data: scalar(dimension, bindings)
            for axis, dimension in zip(allocation.dynamic_axes, allocation.dimensions, strict=True)
        }
        return tuple(
            dynamic[axis] if extent.data == -1 else extent.data
            for axis, extent in enumerate(allocation.ref.type.shape)
        )

    def origins(self, ref, bindings):
        return tuple(scalar(origin, bindings) for origin in view(ref).origins)

    def selection(self, ref, bindings):
        tile = view(ref)
        origins = self.origins(ref, bindings)
        widths = tuple(width.data for width in tile.widths)
        shape = self.shape(root(ref), bindings)
        assert all(
            0 <= origin and origin + width <= extent
            for origin, width, extent in zip(origins, widths, shape, strict=True)
        )
        axes = {axis.data for axis in tile.axes}
        return tuple(
            slice(origin, origin + width) if axis in axes else origin
            for axis, (origin, width) in enumerate(zip(origins, widths, strict=True))
        )

    def bytes(self, ref, bindings):
        # Flatten original-axis cells directly; do not use the lowering's index algebra.
        self.selection(ref, bindings)
        shape = self.shape(root(ref), bindings)
        origins = self.origins(ref, bindings)
        widths = tuple(width.data for width in view(ref).widths)
        itemsize = ITEMSIZE[ref.type.dtype.data]
        result = set()
        for cell in product(*(range(o, o + w) for o, w in zip(origins, widths, strict=True))):
            offset = 0
            for coordinate, extent in zip(cell, shape, strict=True):
                offset = offset * extent + coordinate
            result.update(range(offset * itemsize, (offset + 1) * itemsize))
        return result


@pytest.fixture(scope="module")
def source():
    module = parse(dump(scoped_query_recurrence.to_ir()))
    module.verify()
    program = next(op for op in module.walk() if isinstance(op, ProgramOp))
    serial = next(op for op in program.walk() if isinstance(op, scf.ForOp))
    domains = {op.kind.data: op for op in program.walk() if isinstance(op, DomainOp)}
    children = {
        kind: next(op for op in domain.walk() if isinstance(op, PipelineOp))
        for kind, domain in domains.items()
    }
    return QuerySource(
        program,
        serial,
        next(op for op in serial.body.block.ops if isinstance(op, PipelineOp)),
        children["cube"],
        children["vector"],
        domains,
        {op.sym_name.data: op for op in program.walk() if isinstance(op, AllocationOp)},
        {op.sym_name.data: op for op in program.walk() if isinstance(op, SizeOp)},
        {
            (root(op.source).sym_name.data, root(op.destination).sym_name.data): op
            for op in program.walk()
            if isinstance(op, CopyOp)
        },
    )


def test_runtime_contract_and_reused_storage_are_in_the_actual_source(source):
    assert source.program.kind.data == "mixed" and source.program.groups.data == GROUPS
    assert source.sizes["total_queries"].minimum.data == 0
    assert source.sizes["kv_tiles"].minimum.data == 1
    assert {
        name: allocation.ref.type.dtype.data for name, allocation in source.allocations.items()
    } == {
        "Q": "f16",
        "K": "f16",
        "Score": "f32",
        "O": "f32",
        "q_l1": "f16",
        "k_l1": "f16",
        "a": "f16",
        "b": "f16",
        "c": "f32",
        "loaded": "f32",
        "state": "f32",
    }
    assert all(
        allocation.parent_op() is source.program for allocation in source.allocations.values()
    )
    assert not any(isinstance(op, AllocationOp) for op in source.serial.walk())
    assert source.kv.limit is source.sizes["kv_tiles"].value
    assert source.cube.count.data == source.vector.count.data == 2
    assert (source.kv.stages.data, source.cube.stages.data, source.vector.stages.data) == (2, 4, 3)


@pytest.mark.parametrize("total", range(8))
@pytest.mark.parametrize("kv_tiles", (1, 2, 3))
def test_source_dispatch_and_score_epoch_continue_across_unequal_queries(source, total, kv_tiles):
    q = source.copies["Q", "q_l1"].source
    k = source.copies["K", "k_l1"].source
    score = source.copies["c", "Score"].destination
    output = source.copies["state", "O"].destination
    slot = view(score).origins[1]
    assert isinstance(slot.owner, arith.RemUIOp)
    seen = Counter()
    for group in range(GROUPS):
        bindings = source.bindings(total, kv_tiles, group)
        count = scalar(source.serial.ub, bindings)
        assert count == len(range(group, total, GROUPS))
        epochs, first_slots = [], []
        for query in range(count):
            flat = group + query * GROUPS
            seen[flat] += 1
            for kv in range(kv_tiles):
                bindings = source.bindings(total, kv_tiles, group, query, kv)
                assert source.origins(q, bindings) == (flat, 0, 0, 0)
                assert source.origins(k, bindings) == (kv, 0, 0, 0)
                assert source.origins(output, bindings) == (flat, 0, 0)
                task = scalar(slot.owner.lhs, bindings)
                epochs.append(task)
                assert source.origins(score, bindings) == (group, task % 2, 0, 0)
                if kv == 0:
                    first_slots.append(scalar(slot, bindings))
        assert epochs == list(range(count * kv_tiles))
        assert first_slots == [query % 2 if kv_tiles % 2 else 0 for query in range(count)]
    assert seen == Counter(range(total))


@pytest.mark.parametrize("kv_tiles", (1, 2, 3))
def test_source_physical_bytes_partition_score_and_output(source, kv_tiles):
    total = 7
    export = source.copies["c", "Score"].destination
    load = source.copies["Score", "loaded"].source
    output = source.copies["state", "O"].destination
    q = source.copies["Q", "q_l1"].source
    k = source.copies["K", "k_l1"].source
    score_shape = source.shape(source.allocations["Score"], source.bindings(total, kv_tiles))
    assert score_shape[-1] * ITEMSIZE["f32"] == 128
    assert view(export).widths.data[-1].data * ITEMSIZE["f32"] == 64
    all_output = set()
    owner_slots = {}
    for group in range(GROUPS):
        for query, flat in enumerate(range(group, total, GROUPS)):
            for kv in range(kv_tiles):
                bindings = source.bindings(total, kv_tiles, group, query, kv)
                slot = source.origins(export, bindings)[1]
                cube = []
                for child in range(2):
                    bindings = source.bindings(total, kv_tiles, group, query, kv, child)
                    cube.append(source.bytes(export, bindings))
                    q_base, k_base = (flat * 2 + child) * 512, (kv * 2 + child) * 512
                    assert source.bytes(q, bindings) == set(range(q_base, q_base + 512))
                    assert source.bytes(k, bindings) == set(range(k_base, k_base + 512))
                bands = [
                    source.bytes(load, source.bindings(total, kv_tiles, group, query, kv, r, lane))
                    for lane in range(2)
                    for r in range(2)
                ]
                score_base = (group * 2 + slot) * 2048
                expected = set(range(score_base, score_base + 2048))
                assert cube[0].isdisjoint(cube[1])
                assert all(a.isdisjoint(b) for a, b in product(bands, repeat=2) if a is not b)
                assert set.union(*cube) == set.union(*bands) == expected
                owner_slots[group, slot] = expected
            output_bands = [
                source.bytes(
                    output, source.bindings(total, kv_tiles, group, query, kv_tiles - 1, r, lane)
                )
                for lane in range(2)
                for r in range(2)
            ]
            written = set.union(*output_bands)
            assert sum(map(len, output_bands)) == len(written) == 2048
            assert written == set(range(flat * 2048, (flat + 1) * 2048))
            assert all_output.isdisjoint(written)
            all_output.update(written)
    assert all_output == set(range(total * 2048))
    assert all(
        a.isdisjoint(b)
        for key_a, a in owner_slots.items()
        for key_b, b in owner_slots.items()
        if key_a != key_b
    )


class SequentialSource:
    """Interpret source effects in order; pipeline scheduling and events are unmodeled."""

    def __init__(self, source, total, kv_tiles, q, k):
        self.source = source
        self.bindings = source.bindings(total, kv_tiles)
        self.memory = {
            "Q": q,
            "K": k,
            "Score": np.full((GROUPS, 2, 16, 32), np.nan, np.float32),
            "O": np.full((total, 16, 32), -123.5, np.float32),
        }
        self.locals = {}
        self.output_writes = Counter()

    def tile(self, ref, actor):
        allocation = root(ref)
        name = allocation.sym_name.data
        if allocation.ref.type.memory.data == "GM":
            array = self.memory[name]
        else:
            assert actor is not None
            key = (self.bindings[self.source.program.body.block.args[0]], actor, allocation)
            if key not in self.locals:
                dtype = {"f16": np.float16, "f32": np.float32}[allocation.ref.type.dtype.data]
                self.locals[key] = np.full(
                    self.source.shape(allocation, self.bindings), np.nan, dtype
                )
            array = self.locals[key]
        return array[self.source.selection(ref, self.bindings)]

    def block(self, block, actor=None):
        for op in block.ops:
            if isinstance(op, scf.ForOp):
                for index in range(*(scalar(v, self.bindings) for v in (op.lb, op.ub, op.step))):
                    self.bindings[op.body.block.args[0]] = index
                    self.block(op.body.block, actor)
            elif isinstance(op, PipelineOp):
                count = op.count.data if op.count is not None else scalar(op.limit, self.bindings)
                for index in range(count):
                    self.bindings[op.body.block.args[0]] = index
                    self.block(op.body.block, actor)
            elif isinstance(op, StageOp):
                self.block(op.body.block, actor)
            elif isinstance(op, DomainOp):
                for lane in range(2 if op.kind.data == "vector" else 1):
                    self.bindings[op.body.block.args[0]] = lane
                    self.block(op.body.block, (op.kind.data, lane))
            elif isinstance(op, scf.IfOp):
                region = op.true_region if scalar(op.cond, self.bindings) else op.false_region
                self.block(region.block, actor)
            elif isinstance(op, CopyOp):
                value = self.tile(op.source, actor)
                if op.transpose is not None and op.transpose.data:
                    value = value.T
                np.copyto(self.tile(op.destination, actor), value)
                if root(op.destination).sym_name.data == "O":
                    self.output_writes[self.source.origins(op.destination, self.bindings)] += 1
            elif isinstance(op, MatmulOp):
                a, b = (self.tile(ref, actor).astype(np.float32) for ref in (op.a, op.b))
                value = a @ (b.T if op.transpose_b.data else b)
                destination = self.tile(op.accumulator, actor)
                if not scalar(op.initialize, self.bindings):
                    value = np.add(destination, value, dtype=np.float32)
                np.copyto(destination, value)
            elif isinstance(op, ElementOp):
                destination = self.tile(op.destination, actor)
                inputs = [self.tile(ref, actor) for ref in op.inputs]
                if op.kind.data == "fill":
                    destination.fill(op.scalar.value.data)
                elif op.kind.data == "add":
                    np.add(*inputs, out=destination, dtype=np.float32)
                else:
                    assert op.kind.data == "mul_scalar"
                    np.multiply(
                        inputs[0],
                        np.float32(op.scalar.value.data),
                        out=destination,
                        dtype=np.float32,
                    )
            else:
                assert isinstance(
                    op, (AllocationOp, SizeOp, ViewOp, PredicateOp, YieldOp, scf.YieldOp)
                ) or op.name.startswith("arith.")

    def run(self):
        for group in range(self.source.program.groups.data):
            self.bindings[self.source.program.body.block.args[0]] = group
            self.block(self.source.program.body.block)
        return self.memory["O"]


@pytest.mark.parametrize("total", range(8))
@pytest.mark.parametrize("kv_tiles", (1, 2, 3))
def test_source_recurrence_resets_each_query_with_float16_inputs_and_float32_state(
    source, total, kv_tiles
):
    rng = np.random.default_rng(7900 + total * 10 + kv_tiles)
    q = rng.normal(size=(total, 2, 16, 16)).astype(np.float16)
    k = rng.normal(size=(kv_tiles, 2, 16, 16)).astype(np.float16)
    expected = np.empty((total, 16, 32), np.float32)
    for flat in range(total):
        state = np.zeros((16, 32), np.float32)
        for kv in range(kv_tiles):
            score = np.concatenate(
                [q[flat, j].astype(np.float32) @ k[kv, j].astype(np.float32).T for j in range(2)],
                axis=1,
            )
            state = np.multiply(
                np.add(state, score, dtype=np.float32), np.float32(0.5), dtype=np.float32
            )
        expected[flat] = state
    execution = SequentialSource(source, total, kv_tiles, q, k)
    actual = execution.run()
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert execution.output_writes == Counter(
        {(flat, row, 0): 1 for flat in range(total) for row in (0, 4, 8, 12)}
    )
    if total == 0:
        assert not execution.locals
        assert np.isnan(execution.memory["Score"]).all()
