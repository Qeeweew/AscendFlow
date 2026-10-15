"""Original-axis views and scope-correct scalar SSA, before physical lowering."""

from collections import Counter

import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp, i64
from xdsl.utils.exceptions import VerifyException

from tests.fixtures.attention_target import attention_target
from xdsl_npu import language as N
from xdsl_npu.frontend import FrontendError
from xdsl_npu.ir import dump, parse
from xdsl_npu.model import CompileError
from xdsl_npu.source_ir import AllocationOp, PipelineOp, ProgramOp, SizeOp, StageOp, ViewOp, YieldOp

BNSD_COPY = """
    def tiled(X: N.Tensor((B,H,S,D),N.f32),
              Y: N.Tensor((B,H,S,D),N.f32,access="write"),
              S: N.int64(minimum=1,multiple_of=R),
              B: N.constexpr=2,H: N.constexpr=3,R: N.constexpr=16,
              D: N.constexpr=64,GROUPS: N.constexpr=20):
        with N.Kernel(GROUPS) as group:
            x=N.alloc_ub((2,R,D))
            tiles=S//R
            work=N.ceildiv(N.max(0,B*H*tiles-group),GROUPS)
            for task in N.Pipelined(work,stages=2):
                flat=group+task*GROUPS
                batch=flat//(H*tiles)
                head=(flat//tiles)%H
                row=(flat%tiles)*R
                with N.Stage(0,name="load"):
                    slot=task%2
                    N.copy(X[batch,head,row:row+R,:],x[slot,:,:])
                with N.Stage(1,name="store"):
                    N.copy(x[slot,:,:],Y[batch,head,row:row+R,:])
"""


def evaluate(value, bindings):
    """Independent integer interpretation used only to inspect source coordinates."""
    if value in bindings:
        return bindings[value]
    op = value.owner
    if isinstance(op, arith.ConstantOp):
        return op.value.value.data
    lhs, rhs = (evaluate(operand, bindings) for operand in op.operands)
    if isinstance(op, arith.AddiOp):
        return lhs + rhs
    if isinstance(op, arith.SubiOp):
        return lhs - rhs
    if isinstance(op, arith.MuliOp):
        return lhs * rhs
    if isinstance(op, arith.DivUIOp):
        assert lhs >= 0 and rhs > 0
        return lhs // rhs
    if isinstance(op, arith.RemUIOp):
        assert lhs >= 0 and rhs > 0
        return lhs % rhs
    if isinstance(op, arith.CeilDivUIOp):
        assert lhs >= 0 and rhs > 0
        return (lhs + rhs - 1) // rhs
    if isinstance(op, arith.MaxSIOp):
        return max(lhs, rhs)
    if isinstance(op, arith.MinSIOp):
        return min(lhs, rhs)
    raise AssertionError(op.name)


def test_runtime_bnsd_coordinates_retain_each_original_axis_and_cover_uneven_work():
    module = parse(dump(N.from_source(BNSD_COPY).to_ir()))
    program = next(op for op in module.walk() if isinstance(op, ProgramOp))
    pipeline = next(op for op in module.walk() if isinstance(op, PipelineOp))
    size = next(op for op in module.walk() if isinstance(op, SizeOp))
    external = {
        op.source.owner.sym_name.data: op
        for op in module.walk()
        if isinstance(op, ViewOp) and op.source.type.memory.data == "GM"
    }
    assert (size.minimum.data, size.multiple.data) == (1, 16)
    assert any(
        isinstance(op, arith.DivUIOp) and not isinstance(op.rhs.owner, arith.ConstantOp)
        for op in module.walk()
    )
    assert any(
        isinstance(op, arith.RemUIOp) and not isinstance(op.rhs.owner, arith.ConstantOp)
        for op in module.walk()
    )
    for view in external.values():
        assert tuple(axis.data for axis in view.axes) == (2, 3)
        assert tuple(width.data for width in view.widths) == (1, 1, 16, 64)
        assert tuple(width.data for width in view.ref.type.shape) == (16, 64)
        assert isinstance(view.source.owner, AllocationOp)
    for length in (16, 32, 80):
        seen = Counter()
        empty_owners = 0
        for group in range(20):
            bindings = {size.value: length, program.body.block.args[0]: group}
            count = evaluate(pipeline.limit, bindings)
            empty_owners += count == 0
            for task in range(count):
                bindings[pipeline.body.block.args[0]] = task
                origin = tuple(evaluate(value, bindings) for value in external["X"].origins)
                assert origin == tuple(evaluate(value, bindings) for value in external["Y"].origins)
                batch, head, row, column = origin
                assert 0 <= batch < 2 and 0 <= head < 3 and column == 0
                assert row % 16 == 0 and 0 <= row <= length - 16
                seen.update((batch, head, r) for r in range(row, row + 16))
        assert seen == Counter(
            (batch, head, row) for batch in range(2) for head in range(3) for row in range(length)
        )
        assert (empty_owners > 0) == (6 * (length // 16) < 20)


def test_index_first_used_in_stage_zero_dominates_later_stages():
    module = parse(dump(N.from_source(BNSD_COPY).to_ir()))
    pipeline = next(op for op in module.walk() if isinstance(op, PipelineOp))
    slot = next(
        op
        for op in pipeline.body.block.ops
        if isinstance(op, arith.RemUIOp) and op.lhs is pipeline.body.block.args[0]
    )
    stages = [op for op in pipeline.body.block.ops if isinstance(op, StageOp)]
    assert list(pipeline.body.block.ops).index(slot) < list(pipeline.body.block.ops).index(
        stages[0]
    )
    local = [
        op for op in module.walk() if isinstance(op, ViewOp) and op.source.type.memory.data == "UB"
    ]
    assert len(local) == 2 and all(view.origins[0] is slot.result for view in local)
    assert not any(
        isinstance(op, (arith.DivSIOp, arith.FloorDivSIOp, arith.RemSIOp)) for op in module.walk()
    )


def test_serial_coordinate_times_uniform_factor_and_shifted_division_use_whole_expression():
    module = parse(
        dump(
            N.from_source("""
                def linearized(n: N.int64(minimum=1),m: N.int64(minimum=1)):
                    with N.Kernel():
                        x=N.alloc_ub((2,64))
                        for outer in N.serial(n):
                            for inner in N.Pipelined(m,stages=1):
                                flat=outer*m+inner+3
                                slot=(flat//m)%2
                                N.fill(x[slot,:],0.0)
            """).to_ir()
        )
    )
    serial = next(op for op in module.walk() if isinstance(op, scf.ForOp))
    pipeline = next(op for op in module.walk() if isinstance(op, PipelineOp))
    sizes = {op.sym_name.data: op.value for op in module.walk() if isinstance(op, SizeOp)}
    view = next(op for op in module.walk() if isinstance(op, ViewOp))
    for n, m in ((1, 1), (3, 2), (5, 7)):
        for outer in range(n):
            for inner in range(m):
                bindings = {
                    sizes["n"]: n,
                    sizes["m"]: m,
                    serial.body.block.args[0]: outer,
                    pipeline.body.block.args[0]: inner,
                }
                assert evaluate(view.origins[0], bindings) == ((outer * m + inner + 3) // m) % 2


def test_static_nested_views_compose_original_axes():
    module = N.from_source("""
        def nested():
            with N.Kernel():
                x=N.alloc_ub((4,64,32))
                N.fill(x[1:3,8:40,:][1,4:20,3:11],0.0)
    """).to_ir()
    view = next(op for op in module.walk() if isinstance(op, ViewOp))
    assert tuple(evaluate(value, {}) for value in view.origins) == (2, 12, 3)
    assert tuple(width.data for width in view.widths) == (1, 16, 8)
    assert tuple(axis.data for axis in view.axes) == (1, 2)
    parse(dump(module))


@pytest.mark.parametrize(
    "view,diagnostic",
    [
        ("x[:16][16:32]", "logical axis"),
        ("x[:16][task:task+8]", "dynamic relative indexing"),
        ("x[-1:15]", "nonnegative"),
        ("x[task//0:task//0+16]", "divisor must be positive"),
        ("x[task%0:task%0+16]", "divisor must be positive"),
        ("x[task*task:task*task+16]", "uniform parameter factor"),
    ],
)
def test_invalid_relative_views_and_divisors_are_rejected(view, diagnostic):
    definition = N.from_source(f"""
        def invalid():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(2,stages=1):
                    N.fill({view},0.0)
    """)
    with pytest.raises(FrontendError, match=diagnostic):
        definition.to_ir()


@pytest.mark.parametrize(
    "shape,view",
    [
        ("(64,)", "x[task*32:task*32+64][:16]"),
        ("(64,)", "x[task*16:task*16+16][4:12]"),
        ("(64,8)", "x[task*16:task*16+16,:][0,:]"),
    ],
)
def test_narrowing_dynamic_parent_cannot_discard_its_original_width(shape, view):
    definition = N.from_source(f"""
        def invalid():
            with N.Kernel():
                x=N.alloc_ub({shape})
                for task in N.Pipelined(2,stages=1):
                    N.fill({view},0.0)
    """)
    with pytest.raises(FrontendError, match="dynamic origin needs a parent range proof"):
        definition.to_ir()


@pytest.mark.parametrize("selection", [":", "0:16"])
def test_dynamic_parent_may_keep_the_same_origin_and_width(selection):
    module = N.from_source(f"""
        def nested():
            with N.Kernel():
                x=N.alloc_ub((64,))
                for task in N.Pipelined(2,stages=1):
                    N.fill(x[task*16:task*16+16][{selection}],0.0)
    """).to_ir()
    pipeline = next(op for op in module.walk() if isinstance(op, PipelineOp))
    view = next(op for op in module.walk() if isinstance(op, ViewOp))
    assert tuple(width.data for width in view.widths) == (16,)
    for task in range(2):
        assert evaluate(view.origins[0], {pipeline.body.block.args[0]: task}) == task * 16
    parse(dump(module))


def test_valid_prefix_is_an_explicit_index_operand_with_verified_provenance():
    module = parse(
        dump(
            N.from_source("""
                def masked(n: N.int64):
                    with N.Kernel():
                        x=N.alloc_ub((64,))
                        y=N.alloc_ub((64,))
                        for task in N.Pipelined(n,stages=1):
                            valid=N.min(64,N.max(0,n-task*64))
                            N.copy(x,y,valid=valid)
            """).to_ir()
        )
    )
    views = [op for op in module.walk() if isinstance(op, ViewOp)]
    assert len(views) == 2 and views[0].valid is views[1].valid
    assert isinstance(views[0].valid.owner, arith.MinSIOp)
    view = views[0]
    cast = arith.IndexCastOp(view.valid, i64)
    back = arith.IndexCastOp(cast.result, IndexType())
    view.parent.insert_op_before(cast, view)
    view.parent.insert_op_before(back, view)
    view.operands = (view.source, *view.origins, back.result)
    with pytest.raises(VerifyException, match="unsupported SSA provenance"):
        module.verify()


def escaped_child_index(count):
    trace = N._Trace("exported", False)
    token = N._current.set(trace)
    try:
        child = N.Pipelined(count, stages=1).op
        trace.enter(child)
        coordinate = N._coordinate(child.body.block.args[0])
        index = ((coordinate + 1) % 2) * 16
        trace.leave(child)
        value = trace.scalar(trace.resolve_index(index))
        trace.add(YieldOp.build(operands=[[]]))
        module = ModuleOp([trace.op])
        module.verify()
        return value, module
    finally:
        N._current.reset(token)


def test_static_child_export_substitutes_the_whole_ssa_expression():
    value, module = escaped_child_index(3)
    assert evaluate(value, {}) == 16
    parse(dump(module))


@pytest.mark.parametrize(
    "count,diagnostic",
    [(0, "zero-trip pipeline"), (N.dynamic("n"), "dynamic child coordinate cannot escape")],
)
def test_child_export_needs_a_static_nonempty_last_iteration(count, diagnostic):
    with pytest.raises(CompileError, match=diagnostic):
        escaped_child_index(count)


def simple_views():
    module = N.from_source("""
        def simple():
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.fill(x,0.0)
                N.fill(x,1.0)
    """).to_ir()
    return module, [op for op in module.walk() if isinstance(op, ViewOp)]


@pytest.mark.parametrize(
    "attribute,values,diagnostic",
    [
        ("widths", (0,), "positive static"),
        ("widths", (32,), "result shape"),
        ("widths", (64, 1), "per original axis"),
        ("axes", (), "ordered distinct"),
        ("axes", (0, 0), "ordered distinct"),
        ("axes", (1,), "ordered distinct"),
    ],
)
def test_forged_view_structure_is_rejected(attribute, values, diagnostic):
    module, views = simple_views()
    views[0].attributes[attribute] = ArrayAttr(IntAttr(value) for value in values)
    with pytest.raises(VerifyException, match=diagnostic):
        module.verify()


def test_view_cannot_hide_its_allocation_behind_another_view():
    module, views = simple_views()
    views[1].operands = (views[0].ref, *views[1].operands[1:])
    with pytest.raises(VerifyException, match="original allocation"):
        module.verify()


def test_integer_indexed_axes_must_have_width_one():
    module = N.from_source(BNSD_COPY).to_ir()
    view = next(
        op for op in module.walk() if isinstance(op, ViewOp) and op.source.type.memory.data == "GM"
    )
    view.attributes["widths"] = ArrayAttr(IntAttr(value) for value in (2, 1, 16, 64))
    with pytest.raises(VerifyException, match="integer-indexed"):
        module.verify()


def test_view_rejects_unsupported_index_producer():
    module, views = simple_views()
    view = views[0]
    cast = arith.IndexCastOp(view.origins[0], i64)
    back = arith.IndexCastOp(cast.result, IndexType())
    view.parent.insert_op_before(cast, view)
    view.parent.insert_op_before(back, view)
    view.operands = (view.source, back.result)
    with pytest.raises(VerifyException, match="unsupported SSA provenance"):
        module.verify()


def test_view_checks_definition_order_within_the_same_block():
    module, views = simple_views()
    view = views[0]
    later = arith.ConstantOp.from_int_and_width(0, IndexType())
    view.parent.insert_op_after(later, view)
    view.operands = (view.source, later.result)
    with pytest.raises(VerifyException, match="must precede"):
        module.verify()


def test_view_checks_scope_dominance_across_stages():
    module = N.from_source(BNSD_COPY).to_ir()
    stages = [op for op in module.walk() if isinstance(op, StageOp)]
    value = arith.ConstantOp.from_int_and_width(0, IndexType())
    stages[0].body.block.insert_op_before(value, stages[0].body.block.last_op)
    view = next(op for op in stages[1].walk() if isinstance(op, ViewOp))
    view.operands = (view.source, value.result, *view.origins[1:])
    with pytest.raises(VerifyException, match="does not dominate"):
        module.verify()


@pytest.mark.parametrize("continuous", [False, True])
def test_real_attention_math_and_descriptors_reach_verified_source_ir(continuous):
    from xdsl_npu.source_ir import ElementOp, MatmulOp, ReduceOp

    source = attention_target(continuous=continuous).to_ir()
    source.verify()
    assert sum(isinstance(op, MatmulOp) for op in source.walk()) == 2
    assert sum(isinstance(op, ReduceOp) for op in source.walk()) == 2
    assert sum(isinstance(op, PipelineOp) for op in source.walk()) == 5
    assert any(
        isinstance(op, ElementOp)
        and op.kind.data == "fill"
        and op.scalar.value.data == float("-inf")
        for op in source.walk()
    )
