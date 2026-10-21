"""Actual SSA address facts, with explicit domains and no access admission."""

from dataclasses import FrozenInstanceError

import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr
from xdsl.ir import Block

from ascendflow import indexing, scalar
from ascendflow.analysis.structured_addresses import ScalarEnvironment, StructuredAddresses
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.physical import BufferOp
from ascendflow.dialects.structured import AddressOp
from ascendflow.dialects.types import ref_type
from ascendflow.scalar import Expr, Fact, expression, literal
from ascendflow.target.native import NativeGeometry
from tests.support.structured import index_constant


def _binary(block, operation, left, right):
    result = operation(left, right)
    block.add_op(result)
    return result.result


def _buffer(block, *, bank="UB", name="shared_name", dtype="f32", layout="linear"):
    operation = BufferOp.build(
        operands=[None],
        attributes={
            "sym_name": StringAttr(name),
            "nbytes": IntAttr(16384),
            "external": IntAttr(int(bank == "GM")),
            "initialized": IntAttr(int(bank == "GM")),
        },
        result_types=[ref_type(bank, dtype, layout)],
    )
    block.add_op(operation)
    return operation


def _address(
    block,
    root,
    offset,
    *,
    nbytes=256,
    rows=1,
    row_stride=0,
    origins=(),
    extents=(),
    widths=None,
    axes=None,
    displacement=None,
):
    attributes = {
        "nbytes": IntAttr(nbytes),
        "rows": IntAttr(rows),
        "row_stride": IntAttr(row_stride),
    }
    if widths is not None:
        attributes["widths"] = ArrayAttr(IntAttr(width) for width in widths)
    if axes is not None:
        attributes["axes"] = ArrayAttr(IntAttr(axis) for axis in axes)
    if displacement is not None:
        attributes["displacement"] = IntAttr(displacement)
    operation = AddressOp.build(
        operands=[root, offset, origins, extents, None],
        attributes=attributes,
        result_types=[root.type],
    )
    block.add_op(operation)
    return operation


def _decoder(bindings=None, facts=None):
    return StructuredAddresses(ScalarEnvironment(bindings or {}, facts or {}))


def test_multiple_ancestor_clocks_and_uniform_radix_remain_one_actual_address(monkeypatch):
    block = Block(arg_types=[IndexType()] * 4)
    n, m, outer, inner = block.args
    expressions = tuple(Expr("argument", (name,)) for name in ("n", "m", "$outer", "$inner"))
    bindings = dict(zip(block.args, expressions, strict=True))
    facts = {"n": Fact(minimum=1), "m": Fact(minimum=1)}
    decoder = _decoder(bindings, facts)
    root = _buffer(block, bank="GM")
    zero, width, rowbytes = (index_constant(block, value) for value in (0, 64, 256))
    linear = _binary(block, arith.AddiOp, _binary(block, arith.MuliOp, outer, m), inner)
    offset = _binary(block, arith.MuliOp, linear, rowbytes)
    address = _address(
        block,
        root.ref,
        offset,
        origins=(outer, inner, zero),
        extents=(n, m, width),
        widths=(1, 1, 64),
        axes=(2,),
        displacement=0,
    )

    def no_flatten(*args, **kwargs):
        raise AssertionError("structured addresses must not flatten logical coordinates")

    monkeypatch.setattr(indexing, "physical_terms", no_flatten)
    result = decoder.decode(address.ref)
    assert result.operation is address and result.root is root.ref
    assert result.geometry == NativeGeometry("GM", "f32", "linear", 256)
    assert result.axes.origins == (expressions[2], expressions[3], literal(0))
    assert result.axes.extents == (expressions[0], expressions[1], literal(64))
    assert result.axes.widths == (1, 1, 64) and result.axes.axes == (2,)
    assert result.axes.displacement == 0
    assert indexing.arguments(result.offset) == {"$outer", "$inner", "m"}
    for runtime in ((3, 5, 2, 4), (100_000, 1024, 99_999, 1023)):
        values = dict(zip(("n", "m", "$outer", "$inner"), runtime, strict=True))
        assert result.offset.evaluate(values) == (runtime[2] * runtime[1] + runtime[3]) * 256
    decoder.check_axes(
        result,
        (
            (expressions[2], literal(0), expression("sub", expressions[0], literal(1))),
            (expressions[3], literal(0), expression("sub", expressions[1], literal(1))),
        ),
    )
    assert decoder.decode(address.ref) is result and decoder.decode_count == 1
    with pytest.raises(FrozenInstanceError):
        result.root = root.ref


def test_shared_ssa_memo_and_injected_logical_identities_are_reused(monkeypatch):
    block = Block(arg_types=[IndexType()] * 4)
    stage0, stage1, domain0, domain1 = block.args
    logical = Expr("argument", ("$logical",))
    sid = Expr("argument", ("$sid",))
    env = ScalarEnvironment({stage0: logical, stage1: logical, domain0: sid, domain1: sid}, {})
    one = index_constant(block, 1)
    shared = _binary(block, arith.AddiOp, stage0, one)
    calls = []
    original = scalar.expression

    def tracked(kind, left, right):
        calls.append(kind)
        return original(kind, left, right)

    monkeypatch.setattr(scalar, "expression", tracked)
    first = env.expression(shared)
    assert env.expression(shared) is first
    assert calls == ["add"]
    assert env.expression(stage0) is env.expression(stage1) is logical
    assert env.expression(domain0) is env.expression(domain1) is sid
    decoder = StructuredAddresses(env)
    root = _buffer(block)
    first_address = _address(block, root.ref, shared)
    second_address = _address(block, root.ref, shared)
    assert decoder.decode(first_address.ref).offset is decoder.decode(second_address.ref).offset
    assert decoder.decode_count == 2 and calls == ["add"]


def test_scalar_environment_copies_bindings_and_exposes_readonly_uniform_facts():
    block = Block(arg_types=[IndexType()])
    value = block.args[0]
    symbol = Expr("argument", ("n",))
    bindings, facts = {value: symbol}, {"n": Fact(minimum=1)}
    env = ScalarEnvironment(bindings, facts)
    bindings[value] = literal(7)
    facts["n"] = Fact(minimum=0)
    assert env.expression(value) is symbol
    assert env.facts["n"].minimum == 1
    with pytest.raises(TypeError):
        env.facts["n"] = Fact()
    with pytest.raises(AttributeError):
        env.facts = {}


def test_decode_preserves_rows_and_subview_metadata_without_proving_bounds():
    block = Block()
    root = _buffer(block, dtype="f16", layout="NZ")
    zero, extent, offset = (index_constant(block, value) for value in (0, 16, 20000))
    operation = _address(
        block,
        root.ref,
        offset,
        nbytes=32,
        rows=16,
        row_stride=64,
        origins=(zero,),
        extents=(extent,),
        widths=(16,),
        axes=(0,),
        displacement=20000,
    )
    decoder = _decoder()
    address = decoder.decode(operation.ref)
    assert address.geometry == NativeGeometry("UB", "f16", "NZ", 32, 16, 64)
    assert not address.geometry.contiguous and address.geometry.total_nbytes == 512
    assert address.axes.displacement == address.offset.constant == 20000
    # Original-axis bounds alone say nothing about this out-of-allocation displacement.
    decoder.check_axes(address, ())


def test_absent_axis_metadata_decodes_but_cannot_supply_axis_admission():
    block = Block()
    root = _buffer(block)
    operation = _address(block, root.ref, index_constant(block, 0))
    decoder = _decoder()
    address = decoder.decode(operation.ref)
    assert address.axes is None
    with pytest.raises(CompileError, match="no original-axis metadata") as error:
        decoder.check_axes(address, ())
    assert error.value.kind == ErrorKind.UNSUPPORTED


def test_axis_check_uses_retained_projected_away_axes_and_reports_witness():
    block = Block()
    root = _buffer(block)
    zero, three, four, width = (index_constant(block, value) for value in (0, 3, 4, 64))
    operation = _address(
        block,
        root.ref,
        zero,
        origins=(four, zero),
        extents=(three, width),
        widths=(1, 64),
        axes=(1,),
        displacement=0,
    )
    decoder = _decoder()
    address = decoder.decode(operation.ref)  # Decoding is not a bounds proof.
    with pytest.raises(CompileError, match="not bounded") as error:
        decoder.check_axes(address, ())
    assert error.value.kind == ErrorKind.INVALID
    assert error.value.phase == "structured-addresses"
    assert error.value.witness == ("axis=0", "origin=4", "width=1", "extent=3")


@pytest.mark.parametrize("origin,extent,width", [(-1, 8, 1), (8, 8, 1), (0, 8, 9)])
def test_axis_failure_classification_keeps_explicit_constant_violations_invalid(
    origin, extent, width
):
    block = Block()
    root = _buffer(block)
    zero, start, size = (index_constant(block, value) for value in (0, origin, extent))
    operation = _address(
        block,
        root.ref,
        zero,
        origins=(start,),
        extents=(size,),
        widths=(width,),
        axes=(0,),
        displacement=0,
    )
    decoder = _decoder()
    address = decoder.decode(operation.ref)
    with pytest.raises(CompileError) as error:
        decoder.check_axes(address, ())
    assert error.value.kind == ErrorKind.INVALID
    assert error.value.witness == (
        "axis=0",
        f"origin={origin}",
        f"width={width}",
        f"extent={extent}",
    )


def test_axis_failure_classification_does_not_call_an_unknown_origin_a_counterexample():
    block = Block(arg_types=[IndexType()])
    origin = block.args[0]
    root = _buffer(block)
    zero, extent = (index_constant(block, value) for value in (0, 8))
    operation = _address(
        block,
        root.ref,
        zero,
        origins=(origin,),
        extents=(extent,),
        widths=(1,),
        axes=(0,),
        displacement=0,
    )
    symbol = Expr("argument", ("$unknown",))
    decoder = _decoder({origin: symbol})
    address = decoder.decode(operation.ref)
    with pytest.raises(CompileError, match="origin may be negative") as error:
        decoder.check_axes(address, ())
    assert error.value.kind == ErrorKind.UNSUPPORTED
    assert error.value.witness == ("axis=0", "origin=$unknown", "width=1", "extent=8")


def test_axis_failure_classification_marks_a_missing_cyclic_joint_range_unsupported():
    block = Block(arg_types=[IndexType()] * 3)
    group, ordinal, size = block.args
    symbols = tuple(Expr("argument", (name,)) for name in ("$group", "$loop0", "Sq"))
    decoder = _decoder(
        dict(zip(block.args, symbols, strict=True)),
        {"$group": Fact(maximum=19), "Sq": Fact(minimum=128, multiple=128)},
    )
    root = _buffer(block)
    zero, one, eight, groups, tile = (index_constant(block, value) for value in (0, 1, 8, 20, 128))
    tiles = _binary(block, arith.DivUIOp, size, tile)
    divisor = _binary(block, arith.MuliOp, eight, tiles)
    cyclic = _binary(block, arith.AddiOp, group, _binary(block, arith.MuliOp, groups, ordinal))
    origin = _binary(block, arith.DivUIOp, cyclic, divisor)
    operation = _address(
        block,
        root.ref,
        zero,
        origins=(origin,),
        extents=(one,),
        widths=(1,),
        axes=(0,),
        displacement=0,
    )
    address = decoder.decode(operation.ref)
    total = decoder.environment.expression(divisor)
    count = expression("ceildiv", total, literal(20))
    # A real cyclic activation supplies a joint range for group + 20*g.
    # Separate group/g intervals alone do not supply that missing premise.
    domains = (
        (symbols[0], literal(0), literal(19)),
        (symbols[1], literal(0), expression("sub", count, literal(1))),
    )
    with pytest.raises(CompileError, match="not bounded by its tensor extent") as error:
        decoder.check_axes(address, domains)
    assert error.value.kind == ErrorKind.UNSUPPORTED
    assert error.value.witness == (
        "axis=0",
        f"origin={address.axes.origins[0].text()}",
        "width=1",
        "extent=1",
    )


def test_axis_failure_classification_preserves_an_explicit_analysis_limit(monkeypatch):
    from ascendflow.analysis import structured_addresses

    block = Block()
    root = _buffer(block)
    zero, extent = (index_constant(block, value) for value in (0, 8))
    operation = _address(
        block,
        root.ref,
        zero,
        origins=(zero,),
        extents=(extent,),
        widths=(1,),
        axes=(0,),
        displacement=0,
    )
    decoder = _decoder()
    address = decoder.decode(operation.ref)

    def limited(*args, **kwargs):
        raise CompileError("algebra limit", kind=ErrorKind.ANALYSIS_LIMIT, witness=("budget",))

    monkeypatch.setattr(structured_addresses, "check_axis", limited)
    with pytest.raises(CompileError, match="algebra limit") as error:
        decoder.check_axes(address, ())
    assert error.value.kind == ErrorKind.ANALYSIS_LIMIT
    assert error.value.witness == ("budget", "axis=0", "origin=0", "width=1", "extent=8")


def test_unsigned_dividend_is_checked_before_axis_recombination():
    block = Block(arg_types=[IndexType()])
    g = block.args[0]
    symbol = Expr("argument", ("$g",))
    decoder = _decoder({g: symbol})
    root = _buffer(block)
    zero, three, five, extent = (index_constant(block, value) for value in (0, 3, 5, 8))
    dividend = _binary(block, arith.SubiOp, g, five)
    quotient = _binary(block, arith.DivUIOp, dividend, three)
    remainder = _binary(block, arith.RemUIOp, dividend, three)
    recombined = _binary(
        block, arith.AddiOp, _binary(block, arith.MuliOp, quotient, three), remainder
    )
    operation = _address(
        block,
        root.ref,
        zero,
        origins=(recombined,),
        extents=(extent,),
        widths=(1,),
        axes=(0,),
        displacement=0,
    )
    address = decoder.decode(operation.ref)
    with pytest.raises(CompileError, match="nonnegative dividend"):
        decoder.check_axes(address, ((symbol, literal(0), literal(7)),))


@pytest.mark.parametrize(
    "binding,facts",
    [
        (Expr("argument", ("n",)), {"n": Fact(minimum=0)}),
        (Expr("argument", ("$g",)), {}),
    ],
)
def test_scalar_divisors_need_a_positive_uniform_contract(binding, facts):
    block = Block(arg_types=[IndexType()])
    divisor = block.args[0]
    quotient = _binary(block, arith.DivUIOp, index_constant(block, 64), divisor)
    env = ScalarEnvironment({divisor: binding}, facts)
    with pytest.raises(CompileError, match="uniform and provably positive"):
        env.expression(quotient)


def test_native_operand_must_be_an_actual_address_and_original_buffer_root():
    block = Block()
    root = _buffer(block)
    decoder = _decoder()
    with pytest.raises(CompileError, match="actual AddressOp result") as error:
        decoder.decode(root.ref)
    assert error.value.kind == ErrorKind.UNSUPPORTED
    zero = index_constant(block, 0)
    first = _address(block, root.ref, zero)
    nested = _address(block, first.ref, zero)
    with pytest.raises(CompileError, match="original BufferOp allocation root") as error:
        decoder.decode(nested.ref)
    assert error.value.kind == ErrorKind.INVALID and decoder.decode_count == 0


def test_incomplete_axis_metadata_is_rejected_even_without_prior_ir_verification():
    block = Block()
    root = _buffer(block)
    zero = index_constant(block, 0)
    operation = _address(block, root.ref, zero, origins=(zero,))
    with pytest.raises(CompileError, match="metadata must be complete"):
        _decoder().decode(operation.ref)


@pytest.mark.parametrize("bank,expected", [("UB", True), ("GM", False)])
def test_ub_disjointness_uses_actual_roots_even_when_names_match(bank, expected):
    block = Block()
    roots = (_buffer(block, bank=bank), _buffer(block, bank=bank))
    assert roots[0].sym_name == roots[1].sym_name
    zero = index_constant(block, 0)
    decoder = _decoder()
    left, right = (decoder.decode(_address(block, root.ref, zero).ref) for root in roots)
    assert decoder.disjoint(left, right) is expected


@pytest.mark.parametrize("distance,expected", [(0, False), (128, False), (256, True), (-256, True)])
def test_same_root_copy_alias_proof_keeps_all_clocks_and_uniform_terms(distance, expected):
    block = Block(arg_types=[IndexType()] * 3)
    outer, inner, radix = block.args
    bindings = {
        value: Expr("argument", (name,))
        for value, name in zip(block.args, ("$outer", "$inner", "n"), strict=True)
    }
    decoder = _decoder(bindings, {"n": Fact(minimum=1)})
    root = _buffer(block)
    width = index_constant(block, 256)
    base = _binary(
        block,
        arith.MuliOp,
        _binary(block, arith.AddiOp, _binary(block, arith.MuliOp, outer, radix), inner),
        width,
    )
    offset = _binary(block, arith.AddiOp, base, index_constant(block, distance))
    left = decoder.decode(_address(block, root.ref, base).ref)
    right = decoder.decode(_address(block, root.ref, offset).ref)
    assert decoder.disjoint(left, right) is expected


def test_distinct_offset_ssa_and_unknown_changing_aliases_do_not_prove_disjointness():
    block = Block(arg_types=[IndexType()] * 2)
    g, h = block.args
    clock = Expr("argument", ("$g",))
    decoder = _decoder({g: clock, h: Expr("argument", ("$h",))})
    root = _buffer(block)
    zero = index_constant(block, 0)
    same = _binary(block, arith.AddiOp, g, zero)
    left = decoder.decode(_address(block, root.ref, g).ref)
    equal = decoder.decode(_address(block, root.ref, same).ref)
    changing = decoder.decode(_address(block, root.ref, h).ref)
    assert g is not same and not decoder.disjoint(left, equal)
    assert not decoder.disjoint(left, changing)


@pytest.mark.parametrize(
    "rows,row_stride,distance,expected",
    [
        (1, 0, 64, False),
        (1, 0, 128, True),
        (2, 64, 1024, False),
    ],
)
def test_same_root_alias_proof_requires_contiguity_and_the_larger_span(
    rows, row_stride, distance, expected
):
    block = Block()
    root = _buffer(block)
    decoder = _decoder()
    left = decoder.decode(_address(block, root.ref, index_constant(block, 0), nbytes=128).ref)
    right = decoder.decode(
        _address(
            block,
            root.ref,
            index_constant(block, distance),
            nbytes=32,
            rows=rows,
            row_stride=row_stride,
        ).ref
    )
    assert decoder.disjoint(left, right) is expected
