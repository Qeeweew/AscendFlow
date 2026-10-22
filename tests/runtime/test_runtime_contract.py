"""Caller input preconditions through IR and CPU execution of thin launch wrappers.

The CPU library below contains the actual emitted launcher. Its device launch
line alone is replaced by a counter; this tests ABI forwarding, not NPU work.
"""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as T
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import RuntimeSizeOp
from ascendflow.dialects.source import SizeOp
from ascendflow.frontend import FrontendError
from examples.vector.scale import scale
from tests.fixtures.source_programs import cv_source
from tests.fixtures.state import grouped_state_sum
from tests.support.host_launcher import contracts, host_launcher


def contract_source(*, minimum=1, multiple=4):
    return T.from_source(
        """
        def guarded(X:T.Tensor((n,B),T.f32),Y:T.Tensor((n,B),T.f32,access="write"),
                    n:T.i64(minimum=MINIMUM,multiple_of=MULTIPLE),
                    B:T.constexpr=64, MINIMUM:T.constexpr=1, MULTIPLE:T.constexpr=4):
            with T.Kernel():
                x=T.alloc_ub((2,B))
                y=T.alloc_ub((2,B))
                for i in T.Pipelined(n):
                    with T.Stage(0):
                        T.copy(X[i,:],x[i%2,:])
                    with T.Stage(1):
                        T.mul(x[i%2,:],2.0,out=y[i%2,:])
                    with T.Stage(2):
                        T.copy(y[i%2,:],Y[i,:])
""",
        disjoint_external=True,
    ).specialize(MINIMUM=minimum, MULTIPLE=multiple)


@pytest.fixture(scope="module")
def local():
    return compile(contract_source(minimum=3, multiple=4))


@pytest.fixture(scope="module")
def sequence():
    definition = replace(
        grouped_state_sum,
        source=grouped_state_sum.source.replace(
            "outputs: T.i64", "outputs: T.i64(minimum=1,multiple_of=2)"
        ).replace("tiles: T.i64", "tiles: T.i64(minimum=0,multiple_of=3)"),
    ).specialize(B=64, GROUPS=2)
    return compile(definition)


def test_plain_i64_and_explicit_default_contract_have_identical_code():
    explicit = replace(scale, source=scale.source.replace("tiles: N.i64", "tiles: N.i64()"))
    original, configured = compile(scale), compile(explicit)
    assert original.asc == configured.asc
    assert contracts(parse(original.source_ir), SizeOp) == {"tiles": (0, 1)}


def test_annotation_constructor_exposes_only_static_contract_values():
    parameter = T.i64(minimum=2, multiple_of=8)
    assert (parameter.minimum, parameter.multiple_of) == (2, 8)
    assert (T.i64().minimum, T.i64().multiple_of) == (0, 1)


@pytest.mark.parametrize("dtype,width", [("i32", 4), ("i64", 8)])
def test_integer_types_preserve_storage_width_with_an_i64_runtime_parameter(dtype, width):
    result = compile(
        T.from_source(
            f"""
        def indexed(Meta:T.Tensor((n,),T.{dtype}),
                    Out:T.Tensor((n,64),T.f32,access="write",write_disjoint=True),
                    n:T.i64(minimum=1)):
            with T.Kernel():
                row=T.alloc_ub((64,),T.f32)
                for i in T.serial(n):
                    index=T.load_index(Meta[i:i+1])
                    T.fill(row,0.0)
                    if index<0:
                        T.fill(row,1.0)
                    T.copy(row,Out[i,:])
""",
            disjoint_external=True,
        )
    )
    assert f"__gm__ volatile int{width * 8}_t" in result.asc
    assert "int64_t npu_size_n" in result.asc
    assert contracts(parse(result.source_ir), SizeOp) == {"n": (1, 1)}


@pytest.mark.parametrize("dtype", [T.i32, T.f16, T.bf16, T.f32])
def test_unsupported_scalar_types_do_not_acquire_runtime_parameter_semantics(dtype):
    with pytest.raises(CompileError, match="runtime input contracts require i64"):
        dtype(minimum=1)
    definition = T.from_source(f"""
        def invalid(n:T.{dtype.value}):
            with T.Kernel():
                pass
""")
    with pytest.raises(FrontendError, match="runtime scalar parameters require i64"):
        definition.to_ir()


@pytest.mark.parametrize(
    "options,diagnostic",
    [
        ("minimum=-1", "nonnegative int64"),
        ("minimum=True", "nonnegative int64"),
        ("minimum=1.5", "nonnegative int64"),
        ("minimum=9223372036854775808", "nonnegative int64"),
        ("multiple_of=0", "positive int64"),
        ("multiple_of=-1", "positive int64"),
        ("multiple_of=True", "positive int64"),
        ("multiple_of=1.5", "positive int64"),
        ("multiple_of=9223372036854775808", "positive int64"),
        ("minimum=9223372036854775807,multiple_of=2", "no representable value"),
        ("1", "only minimum and multiple_of"),
        ("maximum=8", "only minimum and multiple_of"),
        ("minimum=1,minimum=2", "cannot repeat"),
    ],
)
def test_invalid_contract_is_diagnosed_at_its_annotation(options, diagnostic):
    definition = T.from_source(
        f"""
        def invalid(n:T.i64({options})):
            with T.Kernel():
                pass
    """,
        filename="contract.dsl",
    )
    with pytest.raises(FrontendError, match=rf"contract.dsl:\d+:\d+: .*{diagnostic}"):
        definition.to_ir()


def test_runtime_contract_values_cannot_be_specialized_or_used_as_host_constants():
    with pytest.raises(FrontendError, match="only constexpr.*n"):
        compile(contract_source().specialize(n=4))
    definition = T.from_source("""
        def invalid(n:T.i64,m:T.i64(minimum=n)):
            with T.Kernel():
                pass
    """)
    with pytest.raises(FrontendError, match="nonnegative int64"):
        definition.to_ir()


def test_contract_annotation_never_calls_captured_python():
    calls = []

    def host_value():
        calls.append(1)
        return 1

    @T.prim_func
    def invalid(n: "T.i64(minimum=host_value())"):
        with T.Kernel():
            host_value()

    with pytest.raises(FrontendError, match="unsupported host|arbitrary Python"):
        invalid.to_ir()
    assert calls == []


@pytest.mark.parametrize("kind", [SizeOp, RuntimeSizeOp])
@pytest.mark.parametrize("field,value", [("minimum", -1), ("multiple", 0), ("minimum", 1 << 63)])
def test_forged_contract_is_rejected_by_ir_verification(local, kind, field, value):
    module = parse(local.source_ir if kind is SizeOp else local.plan_ir)
    size = next(op for op in module.walk() if isinstance(op, kind))
    size.attributes[field] = IntAttr(value)
    with pytest.raises(VerifyException, match="runtime (minimum|multiple)"):
        module.verify()


@pytest.mark.parametrize(
    "fixture,valid",
    [
        ("local", [(4,), (8,), (16,), (1024,)]),
        ("sequence", [(2, 0), (2, 3), (4, 6)]),
        ("mixed", [(2,), (4,), (8,)]),
    ],
)
def test_thin_launcher_reuses_one_binary_for_runtime_inputs(request, tmp_path, fixture, valid):
    artifact = request.getfixturevalue(fixture)
    library, launch = host_launcher(artifact, tmp_path)
    for count, values in enumerate(valid, 1):
        assert launch(*values) == 0
        assert library.dispatches() == count


def test_positive_input_contract_does_not_require_nonempty_kernel_work(tmp_path):
    artifact = compile(
        T.from_source("""
        def empty(n:T.i64(minimum=1)):
            with T.Kernel(2):
                x=T.alloc_ub((64,))
                for i in T.Pipelined(0):
                    T.fill(x,0.0)
""")
    )
    library, launch = host_launcher(artifact, tmp_path)
    assert contracts(parse(artifact.plan_ir), RuntimeSizeOp) == {"n": (1, 1)}
    assert launch(1) == 0
    assert library.dispatches() == 1


@pytest.fixture(scope="module")
def mixed():
    source = cv_source(T.dynamic("n"))
    size = next(op for op in source.walk() if isinstance(op, SizeOp))
    size.attributes.update(minimum=IntAttr(1), multiple=IntAttr(2))
    return compile(source)
