"""Runtime input contracts through IR and CPU execution of generated host guards.

The CPU library below contains the actual emitted launcher. Its device launch
line alone is replaced by a counter; this tests pre-dispatch checks, not NPU work.
"""

import ctypes
import re
import shutil
import subprocess
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntAttr
from xdsl.utils.exceptions import VerifyException

from ascendflow import compile
from ascendflow import language as T
from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import RuntimeSizeOp
from ascendflow.dialects.source import SizeOp
from ascendflow.frontend import FrontendError
from examples.state.grouped import grouped_state_sum
from examples.vector.scale import scale
from tests.fixtures.source_programs import cv_source


def contract_source(*, minimum=1, multiple=4):
    return T.from_source(
        """
        def guarded(X:T.Tensor((n,B),T.f32),Y:T.Tensor((n,B),T.f32,access="write"),
                    n:T.int64(minimum=MINIMUM,multiple_of=MULTIPLE),
                    B:T.constexpr=64, MINIMUM:T.constexpr=1, MULTIPLE:T.constexpr=4):
            with T.Kernel():
                x=T.alloc_ub((2,B))
                y=T.alloc_ub((2,B))
                for i in T.Pipelined(n,stages=3):
                    with T.Stage(0):
                        T.copy(X[i,:],x[i%2,:])
                    with T.Stage(1):
                        T.mul(x[i%2,:],2.0,out=y[i%2,:])
                    with T.Stage(2):
                        T.copy(y[i%2,:],Y[i,:])
        """,
        disjoint_external=True,
    ).specialize(MINIMUM=minimum, MULTIPLE=multiple)


def contracts(module, kind):
    return {
        op.sym_name.data: (op.minimum.data, op.multiple.data)
        for op in module.walk()
        if isinstance(op, kind)
    }


@pytest.fixture(scope="module")
def local():
    return compile(contract_source(minimum=3, multiple=4))


@pytest.fixture(scope="module")
def sequence():
    definition = replace(
        grouped_state_sum,
        source=grouped_state_sum.source.replace(
            "outputs: T.int64", "outputs: T.int64(minimum=1,multiple_of=2)"
        ).replace("tiles: T.int64", "tiles: T.int64(minimum=0,multiple_of=3)"),
    ).specialize(B=64, GROUPS=2)
    return compile(definition)


def test_plain_int64_and_explicit_default_contract_have_identical_code():
    explicit = replace(scale, source=scale.source.replace("tiles: N.int64", "tiles: N.int64()"))
    original, configured = compile(scale), compile(explicit)
    assert original.asc == configured.asc
    assert contracts(parse(original.source_ir), SizeOp) == {"tiles": (0, 1)}


def test_annotation_constructor_exposes_only_static_contract_values():
    parameter = T.int64(minimum=2, multiple_of=8)
    assert (parameter.minimum, parameter.multiple_of) == (2, 8)
    assert (T.int64().minimum, T.int64().multiple_of) == (0, 1)


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
        def invalid(n:T.int64({options})):
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
        def invalid(n:T.int64,m:T.int64(minimum=n)):
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
    def invalid(n: "T.int64(minimum=host_value())"):
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


def host_launcher(artifact, directory):
    """Compile actual generated guards, replacing only the device launch line."""
    marker = f'extern "C" int launch_{artifact.name}('
    launcher = artifact.asc[artifact.asc.index(marker) :]
    launcher, changed = re.subn(
        r"(?m)^  [A-Za-z_]\w*<<<[^\n]+>>>\([^\n]+\);$",
        "  ++npu_dispatches;",
        launcher,
    )
    assert changed == 1
    source = directory / "host_guards.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cstddef>\n"
        "static int npu_dispatches = 0;\n"
        + launcher
        + '\nextern "C" int dispatches() { return npu_dispatches; }\n'
    )
    compiler = shutil.which("c++")
    assert compiler is not None, "the CPU launcher-contract test requires a C++ compiler"
    binary = directory / "host_guards.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    library = ctypes.CDLL(str(binary))
    launch = getattr(library, "launch_" + artifact.name)
    parameters = contracts(parse(artifact.plan_ir), RuntimeSizeOp)
    root_count = launch_source_root_count(artifact)
    launch.argtypes = [ctypes.c_void_p] * (root_count + 1) + [ctypes.c_int64] * len(parameters)
    launch.restype = ctypes.c_int
    library.dispatches.restype = ctypes.c_int
    # The stub never dereferences these disjoint, aligned addresses.
    pointers = [ctypes.c_void_p((i + 1) << 32) for i in range(root_count)]
    return library, lambda *values: launch(None, *pointers, *values)


def launch_source_root_count(artifact):
    signature = artifact.asc.split(f'extern "C" int launch_{artifact.name}(', 1)[1].split(")", 1)[0]
    return signature.count("void* arg_")


@pytest.mark.parametrize(
    "fixture,valid,invalid",
    [
        ("local", [(4,), (8,), (16,), (1024,)], [(0,), (1,), (3,), (5,), (-1,), ((1 << 63) - 1,)]),
        ("sequence", [(2, 0), (2, 3), (4, 6)], [(0, 0), (1, 3), (2, 1), (2, -1)]),
        ("mixed", [(2,), (4,), (8,)], [(0,), (1,), (3,), (-1,)]),
    ],
)
def test_actual_host_guards_reject_before_dispatch_and_reuse_one_binary(
    request, tmp_path, fixture, valid, invalid
):
    artifact = request.getfixturevalue(fixture)
    library, launch = host_launcher(artifact, tmp_path)
    for values in invalid:
        assert launch(*values) == -1
        assert library.dispatches() == 0
    for count, values in enumerate(valid, 1):
        assert launch(*values) == 0
        assert library.dispatches() == count


def test_positive_input_contract_does_not_require_nonempty_kernel_work(tmp_path):
    artifact = compile(
        T.from_source("""
        def empty(n:T.int64(minimum=1)):
            with T.Kernel(2):
                x=T.alloc_ub((64,))
                for i in T.Pipelined(0,stages=1):
                    T.fill(x,0.0)
    """)
    )
    library, launch = host_launcher(artifact, tmp_path)
    assert launch(0) == -1
    assert library.dispatches() == 0
    assert launch(1) == 0
    assert library.dispatches() == 1


@pytest.fixture(scope="module")
def mixed():
    source = cv_source(T.dynamic("n"))
    size = next(op for op in source.walk() if isinstance(op, SizeOp))
    size.attributes.update(minimum=IntAttr(1), multiple=IntAttr(2))
    return compile(source)
