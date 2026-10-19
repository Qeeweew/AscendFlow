"""Static edge specialization and hygienic DSL helpers keep normal contracts."""

import pytest

from ascendflow import compile
from ascendflow import language as T
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from ascendflow.dialects.source import PipelineOp
from ascendflow.frontend import FrontendError
from tests.models.structured_events import check_structured_events


@T.inline
def _copy(source, destination):
    T.copy(source, destination)


@pytest.mark.parametrize("extent", [16, 64, 80, 144])
def test_edge_ranges_and_inline_copies_preserve_byte_versions_and_consumed_events(extent):
    program = T.from_source(
        """
        def edge(A:T.Tensor((E,),T.f32), C:T.Tensor((E,),T.f32,access="write")):
            with T.Kernel(1):
                for base,count,valid in T.tile_ranges(E,64):
                    x=T.alloc_ub((2,valid),T.f32)
                    for i in T.Pipelined(count,stages=2):
                        start=base+i*64
                        with T.Stage(0,name="load"):
                            helper(A[start:start+valid], x[i%2,:])
                        with T.Stage(1,name="store"):
                            helper(x[i%2,:], C[start:start+valid])
        """,
        constants={"E": extent},
        functions={"helper": _copy},
        disjoint_external=True,
    )
    result = compile(program, timeout_s=20)
    before = collect_native_memory(parse(result.before_ir))
    after = collect_native_memory(parse(result.plan_ir))
    trace = check_structured_events(after, logical_reference=before, groups=(0,))
    assert trace.bytes_read and trace.bytes_written and trace.consumptions


@pytest.mark.parametrize("extent", [0, 1, 64, 65, 1 << 26])
def test_static_ranges_skip_empty_work_and_retain_large_device_loops(extent):
    program = T.from_source(
        """
        def ranges():
            with T.Kernel():
                x=T.alloc_ub((64,),T.f32)
                for base,count,valid in T.tile_ranges(E,64):
                    for i in T.Pipelined(count,stages=1):
                        T.fill(x,1.0)
        """,
        constants={"E": extent},
    )
    pipelines = [op for op in program.to_ir().walk() if isinstance(op, PipelineOp)]
    assert sum(op.count.data for op in pipelines) == (extent + 63) // 64
    assert len(pipelines) <= 2


@pytest.mark.parametrize(
    "extent,width,target,diagnostic",
    [
        ("-1", "64", "b,n,v", "nonnegative extent"),
        ("64", "0", "b,n,v", "positive tile width"),
        ("64", "-1", "b,n,v", "positive tile width"),
        ("64", "True", "b,n,v", "constexpr integer"),
        ("n", "64", "b,n,v", "constexpr integer"),
        ("64", "64", "b,b,v", "three distinct names"),
        ("64", "64", "b,n", "three distinct names"),
        ("64", "64", "b", "three distinct names"),
    ],
)
def test_edge_specialization_rejects_invalid_or_dynamic_geometry(extent, width, target, diagnostic):
    program = T.from_source(f"""
        def invalid(n:T.int64):
            with T.Kernel():
                for {target} in T.tile_ranges({extent},{width}):
                    pass
    """)
    with pytest.raises(FrontendError, match=diagnostic):
        program.to_ir()


def test_inline_lexical_constants_defaults_and_local_bindings_are_isolated():
    captured = 16

    @T.inline
    def helper(value=2):
        local = captured
        return local * value, local

    captured = 32

    @T.prim_func
    def program():
        with T.Kernel():
            local = 64
            pair = helper()
            assert pair[0] == 32 and pair[1] == 16 and local == 64, "lexical capture"
            assert helper(value=3)[0] == 48, "named default override"

    program.to_ir().verify()


def test_inline_cannot_capture_the_callers_local_scope():
    @T.inline
    def helper():
        return caller_local  # noqa: F821 - must not resolve from the caller

    program = T.from_source(
        """
        def invalid():
            with T.Kernel():
                caller_local=64
                result=helper()
        """,
        functions={"helper": helper},
    )
    with pytest.raises(FrontendError, match="undefined DSL name caller_local"):
        program.to_ir()


def test_inline_host_calls_are_never_executed_and_diagnostic_names_helper_source():
    calls = []

    def effect():
        calls.append(True)

    @T.inline
    def helper():
        effect()

    program = T.from_source(
        "def invalid():\n    with T.Kernel():\n        helper()",
        filename="caller.py",
        functions={"helper": helper},
    )
    assert calls == []
    with pytest.raises(FrontendError, match=r"test_frontend_helpers.py:\d+:\d+: unsupported host"):
        program.to_ir()
    assert calls == []


@pytest.mark.parametrize(
    "call", ["helper()", "helper(1,2)", "helper(other=1)", "helper(1,value=2)"]
)
def test_inline_argument_errors_are_diagnosed(call):
    @T.inline
    def helper(value):
        return value

    program = T.from_source(
        "def invalid():\n    with T.Kernel():\n        result=" + call,
        functions={"helper": helper},
    )
    with pytest.raises(FrontendError, match="inline argument|inline parameter"):
        program.to_ir()


def test_inline_does_not_bypass_original_axis_bounds():
    @T.inline
    def shifted(source):
        return source[1:65]

    program = T.from_source(
        """
        def invalid(A:T.Tensor((64,),T.f32)):
            with T.Kernel():
                x=T.alloc_ub((64,),T.f32)
                T.copy(shifted(A),x)
        """,
        functions={"shifted": shifted},
    )
    with pytest.raises(CompileError, match="axis|bounds"):
        compile(program, timeout_s=10)


def test_inline_cannot_mask_an_undefined_unsigned_divisor():
    @T.inline
    def offset(i):
        return i // (i - i)

    program = T.from_source(
        """
        def invalid(A:T.Tensor((64,),T.f32)):
            with T.Kernel():
                x=T.alloc_ub((64,),T.f32)
                for i in T.Pipelined(1,stages=1):
                    start=offset(i)
                    T.copy(A[start:start+64],x)
        """,
        functions={"offset": offset},
    )
    with pytest.raises(CompileError, match="divisor|division|unsigned"):
        compile(program, timeout_s=10)


def test_inline_early_returns_cannot_select_a_runtime_branch_on_the_host():
    @T.inline
    def helper(n):
        if n > 0:
            return n
        return 0

    program = T.from_source(
        "def invalid(n:T.int64):\n    with T.Kernel():\n        result=helper(n)",
        functions={"helper": helper},
    )
    with pytest.raises(FrontendError, match="only one final return"):
        program.to_ir()


def test_inline_depth_is_bounded_without_host_recursion():
    @T.inline
    def helper(value):
        return value

    for _ in range(17):
        previous = helper

        @T.inline
        def helper(value):
            return previous(value)  # noqa: B023 - DSL capture freezes each binding

    program = T.from_source(
        "def invalid():\n    with T.Kernel():\n        result=helper(1)",
        functions={"helper": helper},
    )
    with pytest.raises(FrontendError, match="16-call budget"):
        program.to_ir()


def test_explicit_source_bindings_cannot_invoke_arbitrary_python():
    with pytest.raises(CompileError, match="captured DSL inline helpers"):
        T.from_source("def invalid():\n    pass", functions={"helper": lambda: 1})


def test_cli_source_imports_local_dsl_helpers_without_pythonpath(tmp_path):
    import os
    import subprocess
    import sys

    package = tmp_path / "policies"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from ascendflow import language as T\n@T.inline\ndef width():\n    return 64\n"
    )
    entry = tmp_path / "entry.py"
    entry.write_text(
        "from ascendflow import language as T\n"
        "from policies import width\n"
        "@T.prim_func\n"
        "def kernel():\n"
        "    with T.Kernel():\n"
        "        x=T.alloc_ub((width(),),T.f32)\n"
        "        T.fill(x,1.0)\n"
    )
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "ascendflow",
            "--input",
            str(entry),
            "--entry",
            "kernel",
            "--output",
            str(tmp_path / "build"),
            "--timeout",
            "20",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "build/kernel.asc").is_file()
