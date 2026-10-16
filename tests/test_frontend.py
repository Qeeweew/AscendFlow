"""Source semantics, refusal boundaries and real AST-to-synchronized-ASC tests."""

import pytest
from xdsl.dialects import arith, scf

from examples.vector.scale import scale
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import dump, parse
from xdsl_npu.dialects.physical import CallOp, StreamOp, TaskOp
from xdsl_npu.dialects.source import AllocationOp, CopyOp, ElementOp, PipelineOp, SizeOp
from xdsl_npu.frontend import FrontendError, PrimFunction
from xdsl_npu.model import Program


def test_prim_func_captures_body_instead_of_executing_it():
    calls = []

    def host_effect():
        calls.append(1)

    @N.prim_func
    def bad():
        with N.Kernel():
            host_effect()

    assert isinstance(bad, PrimFunction)
    assert calls == []
    with pytest.raises(FrontendError, match=r"test_frontend.py:\d+:\d+: unsupported host"):
        bad.to_ir()
    assert calls == []


def test_captured_namespace_never_executes_rebound_host_operations(monkeypatch):
    syntax = N
    calls = []

    def host_effect(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("public syntax invoked as a host builder")

    @syntax.prim_func
    def captured(n: "syntax.int64(minimum=4, multiple_of=2)"):
        with syntax.Kernel():
            x = syntax.alloc_ub((64,))
            for i in syntax.Pipelined(n, stages=1):
                syntax.fill(x, 1.0)

    for name in ("Kernel", "alloc_ub", "fill"):
        monkeypatch.setattr(syntax, name, host_effect)
    source = captured.to_ir()
    assert calls == []
    size = next(op for op in source.walk() if isinstance(op, SizeOp))
    assert (size.minimum.data, size.multiple.data) == (4, 2)
    assert [op.kind.data for op in source.walk() if isinstance(op, ElementOp)] == ["fill"]


def test_scalar_parameters_are_resolved_before_tensor_annotations_and_slots_are_explicit():
    source = scale.to_ir()
    parse(dump(source))
    sizes = [op for op in source.walk() if isinstance(op, SizeOp)]
    assert [op.sym_name.data for op in sizes] == ["tiles"]
    allocations = [op for op in source.walk() if isinstance(op, AllocationOp)]
    assert all(op.versions.data == 1 for op in allocations)
    assert [tuple(v.data for v in op.ref.type.shape) for op in allocations[-2:]] == [(2, 64)] * 2
    assert next(op for op in source.walk() if isinstance(op, PipelineOp)).limit is sizes[0].value


def test_ast_compile_uses_real_loop_sync_and_includes_frontend_time(monkeypatch):
    def no_instances(*args, **kwargs):
        raise AssertionError("AST compilation expanded input length")

    monkeypatch.setattr(Program, "instances", no_instances)
    result = compile(scale)
    assert result.report["runtime_length"] == "tiles"
    assert result.report["native_templates"] == 3
    assert result.timings["frontend_s"] > 0
    assert result.timings["total_s"] >= sum(
        result.timings[key] for key in ("frontend_s", "lowering_s", "planning_and_final_check_s")
    )
    assert "int64_t npu_size_tiles" in result.asc
    assert result.asc.count("asc_sync_pipe(PIPE_ALL)") == 1
    assert len(result.report["event_ids"]) == 4
    assert "npu_source.pipeline" in result.source_ir


def test_runtime_parameter_cannot_be_silently_specialized():
    with pytest.raises(FrontendError, match="only constexpr.*tiles"):
        compile(scale.specialize(tiles=1000))
    source = scale.specialize(B=128).to_ir()
    assert (
        tuple(next(op for op in source.walk() if isinstance(op, CopyOp)).source.type.shape)[-1].data
        == 128
    )


def test_ast_constant_loop_keeps_one_body_even_for_millions_of_iterations():
    sizes = []
    for count in (0, 4, 1 << 26):
        source = N.from_source(
            """
            def retained(B: N.constexpr = 64):
                with N.Kernel():
                    x = N.alloc_ub((2, B))
                    for i in N.Pipelined(COUNT, stages=1):
                        N.fill(x[i % 2, :], 0.0)
        """,
            constants={"COUNT": count},
        ).to_ir()
        sizes.append(len(list(source.walk())))
        assert next(op for op in source.walk() if isinstance(op, PipelineOp)).count.data == count
    assert len(set(sizes)) == 1


def test_device_if_is_a_real_scf_region_and_both_sides_are_present():
    definition = N.from_source("""
        def branch(n: N.int64):
            with N.Kernel():
                x = N.alloc_ub((2, 64))
                for i in N.Pipelined(n, stages=1):
                    if i < 2:
                        N.fill(x[i % 2, :], 1.0)
                    else:
                        N.fill(x[i % 2, :], 2.0)
    """)
    source = parse(dump(definition.to_ir()))
    assert len([op for op in source.walk() if isinstance(op, scf.IfOp)]) == 1
    assert len([op for op in source.walk() if isinstance(op, arith.CmpiOp)]) == 1
    with pytest.raises(CompileError, match="structured scf.if synchronization"):
        compile(definition)


def test_dynamic_serial_and_ceildiv_are_retained_in_source_ir():
    definition = N.from_source("""
        def nested(n: N.int64, m: N.int64):
            with N.Kernel():
                x = N.alloc_ub((2, 64))
                for i in N.Pipelined(n, stages=1):
                    for j in N.serial(N.ceildiv(m, 64)):
                        N.fill(x[j % 2, :], 1.0)
    """)
    source = parse(dump(definition.to_ir()))
    assert len([op for op in source.walk() if isinstance(op, scf.ForOp)]) == 1
    assert len([op for op in source.walk() if isinstance(op, arith.CeilDivUIOp)]) == 1
    assert len([op for op in source.walk() if isinstance(op, SizeOp)]) == 2
    assert len([op for op in source.walk() if op.name == "npu_source.element"]) == 1
    with pytest.raises(CompileError, match="structured|arithmetic runtime"):
        compile(definition)


def test_ast_constant_nested_pipeline_is_not_sent_to_legacy_unrolling():
    definition = N.from_source("""
        def nested():
            with N.Kernel():
                x = N.alloc_ub((2, 64))
                for i in N.Pipelined(4, stages=1):
                    for j in N.Pipelined(3, stages=1):
                        N.fill(x[j % 2, :], 0.0)
    """)
    assert len([op for op in definition.to_ir().walk() if isinstance(op, PipelineOp)]) == 2
    result = compile(definition)
    physical = parse(result.before_ir)
    assert len([op for op in physical.walk() if isinstance(op, CallOp)]) == 1
    assert [op.count.data for op in physical.walk() if isinstance(op, StreamOp)] == [4, 3]
    assert len([op for op in physical.walk() if isinstance(op, TaskOp)]) == 1
    assert result.report["native_templates"] == 1


@pytest.mark.parametrize(
    "statement,diagnostic",
    [
        ("for i in range(4):\n    pass", "undefined DSL name range"),
        ("while True:\n    pass", "unsupported device statement While"),
        ("x = N.alloc_ub((64,), versions=2)", "unexpected keyword argument.*versions"),
        ("for i in N.unroll(1000000):\n    pass", "64-iteration budget"),
    ],
)
def test_diagnostics_reject_implicit_host_execution(statement, diagnostic):
    import textwrap

    definition = N.from_source(
        "def invalid():\n    with N.Kernel():\n" + textwrap.indent(statement, "        ")
    )
    with pytest.raises(FrontendError, match=diagnostic):
        definition.to_ir()


def test_unroll_is_explicit_and_does_not_add_a_device_loop():
    source = N.from_source("""
        def explicit():
            with N.Kernel():
                x = N.alloc_ub((64,))
                for i in N.Pipelined(4, stages=1):
                    for j in N.unroll(3):
                        N.fill(x, 1.0)
    """).to_ir()
    assert len([op for op in source.walk() if op.name == "npu_source.element"]) == 3
    assert len([op for op in source.walk() if isinstance(op, PipelineOp)]) == 1


def test_branch_rebinding_cannot_silently_select_the_host_value():
    definition = N.from_source("""
        def invalid(n: N.int64):
            with N.Kernel():
                value = 0
                for i in N.Pipelined(n, stages=1):
                    if i == 0:
                        value = 1
    """)
    with pytest.raises(FrontendError, match="loop-carried binding value"):
        definition.to_ir()


def test_explicit_source_does_not_execute_annotations():
    definition = N.from_source("""
        def invalid(X: N.Tensor((open("bad"),), N.f32)):
            with N.Kernel():
                pass
    """)
    with pytest.raises(FrontendError, match="undefined DSL name open"):
        definition.to_ir()


def test_source_contract_decorator_cannot_be_silently_ignored():
    definition = N.from_source("""
        @N.prim_func(disjoint_external=True)
        def invalid():
            with N.Kernel():
                pass
    """)
    with pytest.raises(FrontendError, match="conflicts with the explicit input contract"):
        definition.to_ir()
