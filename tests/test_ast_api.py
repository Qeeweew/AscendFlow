"""The supported AST API has no executable tracing or implicit-storage entry."""

import importlib
import inspect

import pytest

from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.dialects.io import parse
from xdsl_npu.dialects.physical import TaskOp
from xdsl_npu.frontend import FrontendError


def test_public_physical_builder_has_been_removed():
    import xdsl_npu

    assert not hasattr(xdsl_npu, "Kernel")
    assert "Kernel" not in xdsl_npu.__all__
    with pytest.raises(ModuleNotFoundError, match="xdsl_npu.builder"):
        importlib.import_module("xdsl_npu.builder")


@pytest.mark.parametrize("name", ["trace_func", "alloc_workspace", "gemm"])
def test_removed_source_entry_has_no_compatibility_alias(name):
    assert not hasattr(N, name)


def test_pipeline_has_no_python_tracing_iterator():
    assert "__iter__" not in N.Pipelined.__dict__


@pytest.mark.parametrize(
    "operation",
    [lambda: N.Kernel(), lambda: N.alloc_ub((64,)), lambda: N.Pipelined(2, stages=1)],
)
def test_device_syntax_cannot_start_executable_python_tracing(operation):
    with pytest.raises(CompileError, match="AST capture"):
        operation()


@pytest.mark.parametrize("name", ["alloc_ub", "alloc_l1", "alloc_l0a", "alloc_l0b", "alloc_l0c"])
def test_allocations_have_only_explicit_shape_axes(name):
    assert "versions" not in inspect.signature(getattr(N, name)).parameters
    definition = N.from_source(f"""
        def invalid():
            with N.Kernel():
                x=N.{name}((16,16),versions=2)
    """)
    with pytest.raises(FrontendError, match="unexpected keyword argument 'versions'"):
        definition.to_ir()


@pytest.mark.parametrize("name", ["gemm", "alloc_workspace", "trace_func"])
def test_removed_names_are_not_ast_operations(name):
    definition = N.from_source(f"""
        def invalid():
            with N.Kernel():
                N.{name}()
    """)
    with pytest.raises(FrontendError, match=f"unknown DSL operation {name}"):
        definition.to_ir()


@pytest.mark.parametrize("definition", [scale, batched_matmul.specialize(GROUPS=1)])
def test_supported_ast_still_compiles_through_physical_tasks_and_final_check(definition):
    result = compile(definition)
    assert result.source_ir is not None
    assert result.schedule
    assert any(isinstance(op, TaskOp) for op in parse(result.plan_ir).walk())
    assert "npu_source.program" in result.source_ir
    assert result.report["native_templates"] > 0
    assert "PIPE_ALL" not in result.asc


def test_actor_local_allocation_cannot_escape_to_another_source_region():
    definition = N.from_source("""
        def escaped():
            with N.Kernel():
                with N.Vector():
                    x=N.alloc_ub((64,),name="private")
                for i in N.Pipelined(1,stages=1):
                    N.fill(x,0.0)
    """)
    with pytest.raises(FrontendError, match="allocation private escaped its source region"):
        definition.to_ir()


def test_allocation_may_be_used_inside_its_own_source_region():
    definition = N.from_source("""
        def local():
            with N.Kernel():
                for i in N.Pipelined(1,stages=1):
                    with N.Vector():
                        x=N.alloc_ub((64,))
                        N.fill(x,0.0)
    """)
    assert compile(definition).report["native_templates"] == 1
