"""One public source path, immutable artifacts, and no imported proof markers."""

from dataclasses import FrozenInstanceError

import pytest
from xdsl.dialects.builtin import ModuleOp

from xdsl_npu import compile
from xdsl_npu import language as N
from xdsl_npu.compilation import Compilation
from xdsl_npu.diagnostics import CompileError, ErrorKind
from xdsl_npu.dialects.io import dump
from xdsl_npu.dialects.structured import NotificationOp

SOURCE = """
def point(X:N.Tensor((64,),N.f32),Y:N.Tensor((64,),N.f32,access="write")):
    with N.Kernel():
        x,y=N.alloc_ub((64,)),N.alloc_ub((64,))
        value=N.copy(X,x)
        value=N.mul(value,2.0,out=y)
        N.copy(value,Y)
"""


@pytest.fixture(scope="module")
def artifact():
    return compile(N.from_source(SOURCE, disjoint_external=True))


def test_public_source_owns_and_preserves_its_input(monkeypatch):
    source = N.from_source(SOURCE, disjoint_external=True).to_ir()
    before = dump(source)
    result = compile(source)
    assert dump(source) == before
    source.body.block.erase_op(source.body.block.first_op)
    assert result.source_ir == before
    assert result.report["pipeline"] == "structured-regions"
    assert result.report["verification_mode"] == "generated-lifetimes"
    assert "npu.notification" not in result.before_ir
    assert "npu.notification" in result.plan_ir


def test_exported_ir_and_reports_cannot_mutate_an_artifact(artifact):
    plan = artifact.plan
    notify = next(op for op in plan.walk() if isinstance(op, NotificationOp))
    notify.parent.erase_op(notify)
    assert dump(plan) != artifact.plan_ir
    assert dump(artifact.plan) == artifact.plan_ir
    report = artifact.report
    report["sync_instructions"] = -1
    assert artifact.report["sync_instructions"] > 0
    with pytest.raises(FrozenInstanceError):
        artifact.asc = "unsafe"
    with pytest.raises(TypeError):
        Compilation(N.from_source(SOURCE), verified=True)


def test_physical_exports_are_not_proof_carrying_public_inputs(artifact):
    for module in (artifact.plan, ModuleOp([]), object()):
        with pytest.raises(CompileError, match="source ProgramOp") as failure:
            compile(module)
        assert failure.value.kind is ErrorKind.UNSUPPORTED
        assert failure.value.phase == "compilation input"
    # Removal of a Wait does not let a caller bypass the owned source chain.
    changed = artifact.plan
    wait = next(
        op
        for op in changed.walk()
        if isinstance(op, NotificationOp) and op.kind.data == "local_wait"
    )
    wait.parent.erase_op(wait)
    with pytest.raises(CompileError, match="source ProgramOp"):
        compile(changed)


def test_debug_is_explicit_structural_verification_not_a_second_solver(artifact, monkeypatch):
    deep = compile(N.from_source(SOURCE, disjoint_external=True), debug_verify=True)
    assert not artifact.report["debug_structural_verification"]
    assert deep.report["debug_structural_verification"]
    assert deep.asc == artifact.asc
    assert deep.plan_ir == artifact.plan_ir
    deep.plan.verify()


def test_writing_artifacts_uses_only_frozen_snapshots(artifact, tmp_path, monkeypatch):
    artifact.write(tmp_path, diagnostics=True)
    assert (tmp_path / "after.xdsl").read_text() == artifact.plan_ir
    assert (tmp_path / "before.xdsl").read_text() == artifact.before_ir
    assert (tmp_path / "source.xdsl").read_text() == artifact.source_ir
    assert (tmp_path / f"{artifact.name}.asc").read_text() == artifact.asc
    assert (tmp_path / "witnesses.json").exists()
    assert artifact.explain() == artifact.report["witnesses"]
