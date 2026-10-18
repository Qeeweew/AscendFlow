"""Complete owned memory/value admission, without synchronization or expansion."""

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntAttr

from ascendflow import language as N
from ascendflow.analysis.structured_addresses import StructuredAddresses
from ascendflow.analysis.structured_admission import admit_structured_memory
from ascendflow.analysis.structured_bounds import admit_memory_bounds
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError, ErrorKind
from ascendflow.dialects.physical import FixConfigOp
from ascendflow.dialects.source import ElementOp, PipelineOp
from ascendflow.indexing import arguments
from ascendflow.passes.lower_native import lower_native_structure
from ascendflow.target import native
from ascendflow.target.ascend910b import Actor
from examples.flash_attention.attention import attention_target
from examples.gemm.batched import batched_matmul
from examples.vector.scale import scale
from tests.control.test_shared_aiv import SHARED
from tests.models.structured_events import EventModelError, check_structured_events


def _inputs(source):
    structure = lower_native_structure(source)
    memory = collect_native_memory(structure.module)
    return structure, memory, admit_memory_bounds(memory)


def _admit(text):
    return admit_structured_memory(*_inputs(N.from_source(text, disjoint_external=True)))


def _vector(body, signature="n:N.int64(minimum=1)"):
    return f"""
        def vector({signature}):
            with N.Kernel():
                x=N.alloc_ub((128,))
                y=N.alloc_ub((64,))
{body}
    """


def _last_source(minimum, *, overwritten=False):
    tail = "                    N.fill(x[0:64],2.0)\n" if overwritten else ""
    source = N.from_source(
        _vector(
            "                for i in N.Pipelined(n,stages=1):\n"
            "                    N.fill(x[0:64],1.0)\n"
            + tail
            + "                N.mul(x[0:64],2.0,out=y)",
            f"n:N.int64(minimum={minimum})",
        ),
        disjoint_external=True,
    ).to_ir()
    pipeline = next(op for op in source.walk() if isinstance(op, PipelineOp))
    reader = tuple(op for op in source.walk() if isinstance(op, ElementOp))[-1]
    replacement = ElementOp.build(
        operands=[
            reader.inputs,
            reader.destination,
            [pipeline.values[0]],
            reader.scratch,
            None,
            None,
            None,
        ],
        attributes={
            "dependency_inputs": ArrayAttr([IntAttr(0)]),
            "kind": reader.kind,
            "scalar": reader.scalar,
        },
        result_types=[reader.value.type],
    )
    reader.parent.insert_op_before(replacement, reader)
    reader.value.replace_all_uses_with(replacement.value)
    reader.parent.erase_op(reader)
    source.verify()
    return source


@pytest.mark.parametrize(
    "source,templates,reads",
    [
        (scale, 3, 3),
        (batched_matmul, 6, 7),
        (attention_target, 52, 100),
        (N.from_source(SHARED, disjoint_external=True), 9, 13),
    ],
)
def test_real_dsl_admission_keeps_owned_facts_and_reuses_native_address_control(
    source, templates, reads, monkeypatch
):
    structure, memory, bounds = _inputs(source)

    def forbidden(*args, **kwargs):
        pytest.fail("admission rebuilt native/address/control facts")

    monkeypatch.setattr(native, "native_contract", forbidden)
    monkeypatch.setattr(StructuredAddresses, "decode", forbidden)
    admission = admit_structured_memory(structure, memory, bounds)
    assert (
        admission.structure is structure
        and admission.memory is memory
        and admission.bounds is bounds
    )
    assert admission.work["native_templates"] == templates
    assert admission.work["read_proofs"] == reads
    assert admission.work["expanded_iterations"] == 0
    assert all(
        not any(name.startswith("$") for name in arguments(value))
        for value in admission.host_expressions
    )
    assert not hasattr(admission, "completion") and not hasattr(admission, "asc")
    with pytest.raises(TypeError):
        admission.instances[next(iter(admission.instances))] = None
    for (operation, operand, actor), instance in admission.instances.items():
        assert instance.root is operation.refs[operand].owner.source
        assert instance.participant is (None if instance.root.type.bank.data == "GM" else actor)


@pytest.mark.parametrize(
    "body",
    [
        "                N.mul(x[0:64],2.0,out=y)",
        "                if n == 1:\n                    N.fill(x,0.0)\n                N.mul(x[0:64],2.0,out=y)",
        "                for i in N.Pipelined(n,stages=1):\n                    N.mul(x[0:64],2.0,out=y)\n                    if i == 0:\n                        N.fill(x,0.0)",
    ],
)
def test_undefined_local_reads_and_non_dominating_conditional_writers_are_rejected(body):
    with pytest.raises(CompileError, match="initialized writer coverage"):
        _admit(_vector(body))


def test_first_logical_iteration_initialization_is_an_inductive_value_fact():
    admission = _admit(
        _vector(
            """                for i in N.Pipelined(n,stages=1):
                    if i == 0:
                        N.fill(x,1.0)
                    N.mul(x[0:64],2.0,out=y)""",
            "n:N.int64",
        )
    )
    assert admission.work["read_proofs"] == 1


@pytest.mark.parametrize("minimum,accepted", [(0, False), (1, True)])
def test_closed_runtime_loop_needs_a_proven_first_writer_before_an_outside_read(minimum, accepted):
    text = _vector(
        """                for i in N.Pipelined(n,stages=1):
                    N.fill(x,1.0)
                N.mul(x[0:64],2.0,out=y)""",
        f"n:N.int64(minimum={minimum})",
    )
    if accepted:
        assert _admit(text).work["read_proofs"] == 1
    else:
        with pytest.raises(CompileError, match="initialized writer coverage"):
            _admit(text)


@pytest.mark.parametrize(
    "body",
    [
        """                old=N.fill(x[0:64],1.0)
                N.fill(x[0:64],2.0)
                N.mul(old,2.0,out=y)""",
        """                old=N.fill(x[0:64],1.0)
                if n == 1:
                    N.fill(x[0:64],2.0)
                N.mul(old,2.0,out=y)""",
        """                old=N.fill(x[0:64],1.0)
                for i in N.Pipelined(2,stages=1):
                    N.mul(old,2.0,out=y)
                    N.fill(x[0:64],2.0)""",
        """                old=N.fill(x[0:64],1.0)
                for i in N.Pipelined(3,stages=1):
                    if i%2 == 0:
                        N.mul(old,2.0,out=y)
                    else:
                        N.fill(x[0:64],2.0)""",
        """                old=N.fill(x[0:64],1.0)
                start=(n%2)*64
                N.fill(x[start:start+64],2.0)
                N.mul(old,2.0,out=y)""",
    ],
)
def test_tile_stale_conditional_last_iteration_and_unknown_alias_writes_are_rejected(body):
    with pytest.raises(CompileError, match="logical Tile.*overwrite") as error:
        _admit(_vector(body))
    assert error.value.kind == ErrorKind.UNSUPPORTED


def test_a_proved_disjoint_write_preserves_the_borrowed_tile():
    admission = _admit(
        _vector("""                old=N.fill(x[0:64],1.0)
                N.fill(x[64:128],2.0)
                N.mul(old,2.0,out=y)""")
    )
    assert admission.work["tile_uses"] == 1


def test_mutually_exclusive_guard_does_not_clobber_the_else_branch_tile():
    admission = _admit(
        _vector("""                old=N.fill(x[0:64],1.0)
                if n == 1:
                    N.fill(x[0:64],2.0)
                else:
                    N.mul(old,2.0,out=y)""")
    )
    assert admission.work["tile_uses"] == 1


@pytest.mark.parametrize("minimum,accepted", [(0, False), (1, True)])
def test_last_tile_selection_requires_nonempty_and_retains_actual_loop_identity(minimum, accepted):
    source = _last_source(minimum)
    if accepted:
        admission = admit_structured_memory(*_inputs(source))
        assert len(admission.values) == 1
        selected = admission.values[0].selections
        assert len(selected) == 1
        loop, ordinal = selected[0]
        assert admission.memory.control.loops[loop.operation] is loop
        assert ordinal.evaluate({"n": 100_000}) == 99_999
    else:
        with pytest.raises(CompileError, match="last selection is undefined"):
            admit_structured_memory(*_inputs(source))


def test_last_tile_is_rejected_when_a_later_writer_in_its_selected_iteration_clobbers_it():
    with pytest.raises(CompileError, match="logical Tile.*overwrite"):
        admit_structured_memory(*_inputs(_last_source(1, overwritten=True)))


def test_group_independence_is_a_source_contract_checked_by_the_finite_model():
    from ascendflow import compile

    text = """
        def overlapping(Y:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel(2) as group:
                x=N.alloc_ub((64,))
                N.fill(x,1.0)
                N.copy(x,Y)
    """
    result = compile(N.from_source(text, disjoint_external=True))
    with pytest.raises(EventModelError, match="unordered physical WAW"):
        check_structured_events(collect_native_memory(result.plan), groups=(0, 1))


@pytest.mark.parametrize("partitioned", [False, True])
def test_parallel_aiv_peers_keep_private_ub_and_their_actual_shared_gm_effects(partitioned):
    from ascendflow import compile

    destination = "Y[sid*64:sid*64+64]" if partitioned else "Y[0:64]"
    text = f"""
        def lanes(Y:N.Tensor((128,),N.f32,access="write")):
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((64,))
                with N.Vector() as sid:
                    N.fill(x,1.0)
                    N.copy(x,{destination})
    """
    admission = _admit(text)
    ub = [
        instance
        for instance in admission.instances.values()
        if instance.root.type.bank.data == "UB"
    ]
    assert {instance.participant for instance in ub} == {Actor.AIV0, Actor.AIV1}
    # The source body is SPMD. Peer writes must be independent; this is not
    # permission to invent a sequential AIV0 -> AIV1 memory definition.
    result = compile(N.from_source(text, disjoint_external=True))
    memory = collect_native_memory(result.plan)
    if partitioned:
        check_structured_events(memory)
    else:
        with pytest.raises(EventModelError, match="unordered physical WAW"):
            check_structured_events(memory)


def test_native_partial_overlap_requires_an_audited_alias_contract():
    with pytest.raises(CompileError, match="native read/write aliases"):
        _admit(
            _vector("""                N.fill(x,1.0)
                N.mul(x[0:64],2.0,out=x[8:72])""")
        )


def test_shared_vector_gm_tile_retains_the_matching_actual_producer_participant():
    admission = _admit("""
        def borrowed(X:N.Tensor((2,64),N.f32),Y:N.Tensor((2,64),N.f32,access="write")):
            with N.Kernel(1,kind="mixed"):
                x=N.alloc_ub((64,))
                y=N.alloc_ub((64,))
                with N.Vector() as sid:
                    a=N.copy(X[sid,:],x)
                    b=N.copy(a,Y[sid,:])
                    N.copy(b,y)
    """)
    assert admission.work["tile_uses"] == 2
    assert len(admission.values) == 2


def test_unknown_external_pointer_aliases_are_not_inferred_from_allocation_names():
    source = N.from_source("""
        def alias(X:N.Tensor((64,),N.f32),Y:N.Tensor((64,),N.f32,access="write")):
            with N.Kernel():
                x=N.alloc_ub((64,))
                N.copy(X,x)
                N.copy(x,Y)
    """)
    with pytest.raises(CompileError, match="disjoint pointer contract"):
        admit_structured_memory(*_inputs(source))


def test_bounds_from_another_owned_compilation_cannot_authorize_admission():
    structure, memory, _ = _inputs(scale)
    _, _, other_bounds = _inputs(scale)
    with pytest.raises(CompileError, match="different compilation invocations"):
        admit_structured_memory(structure, memory, other_bounds)


def test_fix_configuration_is_a_real_preceding_invariant_writer():
    structure = lower_native_structure(batched_matmul)
    config = next(op for op in structure.module.walk() if isinstance(op, FixConfigOp))
    config.parent.erase_op(config)
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    with pytest.raises(CompileError, match="preceding invariant register configuration"):
        admit_structured_memory(structure, memory, bounds)


def test_register_configuration_has_one_actual_invariant_writer():
    structure = lower_native_structure(batched_matmul)
    root = structure.module.body.block.first_op
    root.body.block.insert_op_before(FixConfigOp(), root.body.block.first_op)
    memory = collect_native_memory(structure.module)
    bounds = admit_memory_bounds(memory)
    with pytest.raises(CompileError, match="one invariant writer"):
        admit_structured_memory(structure, memory, bounds)


def test_host_contract_includes_negative_actual_intermediates_and_large_products():
    admission = _admit("""
        def host(n:N.int64(minimum=1)):
            with N.Kernel(20) as group:
                x=N.alloc_ub((64,))
                for i in N.serial(N.ceildiv(N.max(0,n*n-group),20)):
                    N.fill(x,1.0)
    """)
    expressions = admission.host_expressions
    assert any(value.evaluate({"n": 1}) < 0 for value in expressions)
    assert any(value.evaluate({"n": 1 << 40}) > (1 << 63) - 1 for value in expressions)
    assert all(not any(name.startswith("$") for name in arguments(value)) for value in expressions)


def test_host_contract_rejects_a_static_allocation_byte_extent_outside_int64():
    with pytest.raises(CompileError, match="static host index exceeds signed int64") as error:
        _admit("""
            def huge(X:N.Tensor((4611686018427387904,),N.f32)):
                with N.Kernel():
                    pass
        """)
    assert error.value.kind == ErrorKind.INVALID
