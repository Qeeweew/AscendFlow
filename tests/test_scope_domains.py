"""Actual IR predicates preserve exact boundary/count domains, including rejection."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import DictionaryAttr, IntAttr

from xdsl_npu.completion import CountDomain, Position
from xdsl_npu.model import CompileError
from xdsl_npu.scope_domains import (
    Activation,
    decode_activation,
    encode_activation,
    validate_activation,
)


@pytest.mark.parametrize(
    "activation,distance",
    [
        (Activation(), 0),
        (Activation(CountDomain(1, 0, exact=1), Position(False, 0), Position(False, 0)), 0),
        (Activation(CountDomain(1, 0, minimum=2), Position(False, 0), Position(False, 0)), 1),
        (Activation(CountDomain(1, 0, minimum=2), Position(False, 1), Position(True, 0)), -1),
        (Activation(CountDomain(1, 0, minimum=3), Position(False, 0), Position(True, 2)), 2),
        (Activation(CountDomain(1, 0, minimum=2), Position(True, 1), Position(True, 1)), 1),
    ],
)
def test_ir_activation_round_trip_preserves_source_epoch_contract(activation, distance):
    attr = encode_activation(activation)
    assert decode_activation(attr, distance=distance, period=1, residues=(0,)) == activation


@pytest.mark.parametrize(
    "activation,distance",
    [
        (Activation(), 1),
        (Activation(), -1),
        (Activation(first=Position(False, -1)), 0),
        (Activation(last=Position(True, -1)), 0),
        (Activation(CountDomain(1, 0, minimum=4, maximum=3)), 0),
        (Activation(CountDomain(1, 0, exact=0)), 0),
        (replace(Activation(), period=True), 0),
    ],
)
def test_invalid_predicate_cannot_hide_missing_endpoint_or_empty_count_domain(activation, distance):
    with pytest.raises(CompileError, match="scope"):
        validate_activation(activation, distance=distance)


@pytest.mark.parametrize(
    "key,value",
    [
        ("first_end", IntAttr(2)),
        ("last_end", IntAttr(-1)),
        ("count_residue", IntAttr(1)),
        ("count_period", IntAttr(0)),
        ("unmodelled_guard", IntAttr(1)),
    ],
)
def test_actual_ir_guard_mutations_fail_reconstruction(key, value):
    fields = dict(encode_activation(Activation()).data)
    fields[key] = value
    with pytest.raises(CompileError, match="scope"):
        decode_activation(DictionaryAttr(fields), distance=0, period=1, residues=(0,))


def test_sparse_guarded_barrier_uses_destination_residue_on_the_hardware_graph():
    from tests.test_scope_protocol_review import fifo_proof
    from xdsl_npu.scope_sync import ScopeProtocol

    proof = fifo_proof(4, 3)
    _, destination, _ = proof.access.endpoints
    activation = Activation(last=Position(True, 1), period=3, residues=(0,))
    proof.insert(
        ScopeProtocol(
            "conditional",
            "barrier",
            (),
            (destination,),
            1,
            3,
            (0,),
            activation=activation,
        )
    )
    command = proof.events[destination[0], "conditional", "barrier"]
    assert command.residues == (1,)
    assert all(
        node.residue == 1 and proof.node_activations[index].residues == (1,)
        for index, node in enumerate(proof.graph.nodes)
        if node.command is command and node.point == "begin"
    )


@pytest.mark.parametrize("residue", (0, 1))
@pytest.mark.parametrize("startup_only", (True, False))
def test_storage_period_projection_cannot_promote_a_startup_path_to_steady(residue, startup_only):
    from types import SimpleNamespace

    from xdsl_npu.completion import Site
    from xdsl_npu.loop_graph import Command, Graph
    from xdsl_npu.loop_model import Dependence
    from xdsl_npu.scope_sync import ParentCompletion

    graph = Graph(1, None)
    source, target = Command(0, "aic", None, "task"), Command(1, "aic", None, "task")
    first = graph.node(source, 0, "end")
    last = graph.node(target, 0, "begin")
    gate = graph.node(Command(2, "aic", None, "wait"), 0, "begin")
    graph.edge((first, 0), (gate, 0))
    graph.edge((gate, 0), (last, 0))
    proof = ParentCompletion.__new__(ParentCompletion)
    proof.graph, proof._guarded = graph, None
    proof.node_activations = {
        gate: Activation(last=Position(False, 0)) if startup_only else Activation()
    }
    before = Site(("aic", "source"), Position(False, 0), "begin")
    after = Site(("aic", "target"), Position(False, 0), "begin")
    proof.access = SimpleNamespace(
        endpoints={before.task: ("source", before), after.task: ("target", after)}
    )
    proof.completion = SimpleNamespace(
        commands={
            ("source", replace(before, point="end")): source,
            ("target", after): target,
        }
    )
    obligation = Dependence(before.task, after.task, 0, 2, (residue,), "RAW", "ring", 0, 32)
    # The two-slot alias domain projects onto the single protocol period.
    # Reaching only epoch zero cannot prove either infinite residue stream.
    assert proof.covered(obligation) is (not startup_only)
    assert graph.period == 1
