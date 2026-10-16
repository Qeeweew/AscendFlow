"""Conditional memory premises from real scale and short-MAD native hazards."""

from dataclasses import replace

import pytest

from tests.fixtures.pipelines import cube_pipeline, vector_pipeline
from xdsl_npu.analysis.contracts import ContractWork, WitnessKind, build_memory_contracts
from xdsl_npu.analysis.frontier import Capture, Clock, Partition
from xdsl_npu.analysis.hazards import HazardKind, bind_hazards
from xdsl_npu.analysis.native import extract_native_tasks
from xdsl_npu.analysis.regions import Output, Port
from xdsl_npu.analysis.transitions import transition_layout
from xdsl_npu.diagnostics import CompileError
from xdsl_npu.loop_model import extract_loop, loop_dependencies
from xdsl_npu.target.ascend910b import Milestone


def analyze(module):
    loop = extract_loop(module)
    tasks = extract_native_tasks(loop, clock=Clock("step"))
    return tasks, bind_hazards(loop, tasks, loop_dependencies(loop))


def select(hazards, source, consumer, kind, residue=0):
    return next(
        hazard
        for hazard in hazards
        if hazard.producer.operation.uid.data == source
        and hazard.consumer.operation.uid.data == consumer
        and hazard.kind == kind
        and residue in hazard.activation.residues
    )


def scale_chain(hazards, residue=0):
    target = select(hazards, "load", "load", HazardKind.WAW, residue)
    via = select(hazards, "scale", "load", HazardKind.WAR, residue)
    premise = select(hazards, "load", "scale", HazardKind.RAW, residue)
    return via, target, premise


@pytest.mark.parametrize("factory", [vector_pipeline, cube_pipeline])
def test_index_retains_every_actual_consumer_obligation(factory):
    tasks, hazards = analyze(factory())
    work = ContractWork()
    contracts = build_memory_contracts(tasks, hazards, work=work)
    total = 0
    for task in tasks:
        admission = contracts.admission(task, work=work)
        expected = tuple(hazard for hazard in hazards if hazard.consumer is task)
        assert admission.task is task
        assert len(admission.obligations) == len(expected)
        for actual, original in zip(admission.obligations, expected, strict=True):
            assert actual is original
            assert actual.activation is original.activation
            assert actual.producer.operation is original.producer.operation
            assert actual.consumer.operation is task.operation
        total += len(admission.obligations)
    assert total == len(hazards)
    assert work.tasks_indexed == len(tasks)
    assert work.accesses_indexed == sum(len(task.accesses) for task in tasks)
    assert work.hazards_indexed == len(hazards)
    assert work.admission_lookups == len(tasks)


@pytest.mark.parametrize("residue", [0, 1])
@pytest.mark.parametrize(
    ("source", "capture", "consumer"),
    [("load", "scale", "load"), ("postprocess", "store", "scale")],
)
def test_scale_capture_carries_direct_admission_only_conditionally(
    source, capture, consumer, residue
):
    tasks, hazards = analyze(vector_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    target = select(hazards, source, consumer, HazardKind.WAW, residue)
    via = select(hazards, capture, consumer, HazardKind.WAR, residue)
    premise = select(hazards, source, capture, HazardKind.RAW, residue)
    witness = contracts.direct_witness(via, target)
    assert witness is not None
    assert witness.kind is WitnessKind.ADMISSION_PREMISE
    assert witness.premise is premise
    assert witness.via is via and witness.target is target
    assert witness.capture.producer is via.producer
    assert witness.assumptions.contract is contracts.admission(via.producer)
    assert witness.assumptions.contract is witness.capture.admission
    assert witness.assumptions.generation_offset == -2
    assert len(witness.assumptions.contract.obligations) > 1
    assert any(h is premise for h in witness.assumptions.contract.obligations)
    assert witness.milestone is Milestone.RETIREMENT_CAPTURE
    assert witness.capture.anchor.milestone is Milestone.NATIVE_END
    assert not isinstance(witness.capture, Capture)


@pytest.mark.parametrize("residue", [0, 1])
def test_short_mad_retirement_carries_all_operand_admission_premises(residue):
    tasks, hazards = analyze(cube_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    target = select(hazards, "operand_a", "operand_a", HazardKind.WAW, residue)
    via = select(hazards, "mad", "operand_a", HazardKind.WAR, residue)
    premise = select(hazards, "operand_a", "mad", HazardKind.RAW, residue)
    witness = contracts.direct_witness(via, target)
    assert witness is not None
    assert not via.producer.implicit_completion
    assert witness.kind is WitnessKind.ADMISSION_PREMISE
    assert witness.premise is premise
    assert witness.milestone is Milestone.RETIREMENT_CAPTURE
    admission = contracts.admission(via.producer)
    assert witness.assumptions.contract is admission
    assert {h.producer.operation.uid.data for h in admission.obligations} == {
        "operand_a",
        "operand_b",
        "mad",
        "export",
    }
    own_target = select(hazards, "mad", "export", HazardKind.RAW, residue)
    own = contracts.direct_witness(own_target, own_target)
    assert own is not None
    assert own.kind is WitnessKind.OWN_RETIREMENT
    assert own.premise is None
    assert own.assumptions.contract is admission
    assert own.milestone is Milestone.RETIREMENT_CAPTURE
    with pytest.raises(CompileError, match="actual indexed NativeTask"):
        contracts.capture(via.producer.capture_cut)


def test_passthrough_output_stays_the_same_expression_without_new_guarantees():
    tasks, hazards = analyze(vector_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    via, target, _ = scale_chain(hazards)
    task = via.producer
    layout = transition_layout({task.actor: {t.engine for t in tasks}}, task.clock)
    retire = Port(task.actor, task.engine, Milestone.RETIREMENT_CAPTURE)
    output = Output(frozenset({retire}), layout.frontier.empty())
    capture = contracts.capture(task, passthrough=output)
    witness = contracts.direct_witness(via, target, passthrough=output)
    assert witness is not None
    assert capture.passthrough is output
    assert witness.capture.passthrough is output
    assert output.sources == frozenset({retire})
    assert len(output.guarantees) == 0
    assert not output.guarantees.covers(target.completion_prefixes(Partition())[0])
    assert witness.assumptions.contract.obligations


@pytest.mark.parametrize("component", ["via", "target", "premise"])
@pytest.mark.parametrize("mismatch", ["distance", "residue"])
def test_generation_and_storage_residue_must_match(component, mismatch):
    tasks, hazards = analyze(vector_pipeline())
    via, target, premise = scale_chain(hazards)
    selected = {"via": via, "target": target, "premise": premise}[component]
    activation = (
        replace(selected.activation, distance=4)
        if mismatch == "distance"
        else replace(selected.activation, residues=frozenset({1}))
    )
    wrong = replace(selected, activation=activation)
    if component == "via":
        via = wrong
    elif component == "target":
        target = wrong
    else:
        hazards = tuple(wrong if hazard is premise else hazard for hazard in hazards)
    assert build_memory_contracts(tasks, hazards).direct_witness(via, target) is None


@pytest.mark.parametrize("component", ["via", "premise"])
def test_matching_root_name_does_not_replace_range_coverage(component):
    tasks, hazards = analyze(vector_pipeline())
    via, target, premise = scale_chain(hazards)
    selected = via if component == "via" else premise
    wrong = replace(selected, memory=replace(selected.memory, stop=selected.memory.stop - 1))
    if component == "via":
        via = wrong
    else:
        hazards = tuple(wrong if hazard is premise else hazard for hazard in hazards)
    assert build_memory_contracts(tasks, hazards).direct_witness(via, target) is None


@pytest.mark.parametrize("component", ["via", "premise"])
def test_terminal_endpoint_is_not_assumed_for_an_earlier_generation(component):
    tasks, hazards = analyze(vector_pipeline())
    via, target, premise = scale_chain(hazards)
    selected = via if component == "via" else premise
    terminal = replace(selected, activation=replace(selected.activation, consumer_last=True))
    if component == "via":
        via = terminal
    else:
        target = replace(target, activation=replace(target.activation, consumer_last=True))
        hazards = tuple(terminal if hazard is premise else hazard for hazard in hazards)
    assert build_memory_contracts(tasks, hazards).direct_witness(via, target) is None


def test_required_last_consumer_can_use_a_regular_capture_but_no_fictional_source():
    tasks, hazards = analyze(vector_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    via, target, _ = scale_chain(hazards)
    terminal = replace(target, activation=replace(target.activation, consumer_last=True))
    witness = contracts.direct_witness(via, terminal)
    assert witness is not None
    assert witness.target.activation.consumer_last
    impossible = replace(target, activation=replace(target.activation, source_last=True))
    assert contracts.direct_witness(via, impossible) is None


@pytest.mark.parametrize("component", ["via", "target", "premise"])
def test_individual_admission_end_does_not_discharge_a_reader_family(component):
    tasks, hazards = analyze(vector_pipeline())
    via, target, premise = scale_chain(hazards)
    selected = {"via": via, "target": target, "premise": premise}[component]
    cumulative = replace(selected, reader_prefix=True)
    if component == "via":
        via = cumulative
    elif component == "target":
        target = cumulative
    else:
        hazards = tuple(cumulative if hazard is premise else hazard for hazard in hazards)
    contracts = build_memory_contracts(tasks, hazards)
    assert contracts.direct_witness(via, target) is None
    own = contracts.direct_witness(cumulative, cumulative)
    assert own is not None
    assert own.kind is WitnessKind.OWN_RETIREMENT
    assert own.milestone is Milestone.RETIREMENT_CAPTURE


def test_same_engine_and_later_rank_do_not_replace_actual_source_identity():
    tasks, hazards = analyze(vector_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    via = select(hazards, "postprocess", "scale", HazardKind.WAW)
    scale = via.consumer
    target = replace(via, producer=scale, activation=replace(via.activation, distance=4))
    assert via.producer.engine is target.producer.engine
    assert via.source_cut.rank.compare(target.source_cut.rank) > 0
    assert contracts.direct_witness(via, target) is None


def test_direct_lookup_does_not_search_through_intermediate_admission_contracts():
    tasks, hazards = analyze(vector_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    via = select(hazards, "store", "scale", HazardKind.WAR)
    output_reuse = select(hazards, "postprocess", "scale", HazardKind.WAW)
    target = replace(output_reuse, producer=output_reuse.consumer)
    work = ContractWork()
    # Store admits postprocess; postprocess admits scale. That second hop
    # remains outside this direct conditional index.
    assert contracts.direct_witness(via, target, work=work) is None
    assert work.candidate_visits == 0
    assert contracts.statistics["recursive_queries"] == 0


def test_actual_consumer_snapshot_clock_and_allocation_are_preserved():
    tasks, hazards = analyze(vector_pipeline())
    contracts = build_memory_contracts(tasks, hazards)
    via, target, _ = scale_chain(hazards)
    with pytest.raises(CompileError, match="different native task snapshot"):
        contracts.admission(replace(via.producer))
    with pytest.raises(CompileError, match="indexed native task snapshots"):
        build_memory_contracts(tasks, (replace(target, producer=replace(target.producer)),))
    with pytest.raises(CompileError, match="explicitly bound clock"):
        build_memory_contracts((*tasks[:-1], replace(tasks[-1], clock=Clock("step"))), ())
    different = select(hazards, "postprocess", "store", HazardKind.RAW)
    with pytest.raises(CompileError, match="actual allocation root"):
        contracts.direct_witness(via, replace(target, memory=different.memory))
    with pytest.raises(CompileError, match="actual native tasks"):
        build_memory_contracts((target.source_cut,), ())
    other_consumer = replace(target, consumer=via.producer)
    assert contracts.direct_witness(via, other_consumer) is None


def test_index_and_queries_do_not_repeat_native_or_access_analysis(monkeypatch):
    tasks, hazards = analyze(vector_pipeline())

    def fail(*args, **kwargs):
        raise AssertionError("native/access analysis repeated")

    monkeypatch.setattr("xdsl_npu.analysis.native.extract_native_tasks", fail)
    monkeypatch.setattr("xdsl_npu.analysis.native.native_task_contract", fail)
    monkeypatch.setattr("xdsl_npu.loop_model.loop_dependencies", fail)
    monkeypatch.setattr("xdsl_npu.loop_model.access_cells", fail)
    contracts = build_memory_contracts(tasks, hazards)
    via, target, _ = scale_chain(hazards)
    assert contracts.direct_witness(via, target) is not None


def test_query_work_does_not_grow_with_unrelated_obligations_or_runtime_count():
    results = []
    for count, copies in ((8, 0), (2**20, 1024)):
        tasks, hazards = analyze(vector_pipeline(count=count))
        unrelated = select(hazards, "scale", "postprocess", HazardKind.RAW)
        indexed = (*hazards, *((unrelated,) * copies))
        indexing = ContractWork()
        contracts = build_memory_contracts(tasks, indexed, work=indexing)
        assert indexing.tasks_indexed == len(tasks)
        assert indexing.accesses_indexed == sum(len(task.accesses) for task in tasks)
        assert indexing.hazards_indexed == len(indexed)
        statistics = contracts.statistics
        query = ContractWork()
        via, target, _ = scale_chain(hazards, residue=1)
        for _ in range(3):
            witness = contracts.direct_witness(via, target, work=query)
            assert witness is not None
            assert witness.assumptions.contract is contracts.admission(via.producer)
        assert contracts.statistics == statistics
        assert query.witness_queries == query.capture_lookups == 3
        results.append((query.candidate_visits, query.activation_checks, query.residue_visits))
    assert results[0] == results[1]
    assert results[0][0] == 6
