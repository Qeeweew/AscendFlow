"""Explicit structured-synchronization entry during the M2 migration.

The public compiler still handles nested regions through the old entry until
M3/M4 are complete. This local entry never falls back to that planner. It
admits the physical module once, constructs memory and notification contracts,
then materializes the accepted local schedule at its actual native anchors.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp, StringAttr

from ..analysis.contracts import MemoryContracts, build_memory_contracts
from ..analysis.dependencies import DependencyAnalysis
from ..analysis.frontier import Clock
from ..analysis.hazards import bind_hazards
from ..analysis.native import extract_native_tasks
from ..analysis.progress import ProgressCertificate, establish_local_progress
from ..analysis.protocols import LocalCommand, LocalMechanism, local_commands
from ..analysis.resources import LocalEventPlan, LocalResourceWork, allocate_local_events
from ..compilation_facts import PhysicalFacts
from ..diagnostics import require
from ..dialects.io import dump
from ..dialects.loops import LoopScheduleAttr
from ..dialects.physical import LoopChannelOp, LoopSyncOp
from ..loop_model import Loop
from ..target.ascend910b import Engine
from .local_sync import LocalSelection, SelectionWork, select_local_protocols


@dataclass(frozen=True, slots=True)
class LocalSynchronization:
    module: ModuleOp
    loop: Loop
    dependencies: DependencyAnalysis
    contracts: MemoryContracts
    selection: LocalSelection
    commands: tuple[LocalCommand, ...]
    resources: LocalEventPlan
    progress: ProgressCertificate
    selection_work: SelectionWork
    resource_work: LocalResourceWork
    source_ir: str | None = None
    before_ir: str | None = None
    analysis_work: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))


def _ints(values):
    return ArrayAttr(IntAttr(value) for value in values)


def materialize_local(loop, commands, resources):
    """Write chosen relations/IDs once; native effects and issue order stay fixed.

    LoopChannel/LoopSync are the existing local physical encoding. Their
    replacement by region-independent notification operations belongs to M4;
    using this encoding does not invoke its old planner or recheck a graph.
    """
    data = next(iter(loop.actors.values()))
    scope, stream = data.kernel.body.block, data.stream
    for assignment in resources.assignments:
        protocol = assignment.protocol
        source_residues = sorted(
            (r - protocol.distance) % protocol.activation.period
            for r in protocol.activation.residues
        )
        declaration = LoopChannelOp.create(
            attributes={
                "sym_name": StringAttr(protocol.name),
                "kind": StringAttr("local"),
                "sources": ArrayAttr(
                    [
                        StringAttr(
                            protocol.source.actor.value + "/" + protocol.source.operation.uid.data
                        )
                    ]
                ),
                "destinations": ArrayAttr(
                    [
                        StringAttr(
                            protocol.destination.actor.value
                            + "/"
                            + protocol.destination.operation.uid.data
                        )
                    ]
                ),
                "distance": IntAttr(protocol.distance),
                "period": IntAttr(protocol.activation.period),
                "residues": _ints(source_residues),
                "event_ids": _ints(assignment.ids),
            }
        )
        scope.insert_op_before(declaration, stream)
    after = {}
    for command in commands:
        if command.kind == "task":
            continue
        protocol, task = command.protocol, command.task
        activation = protocol.activation
        is_set = command.kind == "set"
        residues = (
            sorted((r - protocol.distance) % activation.period for r in activation.residues)
            if is_set
            else sorted(activation.residues)
        )
        operation = LoopSyncOp.create(
            operands=[task.coordinate],
            attributes={
                "action": StringAttr(command.kind),
                "channel": StringAttr(
                    "" if protocol.mechanism is LocalMechanism.FENCE else protocol.name
                ),
                "lower": IntAttr(0 if is_set else protocol.distance),
                "upper_gap": IntAttr(protocol.distance if is_set else 0),
                "period": IntAttr(activation.period),
                "residues": _ints(residues),
            },
        )
        block = task.operation.parent
        require(block is not None, "selected native insertion anchor has been detached")
        if is_set:
            anchor = after.get(task.operation, task.operation)
            block.insert_op_after(operation, anchor)
            after[task.operation] = operation
        else:
            block.insert_op_before(operation, task.operation)


def _synchronize_admitted_local(
    owned: ModuleOp, loop: Loop, dependencies: DependencyAnalysis
) -> LocalSynchronization:
    """Consume one invocation's actual admitted facts without rebuilding them."""
    require(
        owned.body.block.first_op is loop.op,
        "admitted local facts must retain their actual owned module",
    )
    require(
        not any(isinstance(op, (LoopChannelOp, LoopSyncOp)) for op in owned.walk()),
        "structured synchronization requires an unsynchronized physical input",
    )
    tasks = extract_native_tasks(loop, clock=Clock("physical-step"))
    hazards = bind_hazards(loop, tasks, dependencies.required)
    contracts = build_memory_contracts(tasks, hazards)
    selection_work = SelectionWork()
    selection = select_local_protocols(
        tasks,
        tuple(need for task in tasks for need in contracts.admission(task).obligations),
        work=selection_work,
    )
    commands = local_commands(tasks, selection.protocols)
    progress = establish_local_progress(commands)
    resource_work = LocalResourceWork()
    reserved = {}
    data = next(iter(loop.actors.values()))
    if data.program.config:
        reserved[tasks[0].actor, Engine.S, Engine.FIX] = (0,)
    resources = allocate_local_events(selection.protocols, reserved=reserved, work=resource_work)
    # Native admission and the selected contracts have succeeded. Serialize
    # only now: a forged external literal may itself be unprintable.
    before_ir = dump(owned)
    data.stream.attributes["npu.loop_schedule"] = LoopScheduleAttr(
        IntAttr(max((task.stage for task in tasks), default=0))
    )
    materialize_local(loop, commands, resources)
    return LocalSynchronization(
        owned,
        loop,
        dependencies,
        contracts,
        selection,
        commands,
        resources,
        progress,
        selection_work,
        resource_work,
        before_ir=before_ir,
    )


def synchronize_local(module: ModuleOp) -> LocalSynchronization:
    """Independently clone/admit external physical IR on every invocation."""
    owned = module.clone()
    facts = PhysicalFacts()
    loop = facts.extract_loop(owned)
    dependencies = facts.dependency_analysis(loop)
    result = _synchronize_admitted_local(owned, loop, dependencies)
    return replace(result, analysis_work=MappingProxyType(facts.statistics()))


def lower_and_synchronize_local(source) -> LocalSynchronization:
    """Own source once and hand its exact lowering facts to local synchronization.

    Source logical-value admission and synchronization share the same Loop and
    DependencyAnalysis. No verified attribute or result from another invocation
    authorizes this handoff. External physical input uses synchronize_local.
    """
    from ..frontend import PrimFunction
    from ..lowering import lower_source

    if isinstance(source, PrimFunction):
        source = source.to_ir()
    require(isinstance(source, ModuleOp), "local source entry requires typed source IR")
    owned_source = source.clone()
    facts = PhysicalFacts()
    owned = lower_source(owned_source, _facts=facts)
    loop = facts.extract_loop(owned)
    dependencies = facts.dependency_analysis(loop)
    result = _synchronize_admitted_local(owned, loop, dependencies)
    return replace(
        result,
        source_ir=dump(owned_source),
        analysis_work=MappingProxyType(facts.statistics()),
    )
