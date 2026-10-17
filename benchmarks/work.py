"""Unweighted source IR inventory and explicitly scoped benchmark work records.

This is report formatting, not execution accounting or a synchronization
analysis. Branches, loop counts, actors, and groups never multiply templates.
Algorithm counts come from each runner's problem definition.
"""

from collections import Counter

from xdsl.dialects import scf
from xdsl.dialects.builtin import ModuleOp

from ascendflow.diagnostics import require
from ascendflow.dialects.physical import BufferOp, KernelOp, MixedOp, RuntimeSizeOp, TaskOp
from ascendflow.dialects.structured import (
    ActorDomainOp,
    NotificationOp,
    PipelineLoopOp,
    PipelineStageOp,
)

WORK_SCHEMA = "source-work-v1"


def structure_inventory(module: ModuleOp):
    """Describe actual IR declarations and templates without evaluating control."""
    root = module.body.block.first_op
    require(
        isinstance(root, (KernelOp, MixedOp))
        and root.next_op is None
        and root.blocks is not None
        and root.blocks.data > 0,
        "work inventory requires one source kernel with an explicit launch grid",
    )
    natives, notifications, allocations, control = Counter(), Counter(), Counter(), Counter()
    for operation in module.walk():
        if isinstance(operation, TaskOp):
            natives[operation.native.data] += 1
        elif isinstance(operation, NotificationOp):
            route = (
                f"{operation.kind.data}:{operation.source.data}->{operation.destination.data}:"
                + ",".join(actor.data for actor in operation.participants)
            )
            notifications[route] += 1
        elif isinstance(operation, BufferOp):
            allocations[operation.ref.type.bank.data] += 1
        elif isinstance(
            operation, (scf.ForOp, scf.IfOp, PipelineLoopOp, PipelineStageOp, ActorDomainOp)
        ):
            control[operation.name] += 1
    return {
        "scope": "generated source IR templates; no runtime, branch, actor, or grid weighting",
        "kernel": root.sym_name.data,
        "groups": root.blocks.data,
        "runtime_parameters": [
            operation.sym_name.data
            for operation in root.body.block.ops
            if isinstance(operation, RuntimeSizeOp)
        ],
        "native_templates": sum(natives.values()),
        "native_templates_by_native": dict(sorted(natives.items())),
        "notification_templates": sum(notifications.values()),
        "notification_templates_by_route": dict(sorted(notifications.items())),
        "allocation_declarations_by_bank": dict(sorted(allocations.items())),
        "control_templates_by_operation": dict(sorted(control.items())),
    }


def work_record(inventory, *, algorithm_counts, algorithm_scope):
    require(
        isinstance(algorithm_scope, str)
        and algorithm_scope
        and all(type(value) is int and value >= 0 for value in algorithm_counts.values()),
        "algorithm work requires a stated scope and nonnegative integer counts",
    )
    return {
        "schema": WORK_SCHEMA,
        "structure_inventory": inventory,
        "algorithm_work": {"scope": algorithm_scope, "counts": dict(algorithm_counts)},
        "runtime_requests": {
            "status": "unavailable",
            "reason": "validated runtime native, transfer, operand-byte, and notification counts are not established for this source pipeline",
        },
    }


def read_work(case, *, required_counts=()):
    """Reject reports whose counts lack the current provenance split."""
    work = case.get("work")
    require(
        isinstance(work, dict)
        and work.get("schema") == WORK_SCHEMA
        and isinstance(work.get("structure_inventory"), dict)
        and type(work["structure_inventory"].get("groups")) is int
        and work["structure_inventory"]["groups"] > 0
        and isinstance(work.get("algorithm_work"), dict)
        and isinstance(work["algorithm_work"].get("counts"), dict)
        and isinstance(work["algorithm_work"].get("scope"), str)
        and isinstance(work.get("runtime_requests"), dict)
        and work["runtime_requests"].get("status") == "unavailable",
        "benchmark analysis requires a source-work-v1 record; runtime request counts are unavailable and older requested_work records cannot establish them",
    )
    counts = work["algorithm_work"]["counts"]
    require(
        all(type(value) is int and value >= 0 for value in counts.values()),
        "benchmark algorithm counts must be nonnegative integers with an explicit scope",
    )
    missing = sorted(set(required_counts) - counts.keys())
    require(
        not missing,
        f"benchmark algorithm work is missing required counts: {', '.join(missing)}; "
        "runtime instruction requests cannot supply them",
    )
    return work
