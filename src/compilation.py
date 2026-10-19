"""One owned source pipeline and immutable public compilation artifacts.

External physical snapshots are inspection output, not proof-carrying input.
Source programs follow one region-based synchronization pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from xdsl.dialects.builtin import ModuleOp

from .budget import limit, poll
from .diagnostics import ErrorKind, require
from .dialects.io import dump, parse
from .dialects.source import ProgramOp


@dataclass(frozen=True, slots=True, init=False)
class Compilation:
    asc: str
    source_ir: str
    before_ir: str
    plan_ir: str
    name: str
    _report_json: str
    _timings_json: str
    _schedule_json: str

    def __init__(
        self,
        module,
        *,
        host_launcher=True,
        timeout_s=60.0,
        cancel=None,
        debug_verify=False,
    ):
        from .frontend import PrimFunction
        from .scalar import expression_scope

        with limit(timeout_s, cancel), expression_scope():
            require(
                isinstance(module, PrimFunction)
                or isinstance(module, ModuleOp)
                and isinstance(module.body.block.first_op, ProgramOp),
                "compile accepts a DSL PrimFunction or source ProgramOp; "
                "physical snapshots cannot establish source compilation contracts",
                kind=ErrorKind.UNSUPPORTED,
                phase="compilation input",
            )
            self._build_structured(
                module,
                host_launcher=host_launcher,
                debug_verify=debug_verify,
            )

    def _build_structured(self, module, *, host_launcher, debug_verify):
        """Orchestrate the sole source pipeline; print its final physical IR."""
        from dataclasses import asdict

        from .backend.structured import EmissionWork, _emit_owned
        from .passes.structured_sync import synchronize_structure

        started = perf_counter()
        require(type(debug_verify) is bool, "debug_verify must be an explicit boolean policy")
        poll("structured source compilation")
        plan = synchronize_structure(module)
        assembled = perf_counter()
        if debug_verify:
            plan.module.verify()  # Explicit structural check, not a completion proof.
        poll("final physical IR emission")
        emission = EmissionWork()
        asc = _emit_owned(plan, host_launcher=host_launcher, work=emission)
        emitted = perf_counter()
        poll("owned compilation snapshots")
        report = {
            "kernel": plan.module.body.block.first_op.sym_name.data,
            "pipeline": "structured-regions",
            "verification_mode": "generated-lifetimes",
            "debug_structural_verification": debug_verify,
            "analysis_work": {
                "lowering": dict(plan.structure.work),
                "bounds": {
                    "accesses": plan.bounds.checked_accesses,
                    "view_contracts": plan.bounds.view_contracts,
                    "domain_contracts": plan.bounds.domain_contracts,
                },
                "admission": dict(plan.admission.work),
                "dependencies": asdict(plan.dependencies.work),
                "resources": asdict(plan.resources.work),
                "generation": {
                    **asdict(plan.generation.order.work),
                    "cell_channels": len(plan.generation.channels),
                    "cell_fragments": sum(len(c.members) for c in plan.generation.channels),
                    "handback_rings": len(plan.generation.leases)
                    - sum(len(c.members) for c in plan.generation.channels),
                },
                "emission": {
                    "native_templates": emission.native_templates,
                    "native_prints": emission.native_prints,
                    "actions_printed": emission.actions_printed,
                    "output_bytes": emission.output_bytes,
                },
                "condition_simplification": asdict(plan.condition_work),
                "integer_queries": {
                    "scope": "SMT backend calls after each consumer's local query cache",
                    "environments": plan.integer_queries,
                },
            },
            "native_templates": len(plan.executable.natives),
            "required_dependencies": len(plan.dependencies.required),
            "selected_protocols": len(plan.selection.notifications),
            "sync_instructions": len(plan.resources.actions),
            "witnesses": [
                {
                    "source": getattr(coverage.requirement.source, "operation", None).uid.data
                    if hasattr(getattr(coverage.requirement.source, "operation", None), "uid")
                    else None,
                    "consumer": (
                        coverage.requirement.consumer.operation.uid.data
                        if hasattr(coverage.requirement.consumer.operation, "uid")
                        else coverage.requirement.consumer.operation.name
                    ),
                    "hazard": coverage.requirement.kind.value,
                    "distance": coverage.requirement.distance,
                    "rule": coverage.rule,
                    "protocol": None
                    if coverage.notification is None
                    else coverage.notification.name,
                }
                for coverage in plan.selection.coverage
            ],
        }
        schedule = {
            scope.name: [
                {
                    "task": task.operation.uid.data,
                    "stage": task.stage,
                    "issue_order": task.lexical,
                    "actor": task.actor.value,
                }
                for task in body
            ]
            for scope, body in plan.selection.dispatch.bodies.items()
            if body
        }
        snapshots = {
            "asc": asc,
            "name": plan.module.body.block.first_op.sym_name.data,
            "source_ir": dump(plan.structure.source),
            "before_ir": plan.before_ir,
            "plan_ir": dump(plan.module),
            "_report_json": json.dumps(report),
            "_schedule_json": json.dumps(schedule),
        }
        frozen = perf_counter()
        snapshots["_timings_json"] = json.dumps(
            {
                "assembly_s": assembled - started,
                "stages": dict(plan.timings),
                "emission_s": emitted - assembled,
                "snapshot_s": frozen - emitted,
                "total_s": frozen - started,
            }
        )
        poll("completed structured compilation")
        for name, value in snapshots.items():
            object.__setattr__(self, name, value)

    @property
    def report(self):
        return json.loads(self._report_json)

    @property
    def timings(self):
        return json.loads(self._timings_json)

    @property
    def schedule(self):
        return json.loads(self._schedule_json)

    @property
    def plan(self):
        """Return a fresh editable IR, which has no cached verification status."""
        return parse(self.plan_ir)

    def explain(self):
        """Return the already established capture/acquire coverage witnesses."""
        return self.report["witnesses"]

    def write(self, directory, *, diagnostics=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "before.xdsl").write_text(self.before_ir)
        (directory / "after.xdsl").write_text(self.plan_ir)
        (directory / "source.xdsl").write_text(self.source_ir)
        (directory / "schedule.json").write_text(self._schedule_json + "\n")
        (directory / f"{self.name}.asc").write_text(self.asc)
        (directory / "verification.json").write_text(self._report_json + "\n")
        (directory / "compilation.json").write_text(self._timings_json + "\n")
        if diagnostics:
            (directory / "witnesses.json").write_text(json.dumps(self.explain(), indent=2) + "\n")


def compile(
    module,
    *,
    host_launcher=True,
    timeout_s=60.0,
    cancel=None,
    debug_verify=False,
):
    """DSL/source IR -> frozen ASC, established facts, and inspection snapshots."""
    return Compilation(
        module,
        host_launcher=host_launcher,
        timeout_s=timeout_s,
        cancel=cancel,
        debug_verify=debug_verify,
    )
