"""Owned compilation snapshots; ordinary reuse never rechecks a mutable IR.

The constructor runs the final physical checker. Only immutable strings leave
this invocation; a requested xDSL plan is parsed into a fresh mutable module.
Passing that module to verification/emission performs a new complete check.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .codegen import _emit_checked
from .ir import KernelOp, TaskOp, dump, parse
from .lowering import lower_source
from .model import require
from .source_ir import ProgramOp
from .sync import autosync, explain_plan, verify_plan


@dataclass(frozen=True, slots=True, init=False)
class Compilation:
    asc: str
    source_ir: str | None
    before_ir: str
    plan_ir: str
    name: str
    _report_json: str
    _timings_json: str
    _schedule_json: str

    def __init__(self, module, *, synchronize=True, host_launcher=True):
        started = perf_counter()
        require(type(synchronize) is bool, "synchronize must be an explicit boolean policy")
        from .frontend import PrimFunction

        if isinstance(module, PrimFunction):
            module = module.to_ir()
        frontend = perf_counter()
        module = module.clone()
        owned = perf_counter()
        source = isinstance(module.body.block.first_op, ProgramOp)
        if source:
            require(synchronize, "source regions must lower through automatic synchronization")
            physical = lower_source(module)
        else:
            physical = module
        lowered = perf_counter()
        checked = []
        if synchronize:
            plan = autosync(physical, _capture=checked.append)
        else:
            # Expert physical synchronization uses the same independent checker.
            plan = physical.clone()
            checking = perf_counter()
            report, analysis = verify_plan(plan, _details=True)
            checked.append((report, analysis, perf_counter() - checking))
        planned = perf_counter()
        report, analysis, final_check_s = checked[0]
        code = _emit_checked(plan, analysis, host_launcher=host_launcher)
        emitted = perf_counter()
        schedule = {}
        for kernel in plan.walk():
            if not isinstance(kernel, KernelOp):
                continue
            actor = kernel.parent_op()
            key = (
                kernel.sym_name.data if actor is not None and actor.name == "npu.mixed" else "local"
            )
            # Dispatch cases have separate tables, preserving their selector scopes.
            key = f"{key}:{len(schedule)}"
            schedule[key] = [
                {
                    "task": op.uid.data,
                    "region": op.attributes["source_region"].data,
                    "stage": op.stage.data,
                    "logical_phase": 0 if op.logical_phase is None else op.logical_phase.data,
                    "issue_order": op.issue_order.data,
                    "logical_generation": "g",
                    "physical_step": f"g + {op.stage.data}",
                }
                for op in kernel.walk()
                if isinstance(op, TaskOp) and "source_region" in op.attributes
            ]
        snapshots = {
            "asc": code,
            "name": plan.body.block.first_op.sym_name.data,
            "source_ir": dump(module) if source else None,
            "before_ir": dump(physical),
            "plan_ir": dump(plan),
            "_report_json": json.dumps(report),
            "_schedule_json": json.dumps(schedule),
        }
        frozen = perf_counter()
        snapshots["_timings_json"] = json.dumps(
            {
                "frontend_s": frontend - started,
                "own_input_s": owned - frontend,
                "lowering_s": lowered - owned,
                "planning_and_final_check_s": planned - lowered,
                "planning_s": planned - lowered - final_check_s,
                "final_check_s": final_check_s,
                "emission_s": emitted - planned,
                "snapshot_s": frozen - emitted,
                "total_s": frozen - started,
            }
        )
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
        """Explicit, potentially expensive per-access synchronization witnesses."""
        return explain_plan(self.plan)

    def write(self, directory, *, diagnostics=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "before.xdsl").write_text(self.before_ir)
        (directory / "after.xdsl").write_text(self.plan_ir)
        if self.source_ir is not None:
            (directory / "source.xdsl").write_text(self.source_ir)
            (directory / "schedule.json").write_text(self._schedule_json + "\n")
        (directory / f"{self.name}.asc").write_text(self.asc)
        (directory / "verification.json").write_text(self._report_json + "\n")
        (directory / "compilation.json").write_text(self._timings_json + "\n")
        if diagnostics:
            (directory / "witnesses.json").write_text(json.dumps(self.explain(), indent=2) + "\n")


def compile(module, *, synchronize=True, host_launcher=True):
    """Typed source or physical IR -> frozen code, proof summary, and IR snapshots."""
    return Compilation(module, synchronize=synchronize, host_launcher=host_launcher)
