"""Join exclusive native cases through the same actual call's physical ports.

SCCs and transitive reductions are computed inside a verified case. A parent
prefix is shared only as transparent incoming state; it never unions native
retirements from mutually exclusive cases into an unconditional certificate.
"""

from collections import defaultdict
from dataclasses import dataclass, replace

from .budget import bounded, poll
from .child_completion import FixedChildProjection
from .completion import Site
from .loop_graph import Command
from .model import require
from .periodic import MAX_PERIOD_TEMPLATES
from .scope_completion import MAX_SCOPE_ARCS, _ClockSlot, _components, compose_calls
from .scope_domains import Activation


@dataclass(frozen=True)
class ProjectionCase:
    label: str
    activation: Activation
    projection: FixedChildProjection


@dataclass(frozen=True)
class ConditionalProjection:
    count: int
    native_templates: int
    cases: tuple[ProjectionCase, ...]

    def statistics(self):
        return {
            "native_templates": self.native_templates,
            "parent_cases": {case.label: case.projection.statistics() for case in self.cases},
        }

    def identity(self):
        inputs = tuple(sorted({p for case in self.cases for p in case.projection.inputs}, key=repr))
        outputs = tuple(replace(p, side="output") for p in inputs)
        return FixedChildProjection(
            self.count,
            inputs,
            outputs,
            (),
            frozenset(zip(inputs, outputs, strict=True)),
            0,
            0,
            self.native_templates,
            0,
            0,
            (),
        )


@bounded
def compose_conditional_calls(calls, *, period, count):
    skeletons, conditional = {}, {}
    for uid, (call, projection) in calls.items():
        if isinstance(projection, FixedChildProjection):
            skeletons[uid] = call, projection
            continue
        require(isinstance(projection, ConditionalProjection), "unknown child projection contract")
        require(
            projection.cases and len({c.label for c in projection.cases}) == len(projection.cases),
            "conditional child needs distinct nonempty analysis cases",
        )
        require(
            all(c.projection.count == projection.count for c in projection.cases),
            "conditional child changes its fixed inner count",
        )
        skeletons[uid] = call, projection.identity()
        conditional[uid] = projection
    # This skeleton carries only input port state to output ports. Inactive
    # calls and inactive pipe cases therefore retain their original prefixes.
    result = compose_calls(skeletons, period=period, count=count)
    graph = result.graph
    number = 1 + max(node.command.index for node in graph.nodes)
    sites, commands, activations, case_ports = defaultdict(list), defaultdict(list), {}, {}
    for uid, variant in conditional.items():
        call = calls[uid][0]
        for case in variant.cases:
            poll("conditional child composition", call=uid, case=case.label)
            groups, mapping, arcs = _components(case.projection)
            require(
                len(graph.nodes) + period * len(groups) <= MAX_PERIOD_TEMPLATES,
                "conditional parent completion node budget exceeded",
            )
            require(
                len(graph.edges) + period * (len(arcs) + len(case.projection.inputs) * 2)
                <= MAX_SCOPE_ARCS,
                "conditional parent completion arc budget exceeded",
            )
            nodes = {}
            for component, members in enumerate(groups):
                actors = {
                    point.task[0] if isinstance(point, Site) else point.actor for point in members
                }
                require(len(actors) == 1, "one native case component changed its physical actor")
                command = Command(
                    number,
                    next(iter(actors)),
                    _ClockSlot(uid, call.stage.data),
                    "scope_component",
                    operation=call,
                )
                number += 1
                for residue in range(period):
                    node = graph.node(command, residue, "component")
                    graph.ids[command.index, residue, "begin"] = node
                    activations[node] = case.activation
                    nodes[component, residue] = node
                for point in members:
                    if isinstance(point, Site):
                        commands[uid, point].append(command)
                        for residue in range(period):
                            sites[uid, point, residue].append(nodes[component, residue])
                    else:
                        for residue in range(period):
                            case_ports[uid, case.label, point, residue] = nodes[component, residue]
            for residue in range(period):
                for source, target in arcs:
                    graph.edge((nodes[source, residue], 0), (nodes[target, residue], 0))
                for point in case.projection.inputs:
                    graph.edge(result.at(uid, point, residue), (nodes[mapping[point], residue], 0))
                for point in case.projection.outputs:
                    graph.edge((nodes[mapping[point], residue], 0), result.at(uid, point, residue))
    return replace(
        result,
        case_sites={key: tuple(value) for key, value in sites.items()},
        case_commands={key: tuple(value) for key, value in commands.items()},
        node_activations=activations,
        case_ports=case_ports,
        projections={uid: projection for uid, (_, projection) in calls.items()},
    )
