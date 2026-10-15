"""Parent memory frontiers of actual fixed children, retaining both clocks.

The derived Loop contains memory-only scope_access tasks, never native IR or
completion facts. Each task maps back to one real CallOp/native/inner position.
Only the child's exact boundary word is projected; captures remain aliases of
the original BufferOp, including private buffers shared by sibling calls.
"""

from dataclasses import dataclass, replace
from math import lcm

from xdsl.ir import SSAValue

from .budget import poll
from .completion import Position, Site
from .indexing import constant_ratio, physical_terms
from .loop_model import (
    Actor,
    Dependence,
    Loop,
    Templates,
    _storage,
    loop_dependencies,
    verify_access_ownership,
)
from .model import ITEMSIZE, Ref, Task, require
from .periodic import bounded_period
from .region_summary import Access, RegionSummary, summarize_fixed_storage
from .scalar import facts_from_sizes, from_ssa
from .scope_model import ScopedProgram

MAX_SCOPE_ACCESS_SEGMENTS = 1_000_000
MAX_SCOPE_ACCESS_VISITS = 1_000_000


@dataclass(frozen=True)
class ScopeAccess:
    loop: Loop
    # The UID is local to a physical actor, preserving matching AIV quorums.
    endpoints: dict[tuple[str, str], tuple[str, Site]]
    obligations: tuple[Dependence, ...]
    origins: dict[str, SSAValue]
    summaries: dict[str, RegionSummary]
    visits: int


def _coordinate(atom):
    modulus, divisor = 0, 1
    if atom.kind == "rem":
        atom, value = atom.args
        require(value.constant is not None, "scope capture needs a static storage period")
        modulus = value.constant
    if atom.kind == "floordiv":
        atom, value = atom.args
        require(value.constant is not None, "scope capture runtime quotient is unsupported")
        divisor = value.constant
    require(
        atom.kind == "argument" and atom.args[0] in {"$parent0", "$core"},
        "scope capture mixes unsupported parent/core coordinates",
    )
    require(
        atom.args[0] == "$core" or modulus > 0 or divisor == 1,
        "scope capture unbounded task quotient is unsupported",
    )
    return atom.args[0], modulus, divisor


def _offset(offset, bound, facts, *, external):
    terms, cores = [], []
    bias = count_offset = count_core_stride = 0
    for atom, coefficient in physical_terms(offset, facts).items():
        if atom is None:
            if coefficient.constant is not None:
                bias += coefficient.constant
            else:
                ratio = None if bound is None else constant_ratio(coefficient, bound, facts)
                require(
                    ratio is not None and ratio >= 0,
                    "scope capture has an unsupported uniform offset",
                )
                count_offset += ratio
            continue
        coordinate, modulus, divisor = _coordinate(atom)
        if coordinate == "$parent0":
            require(coefficient.constant is not None, "scope parent stride must be static")
            require(coefficient.constant >= 0, "scope parent stride must be nonnegative")
            terms.append((0, coefficient.constant, modulus, divisor))
        else:
            require(external, "scope local capture cannot change with its physical group")
            if coefficient.constant is not None:
                require(coefficient.constant >= 0, "scope core stride must be nonnegative")
                cores.append((coefficient.constant, divisor, modulus))
            else:
                ratio = None if bound is None else constant_ratio(coefficient, bound, facts)
                require(
                    ratio is not None and ratio >= 0 and modulus == 0 and divisor == 1,
                    "scope capture has an unsupported runtime core translation",
                )
                count_core_stride += ratio
    require(bias >= 0, "scope capture needs a nonnegative static bias")
    require(external or not count_offset, "scope local capture cannot depend on a runtime extent")
    return bias, tuple(terms), tuple(cores), count_offset, count_core_stride


def _local_uid(leaf, task, generation):
    prefix = leaf.actor + "_"
    call = leaf.call.uid.data.removeprefix(prefix)
    native = task.uid.removeprefix(prefix)
    return f"scope_p{leaf.call.logical_phase.data}_{len(call)}_{call}_{len(native)}_{native}_j{generation}"


def _cut_cost(loop):
    """Charge exact cell splitting before access_cells creates its event lists."""
    roots = {}
    for actor, data in loop.actors.items():
        for task in data.program.tasks:
            for ref in task.refs:
                root = loop.root(actor, ref)
                frame = loop.frames.get(root, 0)
                for generation in range(loop.storage_periods[root]):
                    for lo, hi in ref.segments(generation):
                        roots.setdefault(root, []).append(
                            (lo - generation * frame, hi - generation * frame)
                        )
    cost = 0
    for root, intervals in roots.items():
        points = sorted({point for interval in intervals for point in interval})
        cuts = {point: i for i, point in enumerate(points)}
        cost += len(points) + 3 * sum(cuts[hi] - cuts[lo] for lo, hi in intervals)
        require(cost <= MAX_SCOPE_ACCESS_VISITS, "scope boundary cut/frontier budget exceeded")
        poll("scope boundary cuts", root=root, visits=cost)
    return cost


def analyze_scope_access(program: ScopedProgram) -> ScopeAccess:
    """Build parent byte obligations; the caller separately verifies each leaf.

    Inner native positions are constants in the boundary view. The only
    periodic coordinates in its Refs come from actual ancestor captures.
    Parent runtime N is never instantiated. Unknown parameter-dependent
    strides and mixed parent/core atoms fail explicitly.
    """
    require(isinstance(program, ScopedProgram), "scope accesses need an extracted ScopedProgram")
    facts = facts_from_sizes(program.parameters)
    uniforms = {size.value: from_ssa(size.value, facts=facts) for size in program.parameters}
    for parent in program.parents.values():
        if parent.limit is not None:
            uniforms[parent.limit] = program.bound
    roots, origins, captures, periods = {}, {}, {}, {}
    by_root, aliases = {}, {uid: set() for uid in program.leaves}
    lexical = {
        op: index
        for parent in program.parents.values()
        for index, op in enumerate(parent.body.block.ops)
    }
    storages = {}
    fragments = 0
    for uid, leaf in program.leaves.items():
        require(len(leaf.loop.actors) == 1, "scope boundary leaf needs one actual actor")
        normalized, data = next(iter(leaf.loop.actors.items()))
        require(
            not any(task.last for task in data.program.tasks),
            "scope first/last child branches are unsupported",
        )
        for capture in leaf.captures:
            root = capture.root
            if root.extent_count is not None:
                uniforms[root.extent_count] = from_ssa(root.extent_count, facts=facts)
            storage = _storage(root, program.parents[leaf.actor].limit, uniforms)
            name = storage.name if storage.external else leaf.actor + "/" + storage.name
            require(
                name not in origins or origins[name] is root.ref,
                "scope root name loses actual allocation identity",
            )
            origins[name] = root.ref
            if storage.external:
                roots[name] = root
            storages[leaf.actor, name] = storage
            offset = _offset(capture.offset, program.bound, facts, external=storage.external)
            alias = (
                capture.op.sym_name.data
                if storage.external
                else normalized + "/" + capture.op.sym_name.data
            )
            captures[uid, alias] = name, storage, offset
            by_root[uid, name] = offset
            aliases[uid].add(alias)
            period = periods.setdefault(name, 1)
            for _, stride, modulus, divisor in offset[1]:
                if stride and modulus:
                    period = bounded_period(lcm(period, modulus * divisor), f"scope storage {name}")
            periods[name] = period
        for task in data.program.tasks:
            for ref in task.refs:
                fragments += (
                    leaf.loop.constant_count
                    * ref.rows
                    * (program.groups if ref.storage.external else 1)
                )
    require(
        fragments <= MAX_SCOPE_ACCESS_SEGMENTS, "scope fixed-child fragment/group budget exceeded"
    )
    poll("scope child boundaries", fragments=fragments, calls=len(program.leaves))
    summaries, groups = {}, {}
    visits = fragments
    for uid, leaf in program.leaves.items():
        normalized, data = next(iter(leaf.loop.actors.items()))
        summary = summaries[uid] = summarize_fixed_storage(leaf.loop, aliases[uid])
        visits += summary.visits
        require(
            visits <= MAX_SCOPE_ACCESS_VISITS, "scope fixed-child summary visit budget exceeded"
        )
        native = {task.uid: (index, task) for index, task in enumerate(data.program.tasks)}
        for cell in summary.cells:
            effects = {}
            for access in (cell.first_write, cell.last_write):
                if access is not None:
                    effects.setdefault(access, set()).add("W")
            for span in (*cell.reads_before_write, *cell.pending_reads):
                require(
                    span.first == span.last,
                    "fixed scope boundary needs exact inner reader positions",
                )
                effects.setdefault(Access(span.task, span.first), set()).add("R")
            name = captures[uid, cell.storage][0]
            for access, modes in effects.items():
                generation = access.position.evaluate(summary.constant_count)
                index, task = native[access.task[1]]
                key = uid, task.uid, generation
                group = groups.setdefault(
                    key, {"leaf": leaf, "task": task, "index": index, "parts": {}}
                )
                dtype = min(
                    (
                        ref.dtype
                        for ref in task.refs
                        if leaf.loop.root(normalized, ref) == cell.storage
                    ),
                    key=ITEMSIZE.__getitem__,
                )
                part = name, cell.start, cell.stop, dtype
                group["parts"].setdefault(part, set()).update(modes)
    # Bounds include every repeated parent residue and actual physical GM group.
    parent_fragments = sum(
        periods[name] * (program.groups if storages[group["leaf"].actor, name].external else 1)
        for group in groups.values()
        for name, *_ in group["parts"]
    )
    require(
        parent_fragments <= MAX_SCOPE_ACCESS_SEGMENTS,
        "scope parent fragment/period/group budget exceeded",
    )
    visits += parent_fragments
    require(visits <= MAX_SCOPE_ACCESS_VISITS, "scope parent access visit budget exceeded")
    phases = {
        key: i
        for i, key in enumerate(
            sorted(
                {
                    (
                        g["leaf"].call.logical_phase.data,
                        generation,
                        g["task"].logical_phase,
                        g["index"],
                    )
                    for (_, _, generation), g in groups.items()
                }
            )
        )
    }
    issue_keys = {
        key: i
        for i, key in enumerate(
            sorted(
                {
                    (
                        g["leaf"].call.issue_order.data,
                        lexical[g["leaf"].call],
                        generation + g["task"].stage,
                        g["task"].issue_order,
                        g["index"],
                    )
                    for (_, _, generation), g in groups.items()
                }
            )
        )
    }
    tasks, endpoints = {actor: [] for actor in program.kernels}, {}
    for (uid, _, generation), group in groups.items():
        leaf, task, index = group["leaf"], group["task"], group["index"]
        refs, modes = [], []
        for (name, lo, hi, dtype), mode in sorted(group["parts"].items()):
            storage = storages[leaf.actor, name]
            bias, terms, cores, count_offset, count_core_stride = by_root[uid, name]
            refs.append(
                Ref(
                    storage,
                    ((bias + lo, 0, 0, 1), *terms),
                    hi - lo,
                    dtype,
                    storage.layout,
                    core_terms=cores,
                    count_offset=count_offset,
                    count_core_stride=count_core_stride,
                )
            )
            modes.append("".join(sorted(mode)))
        local = _local_uid(leaf, task, generation)
        endpoint = leaf.actor, local
        require(endpoint not in endpoints, "duplicate scope boundary endpoint identity")
        endpoints[endpoint] = (
            uid,
            Site((leaf.actor, task.uid), Position(False, generation), "begin", region=uid),
        )
        phase = phases[leaf.call.logical_phase.data, generation, task.logical_phase, index]
        issue = issue_keys[
            leaf.call.issue_order.data,
            lexical[leaf.call],
            generation + task.stage,
            task.issue_order,
            index,
        ]
        tasks[leaf.actor].append(
            Task(
                local,
                "scope_access",
                task.pipe,
                leaf.call.stage.data,
                tuple(refs),
                tuple(modes),
                0.0,
                (),
                issue_order=issue,
                logical_phase=phase,
            )
        )
    actors = {}
    for actor, kernel in program.kernels.items():
        actual = next(
            (
                leaf.loop.actors[next(iter(leaf.loop.actors))].program
                for leaf in program.leaves.values()
                if leaf.actor == actor
            ),
            None,
        )
        actors[actor] = Actor(
            actor,
            kernel,
            program.parents[actor],
            Templates(
                actor,
                kernel.core.data,
                tuple(value for (owner, _), value in storages.items() if owner == actor),
                tuple(tasks[actor]),
                False if actual is None else actual.config,
                program.groups,
                None if actual is None else actual.load3d,
            ),
            {},
        )
    loop = Loop(
        program.root,
        actors,
        roots,
        None,
        program.count,
        periods,
        {},
        program.parameters,
        program.bound,
    )
    poll("scope parent ownership", fragments=parent_fragments, visits=visits)
    verify_access_ownership(loop)
    visits += _cut_cost(loop)
    require(visits <= MAX_SCOPE_ACCESS_VISITS, "scope access cumulative visit budget exceeded")
    obligations = []
    for dependency in loop_dependencies(loop):
        if (
            dependency.distance == 0
            and endpoints[dependency.source][0] == endpoints[dependency.destination][0]
        ):
            continue  # Internal native hazards remain the leaf verifier's job.
        if program.count is not None:
            residues = tuple(
                r
                for r in dependency.residues
                if dependency.distance + (r - dependency.distance) % dependency.period
                < program.count
            )
            if not residues:
                continue
            dependency = replace(dependency, residues=residues)
        obligations.append(dependency)
    return ScopeAccess(loop, endpoints, tuple(obligations), origins, summaries, visits)
