"""Actual first/interior/last memory transfers on one retained parent clock.

The fixed child coordinate is part of an endpoint identity. Activation always
describes the parent coordinate; no runtime parent count is instantiated.
Anchored relations retain every reader when one endpoint is a boundary rather
than a constant-distance translation of the other endpoint.
"""

from dataclasses import dataclass, field, fields, replace
from math import lcm

from .budget import poll
from .completion import CountDomain, Position, Site, _first, coalesce
from .completion_flow import ALL
from .diagnostics import require
from .loop_model import verify_access_ownership
from .model import Ref, Task
from .periodic import bounded_period
from .region_summary import summarize_fixed_storage
from .scope_access import ScopeAccess, _cut_cost, _local_uid
from .scope_domains import Activation, ScopedDependence, affine, validate_activation
from .symbolic_completion import le, meet

MAX_GUARDED_ACCESS_VISITS = 1_000_000
MAX_GUARDED_ACCESS_PARTITIONS = 16_384
MAX_GUARDED_ACCESS_RELATIONS = 65_536

FIRST = Position(False, 0)
LAST = Position(True, 0)
ALL_ACTIVATION = Activation()


@dataclass(frozen=True)
class BoundaryEffect:
    endpoint: tuple[str, str]
    storage: str
    start: int
    stop: int
    modes: str
    activation: Activation = ALL_ACTIVATION
    logical_order: tuple = ()


@dataclass(frozen=True)
class AnchoredDependence:
    """One anchored endpoint against every active instance of the other side.

    Activation constrains the unanchored endpoint's epoch. With two anchors,
    only its count predicate is used. Neither a reader span nor its retirement
    may be replaced by one extremal instance without a native prefix proof.
    """

    source: tuple[str, str]
    destination: tuple[str, str]
    source_position: Position | None
    destination_position: Position | None
    activation: Activation
    hazard: str
    storage: str
    start: int
    stop: int
    prefix: bool = False

    def __post_init__(self):
        require(
            self.source_position is not None or self.destination_position is not None,
            "anchored dependence needs an actual boundary endpoint",
        )


@dataclass(frozen=True)
class _Occurrence:
    effect: BoundaryEffect
    position: Position


@dataclass(frozen=True)
class _Span:
    effect: BoundaryEffect
    activation: Activation


@dataclass
class _Budget:
    visits: int = 0

    def charge(self, amount):
        self.visits += amount
        require(
            self.visits <= MAX_GUARDED_ACCESS_VISITS,
            "guarded scope access visit budget exceeded",
        )
        poll("guarded scope access", visits=self.visits)


def _geometry(active):
    first, last = active.first, active.last
    if first == FIRST and last == LAST:
        return {"single", "first", "middle", "last"}
    if first == last == FIRST:
        return {"single", "first"}
    if first == last == LAST:
        return {"single", "last"}
    if not first.end and first.offset in (0, 1) and last.end and last.offset in (0, 1):
        return (
            {"middle"}
            | ({"first"} if not first.offset else set())
            | ({"last"} if not last.offset else set())
        )
    require(False, "guarded scope access supports parent first/interior/last domains only")


def _singleton_count(domain):
    first = _first(domain)
    if domain.exact is not None or (
        domain.maximum is not None and first + domain.period > domain.maximum
    ):
        return CountDomain(1, 0, exact=first)
    return domain


def _domains(records, budget):
    """Exact predicate partition; numeric interval widths never control work."""
    period, cuts = 1, {1, 2, 3}
    for record in records:
        value = validate_activation(record.activation)
        period = bounded_period(
            lcm(period, value.period, value.count.period), "guarded scope access predicates"
        )
        domain = value.count
        if domain.exact is not None:
            cuts.update((max(1, domain.exact), max(1, domain.exact + 1)))
        else:
            cuts.add(max(1, domain.period if domain.minimum is None else domain.minimum))
            if domain.maximum is not None:
                cuts.add(max(1, domain.maximum + 1))
    bounds = sorted(cuts)
    require(
        period * len(bounds) <= MAX_GUARDED_ACCESS_PARTITIONS,
        "guarded scope access predicate partition budget exceeded",
    )
    budget.charge(period * len(bounds))
    for index, lower in enumerate(bounds):
        maximum = bounds[index + 1] - 1 if index + 1 < len(bounds) else None
        for residue in range(period):
            domain = CountDomain(period, residue, minimum=lower, maximum=maximum)
            if maximum is None or _first(domain) <= maximum:
                yield _singleton_count(domain), period


def _cells(records, budget):
    grouped = {}
    for record in records:
        require(
            isinstance(record, BoundaryEffect)
            and type(record.start) is int
            and type(record.stop) is int
            and 0 <= record.start < record.stop
            and record.modes in {"R", "W", "RW"}
            and isinstance(record.activation, Activation)
            and isinstance(record.logical_order, tuple),
            "invalid guarded scope boundary effect",
        )
        _geometry(record.activation)
        grouped.setdefault(record.storage, []).append(record)
    for root, work in sorted(grouped.items()):
        points = sorted({point for record in work for point in (record.start, record.stop)})
        cuts = {point: index for index, point in enumerate(points)}
        incidences = sum(cuts[record.stop] - cuts[record.start] for record in work)
        budget.charge(len(points) + incidences)
        cells = [[] for _ in points[:-1]]
        for record in work:
            for index in range(cuts[record.start], cuts[record.stop]):
                cells[index].append(record)
        for lo, hi, cell in zip(points[:-1], points[1:], cells, strict=True):
            if cell:
                yield root, lo, hi, cell


def _word(records, kind, residue, domain, budget):
    budget.charge(len(records))
    result = {}
    for record in records:
        active = record.activation
        if (
            not active.count.contains(_first(domain))
            or kind not in _geometry(active)
            or residue % active.period not in active.residues
        ):
            continue
        key = record.logical_order, record.endpoint
        old = result.get(key)
        result[key] = (
            record
            if old is None
            else replace(record, modes="".join(sorted(set(old.modes + record.modes))))
        )
    return tuple(result[key] for key in sorted(result))


def _nonempty(active):
    domain = meet(active.count, le(affine(active.first), affine(active.last)))
    return None if domain is None else replace(active, count=_singleton_count(domain))


def _middle_span(record, domain, period, residue):
    first = 1 + (residue - 1) % period
    return _nonempty(
        Activation(domain, Position(False, first), Position(True, 1), period, (residue,))
    )


class _Relations:
    def __init__(self, root, lo, hi, initialized, budget):
        self.root, self.lo, self.hi = root, lo, hi
        self.initialized, self.budget = initialized, budget
        self.relations = set()

    def save(self, relation):
        self.budget.charge(1)
        self.relations.add(relation)
        require(
            len(self.relations) <= MAX_GUARDED_ACCESS_RELATIONS,
            "guarded scope access relation budget exceeded",
        )

    def undefined(self, effect, domain):
        require(
            domain is None or self.initialized,
            f"{effect.endpoint} has an undefined guarded first read on "
            f"{self.root}[{self.lo}:{self.hi}]",
        )

    def points(self, source, destination, hazard, domain):
        if source is None or domain is None:
            return
        a, b = source.position, destination.position
        if domain.exact is not None:
            a, b = (
                Position(False, a.evaluate(domain.exact)),
                Position(False, b.evaluate(domain.exact)),
            )
        left, right = affine(a), affine(b)
        domain = meet(domain, le(left, right))
        domain = meet(domain, meet(le((0, 0), left), le(right, (1, -1))))
        if domain is None:
            return
        source_key, target_key = source.effect.endpoint, destination.effect.endpoint
        if a == b and source_key == target_key:
            return
        fields = hazard, self.root, self.lo, self.hi
        if left[0] == right[0]:
            self.save(
                ScopedDependence(
                    source_key,
                    target_key,
                    right[1] - left[1],
                    Activation(_singleton_count(domain), a, a),
                    *fields,
                )
            )
        else:
            self.save(
                AnchoredDependence(
                    source_key, target_key, a, b, Activation(_singleton_count(domain)), *fields
                )
            )

    def span(self, anchor, span, hazard, domain, *, source_anchor):
        active = _nonempty(replace(span.activation, count=meet(span.activation.count, domain)))
        if active is None:
            return
        source = anchor.effect.endpoint if source_anchor else span.effect.endpoint
        target = span.effect.endpoint if source_anchor else anchor.effect.endpoint
        self.save(
            AnchoredDependence(
                source,
                target,
                anchor.position if source_anchor else None,
                None if source_anchor else anchor.position,
                active,
                hazard,
                self.root,
                self.lo,
                self.hi,
            )
        )

    def scan(self, word, position, domain, writer=None, pending=()):
        pending = list(pending)
        for effect in word:
            destination = _Occurrence(effect, position)
            if "R" in effect.modes:
                if writer is None:
                    self.undefined(effect, domain)
                else:
                    self.points(writer, destination, "RAW", domain)
                pending.append(destination)
            if "W" in effect.modes:
                self.points(writer, destination, "WAW", domain)
                for reader in pending:
                    if isinstance(reader, _Span):
                        self.span(destination, reader, "WAR", domain, source_anchor=False)
                    else:
                        self.points(reader, destination, "WAR", domain)
                writer, pending = destination, []
        return writer, pending

    def periodic(self, cycle, domain, period):
        writes = [index for index, (_, effect) in enumerate(cycle) if "W" in effect.modes]
        if not writes:
            return
        last = writes[-1]
        sg, source = cycle[last]
        writer = sg - period, source
        pending = [(g - period, effect) for g, effect in cycle[last + 1 :] if "R" in effect.modes]

        def edge(source, target, hazard):
            sg, se = source
            dg, de = target
            distance = dg - sg
            if not distance and se.endpoint == de.endpoint:
                return
            residue = sg % period
            first = 1 + (residue - 1) % period
            active = _nonempty(
                Activation(
                    domain,
                    Position(False, first),
                    Position(True, 1 + distance),
                    period,
                    (residue,),
                )
            )
            if active is not None:
                self.save(
                    ScopedDependence(
                        se.endpoint,
                        de.endpoint,
                        distance,
                        active,
                        hazard,
                        self.root,
                        self.lo,
                        self.hi,
                    )
                )

        for generation, effect in cycle:
            target = generation, effect
            if "R" in effect.modes:
                edge(writer, target, "RAW")
                pending.append(target)
            if "W" in effect.modes:
                edge(writer, target, "WAW")
                for reader in pending:
                    edge(reader, target, "WAR")
                writer, pending = target, []

    def long(self, records, domain, period):
        first = _word(records, "first", 0, domain, self.budget)
        writer, pending = self.scan(first, FIRST, domain)
        words = {
            residue: _word(records, "middle", residue, domain, self.budget)
            for residue in range(period)
        }
        cycle = tuple(
            (generation, effect)
            for generation in range(1, period + 1)
            for effect in words[generation % period]
        )
        self.budget.charge(len(cycle))
        self.periodic(cycle, domain, period)
        has_writer = any("W" in effect.modes for _, effect in cycle)
        if has_writer:
            # Only this structurally bounded prefix can still use the first
            # word's version. Later middle words use the periodic frontier.
            current, readers = writer, list(pending)
            for generation, effect in cycle:
                active = meet(domain, CountDomain(1, 0, minimum=generation + 2))
                if active is not None:
                    current, readers = self.scan(
                        (effect,), Position(False, generation), active, current, readers
                    )
                if "W" in effect.modes:
                    break
        else:
            for residue, word in words.items():
                for effect in word:
                    active = _middle_span(effect, domain, period, residue)
                    if active is None:
                        continue
                    if writer is None:
                        self.undefined(effect, active.count)
                    else:
                        self.span(writer, _Span(effect, active), "RAW", domain, source_anchor=True)

        tail_residue = (_first(domain) - 1) % period
        tail = _word(records, "last", tail_residue, domain, self.budget)
        gap = None
        if has_writer:
            gap = next(
                distance
                for distance in range(1, period + 1)
                if any("W" in effect.modes for effect in words[(tail_residue - distance) % period])
            )
            active = meet(domain, CountDomain(1, 0, minimum=gap + 2))
            if active is not None:
                word = words[(tail_residue - gap) % period]
                last = max(index for index, effect in enumerate(word) if "W" in effect.modes)
                before = _Occurrence(word[last], Position(True, gap))
                readers = [
                    _Occurrence(effect, Position(True, distance))
                    for distance in range(gap, 0, -1)
                    for effect in (
                        word[last + 1 :]
                        if distance == gap
                        else words[(tail_residue - distance) % period]
                    )
                    if "R" in effect.modes
                ]
                self.scan(tail, LAST, active, before, readers)
        fallback = (
            domain if gap is None else meet(domain, CountDomain(1, 0, minimum=2, maximum=gap + 1))
        )
        if fallback is not None:
            readers = list(pending)
            for residue, word in words.items():
                for effect in word:
                    active = _middle_span(effect, fallback, period, residue)
                    if active is not None and "R" in effect.modes:
                        readers.append(_Span(effect, active))
            self.scan(tail, LAST, fallback, writer, readers)


def _coalesce(relations):
    families = {}
    for relation in relations:
        key = replace(relation, activation=replace(relation.activation, count=ALL))
        families.setdefault(key, []).append(relation.activation.count)
    return tuple(
        sorted(
            (
                replace(key, activation=replace(key.activation, count=domain))
                for key, domains in families.items()
                for domain in coalesce(domains)
            ),
            key=repr,
        )
    )


def dependent_frontiers(records, *, initialized=(), statistics=None):
    """Derive exact logical RAW/WAW/WAR; no completion or event is implied.

    Predicate and storage periods are static. One period, its startup prefix
    and its terminal suffix suffice; large runtime counts never control work.
    """
    records, budget, relations = tuple(records), _Budget(), []
    initialized = frozenset(initialized)
    for root, lo, hi, cell in _cells(records, budget):
        frontier = _Relations(root, lo, hi, root in initialized, budget)
        for domain, period in _domains(cell, budget):
            if domain.exact == 1:
                frontier.scan(_word(cell, "single", 0, domain, budget), FIRST, domain)
            else:
                frontier.long(cell, domain, period)
        relations.extend(frontier.relations)
        require(
            len(relations) <= MAX_GUARDED_ACCESS_RELATIONS,
            "guarded scope access relation budget exceeded",
        )
    if statistics is not None:
        statistics.update(visits=budget.visits)
    return _coalesce(relations)


@dataclass(frozen=True)
class GuardedScopeAccess(ScopeAccess):
    case_summaries: dict = field(default_factory=dict)
    guarded_roots: frozenset[str] = frozenset()
    effects: tuple[BoundaryEffect, ...] = ()


def _variants(leaf):
    cases = getattr(leaf, "cases", {})
    if not cases:
        return (("all", ALL_ACTIVATION, leaf.loop),)
    require(
        set(cases) == {"single", "first", "middle", "last"},
        "guarded access needs every actual parent case, including empty cases",
    )
    return tuple((label, case.activation, case.loop) for label, case in cases.items())


def _case_effects(program, base, budget):
    """Retain exact fixed-child native positions before any boundary reduction."""
    effects, metadata, signatures, summaries = [], {}, {}, {}
    for uid, leaf in program.leaves.items():
        full = {
            task.uid: (index, task)
            for data in leaf.loop.actors.values()
            for index, task in enumerate(data.program.tasks)
        }
        variants = signatures[uid] = {}
        for label, activation, loop in _variants(leaf):
            signature = variants[label] = {}
            if loop is None:
                summaries[uid, label] = None
                continue
            require(len(loop.actors) == 1, "guarded fixed child needs one actual actor")
            count = loop.constant_count
            require(
                type(count) is int and count >= 0, "guarded child domain must be genuinely fixed"
            )
            aliases = {
                loop.root(actor, ref)
                for actor, data in loop.actors.items()
                for task in data.program.tasks
                for ref in task.refs
            }
            fragments = sum(
                (int(count > 0) if task.last else count) * ref.rows
                for data in loop.actors.values()
                for task in data.program.tasks
                for ref in task.refs
            )
            budget.charge(fragments)
            summary = summaries[uid, label] = summarize_fixed_storage(loop, aliases)
            budget.charge(summary.visits + summary.definition_visits)
            for actor, data in loop.actors.items():
                for task in data.program.tasks:
                    require(
                        task.uid in full, "parent case introduced a native absent from its template"
                    )
                    index, original = full[task.uid]
                    require(
                        (task.native, task.pipe, task.logical_phase)
                        == (original.native, original.pipe, original.logical_phase),
                        "parent case changed its actual native identity",
                    )
                    generations = (
                        (count - 1,) if task.last and count else () if task.last else range(count)
                    )
                    for generation in generations:
                        endpoint = leaf.actor, _local_uid(leaf, task, generation)
                        metadata[endpoint] = leaf, original, generation, index
                        logical = (
                            leaf.call.logical_phase.data,
                            generation,
                            task.logical_phase,
                            index,
                            leaf.actor,
                        )
                        for number, (ref, mode) in enumerate(
                            zip(task.refs, task.modes_for(generation), strict=True)
                        ):
                            alias = loop.root(actor, ref)
                            require(
                                (uid, alias) in base.capture_roots,
                                "guarded native lost its original capture root",
                            )
                            root = base.capture_roots[uid, alias]
                            for lo, hi in ref.segments(generation):
                                signature.setdefault(root, set()).add(
                                    (task.uid, generation, number, lo, hi, mode)
                                )
                                effects.append(
                                    (
                                        uid,
                                        BoundaryEffect(
                                            endpoint, root, lo, hi, mode, activation, logical
                                        ),
                                        ref,
                                    )
                                )
    guarded = set()
    for variants in signatures.values():
        if set(variants) == {"all"}:
            continue
        for root in set().union(*(set(signature) for signature in variants.values())):
            values = [frozenset(signature.get(root, ())) for signature in variants.values()]
            if any(value != values[0] for value in values[1:]):
                guarded.add(root)
    return effects, metadata, frozenset(guarded), summaries


def _augment_projection(program, base, records, metadata, budget):
    """Add real native endpoints; modes here describe geometry, not activation."""
    endpoints = dict(base.endpoints)
    by_actor = {
        actor: {task.uid: task for task in data.program.tasks}
        for actor, data in base.loop.actors.items()
    }
    storages = {
        (actor, storage.name if storage.external else actor + "/" + storage.name): storage
        for actor, data in base.loop.actors.items()
        for storage in data.program.storages
    }
    parts = {}
    for effect, ref in records:
        actor, local = effect.endpoint
        leaf, task, generation, _ = metadata[effect.endpoint]
        endpoints.setdefault(
            effect.endpoint,
            (
                leaf.call.uid.data,
                Site(
                    (actor, task.uid),
                    Position(False, generation),
                    "begin",
                    region=leaf.call.uid.data,
                ),
            ),
        )
        storage = storages[actor, effect.storage]
        _, _, cores, count_offset, count_core = base.capture_offsets[
            leaf.call.uid.data, effect.storage
        ]
        projected = Ref(
            storage,
            ((effect.start, 0, 0, 1),),
            effect.stop - effect.start,
            ref.dtype,
            storage.layout,
            core_terms=cores,
            count_offset=count_offset,
            count_core_stride=count_core,
        )
        parts.setdefault(effect.endpoint, {}).setdefault(projected, set()).update(effect.modes)
        if local not in by_actor[actor]:
            by_actor[actor][local] = Task(
                local, "scope_access", task.pipe, leaf.call.stage.data, (), (), 0.0, ()
            )
    full = {
        (uid, task.uid): (index, task)
        for uid, leaf in program.leaves.items()
        for data in leaf.loop.actors.values()
        for index, task in enumerate(data.program.tasks)
    }
    lexical = {
        op: index
        for parent in program.parents.values()
        for index, op in enumerate(parent.body.block.ops)
    }
    phase_keys, issue_keys = {}, {}
    for endpoint, (uid, site) in endpoints.items():
        leaf = program.leaves[uid]
        index, task = full[uid, site.task[1]]
        generation = site.position.evaluate(leaf.loop.constant_count)
        phase_keys[endpoint] = leaf.call.logical_phase.data, generation, task.logical_phase, index
        issue_keys[endpoint] = (
            leaf.call.issue_order.data,
            lexical[leaf.call],
            generation + task.stage,
            task.issue_order,
            index,
        )
    phases = {key: index for index, key in enumerate(sorted(set(phase_keys.values())))}
    issues = {key: index for index, key in enumerate(sorted(set(issue_keys.values())))}
    actors = {}
    for actor, data in base.loop.actors.items():
        tasks = []
        for local, task in by_actor[actor].items():
            endpoint = actor, local
            combined = {ref: set(mode) for ref, mode in zip(task.refs, task.modes, strict=True)}
            for ref, mode in parts.get(endpoint, {}).items():
                combined.setdefault(ref, set()).update(mode)
            tasks.append(
                replace(
                    task,
                    refs=tuple(combined),
                    modes=tuple("".join(sorted(mode)) for mode in combined.values()),
                    logical_phase=phases[phase_keys[endpoint]],
                    issue_order=issues[issue_keys[endpoint]],
                )
            )
        actors[actor] = replace(data, program=replace(data.program, tasks=tuple(tasks)))
    loop = replace(base.loop, actors=actors)
    verify_access_ownership(loop)
    budget.charge(_cut_cost(loop))
    return loop, endpoints


def analyze_guarded_access(program, *, base=None):
    """Replace only condition-dependent roots, preserving ordinary fast paths.

    ``base`` is the ordinary descriptive full-template projection. It supplies
    alias geometry and unconditional-root dependencies, never permission to
    execute a conditional native or initialize a guarded root.
    """
    if base is None:
        from .scope_access import analyze_scope_access

        base = analyze_scope_access(program, definitions=True)
    if not any(getattr(leaf, "cases", {}) for leaf in program.leaves.values()):
        return base
    budget = _Budget(base.visits)
    projected, metadata, guarded, summaries = _case_effects(program, base, budget)
    for root in guarded:
        translations = {
            offset[2:] for (_, name), offset in base.capture_offsets.items() if name == root
        }
        require(
            len(translations) == 1,
            "conditional scope captures need one common core/count translation",
        )
    records, effects = [], []
    for uid, effect, ref in projected:
        if effect.storage not in guarded:
            continue
        offset = base.capture_offsets[uid, effect.storage]
        bias, terms, _, _, _ = offset
        require(
            base.loop.storage_periods[effect.storage] == 1
            and not any(stride for _, stride, _, _ in terms),
            "conditional scope storage needs stationary parent captures",
        )
        bias += sum(value for value, _, _, _ in terms)
        active = effect.activation
        if program.count is not None:
            if program.count == 0:
                continue
            domain = meet(active.count, CountDomain(1, 0, exact=program.count))
            if domain is None:
                continue
            active = replace(active, count=domain)
        effect = replace(
            effect, start=bias + effect.start, stop=bias + effect.stop, activation=active
        )
        records.append((effect, ref))
        effects.append(effect)
    initialized = {root for root in guarded if bool(base.origins[root].owner.initialized.data)}
    statistics = {}
    obligations = dependent_frontiers(effects, initialized=initialized, statistics=statistics)
    budget.charge(statistics["visits"])
    loop, endpoints = _augment_projection(program, base, records, metadata, budget)
    obligations = tuple(
        dep
        for dep in obligations
        if not (
            isinstance(dep, ScopedDependence)
            and dep.distance == 0
            and endpoints[dep.source][0] == endpoints[dep.destination][0]
        )
    )
    values = {item.name: getattr(base, item.name) for item in fields(ScopeAccess)}
    values.update(
        loop=loop,
        endpoints=endpoints,
        obligations=tuple(dep for dep in base.obligations if dep.storage not in guarded)
        + obligations,
        visits=budget.visits,
        definition_roots=base.definition_roots - guarded,
    )
    return GuardedScopeAccess(
        **values, case_summaries=summaries, guarded_roots=guarded, effects=tuple(effects)
    )
