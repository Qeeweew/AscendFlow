"""Weighted completion graph: node families repeat by a fixed storage period.

An arc (u,v,d) means u[k] precedes v[k+d] for every active translation.
Integer potentials prove well-foundedness. Coverage paths must satisfy every
node's lower/upper guard for the whole query domain, including fill and drain.
This is a parametric proof, not a sampled finite execution graph.
"""

from collections import deque
from dataclasses import dataclass

from .budget import poll
from .ir import EndOp, InvocationSyncOp, LoopChannelOp, LoopSyncOp, RegionSyncOp, TaskOp, ViewOp
from .loop_boundary import stream_operations
from .model import optional_int, require
from .native_order import instruction_points, prefix_inputs
from .periodic import MAX_PERIOD_TEMPLATES, MAX_PROOF_STATES


@dataclass(frozen=True)
class Channel:
    name: str
    kind: str
    sources: tuple[tuple[str, str], ...]
    destinations: tuple[tuple[str, str], ...]
    distance: int
    period: int
    residues: tuple[int, ...]
    ids: tuple[int, ...]
    last: bool = False
    carry: bool = False


@dataclass(frozen=True)
class Command:
    index: int
    actor: str
    task: object
    kind: str
    channel: str = ""
    lower: int = 0
    upper: int = 0
    period: int = 1
    residues: tuple[int, ...] = (0,)
    operation: object = None
    last_gap: int | None = None


@dataclass(frozen=True)
class Node:
    command: Command
    residue: int
    point: str
    guard: tuple[int, int] | None = None

    @property
    def lower(self):
        return self.command.lower if self.guard is None else self.guard[0]

    @property
    def upper(self):
        return self.command.upper if self.guard is None else self.guard[1]

    @property
    def last_gap(self):
        return self.command.last_gap if self.guard is None else None


@dataclass
class Domain:
    index: dict
    successors: list
    predecessors: list
    ancestors: dict

    def reaching(self, target):
        if target not in self.ancestors:
            # Most domains serve only a few completion targets. Computing
            # ALL node-to-node closure multiplies every arc by a large bitset,
            # particularly in counter-lifetime domains spanning 15 sends.
            seen = {target}
            pending = [target]
            while pending:
                for prior in self.predecessors[pending.pop()]:
                    if prior not in seen:
                        seen.add(prior)
                        pending.append(prior)
            bits = bytearray((len(self.predecessors) + 7) // 8)
            for node in seen:
                bits[node // 8] |= 1 << (node % 8)
            self.ancestors[target] = int.from_bytes(bits, "little")
        return self.ancestors[target]


class Graph:
    def __init__(self, period, maximum):
        self.period, self.maximum = period, maximum
        self.nodes, self.ids, self.edges = [], {}, []
        self.outgoing = []
        self.domains = {}
        self._completion = None
        self.potentials = None
        self._arcs = set()
        self.domain_states = 0
        self.domain_edges = 0
        self.path_queries = 0
        self.batch_queries = self.fact_transfers = self.batch_passes = 0

    def node(self, command, residue, point, *, guard=None):
        key = command.index, residue, point
        if key not in self.ids:
            self.ids[key] = len(self.nodes)
            self.nodes.append(Node(command, residue, point, guard))
            self.outgoing.append([])
            if self._completion is not None:
                self._completion.node(self.ids[key])
        else:
            require(self.nodes[self.ids[key]].guard == guard, "completion milestone guard changed")
        return self.ids[key]

    def edge(self, source, destination):
        a, ac = source
        b, bc = destination
        arc = a, b, bc - ac
        if arc in self._arcs:
            return
        self._arcs.add(arc)
        self.edges.append(arc)
        self.outgoing[a].append((b, bc - ac))
        if self._completion is not None:
            self._completion.edge(*arc)

    def potential(self):
        if self.potentials is not None:
            return self.potentials
        poll("completion potential", nodes=len(self.nodes), edges=len(self.edges))
        # Choose T larger than any simple family cycle. An interior cycle
        # with net advance<=0 violates these difference constraints.
        stride = len(self.nodes) + 1
        ranks = [0] * len(self.nodes)
        if self.maximum == 0:
            self.potentials = stride, ranks
            return self.potentials
        # Only successors of an increased rank can need relaxation. Keep the
        # exact full relaxation below as the cycle diagnostic if a difficult
        # graph exceeds the queue budget; the fast path grants no new edges.
        pending = deque(range(len(self.nodes)))
        queued = [True] * len(self.nodes)
        insertions = [1] * len(self.nodes)
        iterations = 0
        while pending:
            iterations += 1
            if iterations % 256 == 0:
                poll("completion potential", nodes=len(self.nodes), relaxations=iterations)
            a = pending.popleft()
            queued[a] = False
            for b, delta in self.outgoing[a]:
                required = ranks[a] + 1 - stride * delta
                if required <= ranks[b]:
                    continue
                ranks[b] = required
                if not queued[b]:
                    insertions[b] += 1
                    if insertions[b] > len(self.nodes):
                        return self._potential_full(stride)
                    pending.append(b)
                    queued[b] = True
        self.potentials = stride, ranks
        return self.potentials

    def _potential_full(self, stride):
        ranks = [0] * len(self.nodes)
        for _ in self.nodes:
            poll("completion cycle diagnosis", nodes=len(self.nodes), edges=len(self.edges))
            changed = False
            for a, b, delta in self.edges:
                required = ranks[a] + 1 - stride * delta
                if required > ranks[b]:
                    ranks[b] = required
                    changed = True
            if not changed:
                self.potentials = stride, ranks
                return self.potentials
        require(False, "symbolic fixed schedule has a completion cycle or non-well-founded carry")

    def bounds(self, node, residue, first, gap, terminal=False):
        """Translations where this milestone exists throughout a query domain."""
        meta = self.nodes[node]
        difference = meta.residue - residue
        lower = -((first + difference - meta.lower) // self.period)
        upper = (gap - meta.upper - difference) // self.period
        if meta.last_gap is not None:
            delta = gap - meta.last_gap - difference
            if not terminal or delta % self.period:
                return 0, -1
            lower, upper = max(lower, delta // self.period), min(upper, delta // self.period)
        return lower, upper

    def domain(self, residue, first, gap, terminal=False):
        key = residue, first, gap, terminal
        if key in self.domains:
            return self.domains[key]
        poll("completion domain", cached_states=self.domain_states, first=first, gap=gap)
        if self.potentials is None:
            self.potential()
        stride, ranks = self.potentials
        bounds = []
        count = 0
        for node in range(len(self.nodes)):
            lower, upper = self.bounds(node, residue, first, gap, terminal)
            count += max(0, upper - lower + 1)
            bounds.append((node, lower, upper))
        require(
            self.domain_states + count <= MAX_PROOF_STATES,
            f"completion proof state budget exceeded: cached={self.domain_states}, "
            f"requested={count}, limit={MAX_PROOF_STATES}, period={self.period}, "
            f"nodes={len(self.nodes)}, first={first}, upper_gap={gap}; "
            "no expansion or serialization fallback",
        )
        states = [
            (node, cycle) for node, lower, upper in bounds for cycle in range(lower, upper + 1)
        ]
        states.sort(key=lambda state: state[1] * stride + ranks[state[0]])
        index = {state: i for i, state in enumerate(states)}
        successors = [[] for _ in states]
        predecessors = [[] for _ in states]
        for i, (node, cycle) in enumerate(states):
            if i % 4096 == 0:
                poll("completion domain", states=len(states), materialized=i)
            for destination, delta in self.outgoing[node]:
                target = index.get((destination, cycle + delta))
                if target is not None:
                    require(
                        target > i, "integer potential does not order an actual translated edge"
                    )
                    successors[i].append((target, (node, destination, delta)))
                    predecessors[target].append(i)
            successors[i].sort()
        value = Domain(index, successors, predecessors, {})
        self.domain_states += len(states)
        self.domain_edges += sum(len(arcs) for arcs in successors)
        self.domains[key] = value
        return value

    def _query(
        self,
        source_command,
        source_g,
        destination_command,
        destination_g,
        minimum=0,
        prefix=False,
        *,
        source_point=None,
        destination_point=None,
        remaining=None,
        at_end=False,
    ):
        """Normalize an obligation to its certified translation domain."""
        p = self.period
        first = max(
            minimum, destination_command.lower, source_command.lower - (source_g - destination_g)
        )
        residue = destination_g % p
        first += (residue - first) % p
        gap = max(destination_command.upper, source_command.upper + source_g - destination_g)
        if remaining is not None:
            require(remaining >= gap, "boundary query excludes an endpoint")
            gap = remaining
        terminal_gaps = [
            last + shift
            for last, shift in [
                (source_command.last_gap, source_g - destination_g),
                (destination_command.last_gap, 0),
            ]
            if last is not None
        ]
        terminal = bool(terminal_gaps) or at_end
        if terminal_gaps:
            if len(set(terminal_gaps)) != 1 or terminal_gaps[0] < gap:
                return None  # The endpoint activation intersection is empty.
            gap = terminal_gaps[0]
            if self.maximum is not None:
                actual = self.maximum - gap - 1
                if actual < first or actual % p != residue:
                    return None
                # Prove the enclosing translation family, not all iterations
                # from zero to this constant N. `actual` only selects the real
                # residue; making it the domain's lower bound expands O(N)
                # history and is unnecessary for this stronger family proof.
        if self.maximum is not None and first + gap >= self.maximum:
            return None
        source_point = source_point or (
            "prefix"
            if prefix
            else "end"
            if source_command.kind in {"task", "set", "cross_set"}
            else "begin"
        )
        source = self.ids[source_command.index, source_g % p, source_point]
        destination_point = destination_point or (
            "end" if destination_command.kind in {"set", "cross_set"} else "begin"
        )
        destination = self.ids[destination_command.index, destination_g % p, destination_point]
        return (
            (residue, first, gap, terminal),
            (source, source_g // p - destination_g // p),
            (destination, 0),
        )

    def completion(self):
        if self._completion is None:
            from .symbolic_completion import NativeCompletion

            self._completion = NativeCompletion(self)
        return self._completion

    def covered_many(self, queries):
        """Batch obligations on native families; no translated instance graph."""
        queries = tuple(queries)
        self.batch_queries += len(queries)
        proof = self.completion()
        proof.register(queries)
        answers = [proof.covered(query) for query in queries]
        self.batch_passes += 1
        self.fact_transfers = proof.transfers
        return answers

    def path(
        self,
        source_command,
        source_g,
        destination_command,
        destination_g,
        minimum=0,
        prefix=False,
        *,
        omitted=(),
        _shorten=True,
        source_point=None,
        destination_point=None,
        remaining=None,
        at_end=False,
    ):
        self.path_queries += 1
        poll("completion path", queries=self.path_queries, states=self.domain_states)
        query = self._query(
            source_command,
            source_g,
            destination_command,
            destination_g,
            minimum,
            prefix,
            source_point=source_point,
            destination_point=destination_point,
            remaining=remaining,
            at_end=at_end,
        )
        if query is None:
            return ()
        if (
            _shorten
            and source_command.kind == "cross_wait"
            and destination_command.kind == "cross_set"
            and source_command.channel == destination_command.channel
        ):
            # Mode-2 sends of one channel retire in FIFO order. Prove a
            # nearby release, then append the ACTUAL send-FIFO arcs up to the
            # requested fifteenth send. Every intervening Set is active when
            # both endpoints of the long query are active. This composes
            # guarded paths; it does not weaken the backlog obligation.
            first_send = source_g - source_command.lower + 1
            for early in range(first_send, destination_g):
                if early % destination_command.period not in destination_command.residues:
                    continue
                path = self.path(
                    source_command,
                    source_g,
                    destination_command,
                    early,
                    omitted=omitted,
                    _shorten=False,
                )
                if path is None:
                    continue
                tail = []
                current = early
                while current < destination_g:
                    following = current + 1
                    while (
                        following % destination_command.period not in destination_command.residues
                    ):
                        following += 1
                    a = self.ids[destination_command.index, current % self.period, "end"]
                    b = self.ids[destination_command.index, following % self.period, "end"]
                    arc = a, b, following // self.period - current // self.period
                    if arc not in self._arcs or a in omitted or b in omitted:
                        break
                    tail.append(arc)
                    current = following
                if current == destination_g:
                    return path + tuple(tail)
        domain = self.domain(*query[0])
        source, destination = query[1][0], query[2][0]
        initial, target = domain.index.get(query[1]), domain.index.get(query[2])
        if initial is None or target is None:
            return None
        reaching = domain.reaching(target)
        successors = domain.successors
        if not reaching & (1 << initial):
            return None
        if omitted:
            if source in omitted or destination in omitted:
                return None
            # The immutable closure is an upper bound after deletion. A
            # concrete search must skip EVERY arc incident on a removed
            # command, including a cross Wait's controller-gate edges.
            if initial == target:
                return ()
            seen = {initial}
            stack = [(initial, iter(reversed(successors[initial])))]
            path = []
            while stack:
                _, following = stack[-1]
                for node, arc in following:
                    if arc[0] in omitted or arc[1] in omitted or node in seen:
                        continue
                    if not reaching & (1 << node):
                        continue
                    seen.add(node)
                    path.append(arc)
                    if node == target:
                        return tuple(path)
                    stack.append((node, iter(reversed(successors[node]))))
                    break
                else:
                    stack.pop()
                    if path:
                        path.pop()
            return None
        path = []
        while initial != target:
            following, arc = next(
                (node, arc) for node, arc in reversed(successors[initial]) if reaching & (1 << node)
            )
            path.append(arc)
            initial = following
        return tuple(path)


def channels(loop, *, virtual=False):
    protocol_period = loop.protocol_period
    result = {}
    tasks = {
        (actor, task.uid): task
        for actor, data in loop.actors.items()
        for task in data.program.tasks
    }
    for op in loop.protocol_scope.body.block.ops:
        if not isinstance(op, LoopChannelOp):
            continue
        name = op.sym_name.data
        sources = tuple(tuple(value.data.split("/", 1)) for value in op.sources)
        destinations = tuple(tuple(value.data.split("/", 1)) for value in op.destinations)
        require(
            name not in result
            and sources
            and destinations
            and set(sources + destinations) <= tasks.keys(),
            "channel references missing/duplicate endpoints",
        )
        require(
            len({actor for actor, _ in sources}) == len(sources)
            and len({actor for actor, _ in destinations}) == len(destinations),
            "channel needs one endpoint per participant",
        )
        require(op.kind.data in {"local", "cross"}, "unknown channel mechanism")
        if op.kind.data == "local":
            require(
                len(sources) == len(destinations) == 1
                and sources[0][0] == destinations[0][0]
                and tasks[sources[0]].pipe != tasks[destinations[0]].pipe,
                "local event needs one actor and different pipes",
            )
        else:
            require(
                ({a for a, _ in sources}, {a for a, _ in destinations})
                in [({"aic"}, {"aiv0", "aiv1"}), ({"aiv0", "aiv1"}, {"aic"})],
                "cross counter requires the real AIC/two-AIV quorum",
            )
        period = op.period.data
        residues = tuple(v.data for v in op.residues)
        ids = tuple(v.data for v in op.event_ids)
        require(
            period > 0
            and protocol_period % period == 0
            and residues == tuple(sorted(set(residues)))
            and residues
            and all(0 <= r < period for r in residues)
            and op.distance.data >= 0,
            "invalid symbolic epoch family",
        )
        require(
            virtual
            or (
                ids
                and len(set(ids)) == len(ids)
                and all(0 <= n < 8 for n in ids)
                and (op.kind.data != "cross" or len(ids) == 1)
            ),
            "event resource assignment missing/invalid",
        )
        terminal = optional_int(op.last, 0)
        carry = optional_int(op.carry, 0)
        require(
            carry in {0, 1}
            and (
                not carry
                or (
                    op.kind.data == "local"
                    and not terminal
                    and period == 1
                    and op.distance.data > 0
                    and len(ids) == op.distance.data
                    and loop.invocation is not None
                )
            ),
            "carried channel requires an invocation and a full local slot-credit ring",
        )
        require(
            terminal in {0, 1} and (not terminal or op.kind.data == "local"),
            "terminal protocols currently require one local actor",
        )
        require(
            all(not tasks[endpoint].last or terminal for endpoint in sources + destinations),
            "a last-only task cannot participate in an unconditional channel",
        )
        result[name] = Channel(
            name,
            op.kind.data,
            sources,
            destinations,
            op.distance.data,
            period,
            residues,
            ids,
            bool(terminal),
            bool(carry),
        )
    return result


def commands(loop, channel_map, *, period=None):
    protocol_period = loop.protocol_period if period is None else period
    result, endpoints, native = {}, {}, {}
    number = 0
    for actor, data in loop.actors.items():
        before, after, pending, previous = {}, {}, [], None
        tasks = {task.uid: task for task in data.program.tasks}
        parent = None
        for op, _ in stream_operations(data.stream):
            if op.parent is not parent:
                require(not pending, "wait cannot escape its physical boundary region")
                parent, previous = op.parent, None
            if isinstance(op, TaskOp):
                before[op.uid.data], pending, previous = pending, [], op.uid.data
            elif isinstance(op, LoopSyncOp):
                require(
                    op.generation is data.stream.body.block.args[0],
                    "sync uses a different task coordinate",
                )
                if op.action.data == "set":
                    require(
                        previous is not None and not pending, "Set must follow its native producer"
                    )
                    after.setdefault(previous, []).append(op)
                else:
                    pending.append(op)
            else:
                require(
                    isinstance(op, (ViewOp, InvocationSyncOp, EndOp))
                    or loop.region is not None
                    and isinstance(op, (LoopChannelOp, RegionSyncOp)),
                    f"unknown loop stream effect {op.name}",
                )
        require(not pending, "trailing waits require an explicit drain contract")
        sequence = []

        def add(task, kind, operation=None, *, actor=actor, sequence=sequence):
            nonlocal number
            if operation is None:
                command = Command(number, actor, task, kind, last_gap=0 if task.last else None)
            else:
                require(
                    operation.action.data in {"set", "wait", "barrier"},
                    "unknown synchronization action",
                )
                period, residues = operation.period.data, tuple(v.data for v in operation.residues)
                require(
                    period > 0
                    and protocol_period % period == 0
                    and residues == tuple(sorted(set(residues)))
                    and residues
                    and all(0 <= r < period for r in residues),
                    "invalid sync epoch family",
                )
                command = Command(
                    number,
                    actor,
                    task,
                    kind,
                    operation.channel.data,
                    operation.lower.data,
                    operation.upper_gap.data,
                    period,
                    residues,
                    operation,
                    optional_int(operation.last_gap, 0)
                    if operation.last_gap is not None
                    else 0
                    if task.last
                    else None,
                )
                require(
                    command.lower >= 0
                    and command.upper >= 0
                    and (command.last_gap is None or command.last_gap >= 0),
                    "negative sync activation bound",
                )
                require(
                    not task.last or command.last_gap == 0,
                    "sync activation disagrees with its last-only enclosing task",
                )
                if kind == "barrier":
                    require(
                        command.channel == "" and command.upper == 0,
                        "barrier cannot acquire a hidden resource",
                    )
                else:
                    require(command.channel in channel_map, "sync refers to unknown channel")
                    channel = channel_map[command.channel]
                    expected_gap = (
                        (channel.distance if kind.endswith("set") else 0) if channel.last else None
                    )
                    require(
                        command.last_gap == expected_gap,
                        "Set/Wait terminal activation does not match its channel",
                    )
                    if kind.endswith("set"):
                        require(
                            (actor, task.uid) in channel.sources
                            and (command.lower, command.upper, command.period, command.residues)
                            == (0, channel.distance, channel.period, channel.residues),
                            "Set activation/producer does not match its channel",
                        )
                    else:
                        shifted = tuple(
                            sorted(
                                (r + channel.distance) % channel.period for r in channel.residues
                            )
                        )
                        require(
                            (actor, task.uid) in channel.destinations
                            and (command.lower, command.upper, command.period, command.residues)
                            == (channel.distance, 0, channel.period, shifted),
                            "Wait activation/consumer does not match its channel",
                        )
                    key = actor, command.channel, "set" if kind.endswith("set") else "wait"
                    require(key not in endpoints, "duplicate symbolic notification endpoint")
                    endpoints[key] = command
            number += 1
            sequence.append(command)
            if kind == "task":
                native[actor, task.uid] = command

        positions = {uid: i for i, uid in enumerate(tasks)}
        for task in sorted(
            tasks.values(), key=lambda task: (task.issue_order, positions[task.uid])
        ):
            for op in before[task.uid]:
                channel = channel_map.get(op.channel.data)
                kind = op.action.data
                if channel is not None and channel.kind == "cross":
                    kind = "cross_" + kind
                add(task, kind, op)
            add(task, "task")
            for op in reversed(after.get(task.uid, ())):
                channel = channel_map.get(op.channel.data)
                add(
                    task,
                    "cross_set" if channel is not None and channel.kind == "cross" else "set",
                    op,
                )
        result[actor] = sequence
    for name, channel in channel_map.items():
        require(
            all(
                (actor, name, kind) in endpoints
                for kind, participants in [("set", channel.sources), ("wait", channel.destinations)]
                for actor, _ in participants
            ),
            "channel has an unpaired symbolic endpoint",
        )
    return result, endpoints, native


def completion_graph(loop, channel_map, *, period=None, _planning=None):
    """Linear guarded prefixes; no pairwise guard-bypass arcs.

    Analysis-only issue/queue/retire/implicit milestones exist at each physical
    clock position, even when that position's instruction is skipped. They
    summarize dispatch, starts and captured retirement SEPARATELY. They add no
    ASC instructions or storage. A removed Wait loses all its semantic edges;
    a pure prefix cannot retain that Wait's controller gate.
    """
    sequences, endpoints, native = commands(loop, channel_map, period=period)
    period = loop.protocol_period if period is None else period
    count = sum(len(sequence) for sequence in sequences.values())
    require(
        period * count <= MAX_PERIOD_TEMPLATES,
        f"protocol residue/command budget exceeded (period={period}, commands={count}); "
        "no serialization fallback",
    )
    graph = Graph(period, loop.constant_count) if _planning is None else _planning
    for actor, sequence in sequences.items():
        maximum = max(command.task.stage for command in sequence)
        timeline = []
        for step in range(period):
            for command in sequence:
                generation = step - command.task.stage
                residue, cycle = generation % period, generation // period
                # These prefixes summarize the physical for-loop position:
                # 0 <= generation + stage < N + maximum.
                guard = -command.task.stage, command.task.stage - maximum
                milestones = {
                    point: (graph.node(command, residue, point, guard=guard), cycle)
                    for point in ("issue", "queue", "retire", "implicit")
                }
                inputs = (
                    milestones
                    if _planning is None
                    else _planning.admit(command, residue, milestones)
                )
                begin = end = prefix = None
                if residue % command.period in command.residues:
                    points = instruction_points(
                        command,
                        inputs,
                        lambda point, c=command, r=residue, k=cycle: (graph.node(c, r, point), k),
                        graph.edge,
                    )
                    begin, end, prefix = points["begin"], points["end"], points.get("prefix")
                timeline.append((command, milestones, begin, end, prefix))

        def shift(point):
            return None if point is None else (point[0], point[1] - 1)

        previous_period = [
            (
                command,
                {name: shift(point) for name, point in milestones.items()},
                shift(begin),
                shift(end),
                shift(prefix),
            )
            for command, milestones, begin, end, prefix in timeline
        ]
        prior_pipe, prior_send = {}, {}
        for item in previous_period:
            command, _, _, end, _ = item
            prior_pipe[command.task.pipe] = item
            if command.kind == "cross_set" and end is not None:
                prior_send[command.channel] = end
        prior_global = previous_period[-1]
        for item in timeline:
            command, milestones, begin, end, prefix = item
            pipe = command.task.pipe
            previous_command, previous_milestones, _, previous_end, _ = prior_global
            graph.edge(previous_milestones["issue"], milestones["issue"])
            if previous_end is not None:
                previous_points = {
                    "begin": prior_global[2],
                    "end": previous_end,
                    "prefix": prior_global[4],
                }
                if gate := prefix_inputs(previous_command, previous_points).get("issue"):
                    graph.edge(gate, milestones["issue"])
            old, old_milestones, old_begin, old_end, old_prefix = prior_pipe[pipe]
            for point in ("queue", "retire", "implicit"):
                graph.edge(old_milestones[point], milestones[point])
            if old_begin is not None:
                previous_points = {"begin": old_begin, "end": old_end, "prefix": old_prefix}
                for point, source in prefix_inputs(old, previous_points).items():
                    if point != "issue":
                        graph.edge(source, milestones[point])
            if command.kind == "cross_set" and end is not None:
                if command.channel in prior_send:
                    graph.edge(prior_send[command.channel], end)
                prior_send[command.channel] = end
            if _planning is not None:
                _planning.link(prior_global, prior_pipe[pipe], item)
            prior_pipe[pipe] = prior_global = item
    for name, channel in channel_map.items():
        for residue in range(period):
            if residue % channel.period not in channel.residues:
                continue
            for actor, _ in channel.sources:
                source = endpoints[actor, name, "set"]
                for destination_actor, _ in channel.destinations:
                    destination = endpoints[destination_actor, name, "wait"]
                    graph.edge(
                        (graph.node(source, residue, "end"), 0),
                        (
                            graph.node(destination, (residue + channel.distance) % period, "begin"),
                            (residue + channel.distance) // period,
                        ),
                    )
    return graph, sequences, endpoints, native
