"""Independent finite mixed-core interleavings, including FIFO counter state.

No dependency frontier, completion graph or compiler witness is consulted.
Starts, retirement and delayed notifications are distinct transitions. Sleeping
commuting transitions uses their explicit state read/write sets, including all
dispatch gates, implicit-pipe gates and physical conflict retirement bits.
Budget exhaustion is an incomplete check, never a proof for larger streams.
"""

from collections import defaultdict
from dataclasses import dataclass

from .check import CheckResult
from .mixed import MAX_CROSS_BACKLOG, merge_ranges, mixed_execution
from .model import CompileError, ranges_cover, require


@dataclass(frozen=True)
class Transition:
    identity: tuple
    state: tuple
    reads: frozenset
    writes: frozenset

    def independent(self, other):
        return not (self.writes & (other.reads | other.writes) or other.writes & self.reads)


def bounded_mixed_check(module, *, max_states=200_000, reduce=True, scope="all"):
    m, channels, commands = mixed_execution(module)
    instances = m.semantic_instances()
    require(scope in {"all", "shared"}, "interleaving scope must be all or shared")
    failure = (
        "shared protocol abstraction counterexample"
        if scope == "shared"
        else "bounded mixed counterexample"
    )
    if scope == "shared":
        # Explicit conservative protocol abstraction: private-bank work and
        # local flags are removed, not treated as instantly completed data.
        # Source arrivals still capture every retained shared-access prefix;
        # cross Wait still stops ALL later dispatch on its actor. This proves
        # only shared GM races/counters, not private banks or numerical data.
        accesses = defaultdict(set)
        for key in instances:
            for root, _, _, mode in m.physical_effects(key):
                for access in ("R", "W"):
                    if access in mode and root in m.roots:
                        accesses[(root, access)].add(key[0])
        shared = {
            root
            for root in m.roots
            if any(a != b for a in accesses[(root, "W")] for b in accesses[(root, "R")])
        }
        instances = [
            key
            for key in instances
            if any(root in shared for root, _, _, _ in m.physical_effects(key))
        ]
        selected = set(instances)
        commands = {
            actor: [
                c
                for c in sequence
                if c.kind.startswith("cross_")
                or (c.kind == "task" and (actor, c.instance) in selected)
            ]
            for actor, sequence in commands.items()
        }
    index = {key: i for i, key in enumerate(instances)}
    effects = [m.physical_effects(key) for key in instances]
    if scope == "shared":
        effects = [[effect for effect in current if effect[0] in shared] for current in effects]
    defined = defaultdict(tuple)
    for key, current in zip(instances, effects):
        for root, a, b, mode in current:
            if "R" in mode:
                initialized = root in m.roots and m.roots[root].initialized.data
                require(
                    initialized or ranges_cover(defined[root], ((a, b),)),
                    f"{failure}: {key} reads undefined {root}[{a}:{b}]",
                )
        for root, a, b, mode in current:
            if "W" in mode:
                defined[root] = merge_ranges((*defined[root], (a, b)))
    conflicts = []
    for i, current in enumerate(effects):
        conflicts.append(
            sum(
                1 << j
                for j, old in enumerate(effects[:i])
                if any(
                    a == b and lo < stop and start < hi and ("W" in mode or "W" in old_mode)
                    for a, lo, hi, mode in current
                    for b, start, stop, old_mode in old
                )
            )
        )
    queues, positions = {}, {}
    for actor, sequence in commands.items():
        for pos, cmd in enumerate(sequence):
            queues.setdefault((actor, cmd.pipe), []).append(cmd)
            positions[id(cmd)] = pos
    queue_keys = tuple(queues)
    queues = tuple(queues.values())
    pipes, implicit = defaultdict(int), defaultdict(int)
    for i, (actor, inst) in enumerate(instances):
        task = m.programs[actor].tasks[inst.task]
        pipes[(actor, task.pipe)] |= 1 << i
        if task.implicit_completion:
            implicit[(actor, task.pipe)] |= 1 << i
    events = sorted(
        {(c.actor, c.event) for seq in commands.values() for c in seq if c.event is not None}
    )
    event_index = {e: i for i, e in enumerate(events)}
    counters = sorted(
        (name, src, dst)
        for name, channel in channels.items()
        for src in channel.sources
        for dst in channel.destinations
    )
    counter_index = {c: i for i, c in enumerate(counters)}
    notifications, notification_index, notification_pipe = [], {}, defaultdict(int)
    tokens, last_cross = {}, {}
    for q, queue in enumerate(queues):
        prefix = 0
        for cmd in queue:
            if cmd.kind == "task":
                prefix |= 1 << index[(cmd.actor, cmd.instance)]
            elif cmd.kind in {"set", "cross_set"}:
                ni = len(notifications)
                prior = (
                    last_cross.get((cmd.actor, cmd.channel)) if cmd.kind == "cross_set" else None
                )
                notifications.append((cmd, prefix, prior))
                notification_index[id(cmd)] = ni
                notification_pipe[queue_keys[q]] |= 1 << ni
                if cmd.kind == "cross_set":
                    last_cross[(cmd.actor, cmd.channel)] = ni
                else:
                    tokens[(cmd.actor, cmd.channel, cmd.epoch)] = ni + 1
    gates = []
    for q, queue in enumerate(queues):
        for head, cmd in enumerate(queue):
            if cmd.kind == "cross_wait" or (cmd.pipe == "S" and cmd.kind in {"task", "wait"}):
                gates.append((positions[id(cmd)], q, head, cmd))

    def bits(mask, field):
        return {(field, i) for i in range(len(instances)) if mask & (1 << i)}

    initial = ((0,) * len(queues), 0, 0, (0,) * len(events), (0,) * len(counters), 0, 0)
    todo, states, explored, terminals = (
        [(initial, frozenset())],
        {initial},
        defaultdict(list),
        set(),
    )
    while todo:
        state, sleeping = todo.pop()
        if any(old <= sleeping for old in explored[state]):
            continue
        explored[state] = [old for old in explored[state] if not sleeping <= old] + [sleeping]
        heads, active, retired, flags, counts, pending, published = state
        blocked = {actor: len(seq) for actor, seq in commands.items()}
        for pos, q, head, cmd in gates:
            completed = (
                bool(retired & (1 << index[(cmd.actor, cmd.instance)]))
                if cmd.kind == "task"
                else heads[q] > head
            )
            if not completed:
                blocked[cmd.actor] = min(blocked[cmd.actor], pos)
        transitions = []
        for i in range(len(instances)):
            if active & (1 << i):
                new = (
                    heads,
                    active ^ (1 << i),
                    retired | (1 << i),
                    flags,
                    counts,
                    pending,
                    published,
                )
                transitions.append(
                    Transition(
                        ("retire", i),
                        new,
                        frozenset({("active", i)}),
                        frozenset({("active", i), ("retired", i)}),
                    )
                )
        for ni, (cmd, prefix, prior) in enumerate(notifications):
            if not pending & (1 << ni) or prefix & ~retired:
                continue
            if prior is not None and not published & (1 << prior):
                continue
            changed_flags, changed_counts = list(flags), list(counts)
            reads = bits(prefix, "retired") | {("pending", ni)}
            writes = {("pending", ni), ("published", ni)}
            if prior is not None:
                reads.add(("published", prior))
            if cmd.kind == "set":
                e = event_index[(cmd.actor, cmd.event)]
                require(
                    flags[e] == 0,
                    f"{failure}: binary re-Set on {cmd.actor}/{cmd.event}",
                )
                changed_flags[e] = tokens[(cmd.actor, cmd.channel, cmd.epoch)]
                reads.add(("flag", e))
                writes.add(("flag", e))
            else:
                channel = channels[cmd.channel]
                for dst in channel.destinations:
                    ci = counter_index[(cmd.channel, cmd.actor, dst)]
                    require(
                        counts[ci] < MAX_CROSS_BACKLOG,
                        f"{failure}: cross backlog exceeds 15",
                    )
                    changed_counts[ci] += 1
                    reads.add(("counter", ci))
                    writes.add(("counter", ci))
            new = (
                heads,
                active,
                retired,
                tuple(changed_flags),
                tuple(changed_counts),
                pending ^ (1 << ni),
                published | (1 << ni),
            )
            transitions.append(
                Transition(("publish", ni), new, frozenset(reads), frozenset(writes))
            )
        for q, queue in enumerate(queues):
            if heads[q] == len(queue):
                continue
            cmd = queue[heads[q]]
            if positions[id(cmd)] > blocked[cmd.actor]:
                continue
            pipe_key = (cmd.actor, cmd.pipe)
            reads = {("head", q)}
            writes = {("head", q)}
            for pos, gq, _, gate in gates:
                if gate.actor == cmd.actor and pos < positions[id(cmd)]:
                    reads.add(
                        ("retired", index[(gate.actor, gate.instance)])
                        if gate.kind == "task"
                        else ("head", gq)
                    )
            new_active, new_pending, new_flags, new_counts = (
                active,
                pending,
                list(flags),
                list(counts),
            )
            if cmd.kind == "task":
                reads |= bits(implicit[pipe_key], "active")
                if active & implicit[pipe_key]:
                    continue
                i = index[(cmd.actor, cmd.instance)]
                reads |= bits(conflicts[i], "retired")
                missing = conflicts[i] & ~retired
                if missing:
                    j = (missing & -missing).bit_length() - 1
                    raise CompileError(
                        f"{failure}: {instances[i]} starts before {instances[j]} retires"
                    )
                new_active |= 1 << i
                writes.add(("active", i))
            elif cmd.kind in {"set", "cross_set"}:
                ni = notification_index[id(cmd)]
                new_pending |= 1 << ni
                writes.add(("pending", ni))
            elif cmd.kind == "barrier":
                reads |= bits(pipes[pipe_key], "active")
                reads |= {
                    ("pending", ni)
                    for ni in range(len(notifications))
                    if notification_pipe[pipe_key] & (1 << ni)
                }
                if active & pipes[pipe_key] or pending & notification_pipe[pipe_key]:
                    continue
            elif cmd.kind == "wait":
                e = event_index[(cmd.actor, cmd.event)]
                if not flags[e]:
                    continue
                require(
                    flags[e] == tokens[(cmd.actor, cmd.channel, cmd.epoch)],
                    f"{failure}: wrong local epoch",
                )
                new_flags[e] = 0
                reads.add(("flag", e))
                writes.add(("flag", e))
            elif cmd.kind == "cross_wait":
                channel = channels[cmd.channel]
                incoming = [counter_index[(cmd.channel, src, cmd.actor)] for src in channel.sources]
                if not all(counts[ci] > 0 for ci in incoming):
                    continue
                for ci in incoming:
                    new_counts[ci] -= 1
                    reads.add(("counter", ci))
                    writes.add(("counter", ci))
            else:
                raise CompileError("unmodeled mixed command")
            new_heads = list(heads)
            new_heads[q] += 1
            new = (
                tuple(new_heads),
                new_active,
                retired,
                tuple(new_flags),
                tuple(new_counts),
                new_pending,
                published,
            )
            transitions.append(
                Transition(("dispatch", q, heads[q]), new, frozenset(reads), frozenset(writes))
            )
        if not transitions:
            require(
                not active
                and not pending
                and not any(flags)
                and not any(counts)
                and all(head == len(queue) for head, queue in zip(heads, queues)),
                f"{failure}: deadlock or unconsumed counter at finite boundary",
            )
            terminals.add(state)
        successors = []
        sleep = set(sleeping)
        enabled = {t.identity: t for t in transitions}
        for transition in transitions:
            if transition.identity in sleep:
                continue
            next_sleep = (
                frozenset(
                    identity
                    for identity in sleep
                    if identity in enabled and transition.independent(enabled[identity])
                )
                if reduce
                else frozenset()
            )
            if transition.state not in states:
                if len(states) >= max_states:
                    return CheckResult(len(states), len(terminals), False, scope)
                states.add(transition.state)
            successors.append((transition.state, next_sleep))
            if reduce:
                sleep.add(transition.identity)
        todo.extend(reversed(successors))
    return CheckResult(len(states), len(terminals), True, scope)
