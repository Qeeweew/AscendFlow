"""Independent bounded interleaving explorer.

Uses physical command queues, active DMA/compute tasks and binary flag state,
NOT the planner's dependency frontier or completion graph. Task starts and
retirements are separate transitions, so same-pipe issue is insufficient.
All earlier conflicting physical accesses must retire before a task begins.
Exhaustion of the state budget is reported as incomplete, never as a proof.
"""

from dataclasses import dataclass

from .model import CompileError, require
from .sync import execution


@dataclass(frozen=True)
class CheckResult:
    states: int
    terminal_states: int
    exhaustive: bool
    scope: str = "all"


def bounded_check(module, *, max_states=200_000) -> CheckResult:
    from .ir import MixedOp
    from .loop_model import is_symbolic

    require(
        not is_symbolic(module),
        "bounded interleaving check requires an explicit finite test instance; "
        "it cannot certify a symbolic runtime loop",
    )

    if isinstance(module.body.block.first_op, MixedOp):
        from .mixed_check import bounded_mixed_check

        return bounded_mixed_check(module, max_states=max_states)
    ex = execution(module)
    p = ex.program
    queues = tuple(ex.queues().values())
    instances = p.instances()
    index = {inst: i for i, inst in enumerate(instances)}
    active_pipe = {
        pipe: sum(1 << i for i, inst in enumerate(instances) if p.tasks[inst.task].pipe == pipe)
        for pipe in ex.queues()
    }
    implicit_pipe = {
        pipe: sum(
            1 << i
            for i, inst in enumerate(instances)
            if p.tasks[inst.task].pipe == pipe and p.tasks[inst.task].implicit_completion
        )
        for pipe in ex.queues()
    }
    access = []
    for inst in instances:
        task = p.tasks[inst.task]
        access.append(
            [
                (r.storage.name, a, b, mode)
                for r, mode in zip(task.refs, task.modes_for(inst.generation))
                for a, b in r.segments(inst.generation)
            ]
        )
    previous_conflicts = []
    for i, current in enumerate(access):
        conflicts = 0
        for j, old in enumerate(access[:i]):
            if any(
                a == b and lo < end and start < hi and ("W" in m or "W" in n)
                for a, lo, hi, m in current
                for b, start, end, n in old
            ):
                conflicts |= 1 << j
        previous_conflicts.append(conflicts)
    event_keys = sorted({c.event for c in ex.commands if c.event is not None})
    event_index = {e: i for i, e in enumerate(event_keys)}
    tokens = {(c.channel, c.epoch): i + 1 for i, c in enumerate(ex.commands) if c.kind == "set"}
    # Issuing Set does not block subsequent source tasks. Its effect is a
    # separate transition after the captured prefix (not after future tasks).
    notifications = []
    notification_indices = {}
    notification_pipe = {pipe: 0 for pipe in ex.queues()}
    for queue in queues:
        prefix = 0
        for cmd in queue:
            if cmd.kind == "task":
                prefix |= 1 << index[cmd.instance]
            elif cmd.kind == "set":
                i = len(notifications)
                notifications.append((cmd, prefix))
                notification_indices[(cmd.channel, cmd.epoch)] = i
                notification_pipe[cmd.pipe] |= 1 << i
    initial = ((0,) * len(queues), 0, 0, (0,) * len(event_keys), 0)
    positions = {id(cmd): pos for pos, cmd in enumerate(ex.commands)}
    scalar_gates = [
        (positions[id(cmd)], q, head, cmd)
        for q, queue in enumerate(queues)
        for head, cmd in enumerate(queue)
        if cmd.pipe == "S" and cmd.kind in {"task", "wait"}
    ]
    todo, seen, terminals = [initial], {initial}, 0
    while todo:
        heads, active, retired, flags, pending = todo.pop()
        blocked_at = len(ex.commands)
        for pos, q, head, cmd in scalar_gates:
            completed = (
                bool(retired & (1 << index[cmd.instance]))
                if cmd.kind == "task"
                else heads[q] > head
            )
            if not completed:
                blocked_at = min(blocked_at, pos)
        next_states = []
        for bit in range(len(instances)):
            if active & (1 << bit):
                next_states.append(
                    (heads, active ^ (1 << bit), retired | (1 << bit), flags, pending)
                )
        for i, (cmd, prefix) in enumerate(notifications):
            if pending & (1 << i) and not prefix & ~retired:
                e = event_index[cmd.event]
                require(flags[e] == 0, f"bounded counterexample: binary re-Set on {cmd.event}")
                changed = list(flags)
                changed[e] = tokens[(cmd.channel, cmd.epoch)]
                next_states.append((heads, active, retired, tuple(changed), pending ^ (1 << i)))
        for q, queue in enumerate(queues):
            if heads[q] == len(queue):
                continue
            cmd = queue[heads[q]]
            if positions[id(cmd)] > blocked_at:
                continue
            pipe_busy = active & active_pipe[cmd.pipe]
            new_active, new_flags, new_pending = active, flags, pending
            if cmd.kind == "task":
                if active & implicit_pipe[cmd.pipe]:
                    continue
                i = index[cmd.instance]
                missing = previous_conflicts[i] & ~retired
                if missing:
                    j = (missing & -missing).bit_length() - 1
                    a, b = instances[j], cmd.instance
                    raise CompileError(
                        f"bounded interleaving counterexample: "
                        f"{p.tasks[b.task].uid}[{b.generation}] starts before "
                        f"{p.tasks[a.task].uid}[{a.generation}] retires"
                    )
                new_active |= 1 << i
            elif cmd.kind == "set":
                new_pending |= 1 << notification_indices[(cmd.channel, cmd.epoch)]
            elif cmd.kind == "barrier":
                if pipe_busy or pending & notification_pipe[cmd.pipe]:
                    continue
            else:
                e = event_index[cmd.event]
                if flags[e] == 0:
                    continue
                require(
                    flags[e] == tokens[(cmd.channel, cmd.epoch)],
                    f"bounded counterexample: wrong epoch/channel consumed on {cmd.event}",
                )
                changed = list(flags)
                changed[e] = 0
                new_flags = tuple(changed)
            new_heads = list(heads)
            new_heads[q] += 1
            next_states.append((tuple(new_heads), new_active, retired, new_flags, new_pending))
        if not next_states:
            require(
                not active
                and not pending
                and all(h == len(q) for h, q in zip(heads, queues))
                and not any(flags),
                "bounded counterexample: deadlock or unconsumed event at finite boundary",
            )
            terminals += 1
        for new_state in next_states:
            if new_state not in seen:
                if len(seen) >= max_states:
                    return CheckResult(len(seen), terminals, False)
                seen.add(new_state)
                todo.append(new_state)
    return CheckResult(len(seen), terminals, True)
