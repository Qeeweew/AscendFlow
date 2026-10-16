"""Emit retained parent calls using the ordinary verified native-region emitter."""

from collections import defaultdict
from itertools import combinations, product

from .asc_scalar import ScalarScope
from .codegen import QUALIFIER
from .diagnostics import require
from .dialects.physical import CallOp, LoopChannelOp, LoopSyncOp, QuerySyncOp, ScopeSyncOp, TaskOp
from .scalar import INT64_MAX, Expr
from .scope_predicates import PARENT_CASES, case_activation, guard_active
from .scope_variants import operation_cases, scope_operations


def _source_epoch(action, distance, scalar):
    """Source epoch for a real instruction at this actor's parent generation."""
    if action == "set" or not distance:
        return scalar.value("npu_parent_g")
    return (
        scalar.sub("npu_parent_g", distance)
        if distance > 0
        else scalar.add("npu_parent_g", -distance)
    )


def _integer_expression(build):
    """Simplify one guard operand without hoisting it before short-circuit guards."""
    scalar = ScalarScope()
    lines = scalar.finish([scalar.use(build(scalar))])
    require(len(lines) == 1, "scope guard unexpectedly requires an unguarded scalar temporary")
    return lines[0]


def _activation_guard(channel, action):
    """Conditions for one explicit source-epoch activation at an actual call.

    Calls already enforce 0 <= parent_g < N. Translate source bounds to this
    call's coordinate before forming a source-epoch remainder: for a negative
    distance, computing parent_g-distance in an inactive tail can overflow.
    Event arithmetic remains inside the resulting guard, as in legacy code.
    """
    from .scope_domains import validate_activation

    activation = channel.activation
    validate_activation(activation, distance=channel.distance)
    require(action in {"set", "wait", "barrier"}, "unsupported scope synchronization action")
    require(
        (channel.period, channel.residues) == (activation.period, activation.residues),
        "scope activation period/residues disagree with the channel",
    )
    count = activation.count
    require(
        all(
            type(value) is int and -INT64_MAX <= value <= INT64_MAX
            for value in (
                channel.distance,
                activation.period,
                *activation.residues,
                activation.first.offset,
                activation.last.offset,
                count.period,
                count.residue,
                *(
                    value
                    for value in (count.exact, count.minimum, count.maximum)
                    if value is not None
                ),
            )
        ),
        "scope activation has an invalid ASC int64 literal",
    )
    terms = []
    if count.exact is not None:
        terms.append(f"npu_parent_n == {count.exact}")
    else:
        minimum = count.period if count.minimum is None else count.minimum
        if minimum:
            terms.append(f"npu_parent_n >= {minimum}")
        if count.maximum is not None:
            terms.append(f"npu_parent_n <= {count.maximum}")
        if count.period != 1:
            remainder = _integer_expression(lambda scalar: scalar.rem("npu_parent_n", count.period))
            terms.append(f"{remainder} == {count.residue}")

    shift = 0 if action == "set" else channel.distance

    def affine(position):
        if count.exact is not None:
            return 0, position.evaluate(count.exact) + shift
        return (1, shift - 1 - position.offset) if position.end else (0, position.offset + shift)

    def expression(boundary):
        coefficient, offset = boundary
        require(
            -INT64_MAX <= offset <= INT64_MAX,
            "scope activation boundary has an invalid ASC int64 literal",
        )

        def build(scalar):
            if not coefficient:
                return scalar.value(offset)
            if offset < 0:
                return scalar.sub("npu_parent_n", -offset)
            if offset:
                return scalar.add("npu_parent_n", offset)
            return scalar.value("npu_parent_n")

        return _integer_expression(build)

    first, last = affine(activation.first), affine(activation.last)
    if first == last:
        terms.append(f"npu_parent_g == {expression(first)}")
    else:
        if first[0] or first[1] > 0:
            terms.append(f"npu_parent_g >= {expression(first)}")
        if not last[0] or last[1] < -1:
            terms.append(f"npu_parent_g <= {expression(last)}")
    if activation.period != 1:
        remainder = _integer_expression(
            lambda scalar: scalar.rem(
                _source_epoch(action, channel.distance, scalar), activation.period
            )
        )
        terms.append("(" + " || ".join(f"{remainder} == {r}" for r in activation.residues) + ")")
    return terms


def _parent_tasks(checked):
    return {
        (actor, task.uid): task
        for actor, data in checked.access.loop.actors.items()
        for task in data.program.tasks
    }


def _parent_case_guard(cases):
    """Minimize a union of the four cases inside an already active parent call.

    First/last are two Boolean coordinates: the single case satisfies both,
    and a middle generation satisfies neither. At most nine Boolean cubes
    exist, so exact simplification costs a fixed amount independent of N.
    Bounds come from the same case activations used by verification.
    """
    selected = frozenset(cases)
    require(selected <= frozenset(PARENT_CASES), "unknown ASC parent case")
    if not selected:
        return "false"
    activations = {case: case_activation(case) for case in PARENT_CASES}
    truth = {
        case: (guard_active("first", case), guard_active("last", case)) for case in PARENT_CASES
    }
    candidates = []
    for cube in product((None, True, False), repeat=2):
        covered = frozenset(
            case
            for case, values in truth.items()
            if all(term is None or term == value for term, value in zip(cube, values))
        )
        if covered and covered <= selected:
            candidates.append((cube, covered))
    best, cost = None, None
    for size in range(1, len(selected) + 1):
        for cover in combinations(candidates, size):
            if frozenset().union(*(covered for _, covered in cover)) != selected:
                continue
            score = (sum(sum(value is not None for value in cube) for cube, _ in cover), size)
            if cost is None or score < cost:
                best, cost = cover, score
    require(best is not None, "ASC parent cases have no Boolean cover")

    def boundary(position):
        return (
            _integer_expression(lambda scalar: scalar.sub("npu_parent_n", position.offset + 1))
            if position.end
            else str(position.offset)
        )

    first = boundary(activations["first"].first)
    last = boundary(activations["last"].last)
    clauses = []
    for cube, covered in best:
        if covered == frozenset(("single",)):
            clauses.append(f"npu_parent_n == {activations['single'].count.exact}")
            continue
        terms = []
        for value, point, comparison in zip(cube, (first, last), (">", "<")):
            if value is not None:
                terms.append(f"npu_parent_g {'==' if value else comparison} {point}")
        if not terms:
            return ""
        clause = " && ".join(terms)
        clauses.append(f"({clause})" if len(terms) > 1 and len(best) > 1 else clause)
    return " || ".join(clauses)


def scope_command_guard(leaf, sequence):
    """Map the complete descriptor back to actual guarded native/sync ops.

    Descriptor clones deliberately have no parent-case fields. Identity, or
    one unique native endpoint and action, locates the original operation;
    only that operation's real annotations determine its activation.
    """
    operations = tuple(scope_operations(leaf.stream))
    if not any(guards or "scope_cases" in op.attributes for op, guards in operations):
        return None
    all_cases = frozenset(PARENT_CASES)
    natives, declarations, sync_cases, by_endpoint = {}, {}, {}, defaultdict(list)
    pending, previous = [], None

    def cases(op, guards):
        return frozenset(operation_cases(op)) & frozenset(
            case for case in PARENT_CASES if all(guard_active(kind, case) for kind in guards)
        )

    def placed(op, uid):
        by_endpoint[op.channel.data, op.action.data, uid].append(op)

    for op, guards in operations:
        if isinstance(op, LoopChannelOp):
            name = op.sym_name.data
            require(name not in declarations, "duplicate ASC child channel")
            declarations[name] = cases(op, guards)
        elif isinstance(op, TaskOp):
            uid = op.uid.data
            require(uid not in natives, "duplicate ASC child native uid")
            natives[uid] = cases(op, guards)
            for sync in pending:
                placed(sync, uid)
            pending, previous = [], uid
        elif isinstance(op, LoopSyncOp):
            sync_cases[op] = cases(op, guards)
            if op.action.data == "set":
                require(previous is not None and not pending, "ASC Set has no native producer")
                placed(op, previous)
            else:
                pending.append(op)
    require(not pending, "ASC acquisition has no following native consumer")
    mapped, native_seen, sync_seen = {}, set(), set()
    for command in sequence:
        uid = command.task.uid
        if command.kind == "task":
            require(uid in natives and uid not in native_seen, "ASC native descriptor changed")
            native_seen.add(uid)
            active = natives[uid]
        else:
            action = command.kind.removeprefix("cross_")
            choices = by_endpoint.get((command.channel, action, uid), ())
            operation = command.operation
            if operation not in choices and isinstance(operation, LoopSyncOp):
                choices = tuple(
                    op
                    for op in choices
                    if {
                        name: value
                        for name, value in op.attributes.items()
                        if name != "scope_cases"
                    }
                    == operation.attributes
                )
            if operation not in choices:
                require(len(choices) == 1, "ASC synchronization has no unique actual operation")
                (operation,) = choices
            require(operation not in sync_seen, "ASC synchronization descriptor is duplicated")
            sync_seen.add(operation)
            active = sync_cases[operation]
            if command.channel:
                require(command.channel in declarations, "ASC child sync has no actual channel")
                active &= declarations[command.channel]
        require(active, "ASC command has contradictory parent cases")
        mapped[id(command)] = _parent_case_guard(active) if active != all_cases else ""
    require(
        native_seen == natives.keys() and sync_seen == sync_cases.keys(),
        "ASC needs the complete actual child descriptor, not one parent case",
    )

    def emit(command, code):
        require(id(command) in mapped, "ASC command is outside its actual child descriptor")
        guard = mapped[id(command)]
        if not guard:
            return code
        # A channel comment may follow an inner protocol guard. Close the
        # parent guard before that comment can consume its enclosing brace.
        body, marker, comment = code.partition("//")
        return f"if ({guard}) {{ {body.rstrip()} }}" + (f" //{comment}" if marker else "")

    return emit


def _endpoint_matches(checked, key, leaf, task, position):
    require(key in checked.access.endpoints, "scope channel has an unknown native endpoint")
    uid, site = checked.access.endpoints[key]
    return (
        key[0] == leaf.actor
        and uid == leaf.call.uid.data
        and site.task[1] == task.uid
        and site.position.evaluate(leaf.stream.count.data) == position
    )


def scope_boundary(checked, leaf):
    """Render actual ScopeSync/QuerySyncs in their single mixed native order."""
    from .loop_codegen import _event
    from .scope_query_protocols import read_query_protocols

    tasks = _parent_tasks(checked)
    query_channels = (
        {p.name: p for p in read_query_protocols(checked.program, checked.access)}
        if checked.program.invocations
        else {}
    )
    before, after, pending, previous, block = {}, {}, [], None, None
    for op, _ in scope_operations(leaf.stream):
        if op.parent is not block:
            require(not pending, "scope acquisition escaped its actual native region")
            previous, block = None, op.parent
        if isinstance(op, TaskOp):
            before[op.uid.data], pending, previous = pending, [], op.uid.data
        elif isinstance(op, (ScopeSyncOp, QuerySyncOp)):
            require(
                op.parent_generation is leaf.call.generation
                and op.child_generation is leaf.stream.body.block.args[0],
                "scope synchronization uses a different parent or child clock",
            )
            require(
                op.channel.data
                in (query_channels if isinstance(op, QuerySyncOp) else checked.channels),
                "scope synchronization has an unknown channel",
            )
            if op.action.data == "set":
                require(
                    previous is not None, "scope notification has no preceding native instruction"
                )
                after.setdefault(previous, []).append(op)
            else:
                require(
                    op.action.data in {"wait", "barrier"},
                    "unsupported scope synchronization action",
                )
                pending.append(op)
        else:
            require(not pending, "scope acquisition must be adjacent to its actual native")
            previous = None
    require(not pending, "scope acquisition has no following native instruction")

    def render(op, task):
        if isinstance(op, QuerySyncOp):
            from .scope_query_codegen import render_query_sync

            return render_query_sync(checked, leaf, op, task, query_channels[op.channel.data])
        channel = checked.channels[op.channel.data]
        require(channel.kind in {"local", "cross", "barrier"}, "unsupported scope channel kind")
        require(
            not channel.last and not channel.carry,
            "scope first/last and serial invocation channels are unsupported",
        )
        action = op.action.data
        require(
            (action == "barrier") == (channel.kind == "barrier"),
            "scope channel kind disagrees with its native synchronization action",
        )
        endpoints = channel.sources if action == "set" else channel.destinations
        require(
            any(_endpoint_matches(checked, key, leaf, task, op.position.data) for key in endpoints),
            "scope synchronization is not beside its checked native endpoint",
        )
        conditions = [f"g == {op.position.data}"]
        if getattr(channel, "activation", None) is not None:
            conditions.extend(_activation_guard(channel, action))
        else:
            require(
                type(channel.distance) is int and channel.distance >= 0,
                "signed scope distance requires an explicit activation",
            )
            if channel.distance:
                conditions.append(
                    f"npu_parent_g < npu_parent_n - {channel.distance}"
                    if action == "set"
                    else f"npu_parent_g >= {channel.distance}"
                )
            epoch = (
                "npu_parent_g"
                if action != "wait" or not channel.distance
                else f"(npu_parent_g - {channel.distance})"
            )
            if channel.period != 1:
                conditions.append(
                    "("
                    + " || ".join(
                        f"{epoch} % {channel.period} == {residue}" for residue in channel.residues
                    )
                    + ")"
                )
        if channel.kind == "barrier":
            sync = f"asc_sync_pipe(PIPE_{task.pipe});"
        elif channel.kind == "cross":
            require(len(channel.ids) == 1, "scope cross channel requires one checked counter")
            function = "asc_sync_block_arrive" if action == "set" else "asc_sync_block_wait"
            sync = f"{function}(PIPE_{task.pipe}, {channel.ids[0]});"
        else:
            require(channel.ids, "scope local channel needs checked event IDs")
            require(
                all(key in tasks for key in (*channel.sources, *channel.destinations)),
                "scope channel has an unknown parent access task",
            )
            sources = {tasks[key].pipe for key in channel.sources}
            destinations = {tasks[key].pipe for key in channel.destinations}
            require(
                len(sources) == len(destinations) == 1,
                "scope local channel needs one source and destination pipe",
            )
            scalar = ScalarScope()
            if getattr(channel, "activation", None) is not None:
                generation = _source_epoch(action, channel.distance, scalar)
            else:
                generation = (
                    scalar.value("npu_parent_g")
                    if action == "set"
                    else scalar.sub("npu_parent_g", channel.distance)
                )
            event = scalar.use(_event(channel, generation, scalar))
            function = "asc_sync_notify" if action == "set" else "asc_sync_wait"
            sync = " ".join(
                scalar.finish(
                    [
                        f"{function}(PIPE_{next(iter(sources))}, PIPE_{next(iter(destinations))}, event_t({event}));"
                    ]
                )
            )
        return f"if ({' && '.join(conditions)}) {{ {sync} }} /* {channel.name} */"

    def emit(task, code):
        return "\n".join(
            [
                *(render(op, task) for op in before.get(task.uid, ())),
                code,
                *(render(op, task) for op in after.get(task.uid, ())),
            ]
        )

    return emit


def emit_scope_actor(checked, actor, uniform_names):
    """Issue real calls in parent schedule order, with one fixed child template."""
    from .loop_codegen import _region_lines

    program = checked.program
    parent = program.parents[actor]
    calls = [op for op in parent.body.block.ops if isinstance(op, CallOp)]
    lexical = {op: index for index, op in enumerate(parent.body.block.ops)}
    calls.sort(key=lambda op: (op.issue_order.data, lexical[op]))
    require(calls, "scope actor has no actual parent call")
    maximum = max(call.stage.data for call in calls)
    lines = [
        "  const int64_t npu_parent_n = n;",
        f"  for (int64_t npu_parent_step = 0; npu_parent_step < npu_parent_n + {maximum}; ++npu_parent_step) {{",
    ]
    names = dict(uniform_names)
    names.update(
        {
            Expr("argument", ("$core",)): "core_id",
            Expr("argument", ("$parent0",)): "npu_parent_g",
        }
    )
    invocation = program.invocations.get(actor)
    if invocation is not None:
        names[Expr("argument", ("$parent1",))] = "npu_query_g"
    root_names = {storage.name for storage in checked.access.loop.actors[actor].program.storages}
    for call in calls:
        uid = call.uid.data
        require(
            uid in program.leaves and uid in checked.analyses,
            "scope call has no checked native leaf",
        )
        leaf = program.leaves[uid]
        require(leaf.call is call and leaf.actor == actor, "scope call changed its physical actor")
        loop, channels, sequences, _, _ = checked.analyses[uid]
        require(len(loop.actors) == 1, "scope child needs one actual native actor")
        inner_actor, data = next(iter(loop.actors.items()))
        require(
            loop.constant_count == leaf.stream.count.data,
            "scope child count changed after verification",
        )
        stage = call.stage.data
        generation = "npu_parent_step" if stage == 0 else f"npu_parent_step - {stage}"
        lines += [
            f"    {{ // call {uid}, parent stage {stage}",
            f"      const int64_t npu_parent_g = {generation};",
            "      if (npu_parent_g >= 0 && npu_parent_g < npu_parent_n) {",
        ]
        sources = {captured.root: captured.root.sym_name.data for captured in leaf.captures}
        if any(captured.op.sym_name.data in root_names for captured in leaf.captures):
            # Capture symbols may legally shadow roots. Save every original
            # source before declaring any such alias, including cross-shadowing.
            for index, captured in enumerate(leaf.captures):
                source = f"npu_capture_root_{index}"
                qualifier = QUALIFIER[captured.root.ref.type.bank.data]
                lines.append(
                    f"        {qualifier} uint8_t* {source} = {captured.root.sym_name.data};"
                )
                sources[captured.root] = source
        for captured in leaf.captures:
            name = captured.op.sym_name.data
            qualifier = QUALIFIER[captured.root.ref.type.bank.data]
            lines.append(
                f"        {qualifier} uint8_t* {name} = {sources[captured.root]} + ({captured.offset.c(names)});"
            )
        lines.append(f"        const int64_t n = {leaf.stream.count.data};")
        native = {(inner_actor, task.uid): task for task in data.program.tasks}
        lines.extend(
            "      " + line
            for line in _region_lines(
                inner_actor,
                data,
                sequences,
                channels,
                native,
                None,
                (),
                uniform_names,
                boundary=scope_boundary(checked, leaf),
                command_guard=scope_command_guard(leaf, sequences[inner_actor]),
                display_actor=actor,
            )
        )
        lines += ["      }", "    } // asynchronous child exit"]
    lines.append("  } // parent pipeline: no per-call drain")
    if invocation is not None:
        from .scope_query_codegen import query_loop_lines

        lines = [
            lines[0],
            *query_loop_lines(invocation, uniform_names),
            "  for (int64_t npu_query_g = 0; npu_query_g < npu_query_n; ++npu_query_g) {",
            *("  " + line for line in lines[1:]),
            "  } // serial query: no per-invocation drain",
        ]
    return lines


def emit_scopes(checked, *, host_launcher=True):
    from .loop_codegen import emit_loop

    require(
        not checked.program.invocations
        or checked.query_report is not None
        and checked.query_report.get("verified") is True,
        "nested serial invocations need verified cross-query protocols before ASC emission",
    )
    return emit_loop(
        (checked.access.loop, {}, {}, None, None),
        host_launcher=host_launcher,
        scopes=checked,
    )
