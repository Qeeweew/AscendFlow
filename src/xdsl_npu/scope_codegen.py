"""Emit retained parent calls using the ordinary verified native-region emitter."""

from .asc_scalar import ScalarScope
from .codegen import QUALIFIER
from .ir import CallOp, ScopeSyncOp, TaskOp
from .loop_boundary import stream_operations
from .model import require
from .scalar import Expr


def _parent_tasks(checked):
    return {
        (actor, task.uid): task
        for actor, data in checked.access.loop.actors.items()
        for task in data.program.tasks
    }


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
    """Render only actual ScopeSyncOps beside their actual native instruction."""
    from .loop_codegen import _event

    tasks = _parent_tasks(checked)
    before, after, pending, previous = {}, {}, [], None
    for op, _ in stream_operations(leaf.stream):
        if isinstance(op, TaskOp):
            before[op.uid.data], pending, previous = pending, [], op.uid.data
        elif isinstance(op, ScopeSyncOp):
            require(
                op.parent_generation is leaf.call.generation
                and op.child_generation is leaf.stream.body.block.args[0],
                "scope synchronization uses a different parent or child clock",
            )
            require(
                op.channel.data in checked.channels, "scope synchronization has an unknown channel"
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
    require(not pending, "scope acquisition has no following native instruction")

    def render(op, task):
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
                display_actor=actor,
            )
        )
        lines += ["      }", "    } // asynchronous child exit"]
    lines.append("  } // parent pipeline: no per-call drain")
    return lines


def emit_scopes(checked, *, host_launcher=True):
    from .loop_codegen import emit_loop

    return emit_loop(
        (checked.access.loop, {}, {}, None, None),
        host_launcher=host_launcher,
        scopes=checked,
    )
