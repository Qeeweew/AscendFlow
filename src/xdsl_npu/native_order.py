"""Native milestone edges shared by periodic and hierarchical completion flows.

Clock placement, activation, and event matching belong to the caller. This
module owns the hardware distinction between issue, begin, retirement capture,
and a notification effect. In particular a Set effect never enters a native
retirement prefix or the following instruction's FIFO-begin prefix.
"""

from .model import require


def instruction_points(command, before, node, edge):
    require(
        command.kind in {"task", "set", "wait", "barrier", "cross_set", "cross_wait"},
        "unknown native completion transition",
    )
    begin = node("begin")
    end = node("end") if command.kind in {"task", "set", "cross_set"} else begin
    points = {"begin": begin, "end": end}
    edge(before["issue"], begin)
    edge(before["queue"], begin)
    if end != begin:
        edge(begin, end)
    if command.kind == "task":
        prefix = points["prefix"] = node("prefix")
        edge(before["implicit"], begin)
        edge(end, prefix)
        edge(before["retire"], prefix)
    elif command.kind in {"set", "cross_set", "barrier"}:
        edge(before["retire"], end)
    return points


def prefix_inputs(command, points):
    """New facts added to monotone prefixes after this instruction position."""
    result = {"queue": points["begin"]}
    if command.kind == "task":
        result["retire"] = points["prefix"]
        if command.task.implicit_completion:
            result["implicit"] = points["end"]
    if command.kind == "cross_wait" or (
        command.task.pipe == "S" and command.kind in {"task", "wait"}
    ):
        result["issue"] = points["end"]
    return result
