"""Finite child event-width choices on unchanged verified command graphs.

Event IDs do not establish completion order. Each candidate reuses the same
leaf graph, while its fixed first-Set/last-Wait endpoints are projected once
for the parent's independent cross-call lifetime proof.
"""

from collections import defaultdict
from dataclasses import dataclass, replace

from .budget import bounded, poll
from .completion import Position, Site
from .diagnostics import require
from .loop_sync import _successor
from .region_summary import EventReuse, fixed_event_reuse


@dataclass(frozen=True)
class EventWidth:
    width: int
    # IDs here are abstract slot indices, before choosing physical bank IDs.
    resources: tuple[EventReuse, ...]


@dataclass(frozen=True)
class EventChoices:
    uid: str
    channel: str
    bank: tuple[str, str, str]
    count: int
    options: tuple[EventWidth, ...]


def resource_sites(uid, actor, count, resource):
    """Bind a fixed resource boundary to its actual parent call and actor."""
    return tuple(
        Site(
            (actor, endpoint.task[1]),
            Position(False, endpoint.position.evaluate(count)),
            point,
            resource.channel,
            action,
            region=uid,
        )
        for endpoint, point, action in (
            (resource.consumed, "begin", "wait"),
            (resource.republished, "end", "set"),
        )
    )


def resource_inputs(choices):
    """Union all candidate boundaries for one parent projection per child."""
    result = defaultdict(dict)
    for choice in choices.values():
        for option in choice.options:
            result[choice.uid].update(dict.fromkeys(option.resources))
    return {uid: tuple(resources) for uid, resources in result.items()}


@bounded
def event_choices(program, analyses):
    """Return locally legal widths 1..8 without rebuilding any child analysis.

    Each child performs one batch of all its candidate lifetime queries on
    analysis[3]. No native/sync placement, capture, buffer or event ID changes.
    Cross-parent legality remains a query on the eventual parent proof graph.
    """
    require(
        set(program.leaves) == set(analyses),
        "child event choices need the actual analysis of every parent call",
    )
    result = {}
    for uid, analysis in analyses.items():
        poll("child event-width choices", call=uid)
        leaf = program.leaves[uid]
        loop, channels, streams, graph, _ = analysis
        require(
            len(loop.actors) == 1 and type(loop.constant_count) is int and loop.constant_count >= 0,
            "child event choices need one actual fixed-count actor",
        )
        endpoints = {
            (actor, command.channel, command.kind): command
            for actor, sequence in streams.items()
            for command in sequence
            if command.kind in {"set", "wait"}
        }
        queries, candidates, banks = [], {}, {}
        for name, channel in channels.items():
            poll("child event-width queries", call=uid, channel=name)
            require(
                channel.kind == "local" and not channel.carry,
                "nested child event choices require local non-carried channels",
            )
            source = endpoints[channel.sources[0][0], name, "set"]
            destination = endpoints[channel.destinations[0][0], name, "wait"]
            banks[name] = leaf.actor, source.task.pipe, destination.task.pipe
            for width in (1,) if channel.last else range(1, 9):
                first = len(queries)
                if not channel.last:
                    queries.extend(
                        (
                            destination,
                            residue + channel.distance,
                            source,
                            _successor(channel, residue, width, graph.period),
                        )
                        for residue in range(graph.period)
                        if residue % channel.period in channel.residues
                    )
                candidates[name, width] = first, len(queries)
        answers = graph.covered_many(queries) if queries else ()
        for name, channel in channels.items():
            options = tuple(
                EventWidth(
                    width,
                    fixed_event_reuse(
                        replace(channel, ids=tuple(range(width))), banks[name], loop.constant_count
                    ),
                )
                for width in ((1,) if channel.last else range(1, 9))
                if all(answers[slice(*candidates[name, width])])
            )
            require(options, f"{uid}/{name}: no locally proved child event width <=8")
            result[uid, name] = EventChoices(uid, name, banks[name], loop.constant_count, options)
    return result
