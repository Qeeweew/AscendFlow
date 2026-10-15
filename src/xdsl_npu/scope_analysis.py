"""Verified native cases alongside an explicitly unproved emission descriptor."""

from .budget import poll
from .loop_graph import channels, commands
from .loop_sync import _verify_leaf


class ScopeAnalyses(dict):
    def __init__(self):
        super().__init__()
        self.cases = {}

    def variants(self, uid):
        selected = [
            (case, analysis) for (owner, case), analysis in self.cases.items() if owner == uid
        ]
        return selected if selected else [(None, self[uid])]


def verify_children(program):
    reports, analyses = {}, ScopeAnalyses()
    for uid, leaf in program.leaves.items():

        def verify(loop):
            roots = {
                loop.root(actor, ref)
                for actor, data in loop.actors.items()
                for task in data.program.tasks
                for ref in task.refs
            }
            return _verify_leaf(loop, live_in=roots, _details=True)

        if not leaf.cases:
            reports[uid], analyses[uid] = verify(leaf.loop)
            continue
        current = {}
        for label, case in leaf.cases.items():
            poll("native parent case verification", call=uid, case=label)
            if case.loop is None:
                continue
            current[label], analyses.cases[uid, label] = verify(case.loop)
        channel_map = channels(leaf.loop)
        sequences, _, _ = commands(leaf.loop, channel_map)
        # This union view is used only by codegen and resource declarations.
        # Parent composition always consumes the separate verified cases.
        analyses[uid] = leaf.loop, channel_map, sequences, None, None
        reports[uid] = {
            "native_templates": sum(len(data.program.tasks) for data in leaf.loop.actors.values()),
            "cases": current,
        }
    return reports, analyses
