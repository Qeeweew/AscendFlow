"""Physical facts owned by one compilation, before synchronization changes.

This is not a verification cache. A compiler-owned physical program is admitted
once; insertion and event-ID assignment preserve its native effects. Callers
must pass the retained descriptors on to those transformations. External IR
entry points create new descriptors and never import this session's results.
"""

from collections import Counter


class PhysicalFacts:
    def __init__(self):
        self._models = {}
        self._dependencies = {}
        self.work = Counter()

    def _extract(self, kind, module, extract):
        key = kind, module
        if key not in self._models:
            self._models[key] = extract(module)
            self.work[kind + "_extractions"] += 1
        else:
            self.work[kind + "_reuses"] += 1
        return self._models[key]

    def extract_loop(self, module):
        from .loop_model import extract_loop

        return self._extract("loop", module, extract_loop)

    def extract_scopes(self, module):
        from .scope_model import extract_scopes

        return self._extract("scope", module, extract_scopes)

    def extract_sequence(self, module):
        from .sequence_model import extract_sequence

        return self._extract("sequence", module, extract_sequence)

    def dependency_analysis(self, loop, *, live_in=()):
        """Retain the complete result shared by value checks and synchronization."""
        from .analysis.dependencies import analyze_dependencies

        # Keep the object alive, not just its id. Names in live_in are the exact
        # admitted root set; a weaker entry assumption needs a new analysis.
        key = id(loop), frozenset(live_in)
        if key not in self._dependencies:
            analysis = analyze_dependencies(loop, live_in=key[1])
            self._dependencies[key] = loop, analysis
            self.work["dependency_builds"] += 1
            self.work["dependency_records"] += len(analysis.obligations)
        else:
            self.work["dependency_reuses"] += 1
        return self._dependencies[key][1]

    def loop_dependencies(self, loop, *, live_in=()):
        """Legacy view of the very same complete analysis, without another scan."""
        return self.dependency_analysis(loop, live_in=live_in).obligations

    def statistics(self):
        return dict(self.work)
