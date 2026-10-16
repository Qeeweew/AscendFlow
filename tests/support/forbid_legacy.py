"""Reject legacy source-planner calls without importing their old test/API tree."""

import sys
from importlib.abc import MetaPathFinder


def forbid_legacy_planning(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("structured source called a retired completion planner")

    names = {
        "xdsl_npu.codegen",
        "xdsl_npu.sync",
        "xdsl_npu.lowering",
        "xdsl_npu.compilation_facts",
        "xdsl_npu.passes.insert_sync",
        "xdsl_npu.passes.local_sync",
        "xdsl_npu.backend.local",
    }

    class Reject(MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname in names or fullname.startswith(("xdsl_npu.loop_", "xdsl_npu.scope_")):
                raise AssertionError(f"structured source imported a retired planner: {fullname}")

    monkeypatch.setattr(sys, "meta_path", [Reject(), *sys.meta_path])
    for name, module in tuple(sys.modules.items()):
        if module is None or not (
            name in names or name.startswith(("xdsl_npu.loop_", "xdsl_npu.scope_"))
        ):
            continue
        # Other test modules may already have loaded a legacy module during
        # collection. Patch its callable aliases without __getattr__ imports.
        for attribute in (
            "autosync",
            "verify_plan",
            "completion_graph",
            "autosync_loop",
            "verify_loop",
            "_verify_leaf",
            "lower_source",
            "autosync_scopes",
            "verify_scopes",
        ):
            if attribute in vars(module):
                monkeypatch.setattr(module, attribute, forbidden)
        if (planning := vars(module).get("DirectCompletion")) is not None:
            monkeypatch.setattr(planning, "__init__", forbidden)
