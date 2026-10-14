"""Small static launch partition, with exact selectors and shared GM bindings.

This is not dynamic task ownership. Every branch is a real verified mixed
program, executed by both AIVs and the AIC of its selected groups. There are
no skipped notifications inside a branch and no inter-branch communication.
"""

import itertools
import re
from dataclasses import dataclass

from xdsl.dialects.builtin import IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from .ir import EndOp, MixedOp, StaticCaseOp, StaticDispatchOp
from .mixed import extract_mixed, merge_ranges
from .model import ranges_overlap, require


@dataclass
class Dispatch:
    op: StaticDispatchOp
    cases: list
    roots: dict


def static_dispatch(name, modules):
    block, cursor = Block(), 0
    require(0 < len(modules) <= 256, "static dispatch requires 1..256 mixed cases")
    for module in modules:
        clone = module.clone()
        mixed = extract_mixed(clone)
        clone.body.block.detach_op(mixed.op)
        size = mixed.op.blocks.data
        block.add_op(
            StaticCaseOp.create(
                attributes={"lower": IntAttr(cursor), "upper": IntAttr(cursor + size)},
                regions=[Region(Block([mixed.op, EndOp()]))],
            )
        )
        cursor += size
    block.add_op(EndOp())
    result = ModuleOp(
        [
            StaticDispatchOp.create(
                attributes={
                    "sym_name": StringAttr(name),
                    "target": StringAttr("dav-2201"),
                    "blocks": IntAttr(cursor),
                },
                regions=[Region(block)],
            )
        ]
    )
    extract_dispatch(result)
    return result


def case_module(case):
    child = case.body.block.first_op
    require(isinstance(child, MixedOp), "static case requires exactly one mixed program")
    return ModuleOp([child.clone()])


def extract_dispatch(module):
    module.verify()
    require(len(list(module.body.block.ops)) == 1, "one static launch per compilation")
    op = module.body.block.first_op
    require(
        isinstance(op, StaticDispatchOp) and op.target.data == "dav-2201",
        "expected dav-2201 static dispatch",
    )
    require(re.fullmatch(r"[A-Za-z_]\w*", op.sym_name.data), "invalid dispatch name")
    require(1 <= op.blocks.data <= 65535, "static mixed launch grid must be in [1,65535]")
    cases, roots, cursor = [], None, 0
    for case in op.body.block.ops:
        if isinstance(case, EndOp):
            continue
        require(isinstance(case, StaticCaseOp), "only static cases belong to a dispatch")
        require(
            case.lower.data == cursor < case.upper.data <= op.blocks.data,
            "static cases must partition the launch grid without holes or overlap",
        )
        children = list(case.body.block.ops)
        require(
            len(children) == 2
            and isinstance(children[0], MixedOp)
            and isinstance(children[1], EndOp),
            "a static case has exactly one mixed region and its terminator",
        )
        mixed = extract_mixed(case_module(case))
        require(
            mixed.op.blocks.data == case.upper.data - cursor,
            "case-local grid does not match its selector range",
        )
        if roots is None:
            roots = mixed.roots
        else:
            require(
                tuple(roots) == tuple(mixed.roots)
                and all(
                    old.ref.type == mixed.roots[name].ref.type
                    and old.nbytes == mixed.roots[name].nbytes
                    and old.initialized == mixed.roots[name].initialized
                    for name, old in roots.items()
                ),
                "static cases must bind identical ordered GM arguments and input contracts",
            )
        cases.append((case, mixed))
        cursor = case.upper.data
    require(
        0 < len(cases) <= 256 and cursor == op.blocks.data,
        "static selector does not cover the complete launch grid",
    )
    # Cases execute on distinct groups concurrently; a branch-local proof
    # alone says nothing about shared workspace/output aliases across them.
    # Read-only roots cannot conflict across cases. Merge all generations at
    # group zero once, then translate the exact unions (including holes). A
    # root with nonuniform reference translations keeps the explicit path;
    # no affine/contiguous assumption or core sample replaces the check.
    writes = {
        root
        for _, mixed in cases
        for key in mixed.semantic_instances()
        for root, _, _, mode in mixed.physical_effects(key)
        if root in mixed.roots and "W" in mode
    }
    accesses = []
    for _, mixed in cases:
        base, refs = {}, {}
        for program in mixed.programs.values():
            for task in program.tasks:
                for ref in task.refs:
                    if ref.storage.name in writes:
                        refs.setdefault(ref.storage.name, set()).add(ref)
        for key in mixed.semantic_instances():
            for root, lo, hi, mode in mixed.physical_effects(key):
                if root in writes:
                    for access in ("R", "W"):
                        if access in mode:
                            base.setdefault((root, access), []).append((lo, hi))
        base = {key: merge_ranges(parts) for key, parts in base.items()}
        effects = {}
        for core in range(mixed.op.blocks.data):
            for root, views in refs.items():
                shifts = {ref.address(0, core) - ref.address(0) for ref in views}
                if len(shifts) == 1:
                    shift = next(iter(shifts))
                    for access in ("R", "W"):
                        effects.setdefault((root, access), []).extend(
                            (lo + shift, hi + shift) for lo, hi in base.get((root, access), ())
                        )
                    continue
                for key in mixed.semantic_instances():
                    for name, lo, hi, mode in mixed.physical_effects(key, core):
                        if name != root:
                            continue
                        for access in ("R", "W"):
                            if access in mode:
                                effects.setdefault((root, access), []).append((lo, hi))
        accesses.append({key: merge_ranges(parts) for key, parts in effects.items()})
    for (left, _), (right, _) in itertools.combinations(enumerate(cases), 2):
        for root in roots:
            a, b = accesses[left], accesses[right]
            require(
                not any(
                    ranges_overlap(a.get((root, mode), ()), b.get((root, other), ()))
                    for mode, other in (("W", "R"), ("W", "W"), ("R", "W"))
                ),
                f"static cases {left}/{right} overlap GM {root}; no inter-case ownership protocol",
            )
    return Dispatch(op, cases, roots)


def autosync_dispatch(module, *, _capture=None):
    from .sync import _finish, autosync

    result = module.clone()
    dispatch = extract_dispatch(result)
    for case, _ in dispatch.cases:
        planned = autosync(case_module(case))
        new = planned.body.block.first_op
        planned.body.block.detach_op(new)
        old = case.body.block.first_op
        case.body.block.insert_op_before(new, old)
        case.body.block.erase_op(old)
    return _finish(result, _capture=_capture)


def verify_dispatch(module, *, _details=False):
    from .mixed import verify_mixed

    dispatch = extract_dispatch(module)
    checked = [verify_mixed(case_module(case), _details=True) for case, _ in dispatch.cases]
    reports = [report for report, _ in checked]
    report = {
        "kernel": dispatch.op.sym_name.data,
        "groups": dispatch.op.blocks.data,
        "static_cases": len(reports),
        "case_ranges": [[case.lower.data, case.upper.data] for case, _ in dispatch.cases],
        "cases": reports,
        "scratch_bytes": max((report["scratch_bytes"] for report in reports), default=0),
        "verification": "exact static selectors, all branch effects and cross-case GM ownership",
    }
    return (report, (dispatch, [analysis for _, analysis in checked])) if _details else report


def explain_dispatch(module):
    from .mixed import explain_mixed

    dispatch = extract_dispatch(module)
    return {
        "verification": verify_dispatch(module),
        "cases": [
            {
                "group_range": [case.lower.data, case.upper.data],
                "plan": explain_mixed(case_module(case)),
            }
            for case, _ in dispatch.cases
        ],
    }
