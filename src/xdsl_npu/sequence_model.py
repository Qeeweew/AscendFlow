"""Actual local region extraction and open physical-access contracts."""

from dataclasses import dataclass, replace

from xdsl.dialects import scf

from .ir import (
    InvocationChannelOp,
    InvocationSyncOp,
    KernelOp,
    LoopChannelOp,
    OnceOp,
    RegionChannelOp,
    StreamOp,
)
from .loop_model import _extract_loop_after_verify, slope
from .model import require, verify_translated_owners
from .region_sequence import Part, summarize_sequence
from .region_summary import allocation, summarize_reused_storage


@dataclass(frozen=True)
class Sequence:
    module: object
    regions: dict
    live_in: dict
    summary: object
    invocation: object = None

    def invocation_dependencies(self):
        # These FrameOps came from extract_invocation's actual partition check.
        fresh = () if self.invocation is None else tuple(f.op.ref for f in self.invocation.frames)
        return self.summary.invocation_dependencies(fresh=fresh)


def sequence_regions(kernel):
    parents = [op for op in kernel.body.block.ops if isinstance(op, scf.ForOp)]
    scope = parents[0] if len(parents) == 1 else kernel
    return [op for op in scope.body.block.ops if isinstance(op, (StreamOp, OnceOp))]


def is_sequence(module):
    root = module.body.block.first_op
    if not isinstance(root, KernelOp):
        return False
    regions = sequence_regions(root)
    return any(isinstance(op, OnceOp) for op in regions) or len(regions) > 1


def extract_sequence(module):
    module.verify()
    kernel = module.body.block.first_op
    require(isinstance(kernel, KernelOp), "sequential regions currently need one local actor")
    regions = sequence_regions(kernel)
    require(regions, "sequence has no executable regions")
    require(
        not any(isinstance(op, (InvocationChannelOp, InvocationSyncOp)) for op in kernel.walk()),
        "sequence calls require their region protocols, not single-child invocation endpoints",
    )
    require(
        all(op.uid is not None and op.uid.data.isidentifier() for op in regions)
        and len({op.uid.data for op in regions}) == len(regions),
        "sequential regions need unique lexical identities",
    )
    require(
        not any(isinstance(op, LoopChannelOp) for op in kernel.body.block.ops),
        "leaf event channels must belong to their actual region",
    )
    loops = {op.uid.data: _extract_loop_after_verify(module, region=op) for op in regions}
    invocation = next(iter(loops.values())).invocation
    if invocation is not None:
        require(
            all(op.parent_op() is invocation.op for op in regions)
            and not any(isinstance(op, (OnceOp, StreamOp)) for op in kernel.body.block.ops),
            "invocation sequence cannot hide execution outside its parent",
        )
    frame_roots = set() if invocation is None else {f.name for f in invocation.frames}
    refs, writes, users = {}, set(), {}
    for name, loop in loops.items():
        for actor, data in loop.actors.items():
            for task in data.program.tasks:
                for ref, mode in zip(task.refs, task.modes):
                    root = loop.root(actor, ref)
                    refs.setdefault(root, []).append((loop, ref, mode))
                    users.setdefault(root, set()).add(name)
                    if "W" in mode:
                        writes.add(root)
    stationary, finite = set(), set()
    for root, accesses in refs.items():
        fixed = all(
            not slope(ref)
            and not ref.count_offset
            and not ref.count_core_stride
            and ref.prefix_extent is None
            and root not in loop.frames
            for loop, ref, _ in accesses
        )
        if root in frame_roots:
            if root in writes and len(users[root]) > 1:
                require(
                    fixed,
                    f"{root}: cross-region drifting GM frame access needs a translated frontier",
                )
                stationary.add(root)
            # Native frame bounds plus the checked invocation partition prove
            # different groups/calls disjoint. Same-call readers/writers still
            # need the ordinary shared frontier; no implicit frame exit fence.
            continue
        if fixed:
            stationary.add(root)
        elif len(users[root]) > 1:
            bounded = all(
                loop.constant_count is not None
                and not ref.count_offset
                and not ref.count_core_stride
                and ref.prefix_extent is None
                and not ref.storage.count_scale
                for loop, ref, _ in accesses
            )
            if root in writes and bounded:
                finite.add(root)
            else:
                require(
                    root not in writes and all(ref.storage.initialized for _, ref, _ in accesses),
                    f"{root}: cross-region drifting writes need a joint ownership proof",
                )
        if root in writes and (fixed or root in finite) and accesses[0][1].storage.external:
            # A per-leaf owner proof is insufficient: two locally disjoint grids
            # could still alias another region's owners. Verify the union.
            physical = [ref for _, ref, _ in accesses]
            from .budget import poll

            counts = [
                loop.constant_count if root in finite else loop.storage_periods[root]
                for loop, _, _ in accesses
            ]
            if root in finite:
                from .region_summary import MAX_FIXED_ACCESS_SEGMENTS

                groups = next(iter(loops.values())).groups
                require(
                    groups * sum(count * ref.rows for count, (_, ref, _) in zip(counts, accesses))
                    <= MAX_FIXED_ACCESS_SEGMENTS,
                    f"{root}: cross-region fixed access budget exceeded",
                )
                require(
                    groups * len(physical) <= MAX_FIXED_ACCESS_SEGMENTS,
                    f"{root}: cross-region fixed translation budget exceeded",
                )
            poll("sequence owner union", root=root, accesses=len(accesses))
            effects = [
                (ref.storage.name, lo, hi, mode)
                for count, (loop, ref, mode) in zip(counts, accesses)
                for g in range(count)
                for lo, hi in ref.segments(g)
            ]
            verify_translated_owners(
                physical,
                effects,
                next(iter(loops.values())).groups,
                translation_error="sequential regions disagree on GM ownership",
                overlap_error="cross-region GM owners overlap",
            )
    live_in, parts = {}, []
    for name, loop in loops.items():
        used = {root for root, regions_using_root in users.items() if name in regions_using_root}
        roots, fixed_roots = stationary & used, finite & used
        live_in[name] = roots | fixed_roots
        summary = summarize_reused_storage(loop, roots)
        if fixed_roots:
            from .region_summary import summarize_fixed_storage

            projected = summarize_fixed_storage(loop, fixed_roots)
            summary = replace(
                summary,
                roots=(*summary.roots, *projected.roots),
                cells=(*summary.cells, *projected.cells),
                origins=(*summary.origins, *projected.origins),
                visits=summary.visits + projected.visits,
            )
        parts.append(Part(name, summary))
    summary = summarize_sequence(parts)
    for cell in summary.cells:
        root = allocation(cell.storage)
        require(
            not cell.reads_before_write or root.external.data and root.initialized.data,
            f"undefined sequence entry read on {root.sym_name.data}[{cell.start}:{cell.stop}] "
            f"in {tuple(r.region for r in cell.reads_before_write)}; empty child is identity",
        )
    # Do not accept a hidden declaration in an unrelated nested scope.
    require(
        all(op.parent_op() is kernel for op in kernel.walk() if isinstance(op, RegionChannelOp)),
        "boundary channels must belong to the enclosing kernel",
    )
    return Sequence(module, loops, live_in, summary, invocation)
