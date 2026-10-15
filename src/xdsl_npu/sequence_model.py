"""Actual local region extraction and open physical-access contracts."""

from dataclasses import dataclass

from xdsl.dialects import scf

from .ir import (
    BufferOp,
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
from .region_summary import summarize_reused_storage


@dataclass(frozen=True)
class Sequence:
    module: object
    regions: dict
    live_in: dict
    summary: object
    invocation: object = None


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
    stationary = set()
    for root, accesses in refs.items():
        if root in frame_roots:
            require(
                root not in writes or len(users[root]) == 1,
                f"{root}: cross-region GM frame writes need a shared frame frontier",
            )
            continue
        fixed = all(
            not slope(ref)
            and not ref.count_offset
            and not ref.count_core_stride
            and ref.prefix_extent is None
            and root not in loop.frames
            for loop, ref, _ in accesses
        )
        if fixed:
            stationary.add(root)
        elif len(users[root]) > 1:
            require(
                root not in writes and all(ref.storage.initialized for _, ref, _ in accesses),
                f"{root}: cross-region drifting writes need a joint ownership proof",
            )
        if root in writes and fixed and accesses[0][1].storage.external:
            # A per-leaf owner proof is insufficient: two locally disjoint grids
            # could still alias another region's owners. Verify the union.
            physical = [ref for _, ref, _ in accesses]
            effects = [
                (ref.storage.name, lo, hi, mode)
                for loop, ref, mode in accesses
                for g in range(loop.storage_periods[root])
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
        roots = stationary & {
            root for root, regions_using_root in users.items() if name in regions_using_root
        }
        live_in[name] = roots
        parts.append(Part(name, summarize_reused_storage(loop, roots)))
    summary = summarize_sequence(parts)
    for cell in summary.cells:
        allocation = cell.storage.owner
        require(isinstance(allocation, BufferOp), "sequence summary lost allocation provenance")
        require(
            not cell.reads_before_write or allocation.external.data and allocation.initialized.data,
            f"undefined sequence entry read on {allocation.sym_name.data}[{cell.start}:{cell.stop}] "
            f"in {tuple(r.region for r in cell.reads_before_write)}; empty child is identity",
        )
    # Do not accept a hidden declaration in an unrelated nested scope.
    require(
        all(op.parent_op() is kernel for op in kernel.walk() if isinstance(op, RegionChannelOp)),
        "boundary channels must belong to the enclosing kernel",
    )
    return Sequence(module, loops, live_in, summary, invocation)
