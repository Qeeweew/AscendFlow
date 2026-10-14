"""Actual local region extraction and open physical-access contracts."""

from dataclasses import dataclass

from .ir import BufferOp, KernelOp, LoopChannelOp, OnceOp, RegionChannelOp, StreamOp
from .loop_model import extract_loop, slope
from .model import require, verify_translated_owners
from .region_sequence import Part, summarize_sequence
from .region_summary import summarize_reused_storage


@dataclass(frozen=True)
class Sequence:
    module: object
    regions: dict
    live_in: dict
    summary: object


def is_sequence(module):
    root = module.body.block.first_op
    if not isinstance(root, KernelOp):
        return False
    regions = [op for op in root.body.block.ops if isinstance(op, (StreamOp, OnceOp))]
    return any(isinstance(op, OnceOp) for op in regions) or len(regions) > 1


def extract_sequence(module):
    module.verify()
    kernel = module.body.block.first_op
    require(isinstance(kernel, KernelOp), "sequential regions currently need one local actor")
    regions = [op for op in kernel.body.block.ops if isinstance(op, (OnceOp, StreamOp))]
    require(regions, "sequence has no executable regions")
    require(
        all(op.uid is not None and op.uid.data.isidentifier() for op in regions)
        and len({op.uid.data for op in regions}) == len(regions),
        "sequential regions need unique lexical identities",
    )
    require(
        not any(isinstance(op, LoopChannelOp) for op in kernel.body.block.ops),
        "leaf event channels must belong to their actual region",
    )
    loops = {op.uid.data: extract_loop(module, region=op) for op in regions}
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
    return Sequence(module, loops, live_in, summary)
