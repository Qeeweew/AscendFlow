"""Exact finite physical-access analysis over trusted target contracts.

No analysis reads a compiler certificate as a premise. Layout changes are legal
only at a named native recipe; a subview/recast preserves allocation identity.
The first version deliberately bounds finite expansion, and emits compact loops.
"""

import re
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scalar import Expr

from xdsl.dialects.builtin import IndexType, ModuleOp, f32
from xdsl.ir import SSAValue

from . import diagnostics
from .dialects.physical import (
    BufferOp,
    ChannelOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    Load3DConfigOp,
    StreamOp,
    SyncOp,
    TaskOp,
    ViewOp,
)
from .dialects.types import RefType
from .target import native as target_native
from .target import scalars as target_scalars

MAX_INSTANCES = 4096


def task_scalar(op):
    """One scalar contract for finite and parametric physical task parsing."""
    diagnostics.require(op.scalar.type == f32, "native Vector scalar literals must be f32")
    return target_scalars.round_f32(op.scalar.value.data, allow_infinity=op.native.data == "fill")


def optional_int(attr, default):
    return default if attr is None else attr.data


def ranges_cover(parent, child):
    parent = sorted(parent)
    index = 0
    for start, stop in sorted(child):
        while index < len(parent) and parent[index][1] <= start:
            index += 1
        cursor = start
        while cursor < stop:
            if index == len(parent) or parent[index][0] > cursor:
                return False
            cursor = max(cursor, parent[index][1])
            if cursor < stop:
                index += 1
    return True


def ranges_overlap(left, right):
    """Linear comparison of two sorted interval unions, including their holes."""
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        if a < d and c < b:
            return True
        if b <= c:
            i += 1
        else:
            j += 1
    return False


@dataclass(frozen=True)
class Storage:
    name: str
    bank: str
    nbytes: int
    dtype: str
    layout: str
    external: bool
    initialized: bool
    count_scale: int = 0
    extent: "Expr | None" = None


@dataclass(frozen=True)
class Ref:
    storage: Storage
    terms: tuple[tuple[int, int, int, int], ...]  # offset, byte stride, modulus, divisor
    nbytes: int
    dtype: str
    layout: str
    tail_nbytes: int = 0
    tail_generation: int = -1
    rows: int = 1
    row_stride: int = 0
    core_terms: tuple[tuple[int, int, int], ...] = ()  # byte stride, divisor, modulus
    count_offset: int = 0
    count_core_stride: int = 0
    prefix_extent: "Expr | None" = None
    prefix_elements: int = 0

    def actual_segments(self, g, parameters, core=0):
        """Concrete footprint for an explicitly supplied runtime invocation."""
        if self.prefix_extent is None:
            diagnostics.require(
                not self.count_offset and not self.count_core_stride,
                "count-translated concrete footprint requires an explicit loop-bound binding",
            )
            return self.segments(g, core)
        valid = min(
            self.prefix_elements, self.prefix_extent.evaluate(parameters) - g * self.prefix_elements
        )
        if valid <= 0:
            return ()
        start = self.address(g, core)
        return ((start, start + valid * target_native.ITEMSIZE[self.dtype]),)

    def address(self, g: int, core=0) -> int:
        return sum(o + s * ((g // d) % m if m else g // d) for o, s, m, d in self.terms) + sum(
            s * ((core // d) % m if m else core // d) for s, d, m in self.core_terms
        )

    def interval(self, g: int, core=0) -> tuple[int, int]:
        parts = self.segments(g, core)
        return parts[0][0], parts[-1][1]

    def segments(self, g: int, core=0):
        start = self.address(g, core)
        return tuple(
            (start + r * self.row_stride, start + r * self.row_stride + self.length(g))
            for r in range(self.rows)
        )

    @property
    def total_nbytes(self):
        return self.nbytes * self.rows

    @property
    def contiguous(self):
        return self.rows == 1 or self.row_stride == self.nbytes

    def length(self, g: int) -> int:
        return self.tail_nbytes if self.tail_nbytes and g == self.tail_generation else self.nbytes


@dataclass(frozen=True)
class Task:
    uid: str
    native: str
    pipe: str
    stage: int
    refs: tuple[Ref, ...]
    modes: tuple[str, ...]
    scalar: float
    params: tuple[int, ...]
    lower: int = 0
    upper: int = -1
    issue_order: int = 0
    scalar_source: str | None = None
    period: int = 1
    phase: int = 0
    logical_phase: int = 0
    phase_width: int = 1
    last: bool = False

    def active(self, g):
        return (
            self.lower <= g < self.upper
            and self.phase <= g % self.period < self.phase + self.phase_width
        )

    def modes_for(self, g):
        if self.native == "mad_first":
            return ("R", "R", "W" if g == 0 else "RW")
        if self.native == "mad_cycle":
            return ("R", "R", "W" if g % self.params[3] == 0 else "RW")
        return self.modes

    @property
    def implicit_completion(self):
        return target_native.implicit_completion(self)


@dataclass(frozen=True, order=True)
class Instance:
    generation: int
    task: int


@dataclass(frozen=True)
class Obligation:
    source: Instance
    destination: Instance
    hazard: str
    storage: str
    start: int
    stop: int
    milestone: str


@dataclass(frozen=True)
class Program:
    name: str
    core: str
    count: int
    storages: tuple[Storage, ...]
    tasks: tuple[Task, ...]
    config: bool
    blocks: int = 1
    epoch_divisor: int = 1
    load3d: tuple[int, int] | None = None

    @cached_property
    def _semantic_instances(self):
        instances = [
            Instance(g, t)
            for g in range(self.count)
            for t, task in enumerate(self.tasks)
            if task.active(g)
        ]
        return tuple(
            sorted(
                instances,
                key=lambda i: (
                    i.generation // self.epoch_divisor,
                    self.tasks[i.task].logical_phase,
                    i.generation,
                    i.task,
                ),
            ),
        )

    def instances(self):
        return self._semantic_instances

    def issued(self):
        return sorted(
            self.instances(),
            key=lambda i: (
                i.generation + self.tasks[i.task].stage,
                self.tasks[i.task].issue_order,
                i.task,
            ),
        )


def extract(module: ModuleOp) -> Program:
    module.verify()
    diagnostics.require(
        len(list(module.body.block.ops)) == 1, "one kernel per compilation is required"
    )
    kernel = module.body.block.first_op
    diagnostics.require(isinstance(kernel, KernelOp), "expected npu.kernel")
    diagnostics.require(
        kernel.target.data == "dav-2201", "only Ascend 910B / dav-2201 is supported"
    )
    diagnostics.require(
        kernel.core.data in {"aic", "aiv"}, "cross AIC/AIV ownership is not implemented"
    )
    blocks = optional_int(kernel.blocks, 1)
    diagnostics.require(1 <= blocks <= 1024, "independent core grid must be static in [1,1024]")
    diagnostics.require(
        re.fullmatch(r"[A-Za-z_]\w*", kernel.sym_name.data), "invalid kernel identifier"
    )
    streams = [op for op in kernel.body.block.ops if isinstance(op, StreamOp)]
    diagnostics.require(
        len(streams) == 1, "one finite stream is required; dynamic/nested streams are unsupported"
    )
    stream = streams[0]
    diagnostics.require(
        stream.limit is None and stream.symbolic is None,
        "symbolic loop must use the loop checker; finite extraction is only for explicit diagnostics",
    )
    count = stream.count.data
    diagnostics.require(
        isinstance(count, int) and 0 <= count <= MAX_INSTANCES,
        f"task count must be a nonnegative static integer <= {MAX_INSTANCES}",
    )
    diagnostics.require(
        len(stream.body.block.args) == 1 and isinstance(stream.body.block.args[0].type, IndexType),
        "stream requires exactly one logical generation argument",
    )
    roots: dict[SSAValue, Storage] = {}
    names = set()
    config = False
    load3d = None
    seen_stream = False
    for op in kernel.body.block.ops:
        if isinstance(op, StreamOp):
            seen_stream = True
        elif isinstance(op, BufferOp):
            diagnostics.require(
                op.extent_count is None and optional_int(op.count_scale, 0) == 0,
                "runtime buffer extent cannot be ignored by finite extraction",
            )
            diagnostics.require(not seen_stream, "storage declarations must precede the stream")
            typ = op.ref.type
            diagnostics.require(isinstance(typ, RefType), "invalid buffer type")
            name, bank, size = op.sym_name.data, typ.bank.data, op.nbytes.data
            diagnostics.require(
                re.fullmatch(r"[A-Za-z_]\w*", name) and name not in names,
                f"duplicate or invalid storage identifier {name!r}",
            )
            names.add(name)
            diagnostics.require(
                name
                not in {
                    "g",
                    "step",
                    "core_id",
                    "stream",
                    "npu_row",
                    "npu_chunk",
                    "npu_fa",
                    "npu_fb",
                    "npu_rpt",
                }
                and not name.startswith("npu_value_"),
                "storage name conflicts with internal ASC variables",
            )
            diagnostics.require(
                bank in {*target_native.CAPACITY, "GM"}
                and typ.dtype.data in target_native.ITEMSIZE,
                "unsupported storage type",
            )
            diagnostics.require(
                op.external.data in {0, 1} and op.initialized.data in {0, 1},
                "storage contracts must be boolean",
            )
            diagnostics.require(
                (bank == "GM") == bool(op.external.data),
                "GM is external; local banks are allocated",
            )
            diagnostics.require(
                size >= 0 and (bank == "GM" or size % 32 == 0), "invalid allocation size/alignment"
            )
            diagnostics.require(
                not op.initialized.data or bank == "GM", "local initialization needs a real writer"
            )
            roots[op.ref] = Storage(
                name,
                bank,
                size,
                typ.dtype.data,
                typ.layout.data,
                bool(op.external.data),
                bool(op.initialized.data),
            )
        elif isinstance(op, FixConfigOp):
            diagnostics.require(
                not seen_stream and not config and kernel.core.data == "aic",
                "FIX configuration must be unique, unconditional and before the stream",
            )
            config = True
        elif isinstance(op, Load3DConfigOp):
            diagnostics.require(
                not seen_stream and load3d is None and kernel.core.data == "aic",
                "Load3D configuration must be unique, unconditional and before the stream",
            )
            load3d = op.a_width.data, op.b_width.data
            diagnostics.require(
                all(16 <= w <= 256 and w % 16 == 0 for w in load3d),
                "invalid invariant Load3D fmatrix width",
            )
        elif not isinstance(op, (ChannelOp, EndOp)):
            raise diagnostics.CompileError(
                f"unsupported kernel operation {op.name}; cannot ignore effects"
            )
    diagnostics.require(kernel.disjoint_external.data in {0, 1}, "noalias contract must be boolean")
    diagnostics.require(
        sum(s.external for s in roots.values()) <= 1 or kernel.disjoint_external.data,
        "external pointer alias is unknown; declare disjoint_external or use views of one root",
    )
    _, peaks = storage_layout(tuple(roots.values()))
    for bank, capacity in target_native.CAPACITY.items():
        usage = peaks.get(bank, 0)
        diagnostics.require(
            usage <= capacity, f"{bank} capacity exceeded: {usage} > {capacity} bytes"
        )
        diagnostics.require(
            not usage or (kernel.core.data == "aiv") == (bank == "UB"),
            f"{bank} storage has incompatible {kernel.core.data} owner",
        )
    views: dict[SSAValue, Ref] = {}
    scalar_values: dict[SSAValue, str] = {}
    tasks = []
    uids = set()
    # These caches belong to this extraction of actual IR. Ref is immutable;
    # equal refs have exactly the same finite addresses, extents, and holes.
    ref_pool = {}
    addresses = {}
    translations = {}
    alignments = {}

    def finite_addresses(ref):
        key = id(ref)
        if key not in addresses:
            addresses[key] = tuple(ref.address(g) for g in range(max(count, 1)))
        return addresses[key]

    def core_translations(ref):
        key = id(ref)
        if key not in translations:
            base = ref.address(0)
            translations[key] = tuple(
                ref.address(0, core) - base for core in range(blocks if ref.storage.external else 1)
            )
        return translations[key]

    def aligned_origins(ref, alignment):
        key = id(ref), alignment
        if key not in alignments:
            alignments[key] = (
                all(address % alignment == 0 for address in finite_addresses(ref))
                and all(shift % alignment == 0 for shift in core_translations(ref))
                and (ref.rows == 1 or ref.row_stride % alignment == 0)
            )
        return alignments[key]

    for op in stream.body.block.ops:
        if isinstance(op, ViewOp):
            diagnostics.require(
                op.extent_count is None
                and optional_int(op.count_offset, 0) == 0
                and optional_int(op.count_core_stride, 0) == 0,
                "runtime view translation cannot be ignored by finite extraction",
            )
            diagnostics.require(
                optional_int(op.prefix_elements, 0) == 0,
                "runtime valid prefix requires the symbolic access checker",
            )
            diagnostics.require(
                op.generation is stream.body.block.args[0], "view uses a different generation clock"
            )
            diagnostics.require(op.source in roots or op.source in views, "unknown view provenance")
            typ = op.ref.type
            diagnostics.require(
                isinstance(typ, RefType) and typ.dtype.data in target_native.ITEMSIZE,
                "unsupported view dtype",
            )
            o, s, m, length = op.offset.data, op.stride.data, op.modulus.data, op.nbytes.data
            divisor = optional_int(op.divisor, 1)
            diagnostics.require(
                o >= 0 and s >= 0 and m >= 0 and length > 0 and 1 <= divisor <= MAX_INSTANCES,
                "invalid view address/footprint/divisor",
            )
            if op.source in roots:
                root, terms = roots[op.source], ()
                bounds = lambda g, core, size=root.nbytes: ((0, size),)
                core_terms = ()
            else:
                parent = views[op.source]
                root, terms = parent.storage, parent.terms
                bounds = parent.segments
                core_terms = parent.core_terms
            diagnostics.require(
                typ.bank.data == root.bank, "a view cannot change physical bank/owner"
            )
            tail = op.tail_nbytes.data
            diagnostics.require(
                0 <= tail <= length and tail % target_native.ITEMSIZE[typ.dtype.data] == 0,
                "tail footprint must fit the view and its dtype",
            )
            rows, row_stride = optional_int(op.rows, 1), optional_int(op.row_stride, 0)
            core_stride = optional_int(op.core_stride, 0)
            core_divisor = optional_int(op.core_divisor, 1)
            core_modulus = optional_int(op.core_modulus, 0)
            diagnostics.require(
                1 <= rows <= MAX_INSTANCES
                and row_stride >= 0
                and (rows == 1 or row_stride >= length),
                "strided view rows must not overlap",
            )
            diagnostics.require(
                core_stride >= 0
                and (root.external or core_stride == 0)
                and 1 <= core_divisor <= 1024
                and 0 <= core_modulus <= 1024,
                "local physical banks are core-private, not indexed by another core",
            )
            if core_stride:
                core_terms += ((core_stride, core_divisor, core_modulus),)
            extra_terms = []
            for term in () if op.index_terms is None else op.index_terms:
                diagnostics.require(
                    len(term) == 3, "view index terms require stride/divisor/modulus"
                )
                coefficient, scale, modulus = (value.data for value in term)
                diagnostics.require(
                    coefficient >= 0
                    and 1 <= scale <= MAX_INSTANCES
                    and 0 <= modulus <= MAX_INSTANCES,
                    "invalid additional generation coordinate",
                )
                extra_terms.append((0, coefficient, modulus, scale))
            for term in () if op.core_terms is None else op.core_terms:
                diagnostics.require(
                    len(term) == 3, "view core terms require stride/divisor/modulus"
                )
                coefficient, scale, modulus = (value.data for value in term)
                diagnostics.require(
                    coefficient >= 0
                    and (root.external or coefficient == 0)
                    and 1 <= scale <= 1024
                    and 0 <= modulus <= 1024,
                    "invalid additional core coordinate",
                )
                core_terms += ((coefficient, scale, modulus),)
            ref = Ref(
                root,
                terms + ((o, s, m, divisor),) + tuple(extra_terms),
                length,
                typ.dtype.data,
                typ.layout.data,
                tail,
                count - 1,
                rows,
                row_stride,
                core_terms,
            )
            ref = ref_pool.setdefault(ref, ref)
            core_count = blocks if root.external else 1
            generations = range(max(count, 1))
            if op.source in roots:
                # Root allocations have no holes. Core/generation terms are
                # additive and independent, so exact finite maxima suffice.
                values = finite_addresses(ref)
                shifts = core_translations(ref)
                last_g = max(generations, key=lambda g: values[g] + ref.length(g))
                last_core = max(range(core_count), key=shifts.__getitem__)
                stop = (
                    values[last_g]
                    + shifts[last_core]
                    + (rows - 1) * row_stride
                    + ref.length(last_g)
                )
                diagnostics.require(
                    count == 0 and root.external or stop <= root.nbytes,
                    f"view {root.name} generation {last_g}, core {last_core} end {stop} exceeds its parent",
                )
                itemsize = target_native.ITEMSIZE[ref.dtype]
                diagnostics.require(
                    aligned_origins(ref, itemsize) and length % itemsize == 0,
                    f"view {root.name} is not aligned for its element type",
                )
                views[op.ref] = ref
                continue
            # With no new core term, a nested view and its already checked
            # parent share a translation. Sparse-row containment is invariant.
            check_cores = (
                range(core_count) if core_stride or op.core_terms is not None else range(1)
            )
            diagnostics.require(
                all(
                    shift % target_native.ITEMSIZE[ref.dtype] == 0
                    for shift in core_translations(ref)
                ),
                f"view {root.name} core translation is not aligned for its element type",
            )
            for core in check_cores:
                for g in range(max(count, 1)):
                    parts = ref.segments(g, core)
                    diagnostics.require(
                        (count == 0 and root.external) or ranges_cover(bounds(g, core), parts),
                        f"view {root.name} generation {g}, core {core} {parts} exceeds its parent",
                    )
                    diagnostics.require(
                        all(a % target_native.ITEMSIZE[ref.dtype] == 0 for a, _ in parts)
                        and length % target_native.ITEMSIZE[ref.dtype] == 0,
                        f"view {root.name} is not aligned for its element type",
                    )
            views[op.ref] = ref
        elif isinstance(op, TaskOp):
            uid, native, stage = op.uid.data, op.native.data, op.stage.data
            diagnostics.require(
                uid not in uids and re.fullmatch(r"[A-Za-z_]\w*", uid), "duplicate/invalid task ID"
            )
            uids.add(uid)
            diagnostics.require(0 <= stage <= 32, "stage must be a static offset in [0,32]")
            operands = tuple(op.refs)
            scalar_source = None
            if native in {"scale_value", "sub_value"}:
                diagnostics.require(
                    len(operands) == 3 and operands[-1] in scalar_values,
                    f"{native} needs a preceding synchronous Scalar f32 result",
                )
                scalar_source = scalar_values[operands[-1]]
                operands = operands[:-1]
            diagnostics.require(
                all(v in views for v in operands),
                "native task operands must be explicit bounded views",
            )
            refs = tuple(views[v] for v in operands)
            params = tuple(p.data for p in op.params)
            pipe, modes = target_native.native_contract(native, refs, params)
            diagnostics.require(
                (op.value is not None) == (native == "scalar_load"),
                "only synchronous Scalar load may return a numerical SSA value",
            )
            if op.value is not None:
                scalar_values[op.value] = uid
            for ref in refs:
                alignment = 4 if pipe == "S" else 32
                diagnostics.require(
                    aligned_origins(ref, alignment),
                    f"{uid}: {pipe} operand origin requires {alignment}-byte alignment",
                )
            if any(r.tail_nbytes for r in refs):
                diagnostics.require(
                    pipe in {"V", "MTE2", "MTE3"} and kernel.core.data == "aiv",
                    "Cube/Fixpipe tail recipes are not implemented",
                )
                tail_refs = tuple(
                    replace(r, nbytes=r.length(count - 1), tail_nbytes=0) for r in refs
                )
                target_native.native_contract(native, tail_refs, params)
            if pipe in {"M", "MTE1", "FIX"}:
                for ref in refs:
                    if ref.storage.bank != "GM":
                        diagnostics.require(
                            all(address % 512 == 0 for address in finite_addresses(ref)),
                            f"{uid}: Cube/Fixpipe local origin must be 512-byte aligned",
                        )
            diagnostics.require(
                (kernel.core.data == "aiv")
                == (
                    pipe in {"V", "MTE3", "S"} or (pipe == "MTE2" and refs[-1].storage.bank == "UB")
                ),
                "native task owner mismatch",
            )
            diagnostics.require(
                native not in {"fix16", "fix"} or config,
                "Fixpipe needs explicit invariant ND_PARA initialization",
            )
            scalar = task_scalar(op)
            lower, upper = optional_int(op.lower, 0), optional_int(op.upper, -1)
            upper = count if upper == -1 else min(upper, count)
            diagnostics.require(
                0 <= lower <= upper <= count,
                "task activation must be a bounded contiguous interval",
            )
            period, phase = optional_int(op.period, 1), optional_int(op.phase, 0)
            phase_width = optional_int(op.phase_width, 1)
            logical_phase = optional_int(op.logical_phase, 0)
            diagnostics.require(0 <= logical_phase <= 255, "logical phase must be in [0,255]")
            diagnostics.require(
                1 <= period <= MAX_INSTANCES
                and 0 <= phase < period
                and 1 <= phase_width <= period - phase,
                "static periodic task activation requires a nonempty phase interval inside period<=4096",
            )
            issue_order = optional_int(op.issue_order, len(tasks))
            diagnostics.require(
                0 <= issue_order <= 65535, "fixed issue order must be a bounded nonnegative rank"
            )
            task = Task(
                uid,
                native,
                pipe,
                stage,
                refs,
                modes,
                scalar,
                params,
                lower,
                upper,
                issue_order,
                scalar_source,
                period,
                phase,
                logical_phase,
                phase_width,
            )
            if scalar_source is not None:
                provider = next(t for t in tasks if t.uid == scalar_source)
                diagnostics.require(
                    provider.stage == stage
                    and provider.lower <= lower
                    and provider.upper >= upper
                    and all(provider.active(g) for g in range(lower, upper) if task.active(g))
                    and (provider.issue_order, tasks.index(provider)) < (issue_order, len(tasks)),
                    "Scalar arguments must be captured earlier in the same physical step; "
                    "delayed descriptors need explicit storage",
                )
            # Native lane overlap must itself be legal; no event can repair it.
            if pipe == "V":
                for r in refs[:-1]:
                    dst = refs[-1]
                    if r.storage != dst.storage or r is dst:
                        continue
                    for g in range(count):
                        old, new = r.segments(g), dst.segments(g)
                        diagnostics.require(
                            old == new or not ranges_overlap(old, new),
                            f"{uid}: shifted in-place Vector overlap is unsupported",
                        )
            tasks.append(task)
        elif not isinstance(op, (SyncOp, EndOp)):
            raise diagnostics.CompileError(
                f"unsupported stream operation {op.name}; dynamic control needs a contract"
            )
    diagnostics.require(tasks, "stream must contain native tasks")
    target_native.check_load3d(tasks, load3d)
    diagnostics.require(
        sum(sum(t.active(g) for g in range(count)) for t in tasks) <= MAX_INSTANCES,
        f"finite verifier budget exceeded ({MAX_INSTANCES} task instances); no unchecked fallback",
    )
    epoch_divisor = optional_int(kernel.epoch_divisor, 1)
    diagnostics.require(
        type(epoch_divisor) is int
        and 1 <= epoch_divisor <= MAX_INSTANCES
        and count % epoch_divisor == 0,
        "native generation clock must form complete static logical epochs",
    )
    program = Program(
        kernel.sym_name.data,
        kernel.core.data,
        count,
        tuple(roots.values()),
        tuple(tasks),
        config,
        blocks,
        epoch_divisor,
        load3d,
    )
    verify_independent_owners(program)
    return program


def verify_independent_owners(program):
    if program.blocks == 1:
        return
    instances = program.instances()
    active_tasks = {i.task for i in instances}
    written_roots = {
        ref.storage.name
        for i in active_tasks
        for ref, mode in zip(program.tasks[i].refs, program.tasks[i].modes)
        if ref.storage.external and "W" in mode
    }
    refs = {
        ref
        for i in active_tasks
        for ref in program.tasks[i].refs
        if ref.storage.name in written_roots
    }
    effects = [
        (ref.storage.name, lo, hi, mode)
        for inst in instances
        for ref, mode in zip(
            program.tasks[inst.task].refs, program.tasks[inst.task].modes_for(inst.generation)
        )
        if ref.storage.name in written_roots
        for lo, hi in ref.segments(inst.generation)
    ]
    verify_translated_owners(
        refs,
        effects,
        program.blocks,
        translation_error="views of a written GM root must share one core translation; core-dependent internal aliases need per-actor synchronization",
        overlap_error="cross-core ownership overlap; handoff/global synchronization is not implemented",
    )


def verify_translated_owners(refs, effects, blocks, *, translation_error, overlap_error):
    """Check every core's additive translation and all exact R/W interval unions.

    Materialize real group-0 effects including holes. Check EVERY written-root
    reference's translation for EVERY core, then sweep ALL translated R/W
    intervals. No generation/core sample or bounding-box alias assumption.
    """
    import heapq

    base = {}
    for root, lo, hi, mode in effects:
        for access in ("R", "W"):
            if access in mode:
                base.setdefault((root, access), []).append((lo, hi))
    unions = {}
    for key, parts in base.items():
        merged = []
        for lo, hi in sorted(parts):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        unions[key] = merged
    translated = {}
    for core in range(blocks):
        shifts = {}
        for ref in refs:
            shifts.setdefault(ref.storage.name, set()).add(ref.address(0, core) - ref.address(0))
        for root, values in shifts.items():
            diagnostics.require(len(values) == 1, f"core {core}, {root}: {translation_error}")
            shift = next(iter(values))
            for access in ("R", "W"):
                translated.setdefault(root, []).extend(
                    (lo + shift, hi + shift, access, core)
                    for lo, hi in unions.get((root, access), ())
                )
    for root, parts in translated.items():
        active = {"R": {}, "W": {}}
        deadlines = []
        for lo, hi, access, core in sorted(parts):
            while deadlines and deadlines[0][0] <= lo:
                end, kind, owner = heapq.heappop(deadlines)
                if active[kind].get(owner) == end:
                    del active[kind][owner]
            conflicts = ("R", "W") if access == "W" else ("W",)
            diagnostics.require(
                all(len(active[kind]) == int(core in active[kind]) for kind in conflicts),
                f"{root}[{lo}:{hi}], core {core}: {overlap_error}",
            )
            active[access][core] = hi
            heapq.heappush(deadlines, (hi, access, core))


def storage_layout(storages):
    offsets, peaks = {}, {}
    for s in storages:
        if s.external:
            continue
        alignment = 32 if s.bank == "UB" else 512
        offset = (peaks.get(s.bank, 0) + alignment - 1) // alignment * alignment
        offsets[s.name] = offset
        peaks[s.bank] = offset + s.nbytes
    return offsets, peaks


def dependencies(program: Program, *, _imports=()) -> tuple[Obligation, ...]:
    """Interval frontier in semantic generation/source order, independent of issue stages.

    Each byte segment has a last writer and ALL unretired readers. Reading one
    subview cannot kill another reader. RW processes the read before the write.
    """
    cuts = {s.name: {0, s.nbytes} for s in program.storages}
    effects = []
    for instance in program.instances():
        task = program.tasks[instance.task]
        accesses = []
        for ref, mode in zip(task.refs, task.modes_for(instance.generation)):
            for start, stop in ref.segments(instance.generation):
                cuts[ref.storage.name].update((start, stop))
                accesses.append((ref.storage.name, start, stop, mode))
        effects.append((instance, accesses))
    points = {name: tuple(sorted(values)) for name, values in cuts.items()}
    positions = {
        name: {point: i for i, point in enumerate(values)} for name, values in points.items()
    }
    # Dense cells index exact cut intervals, including every hole. This avoids
    # re-allocating/hashing a (root, start, stop) key for every native access.
    writers = {name: [None] * (len(values) - 1) for name, values in points.items()}
    readers = {name: [None] * (len(values) - 1) for name, values in points.items()}
    diagnostics.require(
        set(_imports) <= {s.name for s in program.storages if s.external}, "invalid imports"
    )
    initialized = {s.name: s.initialized or s.name in _imports for s in program.storages}
    edges = {}

    def edge(src, dst, kind, root, a, b):
        if src is None or src == dst:
            return
        spans = edges.setdefault((src, dst, kind, root), [])
        if spans and spans[-1][0] <= a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(b, spans[-1][1]))
        else:
            spans.append((a, b))

    for instance, footprints in effects:
        task = program.tasks[instance.task]
        # Process ALL reads before any write, even with multiple aliased refs.
        for root, start, stop, mode in footprints:
            if "R" not in mode:
                continue
            boundaries, index = points[root], positions[root]
            previous, pending = writers[root], readers[root]
            for cell in range(index[start], index[stop]):
                a, b = boundaries[cell], boundaries[cell + 1]
                diagnostics.require(
                    previous[cell] is not None or initialized[root],
                    f"{task.uid}[{instance.generation}] reads undefined {root}[{a}:{b}]",
                )
                edge(previous[cell], instance, "RAW", root, a, b)
                if pending[cell] is None:
                    pending[cell] = set()
                pending[cell].add(instance)
        for root, start, stop, mode in footprints:
            if "W" not in mode:
                continue
            boundaries, index = points[root], positions[root]
            previous, pending = writers[root], readers[root]
            for cell in range(index[start], index[stop]):
                a, b = boundaries[cell], boundaries[cell + 1]
                edge(previous[cell], instance, "WAW", root, a, b)
                for src in pending[cell] or ():
                    edge(src, instance, "WAR", root, a, b)
                pending[cell] = None
                previous[cell] = instance
    obligations = []
    for (src, dst, kind, root), spans in edges.items():
        merged = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        obligations.extend(
            Obligation(
                src, dst, kind, root, a, b, "read_retired" if kind == "WAR" else "write_done"
            )
            for a, b in merged
        )
    return tuple(
        sorted(
            obligations,
            key=lambda o: (
                o.destination.generation,
                o.destination.task,
                o.source.generation,
                o.source.task,
                o.hazard,
                o.storage,
                o.start,
            ),
        )
    )
