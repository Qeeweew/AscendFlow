"""Trusted dav-2201 native contracts and exact finite physical-access analysis.

No analysis reads a compiler certificate as a premise. Layout changes are legal
only at a named native recipe; a subview/recast preserves allocation identity.
The first version deliberately bounds finite expansion, and emits compact loops.
"""

import re
from dataclasses import dataclass, replace
from math import isfinite

from xdsl.dialects.builtin import IndexType, ModuleOp, f32
from xdsl.ir import SSAValue

from .ir import (
    BufferOp,
    ChannelOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    RefType,
    StreamOp,
    SyncOp,
    TaskOp,
    ViewOp,
)

CAPACITY = {
    "UB": 192 * 1024,
    "L1": 512 * 1024,
    "L0A": 64 * 1024,
    "L0B": 64 * 1024,
    "L0C": 128 * 1024,
}
ITEMSIZE = {"f32": 4, "f16": 2}
IMPLICIT_COMPLETION = {"S", "M"}  # NOT V, MTE1/2/3 or FIX on 2201.
MAX_INSTANCES = 4096


class CompileError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CompileError(message)


def optional_int(attr, default):
    return default if attr is None else attr.data


def ranges_cover(parent, child):
    for start, stop in child:
        cursor = start
        for lo, hi in sorted(parent):
            if lo <= cursor < hi:
                cursor = max(cursor, hi)
            if cursor >= stop:
                break
        if cursor < stop:
            return False
    return True


@dataclass(frozen=True)
class Storage:
    name: str
    bank: str
    nbytes: int
    dtype: str
    layout: str
    external: bool
    initialized: bool


@dataclass(frozen=True)
class Ref:
    storage: Storage
    terms: tuple[tuple[int, int, int], ...]  # offset, stride, modulus, in bytes
    nbytes: int
    dtype: str
    layout: str
    tail_nbytes: int = 0
    tail_generation: int = -1
    rows: int = 1
    row_stride: int = 0
    core_stride: int = 0

    def address(self, g: int, core=0) -> int:
        return sum(o + s * (g % m if m else g) for o, s, m in self.terms) + core * self.core_stride

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

    def instances(self):
        return [
            Instance(g, t)
            for g in range(self.count)
            for t, task in enumerate(self.tasks)
            if task.lower <= g < (self.count if task.upper == -1 else task.upper)
        ]

    def issued(self):
        return sorted(
            self.instances(),
            key=lambda i: (
                i.generation + self.tasks[i.task].stage,
                self.tasks[i.task].issue_order,
                i.task,
            ),
        )


def native_contract(native: str, refs: tuple[Ref, ...], params: tuple[int, ...]):
    banks = tuple(r.storage.bank for r in refs)
    dtypes = tuple(r.dtype for r in refs)
    lengths = tuple(r.total_nbytes for r in refs)
    layouts = tuple(r.layout for r in refs)
    if native == "copy":
        require(all(r.contiguous for r in refs), "DMA recipe requires contiguous physical views")
        require(len(refs) == 2 and lengths[0] == lengths[1], "copy needs equal native footprints")
        require(dtypes[0] == dtypes[1], "copy cannot silently convert dtype")
        routes = {
            ("GM", "UB"): "MTE2",
            ("UB", "GM"): "MTE3",
            ("GM", "L1"): "MTE2",
            ("L1", "L0A"): "MTE1",
            ("L1", "L0B"): "MTE1",
        }
        require(banks in routes, f"unsupported dav-2201 DMA route {banks}; no hidden GM staging")
        require(not params, "copy has no unchecked native parameters")
        if banks[0] == "L1":
            expected = ("NZ", "ZZ" if banks[1] == "L0A" else "ZN")
            require(
                dtypes == ("f16", "f16") and lengths == (512, 512) and layouts == expected,
                "L1->L0 v0.1 requires one complete f16 16x16 fractal",
            )
        else:
            require(layouts[0] == layouts[1], "raw DMA must preserve physical layout")
            require(
                lengths[0] % 32 == 0 and 0 < lengths[0] <= 65535 * 32,
                "raw DMA footprint must be positive, 32-byte aligned and fit burst length",
            )
        return routes[banks], ("R", "W")
    if native in {"muls", "adds", "add", "mul", "sub", "div", "exp", "rsqrt", "scale_value"}:
        arity = 3 if native in {"add", "mul", "sub", "div"} else 2
        require(
            len(refs) == arity
            and banks == ("UB",) * arity
            and dtypes == ("f32",) * arity
            and layouts == ("linear",) * arity
            and len(set(lengths)) == 1
            and lengths[0] > 0
            and lengths[0] % 4 == 0,
            "Vector recipe requires equal contiguous f32 UB footprints",
        )
        require(all(r.contiguous for r in refs), "count-mode Vector requires contiguous views")
        require(not params, "Vector recipe has no unchecked native parameters")
        return "V", ("R",) * (arity - 1) + ("W",)
    if native == "scalar_load":
        require(
            banks == ("UB",)
            and dtypes == ("f32",)
            and layouts == ("linear",)
            and lengths == (4,)
            and not params,
            "Scalar load requires one f32 UB word",
        )
        return "S", ("R",)
    if native in {"mul_rows", "sub_rows", "add_rows", "div_rows"}:
        require(len(params) == 2, "row binary recipe needs (rows, columns)")
        rows, cols = params
        require(
            1 <= rows <= 255
            and cols > 0
            and cols % 8 == 0
            and len(refs) == 3
            and banks == ("UB",) * 3
            and dtypes == ("f32",) * 3
            and layouts == ("linear",) * 3
            and all(r.rows == rows and r.nbytes == cols * 4 for r in refs),
            "row binary requires bounded equal f32 rows with aligned origins",
        )
        return "V", ("R", "R", "W")
    if native == "mad16":
        require(
            banks == ("L0A", "L0B", "L0C")
            and dtypes == ("f16", "f16", "f32")
            and lengths == (512, 512, 1024)
            and layouts == ("ZZ", "ZN", "NZ")
            and params in {(0,), (1,)},
            "MAD v0.1 requires f16 16x16x16, f32 accumulation",
        )
        return "M", ("R", "R", "W" if params[0] else "RW")
    if native == "fix16":
        require(
            banks == ("L0C", "GM")
            and dtypes == ("f32", "f32")
            and lengths == (1024, 1024)
            and layouts == ("NZ", "linear")
            and not params,
            "Fixpipe v0.1 requires a complete f32 16x16 NZ->ND transfer",
        )
        return "FIX", ("R", "W")
    raise CompileError(
        f"unmodeled native operation {native!r}: effects/completion contract required"
    )


def extract(module: ModuleOp) -> Program:
    module.verify()
    require(len(list(module.body.block.ops)) == 1, "one kernel per compilation is required")
    kernel = module.body.block.first_op
    require(isinstance(kernel, KernelOp), "expected npu.kernel")
    require(kernel.target.data == "dav-2201", "only Ascend 910B / dav-2201 is supported")
    require(kernel.core.data in {"aic", "aiv"}, "cross AIC/AIV ownership is not implemented")
    blocks = optional_int(kernel.blocks, 1)
    require(1 <= blocks <= 1024, "independent core grid must be static in [1,1024]")
    require(re.fullmatch(r"[A-Za-z_]\w*", kernel.sym_name.data), "invalid kernel identifier")
    streams = [op for op in kernel.body.block.ops if isinstance(op, StreamOp)]
    require(
        len(streams) == 1, "one finite stream is required; dynamic/nested streams are unsupported"
    )
    stream = streams[0]
    count = stream.count.data
    require(
        isinstance(count, int) and 0 <= count <= MAX_INSTANCES,
        f"task count must be a nonnegative static integer <= {MAX_INSTANCES}",
    )
    require(
        len(stream.body.block.args) == 1 and isinstance(stream.body.block.args[0].type, IndexType),
        "stream requires exactly one logical generation argument",
    )
    roots: dict[SSAValue, Storage] = {}
    names = set()
    config = False
    seen_stream = False
    for op in kernel.body.block.ops:
        if isinstance(op, StreamOp):
            seen_stream = True
        elif isinstance(op, BufferOp):
            require(not seen_stream, "storage declarations must precede the stream")
            typ = op.ref.type
            require(isinstance(typ, RefType), "invalid buffer type")
            name, bank, size = op.sym_name.data, typ.bank.data, op.nbytes.data
            require(
                re.fullmatch(r"[A-Za-z_]\w*", name) and name not in names,
                f"duplicate or invalid storage identifier {name!r}",
            )
            names.add(name)
            require(
                name not in {"g", "step", "core_id", "stream", "npu_row", "npu_chunk"}
                and not name.startswith("npu_value_"),
                "storage name conflicts with internal ASC variables",
            )
            require(
                bank in {*CAPACITY, "GM"} and typ.dtype.data in ITEMSIZE, "unsupported storage type"
            )
            require(
                op.external.data in {0, 1} and op.initialized.data in {0, 1},
                "storage contracts must be boolean",
            )
            require(
                (bank == "GM") == bool(op.external.data),
                "GM is external; local banks are allocated",
            )
            require(
                size >= 0 and (bank == "GM" or size % 32 == 0), "invalid allocation size/alignment"
            )
            require(
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
            require(
                not seen_stream and not config and kernel.core.data == "aic",
                "FIX configuration must be unique, unconditional and before the stream",
            )
            config = True
        elif not isinstance(op, (ChannelOp, EndOp)):
            raise CompileError(f"unsupported kernel operation {op.name}; cannot ignore effects")
    require(kernel.disjoint_external.data in {0, 1}, "noalias contract must be boolean")
    require(
        sum(s.external for s in roots.values()) <= 1 or kernel.disjoint_external.data,
        "external pointer alias is unknown; declare disjoint_external or use views of one root",
    )
    _, peaks = storage_layout(tuple(roots.values()))
    for bank, capacity in CAPACITY.items():
        usage = peaks.get(bank, 0)
        require(usage <= capacity, f"{bank} capacity exceeded: {usage} > {capacity} bytes")
        require(
            not usage or (kernel.core.data == "aiv") == (bank == "UB"),
            f"{bank} storage has incompatible {kernel.core.data} owner",
        )
    views: dict[SSAValue, Ref] = {}
    scalar_values: dict[SSAValue, str] = {}
    tasks = []
    uids = set()
    for op in stream.body.block.ops:
        if isinstance(op, ViewOp):
            require(
                op.generation is stream.body.block.args[0], "view uses a different generation clock"
            )
            require(op.source in roots or op.source in views, "unknown view provenance")
            typ = op.ref.type
            require(
                isinstance(typ, RefType) and typ.dtype.data in ITEMSIZE, "unsupported view dtype"
            )
            o, s, m, length = op.offset.data, op.stride.data, op.modulus.data, op.nbytes.data
            require(o >= 0 and s >= 0 and m >= 0 and length > 0, "invalid view address/footprint")
            if op.source in roots:
                root, terms = roots[op.source], ()
                bounds = lambda g, core, size=root.nbytes: ((0, size),)
                inherited_core_stride = 0
            else:
                parent = views[op.source]
                root, terms = parent.storage, parent.terms
                bounds = parent.segments
                inherited_core_stride = parent.core_stride
            require(typ.bank.data == root.bank, "a view cannot change physical bank/owner")
            tail = op.tail_nbytes.data
            require(
                0 <= tail <= length and tail % ITEMSIZE[typ.dtype.data] == 0,
                "tail footprint must fit the view and its dtype",
            )
            rows, row_stride = optional_int(op.rows, 1), optional_int(op.row_stride, 0)
            core_stride = inherited_core_stride + optional_int(op.core_stride, 0)
            require(
                1 <= rows <= 255 and row_stride >= 0 and (rows == 1 or row_stride >= length),
                "strided view rows must not overlap",
            )
            require(
                core_stride >= 0 and (root.external or core_stride == 0),
                "local physical banks are core-private, not indexed by another core",
            )
            ref = Ref(
                root,
                terms + ((o, s, m),),
                length,
                typ.dtype.data,
                typ.layout.data,
                tail,
                count - 1,
                rows,
                row_stride,
                core_stride,
            )
            for core in range(blocks if root.external else 1):
                for g in range(max(count, 1)):
                    parts = ref.segments(g, core)
                    require(
                        (count == 0 and root.external) or ranges_cover(bounds(g, core), parts),
                        f"view {root.name} generation {g}, core {core} {parts} exceeds its parent",
                    )
                    require(
                        all(a % ITEMSIZE[ref.dtype] == 0 for a, _ in parts)
                        and length % ITEMSIZE[ref.dtype] == 0,
                        f"view {root.name} is not aligned for its element type",
                    )
            views[op.ref] = ref
        elif isinstance(op, TaskOp):
            uid, native, stage = op.uid.data, op.native.data, op.stage.data
            require(
                uid not in uids and re.fullmatch(r"[A-Za-z_]\w*", uid), "duplicate/invalid task ID"
            )
            uids.add(uid)
            require(0 <= stage <= 32, "stage must be a static offset in [0,32]")
            operands = tuple(op.refs)
            scalar_source = None
            if native == "scale_value":
                require(
                    len(operands) == 3 and operands[-1] in scalar_values,
                    "scale_value needs a preceding synchronous Scalar f32 result",
                )
                scalar_source = scalar_values[operands[-1]]
                operands = operands[:-1]
            require(
                all(v in views for v in operands),
                "native task operands must be explicit bounded views",
            )
            refs = tuple(views[v] for v in operands)
            params = tuple(p.data for p in op.params)
            pipe, modes = native_contract(native, refs, params)
            require(
                (op.value is not None) == (native == "scalar_load"),
                "only synchronous Scalar load may return a numerical SSA value",
            )
            if op.value is not None:
                scalar_values[op.value] = uid
            for ref in refs:
                alignment = 4 if pipe == "S" else 32
                require(
                    all(
                        a % alignment == 0 for g in range(max(count, 1)) for a, _ in ref.segments(g)
                    ),
                    f"{uid}: {pipe} operand origin requires {alignment}-byte alignment",
                )
            if any(r.tail_nbytes for r in refs):
                require(
                    pipe in {"V", "MTE2", "MTE3"} and kernel.core.data == "aiv",
                    "Cube/Fixpipe tail recipes are not implemented",
                )
                tail_refs = tuple(
                    replace(r, nbytes=r.length(count - 1), tail_nbytes=0) for r in refs
                )
                native_contract(native, tail_refs, params)
            if pipe in {"M", "MTE1", "FIX"}:
                for ref in refs:
                    if ref.storage.bank != "GM":
                        require(
                            all(ref.address(g) % 512 == 0 for g in range(max(count, 1))),
                            f"{uid}: Cube/Fixpipe local origin must be 512-byte aligned",
                        )
            require(
                (kernel.core.data == "aiv")
                == (
                    pipe in {"V", "MTE3", "S"} or (pipe == "MTE2" and refs[-1].storage.bank == "UB")
                ),
                "native task owner mismatch",
            )
            require(
                native != "fix16" or config,
                "Fixpipe needs explicit invariant ND_PARA initialization",
            )
            require(op.scalar.type == f32, "native Vector scalar literals must be f32")
            scalar = op.scalar.value.data
            require(isfinite(scalar), "Vector scalar must be finite")
            lower, upper = optional_int(op.lower, 0), optional_int(op.upper, -1)
            upper = count if upper == -1 else min(upper, count)
            require(
                0 <= lower <= upper <= count,
                "task activation must be a bounded contiguous interval",
            )
            issue_order = optional_int(op.issue_order, len(tasks))
            require(
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
            )
            if scalar_source is not None:
                provider = next(t for t in tasks if t.uid == scalar_source)
                require(
                    provider.stage == stage
                    and provider.lower <= lower
                    and provider.upper >= upper
                    and (provider.issue_order, tasks.index(provider)) < (issue_order, len(tasks)),
                    "Scalar arguments must be captured earlier in the same physical step; "
                    "delayed descriptors need explicit storage",
                )
            # Native lane overlap must itself be legal; no event can repair it.
            if pipe == "V":
                for r in refs[:-1]:
                    dst = refs[-1]
                    for g in range(count):
                        old, new = r.segments(g), dst.segments(g)
                        require(
                            r.storage != dst.storage
                            or old == new
                            or not any(a < d and c < b for a, b in old for c, d in new),
                            f"{uid}: shifted in-place Vector overlap is unsupported",
                        )
            tasks.append(task)
        elif not isinstance(op, (SyncOp, EndOp)):
            raise CompileError(
                f"unsupported stream operation {op.name}; dynamic control needs a contract"
            )
    require(tasks, "stream must contain native tasks")
    require(
        sum(t.upper - t.lower for t in tasks) <= MAX_INSTANCES,
        f"finite verifier budget exceeded ({MAX_INSTANCES} task instances); no unchecked fallback",
    )
    program = Program(
        kernel.sym_name.data,
        kernel.core.data,
        count,
        tuple(roots.values()),
        tuple(tasks),
        config,
        blocks,
    )
    verify_independent_owners(program)
    return program


def verify_independent_owners(program):
    if program.blocks == 1:
        return
    accesses = {}
    for core in range(program.blocks):
        for inst in program.instances():
            task = program.tasks[inst.task]
            for ref, mode in zip(task.refs, task.modes):
                if ref.storage.external:
                    for a, b in ref.segments(inst.generation, core):
                        accesses.setdefault((core, ref.storage.name), []).append((a, b, mode))
    unions = {}
    for key, regions in accesses.items():
        for access in ("R", "W"):
            merged = []
            for a, b in sorted((a, b) for a, b, mode in regions if access in mode):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            unions[(*key, access)] = merged
    for (core, root, access), writes in unions.items():
        if access != "W" or not writes:
            continue
        for other in range(program.blocks):
            if core == other:
                continue
            for mode in ("R", "W"):
                require(
                    not any(
                        a < d and c < b
                        for a, b in writes
                        for c, d in unions.get((other, root, mode), ())
                    ),
                    f"cross-core {root} ownership overlap {core}->{other}; "
                    "handoff/global synchronization is not implemented",
                )


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


def dependencies(program: Program) -> tuple[Obligation, ...]:
    """Interval frontier in semantic generation/source order, independent of issue stages.

    Each byte segment has a last writer and ALL unretired readers. Reading one
    subview cannot kill another reader. RW processes the read before the write.
    """
    cuts = {s.name: {0, s.nbytes} for s in program.storages}
    for instance in program.instances():
        for ref in program.tasks[instance.task].refs:
            for interval in ref.segments(instance.generation):
                cuts[ref.storage.name].update(interval)
    segments = {
        name: tuple(zip(sorted(points), sorted(points)[1:])) for name, points in cuts.items()
    }
    writers = {}
    readers = {}
    initialized = {s.name: s.initialized for s in program.storages}
    obligations = set()

    def edge(src, dst, kind, root, a, b):
        if src is not None and src != dst:
            obligations.add(
                Obligation(
                    src, dst, kind, root, a, b, "read_retired" if kind == "WAR" else "write_done"
                )
            )

    for instance in program.instances():
        task = program.tasks[instance.task]
        accesses = []
        for ref, mode in zip(task.refs, task.modes):
            for start, stop in ref.segments(instance.generation):
                accesses.extend(
                    ((ref.storage.name, a, b), mode)
                    for a, b in segments[ref.storage.name]
                    if start <= a and b <= stop
                )
        for key, mode in accesses:
            if "R" in mode:
                require(
                    key in writers or initialized[key[0]],
                    f"{task.uid}[{instance.generation}] reads undefined {key[0]}[{key[1]}:{key[2]}]",
                )
                edge(writers.get(key), instance, "RAW", *key)
                readers.setdefault(key, set()).add(instance)
        for key, mode in accesses:
            if "W" in mode:
                edge(writers.get(key), instance, "WAW", *key)
                for src in readers.get(key, ()):
                    edge(src, instance, "WAR", *key)
                readers[key] = set()
                writers[key] = instance
    return tuple(
        sorted(obligations, key=lambda o: (o.destination, o.source, o.hazard, o.storage, o.start))
    )
