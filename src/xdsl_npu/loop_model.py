"""Parametric physical effects for fixed schedules and uniform loop lengths.

Only a storage's declared modulo period is enumerated. Runtime N is never
instantiated here. Stream frames are disjoint by an affine ownership contract;
ring cells use an exact cyclic frontier with explicit first-occurrence checks.
"""

import re
from dataclasses import dataclass
from itertools import pairwise
from math import lcm

from xdsl.dialects import arith
from xdsl.dialects.builtin import IndexType

from .ir import (
    BufferOp,
    EndOp,
    FixConfigOp,
    KernelOp,
    LoopChannelOp,
    LoopSyncOp,
    MixedOp,
    RuntimeSizeOp,
    StreamOp,
    TaskOp,
    ViewOp,
)
from .model import (
    CAPACITY,
    ITEMSIZE,
    Ref,
    Storage,
    Task,
    native_contract,
    optional_int,
    ranges_overlap,
    require,
    round_f32,
    storage_layout,
)
from .scalar import ARITHMETIC, Expr, bounded_by, expression, from_ssa, literal, uniform


@dataclass(frozen=True)
class Templates:
    """One actor's native templates; there is no fictitious zero-trip Program."""

    name: str
    core: str
    storages: tuple[Storage, ...]
    tasks: tuple[Task, ...]
    config: bool
    blocks: int


@dataclass(frozen=True)
class Actor:
    name: str
    kernel: KernelOp
    stream: StreamOp
    program: Templates
    operations: dict


@dataclass(frozen=True)
class Loop:
    op: object
    actors: dict[str, Actor]
    roots: dict[str, BufferOp]
    size: RuntimeSizeOp | None
    constant_count: int | None
    period: int
    frames: dict[str, int]
    parameters: tuple[RuntimeSizeOp, ...]
    bound: Expr | None

    @property
    def groups(self):
        return optional_int(self.op.blocks, 1)

    def root(self, actor, ref):
        return ref.storage.name if ref.storage.external else actor + "/" + ref.storage.name


@dataclass(frozen=True)
class Dependence:
    source: tuple[str, str]
    destination: tuple[str, str]
    distance: int
    period: int
    residues: tuple[int, ...]
    hazard: str
    storage: str
    start: int
    stop: int


def is_symbolic(module):
    return any(
        isinstance(op, StreamOp) and (op.limit is not None or op.symbolic is not None)
        for op in module.walk()
    )


def _storage(op, limit, uniform_values):
    typ = op.ref.type
    scale = optional_int(op.count_scale, 0)
    require(
        0 <= scale <= (1 << 63) - 1 and (scale > 0) == (op.extent_count is not None),
        "runtime storage extent must bind its SSA count",
    )
    require(
        op.extent_count is None or op.extent_count in uniform_values,
        "buffer extent needs a declared uniform scalar",
    )
    bank = typ.bank.data
    require(bank in {*CAPACITY, "GM"} and typ.dtype.data in ITEMSIZE, "unknown physical storage")
    require(
        op.external.data in {0, 1} and bool(op.external.data) == (bank == "GM"),
        "GM is external; local banks are private",
    )
    require(
        op.initialized.data in {0, 1} and (not op.initialized.data or bank == "GM"),
        "local initialization needs a real writer",
    )
    require(
        0 <= op.nbytes.data <= (1 << 63) - 1 and (bank == "GM" or op.nbytes.data % 32 == 0),
        "invalid allocation size",
    )
    require(
        not scale or (bank == "GM" and op.nbytes.data == 0),
        "dynamic extent needs a complete stream allocation",
    )
    name = op.sym_name.data
    require(
        re.fullmatch(r"[A-Za-z_]\w*", name)
        and name not in {"n", "g", "step", "core_id", "stream"}
        and not name.startswith("npu_"),
        "allocation name conflicts with ASC variables",
    )
    return Storage(
        name,
        bank,
        op.nbytes.data,
        typ.dtype.data,
        typ.layout.data,
        bool(op.external.data),
        bool(op.initialized.data),
        scale,
        None if op.extent_count is None else uniform_values[op.extent_count],
    )


def _actor(name, kernel, shared, limit, groups, uniform_values):
    require(
        kernel.core.data == ("aic" if name == "aic" else "aiv")
        and optional_int(kernel.blocks, 1) == groups,
        "actor owner/grid mismatch",
    )
    declarations, streams = dict(shared), []
    config = False
    for op in kernel.body.block.ops:
        if isinstance(op, BufferOp):
            declarations[op.ref] = _storage(op, limit, uniform_values)
        elif isinstance(op, StreamOp):
            streams.append(op)
        elif isinstance(op, FixConfigOp):
            require(
                not config and kernel.core.data == "aic" and not streams,
                "FIX config is unique and precedes the stream",
            )
            config = True
        else:
            require(
                isinstance(op, (RuntimeSizeOp, LoopChannelOp, EndOp))
                or (isinstance(op, arith.ConstantOp) or type(op) in ARITHMETIC)
                and all(value in uniform_values for value in op.results),
                f"unsupported loop kernel effect {op.name}",
            )
    require(len(streams) == 1, "one symbolic task stream per actor required")
    stream = streams[0]
    require(
        stream.limit is limit and (stream.count is None) == (limit is not None),
        "all actors must share one immutable task count",
    )
    require(
        len(stream.body.block.args) == 1 and isinstance(stream.body.block.args[0].type, IndexType),
        "loop requires one generation coordinate",
    )
    require(
        len({storage.name for storage in declarations.values()}) == len(declarations),
        "duplicate physical storage names",
    )
    _, peaks = storage_layout(tuple(declarations.values()))
    for bank, size in peaks.items():
        require(
            size <= CAPACITY[bank] and ((kernel.core.data == "aiv") == (bank == "UB")),
            f"{bank} capacity/owner violation",
        )
    views, tasks, operations = {}, [], {}
    scalar_values = {}
    for op in stream.body.block.ops:
        if isinstance(op, ViewOp):
            require(op.generation is stream.body.block.args[0], "view uses a different task clock")
            require(op.source in declarations or op.source in views, "unknown view provenance")
            parent = views.get(op.source)
            storage = parent.storage if parent else declarations[op.source]
            require(
                op.ref.type.bank.data == storage.bank and op.ref.type.dtype.data in ITEMSIZE,
                "view bank/dtype violation",
            )
            require(
                op.tail_nbytes.data == 0, "symbolic tail access needs an explicit mask/extent path"
            )
            extra = []
            for term in () if op.index_terms is None else op.index_terms:
                require(len(term) == 3, "generation term needs coefficient/divisor/modulus")
                coefficient, divisor, modulus = (v.data for v in term)
                extra.append((0, coefficient, modulus, divisor))
            terms = (
                (() if parent is None else parent.terms)
                + ((op.offset.data, op.stride.data, op.modulus.data, optional_int(op.divisor, 1)),)
                + tuple(extra)
            )
            require(
                all(
                    offset >= 0 and stride >= 0 and divisor > 0 and modulus >= 0
                    for offset, stride, modulus, divisor in terms
                ),
                "invalid periodic physical address",
            )
            require(
                all(
                    modulus or divisor == 1 or stride == 0 for _, stride, modulus, divisor in terms
                ),
                "divided unbounded drift needs a segmented access contract",
            )
            core = () if parent is None else parent.core_terms
            stride = optional_int(op.core_stride, 0)
            if stride:
                core += (
                    (stride, optional_int(op.core_divisor, 1), optional_int(op.core_modulus, 0)),
                )
            for term in () if op.core_terms is None else op.core_terms:
                require(len(term) == 3, "core term needs coefficient/divisor/modulus")
                core += (tuple(v.data for v in term),)
            require(
                all(s >= 0 and d > 0 and m >= 0 for s, d, m in core), "invalid core translation"
            )
            count_offset = (0 if parent is None else parent.count_offset) + optional_int(
                op.count_offset, 0
            )
            count_core = (0 if parent is None else parent.count_core_stride) + optional_int(
                op.count_core_stride, 0
            )
            prefix_elements = optional_int(op.prefix_elements, 0)
            require(prefix_elements >= 0, "negative prefix tile extent")
            prefix_extent = None
            if prefix_elements:
                require(
                    parent is None and op.extent_count in uniform_values,
                    "valid prefix requires its uniform extent and direct allocation view",
                )
                require(
                    not count_offset and not count_core,
                    "valid prefix cannot change dynamic ownership",
                )
                prefix_extent = uniform_values[op.extent_count]
            require(
                count_offset >= 0
                and count_core >= 0
                and (
                    op.extent_count is None or op.extent_count is limit or prefix_extent is not None
                ),
                "invalid count-dependent address",
            )
            require(
                not (count_offset or count_core)
                or (limit is not None and op.extent_count is limit),
                "dynamic address must retain its SSA count",
            )
            require(
                storage.external or (not core and not count_offset and not count_core),
                "local bank is private to its actor",
            )
            rows, pitch = optional_int(op.rows, 1), optional_int(op.row_stride, 0)
            require(
                op.nbytes.data > 0 and rows > 0 and (rows == 1 or pitch >= op.nbytes.data),
                "overlapping/empty native rows",
            )
            ref = Ref(
                storage,
                terms,
                op.nbytes.data,
                op.ref.type.dtype.data,
                op.ref.type.layout.data,
                rows=rows,
                row_stride=pitch,
                core_terms=core,
                count_offset=count_offset,
                count_core_stride=count_core,
                prefix_extent=prefix_extent,
                prefix_elements=prefix_elements,
            )
            # Nested periodic views are checked later for every residue, with
            # the same count translation. No bounding box replaces their holes.
            views[op.ref] = ref
            if parent is not None:
                require(
                    (ref.count_offset, ref.count_core_stride)
                    == (parent.count_offset, parent.count_core_stride),
                    "nested views cannot change runtime ownership",
                )
        elif isinstance(op, TaskOp):
            uid = op.uid.data
            require(
                re.fullmatch(r"[A-Za-z_]\w*", uid) and uid not in operations,
                "duplicate/invalid task UID",
            )
            operands, scalar_source = tuple(op.refs), None
            if op.native.data in {"scale_value", "sub_value"}:
                require(
                    len(operands) == 3 and operands[-1] in scalar_values,
                    "native scalar argument needs a preceding Scalar capture",
                )
                scalar_source = scalar_values[operands[-1]]
                operands = operands[:-1]
            require(
                all(ref in views for ref in operands), "native operands need physical bounded views"
            )
            refs = tuple(views[ref] for ref in operands)
            params = tuple(value.data for value in op.params)
            pipe, modes = native_contract(op.native.data, refs, params)
            require(
                0 <= op.stage.data <= 32 and 0 <= optional_int(op.logical_phase, 0) <= 255,
                "invalid stage/phase",
            )
            require(
                optional_int(op.lower, 0) == 0
                and optional_int(op.upper, -1) == -1
                and optional_int(op.period, 1) == 1
                and optional_int(op.phase, 0) == 0
                and optional_int(op.phase_width, 1) == 1,
                "conditional native activation needs an explicit symbolic region contract",
            )
            require(
                (op.value is not None) == (op.native.data == "scalar_load"),
                "only Scalar load may produce a numerical native SSA value",
            )
            require(
                (kernel.core.data == "aiv")
                == (
                    pipe in {"V", "MTE3", "S"} or (pipe == "MTE2" and refs[-1].storage.bank == "UB")
                ),
                "native engine belongs to a different actor",
            )
            require(
                op.native.data not in {"fix", "fix16"} or config, "FIX needs ND_PARA initialization"
            )
            task = Task(
                uid,
                op.native.data,
                pipe,
                op.stage.data,
                refs,
                modes,
                round_f32(op.scalar.value.data),
                params,
                issue_order=optional_int(op.issue_order, len(tasks)),
                logical_phase=optional_int(op.logical_phase, 0),
                scalar_source=scalar_source,
            )
            require(0 <= task.issue_order <= 65535, "issue rank outside supported range")
            if scalar_source is not None:
                provider = next(item for item in tasks if item.uid == scalar_source)
                require(
                    provider.stage == task.stage
                    and (provider.issue_order, tasks.index(provider))
                    < (task.issue_order, len(tasks)),
                    "Scalar capture must precede consumption in the same physical step; "
                    "delayed captures require explicit versioned storage",
                )
            tasks.append(task)
            operations[uid] = op
            if op.value is not None:
                scalar_values[op.value] = uid
        else:
            require(isinstance(op, (LoopSyncOp, EndOp)), f"unsupported symbolic effect {op.name}")
    require(tasks, "actor has no native task template")
    return Actor(
        name,
        kernel,
        stream,
        Templates(
            name, kernel.core.data, tuple(declarations.values()), tuple(tasks), config, groups
        ),
        operations,
    ), views


def slope(ref):
    return sum(stride for _, stride, modulus, _ in ref.terms if not modulus)


def extract_loop(module):
    module.verify()
    require(len(list(module.body.block.ops)) == 1, "one loop program per compilation")
    op = module.body.block.first_op
    require(
        isinstance(op, (KernelOp, MixedOp)) and op.target.data == "dav-2201",
        "expected dav-2201 loop program",
    )
    groups = optional_int(op.blocks, 1)
    require(
        1 <= groups <= 1024 and op.disjoint_external.data in {0, 1},
        "loop needs bounded groups and a boolean external alias contract",
    )
    sizes = [child for child in op.body.block.ops if isinstance(child, RuntimeSizeOp)]
    require(len({s.sym_name.data for s in sizes}) == len(sizes), "duplicate runtime scalar name")
    for size in sizes:
        require(
            re.fullmatch(r"[A-Za-z_]\w*", size.sym_name.data) and size.multiple.data > 0,
            "runtime size needs a valid name and positive multiple",
        )
    names = {s.sym_name.data for s in sizes}
    uniform_values = {}
    for child in op.body.block.ops:
        if isinstance(child, RuntimeSizeOp):
            uniform_values[child.value] = from_ssa(child.value)
        elif isinstance(child, arith.ConstantOp) or type(child) in ARITHMETIC:
            require(
                all(v in uniform_values for v in child.operands),
                "uniform scalar expression has undeclared or nonuniform inputs",
            )
            value = from_ssa(child.results[0])
            require(uniform(value, names), "unsupported nonnegative uniform bound expression")
            uniform_values[child.results[0]] = value
    streams = [child for child in op.walk() if isinstance(child, StreamOp)]
    require(streams, "loop program has no stream")
    limit = streams[0].limit
    require(
        limit is None or limit in uniform_values,
        "stream bound must be a declared uniform expression",
    )
    size = limit.owner if limit is not None and isinstance(limit.owner, RuntimeSizeOp) else None
    roots, shared = {}, {}
    for child in op.body.block.ops:
        if isinstance(child, BufferOp) and child.external.data:
            require(child.sym_name.data not in roots, "duplicate GM root")
            roots[child.sym_name.data] = child
            if isinstance(op, MixedOp):
                shared[child.ref] = _storage(child, limit, uniform_values)
    require(
        len(roots) <= 1 or op.disjoint_external.data,
        "external pointer alias is unknown; declare disjoint_external or use views of one root",
    )
    if isinstance(op, MixedOp):
        kernels = [
            (child.sym_name.data, child)
            for child in op.body.block.ops
            if isinstance(child, KernelOp)
        ]
        require(
            [name for name, _ in kernels] == ["aic", "aiv0", "aiv1"],
            "mixed group is one AIC and two AIVs",
        )
        require(
            all(
                isinstance(child, (RuntimeSizeOp, BufferOp, KernelOp, LoopChannelOp, EndOp))
                for child in op.body.block.ops
            ),
            "unknown mixed loop effect",
        )
    else:
        kernels = [("aic" if op.core.data == "aic" else "aiv0", op)]
    actors, all_views = {}, {}
    constants = set()
    period = 1
    for name, kernel in kernels:
        actor, views = _actor(name, kernel, shared, limit, groups, uniform_values)
        actors[name] = actor
        if actor.stream.count is not None:
            constants.add(actor.stream.count.data)
        all_views[name] = views
        for ref in views.values():
            for _, stride, modulus, divisor in ref.terms:
                if stride and modulus:
                    period = lcm(period, modulus * divisor)
    require(
        len(constants) <= 1 and (limit is not None or len(constants) == 1),
        "all actors need the same loop bound",
    )
    constant = next(iter(constants)) if constants else None
    require(constant is None or constant >= 0, "negative static loop bound")
    require(
        period <= 840 and period * sum(len(a.program.tasks) for a in actors.values()) <= 65536,
        "periodic template budget exceeded; no finite-length fallback",
    )
    loop = Loop(
        op,
        actors,
        roots,
        size,
        constant,
        period,
        {},
        tuple(sizes),
        None if limit is None else uniform_values[limit],
    )
    _check_prefixes(loop, all_views)
    _check_accesses(loop, all_views)
    return loop


def _check_prefixes(loop, all_views):
    """Prove when the fixed-cell frontier is exact for a variable final prefix.

    All accesses to each clipped allocation use the same nonempty prefix of aligned
    equal-sized cells. Only the final generation may be partial, with no subsequent
    generation. Thus removing its unaccessed suffix changes no RAW/WAR/WAW edge or
    initialization obligation. Mixing a full reader with a clipped writer is refused.
    """
    by_root = {}
    for actor, views in all_views.items():
        for ref in views.values():
            by_root.setdefault(loop.root(actor, ref), []).append(ref)
    for root, refs in by_root.items():
        clipped = [ref for ref in refs if ref.prefix_extent is not None]
        if not clipped:
            continue
        require(
            len(loop.actors) == 1 and next(iter(loop.actors.values())).program.core == "aiv",
            "valid prefixes currently require one standalone AIV owner",
        )
        first = clipped[0]
        require(
            len(clipped) == len(refs),
            f"{root}: full and clipped accesses cannot share prefix proof",
        )
        require(
            loop.bound
            == expression("ceildiv", first.prefix_extent, literal(first.prefix_elements)),
            f"{root}: valid prefix disagrees with loop bound",
        )
        for ref in refs:
            require(
                (ref.prefix_extent, ref.prefix_elements, ref.nbytes, ref.dtype)
                == (first.prefix_extent, first.prefix_elements, first.nbytes, first.dtype)
                and ref.dtype == ref.storage.dtype
                and ref.rows == 1
                and ref.nbytes == ref.prefix_elements * ITEMSIZE[ref.dtype],
                f"{root}: prefix aliases require equal cell extent, dtype and clipping",
            )
            require(
                all(ref.address(r) % ref.nbytes == 0 for r in range(loop.period)),
                f"{root}: clipped aliases must select whole aligned cells",
            )
            if ref.storage.external:
                require(
                    loop.groups == 1
                    and not ref.core_terms
                    and not ref.count_core_stride
                    and not ref.count_offset
                    and ref.storage.count_scale == ITEMSIZE[ref.dtype]
                    and ref.storage.extent is not None
                    and bounded_by(ref.prefix_extent, ref.storage.extent)
                    and slope(ref) == ref.nbytes
                    and all(ref.address(r) == r * ref.nbytes for r in range(loop.period)),
                    f"{root}: clipped GM view needs a bounded contiguous stream and one owner",
                )
    for data in loop.actors.values():
        for task in data.program.tasks:
            clips = {(ref.prefix_extent, ref.prefix_elements) for ref in task.refs}
            if any(extent is not None for extent, _ in clips):
                require(
                    len(clips) == 1 and all(ref.prefix_extent is not None for ref in task.refs),
                    f"{task.uid}: native inputs and output need the same valid prefix",
                )
                require(
                    task.native
                    in {
                        "copy",
                        "muls",
                        "adds",
                        "add",
                        "sub",
                        "mul",
                        "div",
                        "max",
                        "exp",
                        "rsqrt",
                        "fill",
                        "convert_rn",
                        "convert_f32",
                    }
                    and all(ref.storage.bank in {"GM", "UB"} for ref in task.refs),
                    f"{task.uid}: variable native footprint is not implemented for this recipe",
                )


def _check_accesses(loop, all_views):
    from .model import ranges_cover

    writes = {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref, mode in zip(task.refs, task.modes)
        if "W" in mode
    }
    refs_by_root = {}
    for actor, views in all_views.items():
        for ref in views.values():
            refs_by_root.setdefault(loop.root(actor, ref), []).append(ref)
    for root, refs in refs_by_root.items():
        storage = refs[0].storage
        drift = {slope(ref) for ref in refs}
        if root in writes and storage.external and any(drift):
            require(
                len(drift) == 1 and next(iter(drift)) > 0,
                f"{root}: stream/state aliases need an explicit ownership contract",
            )
            frame = next(iter(drift))
            loop.frames[root] = frame
            if refs[0].prefix_extent is not None:
                pass  # Exact final-prefix ownership was checked above.
            elif storage.count_scale:
                require(
                    storage.extent == loop.bound,
                    f"{root}: unclipped stream requires its actual loop extent",
                )
                require(
                    storage.count_scale == loop.groups * frame
                    and all(
                        ref.count_core_stride in ({0, frame} if loop.groups == 1 else {frame})
                        and ref.count_offset == 0
                        and not ref.core_terms
                        for ref in refs
                    ),
                    f"{root}: group-private stream frame translations disagree",
                )
            else:
                require(
                    loop.constant_count is not None,
                    f"{root}: unbounded stream exceeds its fixed allocation",
                )
                translations = {
                    tuple(ref.address(0, core) - ref.address(0) for core in range(loop.groups))
                    for ref in refs
                }
                require(
                    len(translations) == 1, f"{root}: written stream aliases change between groups"
                )
                shifts = next(iter(translations))
                require(
                    all(b - a >= loop.constant_count * frame for a, b in pairwise(shifts)),
                    f"{root}: cross-group stream ownership overlap",
                )
        elif root in writes and storage.external:
            require(
                all(not ref.count_offset and not ref.count_core_stride for ref in refs),
                f"{root}: ring ownership cannot depend on stream length",
            )
            shifts = {
                tuple(ref.address(0, core) - ref.address(0) for core in range(loop.groups))
                for ref in refs
            }
            require(len(shifts) == 1, f"{root}: ring translations disagree")
            used = [
                (lo, hi) for ref in refs for r in range(loop.period) for lo, hi in ref.segments(r)
            ]
            from .model import verify_translated_owners

            verify_translated_owners(
                refs,
                [(root, lo, hi, "RW") for lo, hi in used],
                loop.groups,
                translation_error="ring group translations disagree",
                overlap_error="cross-group ring ownership overlap",
            )
        for ref in refs:
            require(
                storage.external or slope(ref) == 0,
                f"{root}: local lifetime is unbounded; declare a ring",
            )
            itemsize = ITEMSIZE[ref.dtype]
            require(ref.nbytes % itemsize == 0, "view extent does not match its dtype")
            for core in range(loop.groups if storage.external else 1):
                coefficient = ref.count_offset + ref.count_core_stride * core
                for r in range(loop.period):
                    parts = ref.segments(r, core)
                    if ref.prefix_extent is not None and storage.external:
                        pass  # The upper address is min((g+1)*tile, raw_extent), not padded.
                    elif storage.count_scale:
                        require(
                            storage.extent == loop.bound,
                            f"{root}: unclipped access needs matching loop/storage extent",
                        )
                        require(
                            slope(ref) + coefficient <= storage.count_scale
                            and all(
                                hi + (r + 1) * (coefficient - storage.count_scale) <= 0
                                for _, hi in parts
                            ),
                            f"{root}: symbolic access exceeds its GM extent",
                        )
                    elif loop.constant_count is None:
                        require(
                            slope(ref) == 0 and all(hi <= storage.nbytes for _, hi in parts),
                            f"{root}: periodic access exceeds its allocation",
                        )
                    elif r < loop.constant_count:
                        last = r + (loop.constant_count - 1 - r) // loop.period * loop.period
                        require(
                            all(hi <= storage.nbytes for _, hi in ref.segments(last, core)),
                            f"{root}: affine static bound exceeds allocation",
                        )
                    require(
                        coefficient % itemsize == 0 and all(lo % itemsize == 0 for lo, _ in parts),
                        f"{root}: dtype alignment violated",
                    )
                    if root in loop.frames:
                        translation = ref.address(0, core) - ref.address(0)
                        require(
                            all(
                                0
                                <= lo - r * loop.frames[root] - translation
                                < hi - r * loop.frames[root] - translation
                                <= loop.frames[root]
                                for lo, hi in parts
                            ),
                            f"{root}: access crosses its task frame",
                        )
    for actor, data in loop.actors.items():
        views = all_views[actor]
        for value, ref in views.items():
            parent = views.get(value.owner.source)
            if parent:
                require(
                    slope(parent) == slope(ref),
                    "nested view changes affine drift; its parent bounds cannot be reused",
                )
                require(
                    all(
                        ranges_cover(parent.segments(r, core), ref.segments(r, core))
                        for r in range(loop.period)
                        for core in range(loop.groups if ref.storage.external else 1)
                    ),
                    "nested view exceeds actual parent rows/holes",
                )
        for task in data.program.tasks:
            for ref in task.refs:
                alignment = 4 if task.pipe == "S" else 32
                require(
                    (ref.count_offset % alignment == ref.count_core_stride % alignment == 0)
                    and all(
                        lo % alignment == 0
                        for r in range(loop.period)
                        for core in range(loop.groups if ref.storage.external else 1)
                        for lo, _ in ref.segments(r, core)
                    ),
                    f"{task.uid}: native origin alignment violated",
                )
                if task.pipe in {"M", "MTE1", "FIX"} and not ref.storage.external:
                    require(
                        all(ref.address(r) % 512 == 0 for r in range(loop.period)),
                        f"{task.uid}: Cube local origin needs 512-byte alignment",
                    )
            if task.pipe == "V":
                destination = task.refs[-1]
                for ref in task.refs[:-1]:
                    if ref.storage != destination.storage:
                        continue
                    require(
                        all(
                            ref.segments(r) == destination.segments(r)
                            or not ranges_overlap(ref.segments(r), destination.segments(r))
                            for r in range(loop.period)
                        ),
                        f"{task.uid}: shifted in-place Vector alias",
                    )


def loop_dependencies(loop):
    """Exact periodic frontier and startup obligations, independent of N."""
    cells = {}
    written = {
        loop.root(actor, ref)
        for actor, data in loop.actors.items()
        for task in data.program.tasks
        for ref, mode in zip(task.refs, task.modes)
        if "W" in mode
    }
    work = sorted(
        (task.logical_phase, owner, index, actor, task)
        for owner, (actor, data) in enumerate(loop.actors.items())
        for index, task in enumerate(data.program.tasks)
    )
    for r in range(loop.period):
        for _, _, index, actor, task in work:
            accesses = {}
            for ref, mode in zip(task.refs, task.modes):
                root = loop.root(actor, ref)
                if ref.storage.initialized and root not in written:
                    continue
                frame = loop.frames.get(root, 0)
                for lo, hi in ref.segments(r):
                    accesses.setdefault(root, []).append(
                        (lo - r * frame, hi - r * frame, mode, ref.storage.initialized)
                    )
            for root, parts in accesses.items():
                cells.setdefault(root, []).append((r, actor, task.uid, parts))
    edges = {}

    def edge(source, destination, hazard, root, lo, hi):
        if source is None or source == destination:
            return
        sg, sa, su = source
        dg, da, du = destination
        distance = dg - sg
        require(distance >= 0, "logical effect order is inconsistent")
        key = (sa, su, da, du, distance, hazard, root, lo, hi)
        edges.setdefault(key, set()).add(dg)

    for root, work in cells.items():
        points = sorted({p for _, _, _, parts in work for lo, hi, _, _ in parts for p in (lo, hi)})
        cuts = {point: i for i, point in enumerate(points)}
        per_cell = [[] for _ in points[:-1]]
        for g, actor, uid, parts in work:
            touched = {}
            for start, stop, mode, init in parts:
                for cell in range(cuts[start], cuts[stop]):
                    old_read, old_write, old_init = touched.get(cell, (False, False, False))
                    touched[cell] = (
                        old_read or "R" in mode,
                        old_write or "W" in mode,
                        old_init or init,
                    )
            for cell, (read, write, init) in touched.items():
                per_cell[cell].append(((g, actor, uid), read, write, init))
        for lo, hi, cell_events in zip(points, points[1:], per_cell):
            if not cell_events:
                continue
            initialized = cell_events[-1][3]
            events = [(event, read, write) for event, read, write, _ in cell_events]
            writes = [i for i, (_, _, write) in enumerate(events) if write]
            last = writes[-1] if writes else None
            if root in loop.frames:
                last = None
            writer = (
                None if last is None else (events[last][0][0] - loop.period, *events[last][0][1:])
            )
            pending = (
                []
                if last is None
                else [
                    (event[0][0] - loop.period, *event[0][1:])
                    for event in events[last + 1 :]
                    if event[1]
                ]
            )
            previous_generation = None
            for destination, read, write in events:
                g = destination[0]
                if root in loop.frames and g != previous_generation:
                    writer, pending = None, []
                previous_generation = g
                active_first = loop.constant_count is None or g < loop.constant_count
                if read:
                    require(
                        initialized or not active_first or (writer is not None and writer[0] >= 0),
                        f"{destination[1]}/{destination[2]} has an undefined first read on {root}[{lo}:{hi}]",
                    )
                    edge(writer, destination, "RAW", root, lo, hi)
                    pending.append(destination)
                if write:
                    edge(writer, destination, "WAW", root, lo, hi)
                    for reader in pending:
                        edge(reader, destination, "WAR", root, lo, hi)
                    writer, pending = destination, []
    return tuple(
        Dependence(
            (sa, su), (da, du), distance, loop.period, tuple(sorted(residues)), hazard, root, lo, hi
        )
        for (sa, su, da, du, distance, hazard, root, lo, hi), residues in sorted(edges.items())
    )


def compose_loop(name, cube, vector0, vector1, *, disjoint_external):
    from xdsl.dialects.builtin import IntAttr, ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    block, roots, size, groups = Block(), {}, None, None
    kernels = []
    for actor, original in zip(("aic", "aiv0", "aiv1"), (cube, vector0, vector1)):
        module = original.clone()
        kernel = module.body.block.first_op
        require(isinstance(kernel, KernelOp), "mixed loop requires physical kernel regions")
        if groups is None:
            groups = optional_int(kernel.blocks, 1)
        require(optional_int(kernel.blocks, 1) == groups, "mixed grids disagree")
        for child in list(kernel.body.block.ops):
            if isinstance(child, RuntimeSizeOp):
                if size is None:
                    size = child.clone()
                    block.add_op(size)
                require(
                    child.sym_name == size.sym_name and child.multiple == size.multiple,
                    "actors need the same uniform count contract",
                )
                child.value.replace_all_uses_with(size.value)
                kernel.body.block.erase_op(child)
        for child in list(kernel.body.block.ops):
            if isinstance(child, BufferOp) and child.external.data:
                root = roots.get(child.sym_name.data)
                if root is None:
                    root = child.clone()
                    roots[child.sym_name.data] = root
                    block.add_op(root)
                require(
                    root.ref.type == child.ref.type
                    and root.nbytes == child.nbytes
                    and root.count_scale == child.count_scale
                    and root.extent_count is child.extent_count
                    and root.initialized == child.initialized,
                    "shared root extent/contract differs between actors",
                )
                child.ref.replace_all_uses_with(root.ref)
                kernel.body.block.erase_op(child)
        kernel.attributes["sym_name"] = StringAttr(actor)
        module.body.block.detach_op(kernel)
        kernels.append(kernel)
    block.add_ops(kernels)
    block.add_op(EndOp())
    result = ModuleOp(
        [
            MixedOp.create(
                attributes={
                    "sym_name": StringAttr(name),
                    "target": StringAttr("dav-2201"),
                    "blocks": IntAttr(groups),
                    "disjoint_external": IntAttr(int(disjoint_external)),
                },
                regions=[Region(block)],
            )
        ]
    )
    extract_loop(result)
    return result
