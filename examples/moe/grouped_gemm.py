"""b16 grouped GEMM consuming cumulative expert ends directly (type 0)."""

from __future__ import annotations

from ascendflow import language as T

# ruff: noqa: F821


@T.inline
def compute_panel(inputs, buffers, coordinates, geometry, boundaries, UNIT_FLAG):
    X, W, Y = inputs
    a1, b1, a0, b0, c0 = buffers
    expert, panel, row, column, k, valid = coordinates
    BM, BK, CK, nvalid, kvalid = geometry
    first, last, cache_bypass = boundaries
    with T.Stage(0, name="load"), T.CopyGroup(name="panels"):
        av = T.copy(
            X[row : row + BM, k : k + kvalid],
            a1.at(panel),
            valid_rows=None if UNIT_FLAG else valid,
            pad_value=0,
        )
        bv = T.copy(
            W[expert, k : k + kvalid, column : column + nvalid],
            b1.at(panel),
            pad_value=0,
            cache_bypass=cache_bypass,
        )
    with T.Stage(1, name="compute"):
        for chunk in T.unroll(BK // CK):
            flag = 0
            if UNIT_FLAG:
                flag = (3 if last else 2) if chunk == BK // CK - 1 else 2
            slot = panel * (BK // CK) + chunk
            aa = T.copy(av[:, chunk * CK : (chunk + 1) * CK], a0.at(slot))
            bb = T.copy(bv[chunk * CK : (chunk + 1) * CK, :], b0.at(slot))
            T.matmul(aa, bb, accum=c0, initialize=first if chunk == 0 else False, unit_flag=flag)
            if chunk == BK // CK - 1:  # noqa: SIM102
                if last:
                    T.copy(
                        c0[:, : T.ceildiv(nvalid, 16) * 16],
                        Y[row : row + BM, column : column + nvalid],
                        valid_rows=None if UNIT_FLAG else T.min(BM, T.max(0, valid)),
                        unit_flag=3 if UNIT_FLAG else 0,
                    )


@T.inline
def tile_coordinates(index, mtiles, ntiles, SWIZZLE):
    mi, ni = index // ntiles, index % ntiles
    if SWIZZLE > 1:  # noqa: SIM102
        if mtiles > 5:
            # Each M/N window is a cyclic column permutation for every row.
            mbase = (index // (SWIZZLE * ntiles)) * SWIZZLE
            height = T.min(SWIZZLE, mtiles - mbase)
            relative = index % (SWIZZLE * ntiles)
            nbase = (relative // (height * SWIZZLE)) * SWIZZLE
            width = T.min(SWIZZLE, ntiles - nbase)
            local = relative % (height * SWIZZLE)
            mi = mbase + local % height
            ni = nbase + (local // height + local % height) % width
    return mi, ni


@T.inline
def compute_expert_impl(inputs, buffers, coordinates, geometry, UNIT_FLAG, WEIGHT_CACHE):
    expert, group, workers, begin, end = coordinates
    K, N, BM, BN, BK, CK, SWIZZLE = geometry
    mtiles = T.ceildiv(T.max(0, end - begin), BM)
    cache_bypass = False
    if WEIGHT_CACHE == "auto":
        cache_bypass = mtiles == 1
    if K % BK == 0 and N % BN == 0:
        # Continue the two-slot K pipeline across output tiles.
        ntiles, panels = N // BN, K // BK
        tiles = T.ceildiv(T.max(0, mtiles * ntiles - group), workers)
        for panel in T.Pipelined(tiles * panels, issue_order=("load", "compute")):
            index = group + (panel // panels) * workers
            mi, ni = tile_coordinates(index, mtiles, ntiles, SWIZZLE)
            row, column = begin + mi * BM, ni * BN
            compute_panel(
                inputs,
                buffers,
                (expert, panel, row, column, (panel % panels) * BK, T.min(BM, end - row)),
                (BM, BK, CK, BN, BK),
                (panel % panels == 0, panel % panels == panels - 1, cache_bypass),
                UNIT_FLAG,
            )
    else:
        # Unaligned output rows keep all columns with the same owner.
        for nbase, ntiles, nvalid in T.tile_ranges(N, BN):
            count = mtiles * ntiles if N % BN == 0 else mtiles
            for tile in T.serial(T.ceildiv(T.max(0, count - group), workers)):
                index = group + tile * workers
                mi, ni = tile_coordinates(index, mtiles, ntiles, SWIZZLE)
                row = begin + (mi if N % BN == 0 else index) * BM
                first_column = nbase + ni * BN if N % BN == 0 else nbase
                for col in T.serial(1 if N % BN == 0 else ntiles):
                    column = first_column + col * BN
                    for kbase, panels, kvalid in T.tile_ranges(K, BK):
                        for panel in T.Pipelined(panels, issue_order=("load", "compute")):
                            compute_panel(
                                inputs,
                                buffers,
                                (
                                    expert,
                                    panel,
                                    row,
                                    column,
                                    kbase + panel * BK,
                                    T.min(BM, end - row),
                                ),
                                (BM, BK, CK, nvalid, kvalid),
                                (
                                    (panel == 0) if kbase == 0 else False,
                                    (panel == panels - 1) if kbase + panels * BK >= K else False,
                                    cache_bypass,
                                ),
                                UNIT_FLAG,
                            )


@T.inline
def compute_expert(inputs, buffers, coordinates, geometry, UNIT_FLAG, WEIGHT_CACHE):
    _expert, _group, _workers, begin, end = coordinates
    _K, N, BM, BN, _BK, _CK, _SWIZZLE = geometry
    if UNIT_FLAG and N % BN == 0:
        # Partial experts keep the explicit row-prefix access/event protocol.
        if (end - begin) % BM == 0:
            compute_expert_impl(inputs, buffers, coordinates, geometry, True, WEIGHT_CACHE)
        else:
            compute_expert_impl(inputs, buffers, coordinates, geometry, False, WEIGHT_CACHE)
    else:
        compute_expert_impl(inputs, buffers, coordinates, geometry, False, WEIGHT_CACHE)


@T.inline
def clear_tail(inputs, buffers, coordinates, geometry):
    GroupList, Y = inputs
    a1, b1, a0, b0, c0 = buffers
    group, workers, capacity = coordinates
    E, N, BM, BN, CK = geometry
    begin = T.load_index(GroupList[E - 1 : E], minimum=0)
    if begin < capacity:
        T.fill(a1.at(0), 0.0)
        T.fill(b1.at(0), 0.0)
        za = T.copy(a1.at(0)[:BM, :CK], a0.at(0))
        zb = T.copy(b1.at(0)[:CK, :BN], b0.at(0))
        mtiles = T.ceildiv(capacity - begin, BM)
        for nbase, ntiles, nvalid in T.tile_ranges(N, BN):
            count = mtiles * ntiles if N % BN == 0 else mtiles
            for tile in T.serial(T.ceildiv(T.max(0, count - group), workers)):
                index = group + tile * workers
                row = begin + (index // ntiles if N % BN == 0 else index) * BM
                first_column = nbase + (index % ntiles) * BN if N % BN == 0 else nbase
                for col in T.serial(1 if N % BN == 0 else ntiles):
                    column = first_column + col * BN
                    T.matmul(za, zb, accum=c0, initialize=True)
                    T.copy(
                        c0[:, : T.ceildiv(nvalid, 16) * 16],
                        Y[row : row + BM, column : column + nvalid],
                        valid_rows=T.min(BM, capacity - row),
                    )


@T.prim_func(disjoint_external=True)
def grouped_gemm(
    X: T.Tensor((capacity, K), DTYPE),
    W: T.Tensor(
        (E, K, N),
        DTYPE,
        layout=WEIGHT_LAYOUT,
        cache="normal" if WEIGHT_CACHE == "auto" else WEIGHT_CACHE,
    ),
    GroupList: T.Tensor((E,), T.i64),
    Y: T.Tensor((capacity, N), DTYPE, access="write", write_disjoint=True),
    capacity: T.i64,
    E: T.constexpr = 4,
    K: T.constexpr = 256,
    N: T.constexpr = 256,
    DTYPE: T.constexpr = "bf16",
    WEIGHT_LAYOUT: T.constexpr = "linear",
    WEIGHT_CACHE: T.constexpr = "auto",
    BM: T.constexpr = 128,
    BN: T.constexpr = 256,
    BK: T.constexpr = 256,
    CK: T.constexpr = 64,
    GROUPS: T.constexpr = 20,
    SWIZZLE: T.constexpr = 8,
    UNIT_FLAG: T.constexpr = True,
):
    with T.Kernel(GROUPS, kind="cube") as block:
        assert E > 0 and K > 0 and N > 0, "positive expert and matrix dimensions required"
        assert DTYPE == "f16" or DTYPE == "bf16", "grouped GEMM requires b16 operands"
        assert BM % 16 == 0 and BN % 16 == 0 and BK % 16 == 0 and CK % 16 == 0, (
            "complete Cube fractals required"
        )
        assert BK % CK == 0, "L1 panel must contain complete Cube chunks"
        assert SWIZZLE > 0, "positive tile window required"
        a1 = T.alloc_l1((BM, BK), DTYPE, slots=2)
        b1 = T.alloc_l1((BK, BN), DTYPE, slots=2)
        a0 = T.alloc_l0a((BM, CK), DTYPE, slots=2)
        b0 = T.alloc_l0b((CK, BN), DTYPE, slots=2)
        c0 = T.alloc_l0c((BM, BN), T.f32)
        buffers = (a1, b1, a0, b0, c0)
        tile_prefix, begin = 0, 0
        for expert in T.serial(E):
            end = T.load_index(GroupList[expert : expert + 1], minimum=0)
            worker = (block + GROUPS - tile_prefix % GROUPS) % GROUPS
            compute_expert(
                (X, W, Y),
                buffers,
                (expert, worker, GROUPS, begin, end),
                (K, N, BM, BN, BK, CK, SWIZZLE),
                UNIT_FLAG,
                WEIGHT_CACHE,
            )
            tile_prefix = tile_prefix + T.ceildiv(T.max(0, end - begin), BM) * (
                N // BN if N % BN == 0 else 1
            )
            begin = end
        clear_tail((GroupList, Y), buffers, (block, GROUPS, capacity), (E, N, BM, BN, CK))
