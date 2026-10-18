"""Block triangular inverse: Vector diagonal solve, then IEEE FP32 Cube merges."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.prim_func(disjoint_external=True)
def prefill_solve(
    A: N.Tensor((CHUNKS, H, BT, BT), N.f32),
    Negative: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Diagonal: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    Ai32: N.Tensor((CHUNKS, H, BT, BT), N.f32, access="write", write_disjoint=True),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    BT: N.constexpr = 64,
    GROUPS: N.constexpr = 40,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="vector") as owner:
        assert BT == 64, "four 16-row diagonal blocks"
        negative = N.alloc_ub((BT, BT), N.f32)
        inverse = N.alloc_ub((16, 16), N.f32)
        a = N.alloc_ub((16,), N.f32)
        work = N.alloc_ub((16,), N.f32)
        zeros = N.alloc_ub((BT, BT), N.f32)
        N.fill(zeros, 0.0)
        for task in N.serial(N.ceildiv(N.max(0, CHUNKS * H - owner), GROUPS)):
            flat = owner + task * GROUPS
            chunk, head = flat // H, flat % H
            N.copy(A[chunk, head, :, :], negative)
            N.mul(negative, -1.0, out=negative)
            N.copy(negative, Negative[chunk, head, :, :])
            N.copy(zeros, Ai32[chunk, head, :, :])
            N.copy(zeros, Diagonal[chunk, head, :, :])
            for block in N.serial(BT // 16):
                begin = block * 16
                N.fill(inverse, 0.0)
                for row in N.serial(16):
                    N.copy(negative[begin + row, begin : begin + 16], a)
                    N.copy(a, inverse[row, :])
                    for previous in N.serial(row):
                        coefficient = N.load_scalar(a[previous : previous + 1])
                        N.mul(inverse[previous, :], coefficient, out=work)
                        N.add(inverse[row, :], work, out=inverse[row, :])
                N.fill(inverse, 1.0, mask=N.arange(16)[:, None] == N.arange(16)[None, :])
                N.copy(inverse, Ai32[chunk, head, begin : begin + 16, begin : begin + 16])
                N.copy(inverse, Diagonal[chunk, head, begin : begin + 16, begin : begin + 16])


@N.prim_func(disjoint_external=True)
def prefill_solve_merge(
    Negative: N.Tensor((CHUNKS, H, BT, BT), N.f32),
    Blocks: N.Tensor((CHUNKS, H, BT, BT), N.f32),
    Merge: N.Tensor((GROUPS, 2, WIDTH, WIDTH), N.f32, access="write"),
    Off: N.Tensor(
        (CHUNKS, H, BT, BT) if WIDTH == 16 else (GROUPS, 2, WIDTH, WIDTH),
        N.f32,
        access="write",
        write_disjoint=WIDTH == 16,
    ),
    Inverse: N.Tensor((CHUNKS, H, BT, BT), DTYPE, access="write", write_disjoint=True),
    CHUNKS: N.constexpr = 2,
    H: N.constexpr = 32,
    BT: N.constexpr = 64,
    WIDTH: N.constexpr = 16,
    GROUPS: N.constexpr = 20,
    DTYPE: N.constexpr = N.f16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert BT == 64 and (WIDTH == 16 or WIDTH == 32), "hierarchical block merge"
        raw = N.alloc_ub((32, BT), N.f32)
        packed = N.alloc_ub((32, BT), DTYPE)
        a1 = N.alloc_l1((2, WIDTH, WIDTH), N.f32)
        b1 = N.alloc_l1((2, WIDTH, WIDTH), N.f32)
        a0 = N.alloc_l0a((WIDTH, WIDTH), N.f32)
        b0 = N.alloc_l0b((WIDTH, WIDTH), N.f32)
        c0 = N.alloc_l0c((WIDTH, WIDTH), N.f32)
        for item in N.Pipelined(
            N.ceildiv(N.max(0, CHUNKS * H * (BT // (2 * WIDTH)) - owner), GROUPS), stages=3
        ):
            work = owner + item * GROUPS
            chunk, head = work // (H * (BT // (2 * WIDTH))), (work // (BT // (2 * WIDTH))) % H
            left = (work % (BT // (2 * WIDTH))) * 2 * WIDTH
            right = left + WIDTH
            ring = item % 2
            with N.Stage(0, name="import"), N.Cube():
                N.copy(
                    Blocks[chunk, head, right : right + WIDTH, right : right + WIDTH],
                    a1[ring, :, :],
                )
                N.copy(
                    Negative[chunk, head, right : right + WIDTH, left : left + WIDTH],
                    b1[ring, :, :],
                )
            with N.Stage(1, name="merge"), N.Cube():
                aa = N.copy(a1[ring, :, :], a0)
                bb = N.copy(b1[ring, :, :], b0)
                N.matmul(aa, bb, accum=c0, initialize=True)
                N.copy(c0, Merge[owner, ring, :, :])
                aa = N.copy(Merge[owner, ring, :, :], a1[ring, :, :])
                bb = N.copy(
                    Blocks[chunk, head, left : left + WIDTH, left : left + WIDTH], b1[ring, :, :]
                )
                aa = N.copy(aa, a0)
                bb = N.copy(bb, b0)
                N.matmul(aa, bb, accum=c0, initialize=True)
                if WIDTH == 16:
                    N.copy(c0, Off[chunk, head, right : right + WIDTH, left : left + WIDTH])
                else:
                    N.copy(c0, Off[owner, ring, :, :])
            with N.Stage(2, name="pack"):
                if WIDTH == 32:
                    with N.Vector() as lane:
                        row = lane * 16
                        N.copy(Blocks[chunk, head, row : row + 16, :], raw[:16, :])
                        N.copy(Blocks[chunk, head, row + 32 : row + 48, 32:64], raw[16:32, 32:64])
                        N.copy(Off[owner, ring, row : row + 16, :], raw[16:32, :32])
                        N.cast(raw, out=packed)
                        N.copy(packed[:16, :], Inverse[chunk, head, row : row + 16, :])
                        N.copy(packed[16:32, :], Inverse[chunk, head, row + 32 : row + 48, :])
