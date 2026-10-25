"""Merge the Vector diagonal inverses with pipelined IEEE FP32 Cube products."""

from __future__ import annotations

from ascendflow import language as N

# ruff: noqa: F821


@N.inline
def solve_import(inputs, buffers, geometry, coordinates):
    Blocks, Negative = inputs
    a1, b1 = buffers
    (WIDTH,) = geometry
    right, left = coordinates
    with N.CopyGroup(name="inverse_operands"):
        diagonal = N.copy(Blocks[right : right + WIDTH, right : right + WIDTH], a1)
        negative = N.copy(Negative[right : right + WIDTH, left : left + WIDTH], b1)
    return diagonal, negative


@N.inline
def merge_inverse_blocks(inputs, operands, buffers):
    Merge, Blocks, Off = inputs
    a0, a1, b0, b1, c0 = buffers
    diagonal, negative = operands
    aa = N.copy(diagonal, a0)
    bb = N.copy(negative, b0)
    result = N.matmul(aa, bb, accum=c0, initialize=True)
    N.copy(result, Merge)
    aa = N.copy(Merge, a1)
    bb = N.copy(Blocks, b1)
    aa = N.copy(aa, a0)
    bb = N.copy(bb, b0)
    result = N.matmul(aa, bb, accum=c0, initialize=True)
    N.copy(result, Off)


@N.inline
def read_inverse_blocks(inputs, raw, coordinates):
    Blocks, Off = inputs
    (lane,) = coordinates
    row = lane * 16
    with N.CopyGroup(name="inverse_blocks"):
        N.copy(Blocks[row : row + 16, :], raw[:16, :])
        N.copy(Blocks[row + 32 : row + 48, 32:64], raw[16:32, 32:64])
        N.copy(Off[row : row + 16, :], raw[16:32, :32])


@N.inline
def publish_inverse_blocks(Inverse, buffers, coordinates):
    raw, packed = buffers
    (lane,) = coordinates
    row = lane * 16
    rounded = N.cast(raw, out=packed)
    N.copy(rounded[:16, :], Inverse[row : row + 16, :])
    N.copy(rounded[16:32, :], Inverse[row + 32 : row + 48, :])


@N.inline
def pack_inverse_blocks(inputs, buffers, coordinates):
    Blocks, Off, Inverse = inputs
    read_inverse_blocks((Blocks, Off), buffers[0], coordinates)
    publish_inverse_blocks(Inverse, buffers, coordinates)


@N.inline
def solve_chunk_inverse(Diagonal, Negative, Ai32, Merge16, Merge32, Off32, a1, b1, cube16, cube32):
    a016, b016, c016 = cube16
    a032, b032, c032 = cube32
    # Rows 0:16 of b1 serve the two small merges. Rows 32:64 preserve
    # the large negative block, acquired first to capture both AIV prefixes.
    # All B loads use the same physical leading M=64, including this subview.
    negative32 = N.copy(Negative[32:64, :32], b1[32:64, :32])
    for pair in N.unroll(2):
        left, right = pair * 32, pair * 32 + 16
        diagonal16, negative16 = solve_import(
            (Diagonal, Negative), (a1[:16, :16], b1[:16, :16]), (16,), (right, left)
        )
        merge_inverse_blocks(
            (
                Merge16,
                Diagonal[left : left + 16, left : left + 16],
                Ai32[right : right + 16, left : left + 16],
            ),
            (diagonal16, negative16),
            (a016, a1[:16, :16], b016, b1[:16, :16], c016),
        )
    diagonal32 = N.copy(Ai32[32:64, 32:64], a1[:32, :32])
    merge_inverse_blocks(
        (Merge32, Ai32[:32, :32], Off32),
        (diagonal32, negative32),
        (a032, a1[:32, :32], b032, b1[:32, :32], c032),
    )


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
    DTYPE: N.constexpr = N.bf16,
):
    with N.Kernel(GROUPS, kind="mixed") as owner:
        assert BT == 64 and (WIDTH == 16 or WIDTH == 32), "hierarchical block merge"
        raw = N.alloc_ub((32, BT), N.f32)
        packed = N.alloc_ub((32, BT), DTYPE)
        a1 = N.alloc_l1((WIDTH, WIDTH), N.f32, slots=2)
        b1 = N.alloc_l1((WIDTH, WIDTH), N.f32, slots=2)
        a0 = N.alloc_l0a((WIDTH, WIDTH), N.f32)
        b0 = N.alloc_l0b((WIDTH, WIDTH), N.f32)
        c0 = N.alloc_l0c((WIDTH, WIDTH), N.f32)
        merges = N.slotted_view(Merge[owner, :, :, :])
        if WIDTH == 32:
            off_values = N.slotted_view(Off[owner, :, :, :])
        for item in N.Dataflow(
            N.ceildiv(N.max(0, CHUNKS * H * (BT // (2 * WIDTH)) - owner), GROUPS),
            issue_order={"cube": ("import", "merge"), "vector": ("pack",)}
            if WIDTH == 32
            else {"cube": ("import", "merge")},
        ):
            work = owner + item * GROUPS
            chunk, head = work // (H * (BT // (2 * WIDTH))), (work // (BT // (2 * WIDTH))) % H
            left = (work % (BT // (2 * WIDTH))) * 2 * WIDTH
            right = left + WIDTH
            with N.Cube(), N.Stage(0, name="import"):
                diagonal, negative = solve_import(
                    (Blocks[chunk, head, :, :], Negative[chunk, head, :, :]),
                    (a1.at(item), b1.at(item)),
                    (WIDTH,),
                    (right, left),
                )
            with N.Cube(), N.Stage(1, name="merge"):
                if WIDTH == 16:
                    target = Off[chunk, head, right : right + WIDTH, left : left + WIDTH]
                else:
                    target = off_values.at(item)
                merge_inverse_blocks(
                    (
                        merges.at(item),
                        Blocks[chunk, head, left : left + WIDTH, left : left + WIDTH],
                        target,
                    ),
                    (diagonal, negative),
                    (a0, a1.at(item), b0, b1.at(item), c0),
                )
            if WIDTH == 32:
                with N.Vector() as lane, N.Stage(2, name="pack"):
                    pack_inverse_blocks(
                        (
                            Blocks[chunk, head, :, :],
                            off_values.at(item),
                            Inverse[chunk, head, :, :],
                        ),
                        (raw, packed),
                        (lane,),
                    )
