"""CSA cache movement: direct window pages and streamed sparse compressed rows."""

from ascendflow import language as N

NEG_INF = float("-inf")


@N.inline
def query_metadata(cu_seqlens_q, seqused_ori_kv, ori_topk_length, token, B, T, MAX_SEQ, WINDOW):
    # upper_bound(Cu, token)-1, including empty requests. The constexpr span
    # halves on each expansion; only ceil(log2(B)) GM loads survive.
    batch = 0
    span = B
    for step in N.unroll(32):
        if span > 1:
            half = span // 2
            candidate = N.min(B - 1, batch + half)
            start = N.load_index(cu_seqlens_q[candidate : candidate + 1])
            if start <= token:
                batch = candidate
            span = span - half
    # Monotone Cu and actual access legality are input contracts; the
    # upper_bound result needs no caller-written range proof.
    begin = N.load_index(cu_seqlens_q[batch : batch + 1])
    end = N.load_index(cu_seqlens_q[batch + 1 : batch + 2])
    length = N.load_index(seqused_ori_kv[batch : batch + 1])
    offset = token - begin
    count = end - begin
    right = N.max(0, length - count + offset + 1)
    visible = N.min(WINDOW, right)
    requested = N.load_index(ori_topk_length[token, :])
    # Official band mode treats nonpositive ori_topk_length as "no narrowing".
    if requested > 0:
        visible = N.min(visible, requested)
    return batch, count, offset, right, visible


@N.inline
def copy_window_pages(
    Ori, OriTable, destination, batch, length, visible, feature, PAGE, C, MAX_SEQ, ORI_BLOCKS
):
    position = N.max(0, length - visible)
    first_rows = N.min(visible, PAGE - position % PAGE)
    table_column = N.min(position // PAGE, MAX_SEQ // PAGE - 1)
    page = N.load_index(OriTable[batch, table_column : table_column + 1])
    # Overlay only real page fragments onto an initialized resident NZ plane.
    # Rows outside visible are ignored by the per-query softmax mask.
    N.copy(
        Ori[page, position % PAGE : position % PAGE + 128, 0, feature : feature + C],
        destination,
        valid_rows=first_rows,
    )
    second_rows = visible - first_rows
    table_column2 = N.min((position + first_rows) // PAGE, MAX_SEQ // PAGE - 1)
    page2 = N.load_index(OriTable[batch, table_column2 : table_column2 + 1])
    N.copy(
        Ori[page2, :128, 0, feature : feature + C],
        destination[first_rows : first_rows + 128, :],
        valid_rows=second_rows,
    )


@N.inline
def gather_compressed(
    Cmp,
    CmpTable,
    Indices,
    KV,
    Bias,
    zeros,
    rows,
    masks,
    batch,
    token,
    available,
    cutoff,
    group,
    slot,
    tile,
    sid,
    BK,
    ROWS,
    PAGE,
    RATIO,
    MAX_SEQ,
    CMP_BLOCKS,
    TOPK,
):
    count = N.min(TOPK, N.min(available, cutoff))
    zero = N.load_scalar(zeros[0, 0:1])
    for block in N.Pipelined(BK // 2 // ROWS):
        bank = block % 2
        start = sid * (BK // 2) + block * ROWS
        with N.Stage(0, name="sparse_load"):
            N.fill(rows[bank, :, :], 0.0)
            N.fill(masks[bank, :], NEG_INF)
            for row in N.serial(ROWS):
                index_slot = tile * BK + start + row
                if index_slot < count:
                    index = N.load_index(Indices[token, 0, index_slot : index_slot + 1])
                    if index >= 0:  # noqa: SIM102 - each guard retains its address premises.
                        if index < cutoff:
                            page = N.load_index(
                                CmpTable[
                                    batch, index // (PAGE // RATIO) : index // (PAGE // RATIO) + 1
                                ]
                            )
                            N.copy(Cmp[page, index % (PAGE // RATIO), 0, :], rows[bank, row, :])
                            N.store_scalar(zero, out=masks[bank, row : row + 1])
        with N.Stage(1, name="sparse_publish"):
            # Masks and rows are in the same MTE3 publication prefix.
            N.copy(masks[bank, :], Bias[group, slot, start : start + ROWS])
            N.copy(rows[bank, :, :], KV[group, slot, start : start + ROWS, :])
