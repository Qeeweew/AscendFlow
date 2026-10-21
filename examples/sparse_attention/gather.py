"""CSA cache movement: direct window pages and streamed sparse compressed rows."""

from ascendflow import language as N

NEG_INF = float("-inf")


@N.inline
def copy_window_pages(
    Ori, OriTable, destination, batch, length, visible, feature, PAGE, C, MAX_SEQ, ORI_BLOCKS
):
    position = N.max(0, length - visible)
    first_rows = N.min(visible, PAGE - position % PAGE)
    table_column = N.min(position // PAGE, MAX_SEQ // PAGE - 1)
    page = N.load_index(
        OriTable[batch, table_column : table_column + 1], minimum=0, maximum=ORI_BLOCKS - 1
    )
    # One real page fragment and explicitly defined zero padding. For a full
    # aligned page the native packet emits just ND2NZ, without a clear.
    N.copy(
        Ori[page, position % PAGE : position % PAGE + 128, feature : feature + C],
        destination,
        valid_rows=first_rows,
        pad_value=0,
    )
    second_rows = visible - first_rows
    table_column2 = N.min((position + first_rows) // PAGE, MAX_SEQ // PAGE - 1)
    page2 = N.load_index(
        OriTable[batch, table_column2 : table_column2 + 1], minimum=0, maximum=ORI_BLOCKS - 1
    )
    N.copy(
        Ori[page2, :128, feature : feature + C],
        destination[first_rows : first_rows + 128, :],
        valid_rows=second_rows,
    )


@N.inline
def gather_compressed(
    Cmp,
    CmpTable,
    Lengths,
    Counts,
    Indices,
    KV,
    Bias,
    zeros,
    rows,
    masks,
    batch,
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
    length = N.load_index(Lengths[batch : batch + 1], minimum=0, maximum=MAX_SEQ)
    count = N.load_index(Counts[batch : batch + 1], minimum=0, maximum=TOPK)
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
                    index = N.load_index(
                        Indices[batch, index_slot : index_slot + 1],
                        minimum=0,
                        maximum=MAX_SEQ // RATIO - 1,
                    )
                    if index < length // RATIO:
                        page = N.load_index(
                            CmpTable[
                                batch, index // (PAGE // RATIO) : index // (PAGE // RATIO) + 1
                            ],
                            minimum=0,
                            maximum=CMP_BLOCKS - 1,
                        )
                        N.copy(Cmp[page, index % (PAGE // RATIO), :], rows[bank, row, :])
                        N.store_scalar(zero, out=masks[bank, row : row + 1])
        with N.Stage(1, name="sparse_publish"):
            # Masks and rows are in the same MTE3 publication prefix.
            N.copy(masks[bank, :], Bias[group, slot, start : start + ROWS])
            N.copy(rows[bank, :, :], KV[group, slot, start : start + ROWS, :])
