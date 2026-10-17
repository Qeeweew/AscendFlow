"""L2 traversal policy for the GEMM example; no compiler-specific shortcut."""

from ascendflow import language as T


@T.inline
def l2_tile(index, mcount, ncount, bm, bw, k, tile, cache_bytes):
    # Keep A/B residency within half of L2, including simultaneous regions.
    edge = T.max(1, cache_bytes // (4 * k * (bm + bw))) if tile == 0 else tile
    rows, cols = T.min(mcount, edge), T.min(ncount, edge)
    tile_m = rows if mcount % rows == 0 else 1
    tile_n = cols if ncount % cols == 0 else 1
    nregions = ncount // tile_n
    if tile_m > 1 and nregions > 1:
        row = index // (tile_m * tile_n * nregions) * tile_m + (index // tile_n) % tile_m
        col = index // (tile_m * tile_n) % nregions * tile_n + index % tile_n
    else:
        row, col = index // ncount, index % ncount
    return row * bm, col * bw
