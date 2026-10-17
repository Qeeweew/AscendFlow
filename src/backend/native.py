"""Pure dav-2201 native formatting; no analysis or synchronization planning.

Native selection, physical geometry and completion contracts are established
before this backend. Pointers are bound directly to final SSA addresses.
The formatter does not decode programs or plan synchronization.
"""

from __future__ import annotations

from math import isinf

from ..target.native import ITEMSIZE

QUALIFIER = {
    "GM": "__gm__",
    "UB": "__ubuf__",
    "L1": "__cbuf__",
    "L0A": "__ca__",
    "L0B": "__cb__",
    "L0C": "__cc__",
}
CTYPE = {"f32": "float", "f16": "half", "bf16": "bfloat16_t"}


def native_line(task, *, pointer_renderer, generation):
    """Format an admitted native using final IR operand addresses."""
    ptr = pointer_renderer
    refs = task.refs
    p = [ptr(ref) for ref in refs]

    def length(ref, unit=1):
        return str(ref.total_nbytes // unit)

    if task.native == "copy":
        route = tuple(r.storage.bank for r in refs)
        function = {
            ("GM", "UB"): "asc_copy_gm2ub",
            ("UB", "GM"): "asc_copy_ub2gm",
            ("GM", "L1"): "asc_copy_gm2l1",
            ("UB", "UB"): "asc_copy_ub2ub",
        }.get(route)
        if function:
            return f"{function}({p[1]}, {p[0]}, uint32_t({length(refs[0])}));"
        if route == ("L1", "L0A"):
            # CATLASS AtlasA2 zN->zZ uses classic Load2D with ifTranspose=false.
            return f"asc_copy_l12l0a({p[1]}, {p[0]}, 0, 1, 1, 0);"
        return f"asc_copy_l12l0b({p[1]}, {p[0]}, 0, 1, 1, 0);"
    if task.native == "copy_rows":
        function = "asc_copy_gm2ub" if refs[0].storage.bank == "GM" else "asc_copy_ub2gm"
        gaps = [(r.row_stride - r.nbytes) // 32 if r.rows > 1 else 0 for r in refs]
        return (
            f"{function}({p[1]}, {p[0]}, uint16_t({refs[0].rows}), "
            f"uint16_t({refs[0].nbytes // 32}), uint16_t({gaps[0]}), uint16_t({gaps[1]}));"
        )
    if task.native in {"convert_f32", "convert_rn"}:
        source, destination = refs[0].dtype, refs[1].dtype
        function = (
            ("asc_half2float" if source == "f16" else "asc_bfloat162float")
            if task.native == "convert_f32"
            else ("asc_float2half_rn" if destination == "f16" else "asc_float2bfloat16_rn")
        )
        return f"{function}({p[1]}, {p[0]}, uint32_t({length(refs[0], ITEMSIZE[source])}));"
    if task.native in {"muls", "adds"}:
        function = "asc_mul_scalar" if task.native == "muls" else "asc_add_scalar"
        return (
            f"{function}({p[-1]}, {p[0]}, float({task.scalar!r}), "
            f"uint32_t({length(refs[0], ITEMSIZE[refs[0].dtype])}));"
        )
    if task.native in {"add", "mul", "sub", "div", "max"}:
        return f"asc_{task.native}({p[2]}, {p[0]}, {p[1]}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"exp", "rsqrt", "sqrt"}:
        return f"asc_{task.native}({p[1]}, {p[0]}, uint32_t({length(refs[0], 4)}));"
    if task.native == "scalar_load":
        source = ptr(refs[0]).replace("__ubuf__ float*", "__ubuf__ volatile float*")
        return f"npu_value_{task.uid} = *({source});"
    if task.native in {"vector_copy", "brcb8", "broadcast_rows64"}:
        bits = [value.replace("float*", "uint32_t*") for value in p]
        if task.native == "vector_copy":
            return f"asc_copy({bits[1]}, {bits[0]}, uint32_t({length(refs[0], 4)}));"
        if task.native == "brcb8":
            return f"asc_brcb({bits[1]}, {bits[0]}, 1, 8, uint8_t({task.params[0] // 8}));"
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"asc_copy({bits[1]}, {bits[0]}, uint8_t({task.params[0]}), 1, 0, 8, 1);"
        )
    if task.native in {"fill", "fill_neg_inf"}:
        value = "-__builtin_inff()" if task.native == "fill_neg_inf" else f"float({task.scalar!r})"
        if task.native == "fill" and isinf(task.scalar):
            value = ("-" if task.scalar < 0 else "") + "__builtin_inff()"
        return f"asc_duplicate({p[0]}, {value}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"mul_row_broadcast", "sub_row_broadcast", "div_row_broadcast"}:
        rows, cols = task.params
        function = task.native.removesuffix("_row_broadcast")
        if rows < cols // 64:
            # Repeat across the columns of one row. The eight identical
            # scalar words stay fixed (block/repeat strides 0/0); the full
            # matrix footprint and the exact in-place policy are unchanged.
            src = ptr(refs[0], extra=f"npu_row * {cols * 4}")
            scalar = ptr(refs[1], extra="npu_row * 32")
            dst = ptr(refs[2], extra=f"npu_row * {cols * 4}")
            return (
                "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
                f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
                f"asc_{function}({dst}, {src}, {scalar}, uint8_t({cols // 64}), "
                "1, 1, 0, 8, 8, 0); }"
            )
        src = ptr(refs[0], extra="npu_chunk * 256")
        dst = ptr(refs[2], extra="npu_chunk * 256")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"asc_{function}({dst}, {src}, {p[1]}, uint8_t({rows}), "
            f"1, 1, 0, {cols // 8}, {cols // 8}, 1); }}"
        )
    if task.native in {"merge_sum_packed", "merge_max_packed"}:
        rows, partials = task.params
        function = (
            "asc_repeat_reduce_sum"
            if task.native == "merge_sum_packed"
            else "asc_repeat_reduce_max_only_value"
        )
        return (
            f"asc_set_mask_norm(); asc_set_vector_mask(0, uint64_t({(1 << partials) - 1})); "
            f"{function}({p[1]}, {p[0]}, uint8_t({rows}), 1, 1, {refs[0].row_stride // 32});"
        )
    if task.native in {"mul_columns64", "fma_columns64"}:
        function = "asc_fma" if task.native == "fma_columns64" else "asc_mul"
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"{function}({p[2]}, {p[0]}, {p[1]}, uint8_t({task.params[0]}), 1, 1, 1, 8, 8, 0);"
        )
    if task.native == "reduce_rows64":
        return f"asc_repeat_reduce_sum({p[1]}, {p[0]}, uint32_t({task.params[0] * 64}));"
    if task.native in {"scale_value", "sub_value"}:
        function = "asc_mul_scalar" if task.native == "scale_value" else "asc_add_scalar"
        value = f"npu_value_{task.scalar_source}"
        if task.native == "sub_value":
            value = f"-{value}"
        return f"{function}({p[1]}, {p[0]}, {value}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"mul_rows", "sub_rows", "add_rows", "div_rows"}:
        rows, cols = task.params
        pp = [ptr(r, extra=f"npu_row * {r.row_stride}") for r in refs]
        function = task.native.removesuffix("_rows")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_{function}({pp[2]}, {pp[0]}, {pp[1]}, uint32_t({cols})); }}"
        )
    if task.native in {"reduce_tile_sum64", "reduce_tile_max64"}:
        rows, cols = task.params
        function = (
            "asc_repeat_reduce_sum"
            if task.native == "reduce_tile_sum64"
            else "asc_repeat_reduce_max_only_value"
        )
        src = ptr(refs[0], extra="npu_chunk * 256")
        dst = ptr(refs[1], extra=f"npu_chunk * {rows * 4}")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"{function}({dst}, {src}, uint8_t({rows}), 1, 1, {cols // 8}); }}"
        )
    if task.native in {"reduce_max8", "reduce_sum8"}:
        function = "max" if task.native == "reduce_max8" else "sum"
        return f"asc_datablock_reduce_{function}({p[1]}, {p[0]}, uint32_t({length(refs[0], 4)}));"
    if task.native in {"reduce_sum64", "reduce_max64"}:
        rows, cols = task.params
        function = (
            "asc_repeat_reduce_sum"
            if task.native == "reduce_sum64"
            else "asc_repeat_reduce_max_only_value"
        )
        if refs[1].contiguous:
            return f"{function}({p[1]}, {p[0]}, uint32_t({rows * cols}));"
        src = ptr(refs[0], extra=f"npu_row * {cols * 4}")
        dst = ptr(refs[1], extra=f"npu_row * {refs[1].row_stride}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"{function}({dst}, {src}, uint32_t({cols})); }}"
        )
    if task.native in {
        "merge_sum",
        "merge_max",
        "stat_muls",
        "stat_adds",
        "stat_rsqrt",
        "stat_sqrt",
    }:
        rows = task.params[0]
        src = ptr(refs[0], extra=f"npu_row * {refs[0].row_stride}")
        dst = ptr(refs[1], extra=f"npu_row * {refs[1].row_stride}")
        if task.native.startswith("merge"):
            function = (
                "asc_repeat_reduce_sum"
                if task.native == "merge_sum"
                else "asc_repeat_reduce_max_only_value"
            )
            call = f"{function}({dst}, {src}, uint32_t({task.params[1]}));"
        elif task.native in {"stat_rsqrt", "stat_sqrt"}:
            call = f"asc_{task.native.removeprefix('stat_')}({dst}, {src}, uint32_t(1));"
        else:
            function = "asc_mul_scalar" if task.native == "stat_muls" else "asc_add_scalar"
            call = f"{function}({dst}, {src}, float({task.scalar!r}), uint32_t(1));"
        return f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ {call} }}"
    if task.native in {"mul_gamma", "add_bias"}:
        rows, cols = task.params
        src = ptr(refs[0], extra=f"npu_row * {cols * 4}")
        dst = ptr(refs[2], extra=f"npu_row * {cols * 4}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_{'mul' if task.native == 'mul_gamma' else 'add'}"
            f"({dst}, {src}, {p[1]}, uint32_t({cols})); }}"
        )
    if task.native in {"stat_fill", "stat_div"}:
        rows = task.params[0]
        pp = [ptr(r, extra=f"npu_row * {r.row_stride}") for r in refs]
        if task.native == "stat_fill":
            call = f"asc_duplicate({pp[0]}, float({task.scalar!r}), uint32_t(1));"
        else:
            call = f"asc_div({pp[2]}, {pp[0]}, {pp[1]}, uint32_t(1));"
        return f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ {call} }}"
    if task.native in {"nd2nz", "nd2nz_pad"}:
        rows, cols = task.params[:2]
        padded = task.native == "nd2nz_pad"
        resident_rows = task.params[2] if padded else rows
        setup = ""
        if padded:
            setup = (
                "{ asc_fill_value_config npu_zero = { .config = uint64_t(1) | "
                f"(uint64_t({refs[1].total_nbytes // 32}) << 16) }}; "
                f"asc_fill_l1({p[1]}, uint32_t(0), npu_zero); asc_sync_pipe(PIPE_MTE2); "
            )
        if refs[0].row_stride // 2 > 65535:
            # The hardware source leading dimension is uint16. Row-wise
            # descriptors retain exact byte effects without truncating it.
            source = ptr(refs[0], extra=f"npu_row * {refs[0].row_stride}")
            destination = ptr(refs[1], extra="npu_row * 32")
            copy = (
                f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
                f"asc_copy_gm2l1_nd2nz({destination}, {source}, 1, 1, {cols}, 0, "
                f"0, {resident_rows}, 1, 0); }}"
            )
        else:
            copy = (
                f"asc_copy_gm2l1_nd2nz({p[1]}, {p[0]}, 1, {rows}, {cols}, 0, "
                f"{refs[0].row_stride // 2}, {resident_rows}, 1, 0);"
            )
        return setup + copy + (" }" if padded else "")
    if task.native in {"load3d_a", "load3d_b_nn"}:
        rows, cols = task.params
        a = task.native == "load3d_a"
        dst = p[1].replace("bfloat16_t*", "half*")
        src = p[0].replace("bfloat16_t*", "half*")
        return (
            f"asc_copy_l12l0{'a' if a else 'b'}({dst}, {src}, {cols}, {rows}, 0, 0, "
            f"1, 1, 1, 1, 1, 1, 0, 0, false, {str(not a).lower()}, {cols});"
        )
    if task.native == "load2d_a":
        rows, cols = task.params
        src = ptr(refs[0], extra="npu_chunk * 512")
        dst = ptr(refs[1], extra=f"npu_chunk * {cols * 32}")
        stride = refs[0].row_stride // 512 if not refs[0].contiguous else rows // 16
        return (
            f"for (int npu_chunk = 0; npu_chunk < {rows // 16}; ++npu_chunk) {{ "
            f"asc_copy_l12l0a({dst}, {src}, 0, {cols // 16}, {stride}, 0); }}"
        )
    if task.native == "load2d_b_nt":
        rows, cols = task.params
        return f"asc_copy_l12l0b({p[1]}, {p[0]}, 0, {rows * cols // 256}, 1, 0);"
    if task.native == "load2d_b_nn":
        rows, cols = task.params  # B[K,N] compact zN -> L0B ZN
        src = ptr(refs[0], extra="npu_chunk * 512")
        dst = ptr(refs[1], extra=f"npu_chunk * {cols * 32}")
        stride = refs[0].row_stride // 512 if not refs[0].contiguous else rows // 16
        # CANN 9.1/2201's public asc_copy_l12l0b has no classic transpose
        # parameter; the similarly named *_transpose is a 3510 API. Use
        # the exact audited dav_c220 LoadData2D intrinsic with transpose=true,
        # not the distinct LoadDataWithTranspose instruction/footprint.
        return (
            f"for (int npu_chunk = 0; npu_chunk < {rows // 16}; ++npu_chunk) {{ "
            f"load_cbuf_to_cb({dst}, {src}, 0, {cols // 16}, {stride}, 0, 0, true, inc); }}"
        )
    if task.native in {"mad", "mad_cycle", "mad_first"}:
        m, n, k = task.params[:3]
        clear = task.params[3] if task.native != "mad_first" else 0
        flag = (
            f"({generation} == 0)"
            if task.native == "mad_first"
            else f"({generation} % {clear} == 0)"
            if task.native == "mad_cycle"
            else ("true" if clear else "false")
        )
        return f"asc_mmad({p[2]}, {p[0]}, {p[1]}, {m}, {k}, {n}, 0, false, false, {flag});"
    if task.native == "fix":
        m, n = task.params[:2]
        source_m = task.params[2] if len(task.params) == 3 else m
        dtype = refs[1].dtype
        quant = {
            "f32": "0",
            "f16": "uint64_t(QuantMode_t::F322F16)",
            "bf16": "uint64_t(QuantMode_t::F322BF16)",
        }[dtype]
        return (
            f"asc_copy_l0c2gm({p[1]}, {p[0]}, {n}, {m}, {refs[1].row_stride // ITEMSIZE[dtype]}, "
            f"{source_m}, 0, {quant}, 0, false, true);"
        )
    if task.native == "mad16":
        clear = "true" if task.params[0] else "false"
        return f"asc_mmad({p[2]}, {p[0]}, {p[1]}, 16, 16, 16, 0, false, false, {clear});"
    if task.native == "fix16":
        return f"asc_copy_l0c2gm({p[1]}, {p[0]}, 16, 16, 16, 16, 0, 0, 0, false, true);"
    raise AssertionError("unvalidated native recipe")


def load3d_init(config):
    if config is None:
        return []
    a, b = config
    return [
        "  // Invariant dav-2201 Load3D SPRs; width low, height high (raw contract).",
        f"  asc_l13d_fmatrix_config npu_fa = {{ .config = uint64_t({a}) | (uint64_t(1) << 16) }};",
        f"  asc_l13d_fmatrix_config npu_fb = {{ .config = uint64_t({b}) | (uint64_t(1) << 16) }};",
        "  asc_set_l13d_fmatrix(npu_fa); asc_set_l13d_fmatrix_b(npu_fb);",
        "  asc_load3d_v2_config npu_rpt = { .config = uint64_t(1) << 16 };",
        "  asc_set_l13d_rpt(npu_rpt); asc_set_l13d_padding(uint64_t(0)); asc_set_l13d_size(uint64_t(0));",
        "  asc_sync_notify(PIPE_S, PIPE_MTE1, event_t(0));",
        "  asc_sync_wait(PIPE_S, PIPE_MTE1, event_t(0));",
    ]
