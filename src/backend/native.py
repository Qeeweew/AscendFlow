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
CTYPE = {
    "f32": "float",
    "f16": "half",
    "bf16": "bfloat16_t",
    "u8": "uint8_t",
    "i32": "int32_t",
    "i64": "int64_t",
}


_MASKED_ELEMENTS = frozenset(
    {
        "mask_fill",
        "mask_muls",
        "mask_adds",
        "mask_add",
        "mask_mul",
        "mask_sub",
        "mask_div",
        "mask_max",
        "mask_exp",
        "mask_rsqrt",
        "mask_sqrt",
    }
)


def native_line(task, *, pointer_renderer, generation):
    """Format an admitted native using final IR operand addresses."""
    ptr = pointer_renderer
    refs = task.spec.operands
    p = [ptr(index) for index in range(len(refs))]
    if task.spec.native in {"set_fmatrix_a", "set_fmatrix_b"}:
        function = "asc_set_l13d_fmatrix" + ("_b" if task.spec.native.endswith("_b") else "")
        return (
            "{ asc_l13d_fmatrix_config config = { .config = "
            f"uint64_t({task.spec.params[0]}) | (uint64_t(1) << 16) "
            "}; " + function + "(config); }"
        )
    if task.spec.native == "set_load3d":
        return (
            "{ asc_load3d_v2_config config = { .config = uint64_t(1) << 16 }; "
            "asc_set_l13d_rpt(config); asc_set_l13d_padding(uint64_t(0)); "
            "asc_set_l13d_size(uint64_t(0)); }"
        )
    if task.spec.native == "set_fp32":
        return "asc_set_fp32_mode();"
    if task.spec.native in _MASKED_ELEMENTS:
        from . import vector_mask

        return "\n".join(vector_mask.masked_element_lines(task, p))
    if task.spec.native == "mask_bits":
        from . import vector_mask

        return "\n".join(vector_mask.mask_bits_lines(task, p))
    if task.spec.native == "mask_zero":
        from . import vector_mask

        return "\n".join(vector_mask.mask_zero_lines(task, p))
    if task.spec.native in {"mask_and", "mask_or", "mask_not"}:
        from . import vector_mask

        return "\n".join(vector_mask.combine_lines(task, p))
    if task.spec.native in {"select_masked", "select_masked_update"}:
        from . import vector_mask

        return "\n".join(vector_mask.select_lines(task, p))
    if task.spec.native.startswith("compare_"):
        from . import vector_mask

        return "\n".join(vector_mask.compare_lines(task, p))

    def length(ref, unit=1):
        return str(ref.total_nbytes // unit)

    if task.spec.native == "copy_bytes":
        function = "asc_copy_gm2ub_align" if refs[0].bank == "GM" else "asc_copy_ub2gm_align"
        return f"{function}({p[1]}, {p[0]}, uint32_t({length(refs[0])}));"
    if task.spec.native == "copy":
        route = tuple(r.bank for r in refs)
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
    if task.spec.native == "vector_copy_rows":
        columns = refs[0].nbytes // 4
        bits = [
            value.replace(CTYPE[ref.dtype] + "*", "uint32_t*")
            for value, ref in zip(p, refs, strict=True)
        ]
        strides = [ref.row_stride // 32 if ref.rows > 1 else 1 for ref in refs]
        return (
            "asc_set_mask_norm(); "
            f"asc_set_vector_mask(0, UINT64_C({(1 << columns) - 1})); "
            f"asc_copy({bits[1]}, {bits[0]}, uint8_t({refs[0].rows}), "
            f"1, 1, {strides[1]}, {strides[0]});"
        )
    if task.spec.native in {"copy_nz", "copy_nz_pad"}:
        rows, columns = task.spec.params[:2]
        resident_rows = refs[1].leading_rows
        call = (
            f"asc_copy_gm2l1((__cbuf__ void*){p[1]}, (__gm__ void*){p[0]}, "
            f"uint16_t({columns // 16}), "
            f"uint16_t({rows}), uint16_t({refs[0].leading_rows - rows}), "
            f"uint16_t({resident_rows - rows}));"
        )
        if task.spec.native == "copy_nz_pad":
            zero = (
                "asc_fill_value_config npu_zero = { .config = uint64_t(1) | "
                f"(uint64_t({refs[1].total_nbytes // 32}) << 16) }}; "
                f"asc_fill_l1({p[1]}, uint32_t(0), npu_zero); asc_sync_pipe(PIPE_MTE2); "
            )
            return "{ " + zero + call + " }"
        return call
    if task.spec.native == "copy_rows_prefix":
        rows = task.operands[0]
        if all(ref.contiguous for ref in refs):
            call = f"asc_copy_ub2gm({p[1]}, {p[0]}, uint32_t(({rows}) * {refs[0].nbytes}));"
        else:
            gaps = [(r.row_stride - r.nbytes) // 32 if r.rows > 1 else 0 for r in refs]
            call = (
                f"asc_copy_ub2gm({p[1]}, {p[0]}, uint16_t({rows}), "
                f"uint16_t({refs[0].nbytes // 32}), uint16_t({gaps[0]}), uint16_t({gaps[1]}));"
            )
        return f"if (({rows}) > 0) {{ {call} }}"
    if task.spec.native == "copy_rows":
        function = "asc_copy_gm2ub" if refs[0].bank == "GM" else "asc_copy_ub2gm"
        gaps = [(r.row_stride - r.nbytes) // 32 if r.rows > 1 else 0 for r in refs]
        return (
            f"{function}({p[1]}, {p[0]}, uint16_t({refs[0].rows}), "
            f"uint16_t({refs[0].nbytes // 32}), uint16_t({gaps[0]}), uint16_t({gaps[1]}));"
        )
    if task.spec.native in {"convert_f32", "convert_rn"}:
        source, destination = refs[0].dtype, refs[1].dtype
        function = (
            ("asc_half2float" if source == "f16" else "asc_bfloat162float")
            if task.spec.native == "convert_f32"
            else ("asc_float2half_rn" if destination == "f16" else "asc_float2bfloat16_rn")
        )
        return f"{function}({p[1]}, {p[0]}, uint32_t({length(refs[0], ITEMSIZE[source])}));"
    if task.spec.native in {"muls", "adds"}:
        function = "asc_mul_scalar" if task.spec.native == "muls" else "asc_add_scalar"
        return (
            f"{function}({p[-1]}, {p[0]}, float({task.scalar!r}), "
            f"uint32_t({length(refs[0], ITEMSIZE[refs[0].dtype])}));"
        )
    if task.spec.native in {"add", "mul", "sub", "div", "max"}:
        return f"asc_{task.spec.native}({p[2]}, {p[0]}, {p[1]}, uint32_t({length(refs[0], 4)}));"
    if task.spec.native == "add_strided":
        operands = [ptr(i, extra="npu_chunk * 256") for i in range(3)]
        strides = [ref.row_stride // 32 for ref in refs]
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {refs[0].nbytes // 256}; ++npu_chunk) {{ "
            f"asc_add({operands[2]}, {operands[0]}, {operands[1]}, uint8_t({refs[0].rows}), "
            f"1, 1, 1, {strides[2]}, {strides[0]}, {strides[1]}); }}"
        )
    if task.spec.native in {"exp", "rsqrt", "sqrt"}:
        return f"asc_{task.spec.native}({p[1]}, {p[0]}, uint32_t({length(refs[0], 4)}));"
    if task.spec.native in {"scalar_load", "index_load"}:
        element = CTYPE[refs[0].dtype]
        source = ptr(0).replace(f"{element}*", f"volatile {element}*")
        cast = "int64_t" if task.spec.native == "index_load" else "float"
        return f"npu_value_{task.uid} = {cast}(*({source}));"
    if task.spec.native == "scalar_store":
        destination = p[0].replace("float*", "volatile float*")
        return f"*({destination}) = npu_value_{task.scalar_source};"
    if task.spec.native in {"mul_columns", "add_columns", "broadcast_columns", "broadcast_rows"}:
        rows, cols = task.spec.params
        dst = ptr(-1, extra="npu_chunk * 256")
        setup = "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
        if task.spec.native in {"mul_columns", "add_columns"}:
            src = ptr(0, extra="npu_chunk * 256")
            vector = ptr(1, extra="npu_chunk * 256")
            call = f"asc_{task.spec.native.removesuffix('_columns')}({dst}, {src}, {vector}, uint8_t({rows}), 1, 1, 1, {cols // 8}, {cols // 8}, 0);"
        else:
            src = p[0] if task.spec.native == "broadcast_rows" else ptr(0, extra="npu_chunk * 256")
            dst = dst.replace("float*", "uint32_t*")
            src = src.replace("float*", "uint32_t*")
            strides = (0, 1) if task.spec.native == "broadcast_rows" else (1, 0)
            call = f"asc_copy({dst}, {src}, uint8_t({rows}), 1, {strides[0]}, {cols // 8}, {strides[1]});"
        return (
            setup + f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ {call} }}"
        )
    if task.spec.native in {"vector_copy", "brcb8", "broadcast_rows64"}:
        bits = [value.replace("float*", "uint32_t*") for value in p]
        if task.spec.native == "vector_copy":
            return f"asc_copy({bits[1]}, {bits[0]}, uint32_t({length(refs[0], 4)}));"
        if task.spec.native == "brcb8":
            return f"asc_brcb({bits[1]}, {bits[0]}, 1, 8, uint8_t({task.spec.params[0] // 8}));"
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"asc_copy({bits[1]}, {bits[0]}, uint8_t({task.spec.params[0]}), 1, 0, 8, 1);"
        )
    if task.spec.native == "zero_l1":
        return (
            "{ asc_fill_value_config npu_zero = { .config = uint64_t(1) | "
            f"(uint64_t({refs[0].total_nbytes // 32}) << 16) }}; "
            f"asc_fill_l1({p[0]}, uint32_t(0), npu_zero); }}"
        )
    if task.spec.native in {"fill", "fill_neg_inf"}:
        if refs[0].dtype in {"f16", "bf16"}:
            from ..target.scalars import b16_fill_bits

            value = float("-inf") if task.spec.native == "fill_neg_inf" else task.scalar
            bits = b16_fill_bits(value, refs[0].dtype)
            destination = p[0].replace(CTYPE[refs[0].dtype] + "*", "uint16_t*")
            return (
                f"asc_duplicate({destination}, uint16_t({bits}), uint32_t({length(refs[0], 2)}));"
            )
        value = (
            "-__builtin_inff()" if task.spec.native == "fill_neg_inf" else f"float({task.scalar!r})"
        )
        if task.spec.native == "fill" and isinf(task.scalar):
            value = ("-" if task.scalar < 0 else "") + "__builtin_inff()"
        return f"asc_duplicate({p[0]}, {value}, uint32_t({length(refs[0], 4)}));"
    if task.spec.native == "mul_outer":
        rows, cols = task.spec.params
        column = ptr(0, extra="npu_chunk * 256")
        dst = ptr(2, extra="npu_chunk * 256")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"asc_mul({dst}, {column}, {p[1]}, uint8_t({rows}), "
            f"1, 1, 0, {cols // 8}, 0, 1); }}"
        )
    if task.spec.native in {"mul_row_broadcast", "sub_row_broadcast", "div_row_broadcast"}:
        rows, cols = task.spec.params
        function = task.spec.native.removesuffix("_row_broadcast")
        if cols < 64:
            # Each narrow row has a 32-byte aligned origin. Mask the exact
            # columns before repeating across rows; neither reads nor writes
            # touch the next row's lanes or allocation padding.
            return (
                "asc_set_mask_norm(); "
                f"asc_set_vector_mask(0, UINT64_C({(1 << cols) - 1})); "
                f"asc_{function}({p[2]}, {p[0]}, {p[1]}, uint8_t({rows}), "
                f"1, 1, 0, {cols // 8}, {cols // 8}, 1);"
            )
        if rows < cols // 64:
            # Repeat across the columns of one row. The eight identical
            # scalar words stay fixed (block/repeat strides 0/0); the full
            # matrix footprint and the exact in-place policy are unchanged.
            src = ptr(0, extra=f"npu_row * {cols * 4}")
            scalar = ptr(1, extra="npu_row * 32")
            dst = ptr(2, extra=f"npu_row * {cols * 4}")
            return (
                "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
                f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
                f"asc_{function}({dst}, {src}, {scalar}, uint8_t({cols // 64}), "
                "1, 1, 0, 8, 8, 0); }"
            )
        src = ptr(0, extra="npu_chunk * 256")
        dst = ptr(2, extra="npu_chunk * 256")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"asc_{function}({dst}, {src}, {p[1]}, uint8_t({rows}), "
            f"1, 1, 0, {cols // 8}, {cols // 8}, 1); }}"
        )
    if task.spec.native in {"merge_sum_packed", "merge_max_packed"}:
        rows, partials = task.spec.params
        function = (
            "asc_repeat_reduce_sum"
            if task.spec.native == "merge_sum_packed"
            else "asc_repeat_reduce_max_only_value"
        )
        return (
            f"asc_set_mask_norm(); asc_set_vector_mask(0, uint64_t({(1 << partials) - 1})); "
            f"{function}({p[1]}, {p[0]}, uint8_t({rows}), 1, 1, {refs[0].row_stride // 32});"
        )
    if task.spec.native in {"mul_columns64", "fma_columns64"}:
        function = "asc_fma" if task.spec.native == "fma_columns64" else "asc_mul"
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"{function}({p[2]}, {p[0]}, {p[1]}, uint8_t({task.spec.params[0]}), 1, 1, 1, 8, 8, 0);"
        )
    if task.spec.native == "reduce_rows64":
        return f"asc_repeat_reduce_sum({p[1]}, {p[0]}, uint32_t({task.spec.params[0] * 64}));"
    if task.spec.native in {"scale_value", "sub_value"}:
        function = "asc_mul_scalar" if task.spec.native == "scale_value" else "asc_add_scalar"
        value = f"npu_value_{task.scalar_source}"
        if task.spec.native == "sub_value":
            value = f"-{value}"
        return f"{function}({p[1]}, {p[0]}, {value}, uint32_t({length(refs[0], 4)}));"
    if task.spec.native in {"mul_rows", "sub_rows", "add_rows", "div_rows"}:
        rows, cols = task.spec.params
        pp = [ptr(index, extra=f"npu_row * {ref.row_stride}") for index, ref in enumerate(refs)]
        function = task.spec.native.removesuffix("_rows")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_{function}({pp[2]}, {pp[0]}, {pp[1]}, uint32_t({cols})); }}"
        )
    if task.spec.native in {"reduce_tile_sum64", "reduce_tile_max64"}:
        rows, cols = task.spec.params
        function = (
            "asc_repeat_reduce_sum"
            if task.spec.native == "reduce_tile_sum64"
            else "asc_repeat_reduce_max_only_value"
        )
        src = ptr(0, extra="npu_chunk * 256")
        dst = ptr(1, extra=f"npu_chunk * {rows * 4}")
        return (
            "asc_set_mask_norm(); asc_set_vector_mask(0, UINT64_MAX); "
            f"for (int npu_chunk = 0; npu_chunk < {cols // 64}; ++npu_chunk) {{ "
            f"{function}({dst}, {src}, uint8_t({rows}), 1, 1, {cols // 8}); }}"
        )
    if task.spec.native in {"reduce_max8", "reduce_sum8"}:
        function = "max" if task.spec.native == "reduce_max8" else "sum"
        return f"asc_datablock_reduce_{function}({p[1]}, {p[0]}, uint32_t({length(refs[0], 4)}));"
    if task.spec.native in {"reduce_sum64", "reduce_max64"}:
        rows, cols = task.spec.params
        function = (
            "asc_repeat_reduce_sum"
            if task.spec.native == "reduce_sum64"
            else "asc_repeat_reduce_max_only_value"
        )
        if refs[1].contiguous:
            return f"{function}({p[1]}, {p[0]}, uint32_t({rows * cols}));"
        src = ptr(0, extra=f"npu_row * {cols * 4}")
        dst = ptr(1, extra=f"npu_row * {refs[1].row_stride}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"{function}({dst}, {src}, uint32_t({cols})); }}"
        )
    if task.spec.native in {
        "merge_sum",
        "merge_max",
        "stat_muls",
        "stat_adds",
        "stat_rsqrt",
        "stat_sqrt",
    }:
        rows = task.spec.params[0]
        src = ptr(0, extra=f"npu_row * {refs[0].row_stride}")
        dst = ptr(1, extra=f"npu_row * {refs[1].row_stride}")
        if task.spec.native.startswith("merge"):
            function = (
                "asc_repeat_reduce_sum"
                if task.spec.native == "merge_sum"
                else "asc_repeat_reduce_max_only_value"
            )
            call = f"{function}({dst}, {src}, uint32_t({task.spec.params[1]}));"
        elif task.spec.native in {"stat_rsqrt", "stat_sqrt"}:
            call = f"asc_{task.spec.native.removeprefix('stat_')}({dst}, {src}, uint32_t(1));"
        else:
            function = "asc_mul_scalar" if task.spec.native == "stat_muls" else "asc_add_scalar"
            call = f"{function}({dst}, {src}, float({task.scalar!r}), uint32_t(1));"
        return f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ {call} }}"
    if task.spec.native in {"mul_gamma", "add_bias"}:
        rows, cols = task.spec.params
        src = ptr(0, extra=f"npu_row * {cols * 4}")
        dst = ptr(2, extra=f"npu_row * {cols * 4}")
        return (
            f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
            f"asc_{'mul' if task.spec.native == 'mul_gamma' else 'add'}"
            f"({dst}, {src}, {p[1]}, uint32_t({cols})); }}"
        )
    if task.spec.native in {"stat_fill", "stat_div"}:
        rows = task.spec.params[0]
        pp = [ptr(index, extra=f"npu_row * {ref.row_stride}") for index, ref in enumerate(refs)]
        if task.spec.native == "stat_fill":
            call = f"asc_duplicate({pp[0]}, float({task.scalar!r}), uint32_t(1));"
        else:
            call = f"asc_div({pp[2]}, {pp[0]}, {pp[1]}, uint32_t(1));"
        return f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ {call} }}"
    if task.spec.native == "nd2nz_overlay":
        maximum, cols, _ = task.spec.params
        resident_rows = refs[1].leading_rows
        rows = task.operands[0]
        itemsize = ITEMSIZE[refs[0].dtype]
        copy = (
            f"asc_copy_gm2l1_nd2nz({p[1]}, {p[0]}, 1, uint16_t({rows}), {cols}, 0, "
            f"{refs[0].row_stride // itemsize}, {resident_rows}, 1, 0);"
        )
        if refs[0].row_stride // itemsize > 65535:
            source = ptr(0, extra="npu_row * " + str(refs[0].row_stride))
            destination = ptr(1, extra="npu_row * 32")
            copy = (
                f"for (int64_t npu_row = 0; npu_row < ({rows}); ++npu_row) {{ "
                f"asc_copy_gm2l1_nd2nz({destination}, {source}, 1, 1, {cols}, 0, 0, {resident_rows}, 1, 0); }}"
            )
        return f"if (({rows}) > 0) {{ {copy} }}"
    if task.spec.native == "nd2nz_rows":
        maximum, cols, _, resident_cols = task.spec.params
        resident_rows = refs[1].leading_rows
        rows = task.operands[0]
        itemsize = ITEMSIZE[refs[0].dtype]
        copy = (
            f"asc_copy_gm2l1_nd2nz({p[1]}, {p[0]}, 1, uint16_t({rows}), {cols}, 0, "
            f"{refs[0].row_stride // itemsize}, {resident_rows}, 1, 0);"
        )
        if refs[0].row_stride // itemsize > 65535:
            source = ptr(0, extra=f"npu_row * {refs[0].row_stride}")
            destination = ptr(1, extra="npu_row * 32")
            copy = (
                f"for (int64_t npu_row = 0; npu_row < ({rows}); ++npu_row) {{ "
                f"asc_copy_gm2l1_nd2nz({destination}, {source}, 1, 1, {cols}, 0, "
                f"0, {resident_rows}, 1, 0); }}"
            )
        zero = (
            "asc_fill_value_config npu_zero = { .config = uint64_t(1) | "
            f"(uint64_t({refs[1].total_nbytes // 32}) << 16) }}; "
            f"asc_fill_l1({p[1]}, uint32_t(0), npu_zero);"
        )
        tail = f"{{ {zero} if (({rows}) > 0) {{ asc_sync_pipe(PIPE_MTE2); {copy} }} }}"
        return (
            f"if (({rows}) == {resident_rows}) {{ {copy} }} else {tail}"
            if maximum == resident_rows and cols == resident_cols
            else tail
        )
    if task.spec.native in {"nd2nz", "nd2nz_pad"}:
        rows, cols = task.spec.params[:2]
        padded = task.spec.native == "nd2nz_pad"
        resident_rows = refs[1].leading_rows
        setup = ""
        if padded:
            setup = (
                "{ asc_fill_value_config npu_zero = { .config = uint64_t(1) | "
                f"(uint64_t({refs[1].total_nbytes // 32}) << 16) }}; "
                f"asc_fill_l1({p[1]}, uint32_t(0), npu_zero); asc_sync_pipe(PIPE_MTE2); "
            )
        itemsize = ITEMSIZE[refs[0].dtype]
        if refs[0].row_stride // itemsize > 65535:
            # The hardware source leading dimension is uint16. Row-wise
            # descriptors retain exact byte effects without truncating it.
            source = ptr(0, extra=f"npu_row * {refs[0].row_stride}")
            destination = ptr(1, extra="npu_row * 32")
            copy = (
                f"for (int npu_row = 0; npu_row < {rows}; ++npu_row) {{ "
                f"asc_copy_gm2l1_nd2nz({destination}, {source}, 1, 1, {cols}, 0, "
                f"0, {resident_rows}, 1, 0); }}"
            )
        else:
            copy = (
                f"asc_copy_gm2l1_nd2nz({p[1]}, {p[0]}, 1, {rows}, {cols}, 0, "
                f"{refs[0].row_stride // itemsize}, {resident_rows}, 1, 0);"
            )
        return setup + copy + (" }" if padded else "")
    if task.spec.native in {"load3d_a", "load3d_b_nn"}:
        rows, cols = task.spec.params
        a = task.spec.native == "load3d_a"
        dst = p[1].replace("bfloat16_t*", "half*")
        src = p[0].replace("bfloat16_t*", "half*")
        return (
            f"asc_copy_l12l0{'a' if a else 'b'}({dst}, {src}, {cols}, {rows}, 0, 0, "
            f"1, 1, 1, 1, 1, 1, 0, 0, false, {str(not a).lower()}, {cols});"
        )
    if task.spec.native == "load2d_a":
        rows, cols = task.spec.params
        src = ptr(0, extra="npu_chunk * 512")
        dst = ptr(1, extra=f"npu_chunk * {cols * 32}")
        stride = refs[0].leading_rows // 16
        return (
            f"for (int npu_chunk = 0; npu_chunk < {rows // 16}; ++npu_chunk) {{ "
            f"asc_copy_l12l0a({dst}, {src}, 0, {cols // 16}, {stride}, 0); }}"
        )
    if task.spec.native == "load2d_a_t":
        rows, cols = task.spec.params
        return f"load_cbuf_to_ca({p[1]}, {p[0]}, 0, {rows * cols // 256}, 1, 0, 0, true, inc);"
    if task.spec.native == "load2d_b_nt":
        rows, cols = task.spec.params
        return f"asc_copy_l12l0b({p[1]}, {p[0]}, 0, {rows * cols * ITEMSIZE[refs[0].dtype] // 512}, 1, 0);"
    if task.spec.native == "load2d_b_nn":
        rows, cols = task.spec.params  # B[K,N] compact zN -> L0B ZN
        src = ptr(0, extra="npu_chunk * 512")
        dst = ptr(1, extra=f"npu_chunk * {cols * 32}")
        stride = refs[0].leading_rows // 16
        # CANN 9.1/2201's public asc_copy_l12l0b has no classic transpose
        # parameter; the similarly named *_transpose is a 3510 API. Use
        # the exact audited dav_c220 LoadData2D intrinsic with transpose=true,
        # not the distinct LoadDataWithTranspose instruction/footprint.
        return (
            f"for (int npu_chunk = 0; npu_chunk < {rows // 16}; ++npu_chunk) {{ "
            f"load_cbuf_to_cb({dst}, {src}, 0, {cols // 16}, {stride}, 0, 0, true, inc); }}"
        )
    if task.spec.native in {"mad", "mad_cycle", "mad_first"}:
        m, n, k = task.spec.params[:3]
        clear = task.spec.params[3] if task.spec.native != "mad_first" else 0
        flag = (
            f"({generation} == 0)"
            if task.spec.native == "mad_first"
            else f"({generation} % {clear} == 0)"
            if task.spec.native == "mad_cycle"
            else ("true" if clear else "false")
        )
        unit = task.operands[0] if task.spec.unit_flag else "0"
        return f"asc_mmad({p[2]}, {p[0]}, {p[1]}, {m}, {k}, {n}, {unit}, false, false, {flag});"
    if task.spec.native in {"fix", "fix_rows"}:
        m, n = task.spec.params[:2]
        source_m = task.spec.params[2] if len(task.spec.params) == 3 else m
        dtype = refs[1].dtype
        quant = {
            "f32": "0",
            "f16": "uint64_t(QuantMode_t::F322F16)",
            "bf16": "uint64_t(QuantMode_t::F322BF16)",
        }[dtype]
        rows = task.operands[0] if task.spec.native == "fix_rows" else str(m)
        call = (
            f"asc_copy_l0c2gm({p[1]}, {p[0]}, {n}, {rows}, {refs[1].row_stride // ITEMSIZE[dtype]}, "
            f"{source_m}, {task.spec.unit_flag}, {quant}, 0, false, true);"
        )
        return f"if (({rows}) > 0) {{ {call} }}" if task.spec.native == "fix_rows" else call
    if task.spec.native == "mad16":
        clear = "true" if task.spec.params[0] else "false"
        return f"asc_mmad({p[2]}, {p[0]}, {p[1]}, 16, 16, 16, 0, false, false, {clear});"
    if task.spec.native == "fix16":
        return f"asc_copy_l0c2gm({p[1]}, {p[0]}, 16, 16, 16, 16, 0, 0, 0, false, true);"
    raise AssertionError("unvalidated native recipe")
