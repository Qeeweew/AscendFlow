"""Real masked source setup and native-trace inspection; no independent oracle."""

from __future__ import annotations

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.dialects.io import parse


def _task(memory, native):
    return next(
        template.operation for template in memory.templates if template.spec.native == native
    )


def _ordered(execution):
    indexes = {
        command.occurrence.operation.uid.data: index
        for index, command in enumerate(execution.commands)
        if command.kind == "native"
    }

    def ordered(source, consumer):
        first = execution.commands[indexes[source.uid.data]]
        second = execution.commands[indexes[consumer.uid.data]]
        return execution.happens_before(first.end, second.begin)

    return ordered


def _masked_fill(mask_expression):
    return N.from_source(
        f"""
        def kernel(X:N.Tensor((16,256),N.f32), Y:N.Tensor((16,256),N.f32,access="write"),
                   q:N.int64, lo:N.int64, hi:N.int64):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((16,256),N.f32)
                N.copy(X,work)
                rows=N.arange(16)[:, None]
                cols=N.arange(256)[None, :]
                mask={mask_expression}
                N.fill(work,-7.0,mask=mask)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )


def _mask_spec(expression, shape=(16, 256)):
    """The filled tile's predicate program and its admitted replay facts."""
    rank = len(shape)
    coords = []
    for axis, extent in enumerate(shape):
        parts = ["None"] * rank
        parts[axis] = ":"
        coords.append(f"i{axis}=N.arange({extent})[{', '.join(parts)}]")
    body = [
        f'def kernel(X:N.Tensor({shape},N.f32), Y:N.Tensor({shape},N.f32,access="write"),',
        "           q:N.int64, lo:N.int64, hi:N.int64):",
        '    with N.Kernel(1, kind="vector"):',
        f"        work=N.alloc_ub({shape},N.f32)",
        "        N.copy(X,work)",
        *(f"        {coordinate}" for coordinate in coords),
        f"        N.fill(work,-7.0,mask={expression})",
        "        N.copy(work,Y)",
    ]
    source = N.from_source("\n".join(body), disjoint_external=True)
    result = compile(source, timeout_s=120)
    module = parse(result.source_ir)
    fill = next(
        operation
        for operation in module.walk()
        if operation.name == "npu_source.element" and operation.kind.data == "fill"
    )
    op = fill.mask.owner
    task = next(
        template
        for template in collect_native_memory(result.plan).templates
        if template.spec.native == "mask_fill"
    )
    attributes = task.operation.attributes
    return (
        op,
        op.program(),
        [value.owner.sym_name.data for value in op.scalars],
        attributes["mask_word"].data,
        attributes["mask_repeat"].data,
        attributes["mask_iteration"].data,
    )


def _row_fill_asc(shape, expression):
    """Compile the masked fill at an arbitrary shape and return its ASC."""
    rank = len(shape)
    coords = []
    for axis, extent in enumerate(shape):
        parts = ["None"] * rank
        parts[axis] = ":"
        coords.append(f"i{axis}=N.arange({extent})[{', '.join(parts)}]")
    source = N.from_source(
        "\n".join(
            [
                f'def kernel(X:N.Tensor({shape},N.f32), Y:N.Tensor({shape},N.f32,access="write"),',
                "           q:N.int64, lo:N.int64, hi:N.int64):",
                '    with N.Kernel(1, kind="vector"):',
                f"        work=N.alloc_ub({shape},N.f32)",
                "        N.copy(X,work)",
                *(f"        {coordinate}" for coordinate in coords),
                f"        N.fill(work,-7.0,mask={expression})",
                "        N.copy(work,Y)",
            ]
        ),
        disjoint_external=True,
    )
    return compile(source, timeout_s=120).asc


def _data_predicate_program():
    return N.from_source(
        """
        def kernel(X:N.Tensor((8,128),N.f32), T:N.Tensor((8,128),N.f32),
                   Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                other=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                bits2=N.alloc_mask((8,128))
                bits3=N.alloc_mask((8,128))
                N.copy(X,work)
                N.copy(T,other)
                p=N.compare(work,0.5,kind="gt",out=bits)
                N.mul(work,2.0,out=work)
                q=N.compare(work,other,kind="lt",out=bits2)
                both=N.mask_and(p,q,out=bits3)
                N.fill(other,3.0)
                N.where(both,work,other,out=work)
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )


def _reuse_program(*, consume_before_reuse):
    first = "N.fill(work,-9.0,mask=p,scratch=temp)" if consume_before_reuse else ""
    return N.from_source(
        f"""
        def kernel(X:N.Tensor((8,128),N.f32), Y:N.Tensor((8,128),N.f32,access="write")):
            with N.Kernel(1, kind="vector"):
                work=N.alloc_ub((8,128),N.f32)
                temp=N.alloc_ub((8,128),N.f32)
                bits=N.alloc_mask((8,128))
                N.copy(X,work)
                p=N.compare(work,0.5,kind="gt",out=bits)
                {first}
                q=N.compare(work,1.5,kind="gt",out=bits)
                N.fill(work,-1.0,mask=q,scratch=temp)
                {"N.fill(work,-9.0,mask=p,scratch=temp)" if not consume_before_reuse else ""}
                N.copy(work,Y)
        """,
        disjoint_external=True,
    )
