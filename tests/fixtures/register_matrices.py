"""Small real FP32 GEMMs requiring different FMATRIX register values."""

from ascendflow import language as N


def register_matrices(*, branch=False):
    sizes = (16, 32, 16)
    arguments = []
    allocations, operations = [], []
    for index, size in enumerate(sizes):
        arguments.extend(
            (
                f"A{index}:N.Tensor(({size},{size}),N.f32)",
                f"B{index}:N.Tensor(({size},{size}),N.f32)",
                f"Y{index}:N.Tensor((n,{size},{size}),N.f32,access='write',write_disjoint=True)",
            )
        )
        for name, bank in (("a1", "l1"), ("b1", "l1"), ("a0", "l0a"), ("b0", "l0b"), ("c0", "l0c")):
            allocations.append(f"        {name}_{index}=N.alloc_{bank}(({size},{size}),N.f32)\n")
        operations.append(
            [
                f"N.copy(A{index},a1_{index})",
                f"N.copy(B{index},b1_{index})",
                f"N.copy(a1_{index},a0_{index})",
                f"N.copy(b1_{index},b0_{index})",
                f"N.matmul(a0_{index},b0_{index},accum=c0_{index},initialize=True)",
                f"N.copy(c0_{index},Y{index}[i,:,:])",
            ]
        )
    arguments.extend(("n:N.i64(minimum=0)", "choice:N.i64(minimum=0)"))
    source = "def matrices(" + ",".join(arguments) + "):\n    with N.Kernel(1,kind='cube'):\n"
    source += "".join(allocations) + "        for i in N.serial(n):\n"
    for index, body in enumerate(operations):
        indent = "            "
        if branch and index == 1:
            source += indent + "if choice==0:\n"
            indent += "    "
        source += "".join(indent + line + "\n" for line in body)
    return N.from_source(source, disjoint_external=True)
