"""Execute the actual host wrapper with only the device launch replaced by a counter."""

import ctypes
import re
import shutil
import subprocess

from ascendflow.dialects.io import parse
from ascendflow.dialects.physical import RuntimeSizeOp


def contracts(module, kind):
    return {
        op.sym_name.data: (op.minimum.data, op.multiple.data)
        for op in module.walk()
        if isinstance(op, kind)
    }


def host_launcher(artifact, directory):
    """Compile the actual wrapper, replacing only the device launch line."""
    marker = f'extern "C" int launch_{artifact.name}('
    launcher = artifact.asc[artifact.asc.index(marker) :]
    launcher, changed = re.subn(
        r"(?m)^  [A-Za-z_]\w*<<<[^\n]+>>>\([^\n]+\);$",
        "  ++npu_dispatches;",
        launcher,
    )
    assert changed == 1
    assert not re.search(r"\b(if|for|while)\b|npu_host_|overflow", launcher)
    source = directory / "launcher.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cstddef>\n"
        "static int npu_dispatches = 0;\n"
        + launcher
        + '\nextern "C" int dispatches() { return npu_dispatches; }\n'
    )
    compiler = shutil.which("c++")
    assert compiler is not None, "the CPU launcher-contract test requires a C++ compiler"
    binary = directory / "launcher.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", "-O2", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    library = ctypes.CDLL(str(binary))
    launch = getattr(library, "launch_" + artifact.name)
    parameters = contracts(parse(artifact.plan_ir), RuntimeSizeOp)
    root_count = launch_source_root_count(artifact)
    launch.argtypes = [ctypes.c_void_p] * (root_count + 1) + [ctypes.c_int64] * len(parameters)
    launch.restype = ctypes.c_int
    library.dispatches.restype = ctypes.c_int
    # The stub never dereferences these disjoint, aligned addresses.
    pointers = [ctypes.c_void_p((i + 1) << 32) for i in range(root_count)]
    return library, lambda *values: launch(None, *pointers, *values)


def launch_source_root_count(artifact):
    signature = artifact.asc.split(f'extern "C" int launch_{artifact.name}(', 1)[1].split(")", 1)[0]
    return signature.count("void* arg_")
