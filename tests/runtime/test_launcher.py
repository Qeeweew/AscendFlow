"""launcher contracts."""

import re
import subprocess

from ascendflow.backend.structured import _emit_owned
from ascendflow.passes.structured_sync import synchronize_structure
from examples.vector.scale import scale
from tests.support.asc_trace import compile_cpp


def test_launcher_forwards_stream_pointers_and_scalars_without_runtime_checks(tmp_path):
    asc = _emit_owned(synchronize_structure(scale))
    launcher = asc[asc.index('extern "C" int launch_') :]
    assert not re.search(r"\b(if|for|while)\b|npu_host_|npu_bytes|npu_ptrs|overflow", launcher)
    launcher = re.sub(
        r"  scale<<<[^\n]+",
        "  seen_stream = stream; seen_x = arg_X; seen_y = arg_Y; seen_tiles = npu_size_tiles; ++launches;",
        launcher,
    )
    source = (
        """#include <cstdint>
#include <cassert>
void *seen_stream, *seen_x, *seen_y;
int64_t seen_tiles;
int launches=0;
"""
        + launcher
        + r"""
int main() {
  void* stream=(void*)uintptr_t(4096);
  void* x=(void*)uintptr_t(8192);
  void* y=(void*)uintptr_t(16384);
  for (int64_t n : {int64_t(0), int64_t(1), int64_t(31)}) {
    assert(launch_scale(stream,x,y,n)==0);
    assert(seen_stream==stream && seen_x==x && seen_y==y && seen_tiles==n);
  }
  assert(launches==3);
}
"""
    )
    binary = compile_cpp(tmp_path, "#include <initializer_list>\n" + source, sanitize=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)
