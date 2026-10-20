"""One public source build per attention policy, shared within its test module."""

from pathlib import Path

from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.build import build_file
from ascendflow.dialects.io import parse


def compile_attention(tmp_path_factory, policy):
    directory = tmp_path_factory.mktemp("attention")
    source = Path(__file__).parents[2] / "examples/flash_attention/attention.py"
    build_file(
        source, directory, entry="attention_target", parameters=policy.parameters(), timeout_s=300
    )
    return (
        collect_native_memory(parse((directory / "after.xdsl").read_text())),
        collect_native_memory(parse((directory / "before.xdsl").read_text())),
        (directory / "attention_target.asc").read_text(),
        directory,
        policy,
    )
