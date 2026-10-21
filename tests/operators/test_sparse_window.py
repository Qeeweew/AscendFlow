"""Paged window import reads exactly the visible rows, including NZ overlays."""

from dataclasses import replace

import pytest

from ascendflow import compile
from ascendflow import language as N
from ascendflow.analysis.structured_memory import collect_native_memory
from ascendflow.diagnostics import CompileError
from ascendflow.dialects.io import parse
from examples.sparse_attention.gather import copy_window_pages
from tests.models.structured_events import check_structured_events


@N.prim_func(disjoint_external=True)
def window_import(
    ori_kv: N.Tensor((2, 128, 1, 128), N.f16),
    Table: N.Tensor((1, 2), N.i32),
    length: N.int64,
    requested: N.int64,
):
    with N.Kernel(kind="cube"):
        x = N.alloc_l1((128, 128), N.f16)
        a0 = N.alloc_l0a((128, 128), N.f16)
        N.fill(x, 0.0)
        length = N.min(length, 256)
        visible = N.min(N.min(requested, 128), length)
        copy_window_pages(
            ori_kv,
            Table,
            x,
            batch=0,
            length=length,
            visible=visible,
            feature=0,
            PAGE=128,
            C=128,
            MAX_SEQ=256,
            ORI_BLOCKS=2,
        )
        N.copy(x, a0)


@pytest.fixture(scope="module")
def artifact():
    result = compile(window_import)
    return collect_native_memory(result.plan), collect_native_memory(parse(result.before_ir))


def test_window_overlay_cannot_initialize_untouched_l1_rows():
    source = window_import.source.replace("        N.fill(x, 0.0)\n", "")
    with pytest.raises(CompileError, match="initialized writer coverage"):
        compile(replace(window_import, source=source))


def test_l1_zero_does_not_accept_nonzero_values_or_clear_physical_gaps():
    for replacement, diagnostic in (
        ("N.fill(x, 1.0)", "L1 fill currently requires zero"),
        ("N.fill(x[:64, :], 0.0)", "contiguous NZ plane"),
    ):
        source = window_import.source.replace("N.fill(x, 0.0)", replacement)
        with pytest.raises(CompileError, match=diagnostic):
            compile(replace(window_import, source=source))


def test_visible_window_bytes_across_page_boundaries_and_empty_fragments(artifact, monkeypatch):
    from tests.models.structured_events import _Interpret

    original = _Interpret.native

    def identity_pages(self, operation, values, invocation, group, actor, coordinates):
        original(self, operation, values, invocation, group, actor, coordinates)
        if operation.native.data == "index_load":
            _, offset, _ = values[operation.refs[0]]
            values[operation.value] = offset // 4

    monkeypatch.setattr(_Interpret, "native", identity_pages)
    final, before = artifact
    for length, requested in (
        (0, 0),
        (128, 128),
        (129, 128),
        (131, 128),
        (191, 128),
        (255, 128),
        (191, 3),
        (200, 100),
    ):
        checked = check_structured_events(
            final,
            {"length": length, "requested": requested},
            logical_reference=before,
            invocations=2,
        )
        for invocation in (0, 1):
            spans = sorted(
                (span.start, span.end)
                for command in checked.natives
                if command.occurrence.invocation == invocation
                for span in command.spans
                if span.root.sym_name.data == "ori_kv"
            )
            visible = min(length, requested)
            assert sum(end - start for start, end in spans) == visible * 256
            cursor = (length - visible) * 256
            for start, end in spans:
                assert start == cursor
                cursor = end
            assert cursor == length * 256
