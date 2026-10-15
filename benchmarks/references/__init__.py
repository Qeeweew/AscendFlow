"""Handwritten baselines; not part of the compiler package."""

from pathlib import Path


def ascendc_source(name: str) -> str:
    return (Path(__file__).parent / "ascendc" / name).read_text()
