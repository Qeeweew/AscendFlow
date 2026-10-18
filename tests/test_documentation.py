"""Repository documentation must keep its local navigation usable."""

import re
from pathlib import Path
from urllib.parse import unquote


def test_local_documentation_links_resolve():
    root = Path(__file__).resolve().parents[1]
    pages = [
        root / name
        for name in ("README.md", "examples/README.md", "benchmarks/README.md", "tests/README.md")
    ]
    pages.extend((root / "docs").rglob("*.md"))
    failures = []
    for page in pages:
        for href in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", page.read_text()):
            if href.startswith(("https:", "http:", "mailto:", "#")):
                continue
            path = Path(unquote(href.split("#", 1)[0]))
            if path.is_absolute() or not (page.parent / path).exists():
                failures.append(f"{page.relative_to(root)}: {href}")
    assert not failures, "broken or nonportable documentation links:\n" + "\n".join(failures)
