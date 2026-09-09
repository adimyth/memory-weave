"""MkDocs hooks: rewrite links that point outside ``docs/`` at the pages that include those files.

The README and the design documents link to three files at the repository root. On the site each of them is
included by a page under ``docs/``, so the links are rewritten to those pages instead of being reported as
unresolved by strict mode.
"""

from __future__ import annotations

from typing import Any

_REWRITES = (
    ("../BENCHMARK_HANDOFF.md", "benchmark-handoff.md"),
    ("../benchmarks/README.md", "benchmarks.md"),
    ("../README.md", "index.md"),
    ("BENCHMARK_HANDOFF.md", "benchmark-handoff.md"),
    ("benchmarks/README.md", "benchmarks.md"),
)


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    for old, new in _REWRITES:
        markdown = markdown.replace(f"]({old}", f"]({new}")
    return markdown
