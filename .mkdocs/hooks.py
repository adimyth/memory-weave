"""MkDocs hooks: rewrite links that point outside ``docs/`` at the pages that include those files.

The README, the changelog, and the design documents link to files at the repository root. On the site each
of those files is included by a page under ``docs/``, so the links are rewritten to site-absolute paths,
which ``validation.links.absolute_links: relative_to_docs`` then resolves and checks.
"""

from __future__ import annotations

import re
from typing import Any

_REWRITES = (
    (re.compile(r"\]\((?:\.\./)*BENCHMARK_HANDOFF\.md"), "](/benchmark-handoff.md"),
    (re.compile(r"\]\((?:\.\./)*benchmarks/README\.md"), "](/benchmarks.md"),
    (re.compile(r"\]\((?:\.\./)*CHANGELOG\.md"), "](/changelog.md"),
    (re.compile(r"\]\((?:\.\./)+README\.md"), "](/index.md"),
)


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    for pattern, replacement in _REWRITES:
        markdown = pattern.sub(replacement, markdown)
    return markdown
