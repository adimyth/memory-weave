"""Rewrite links to the README and changelog when their content is included in the documentation site.

The site exposes those root files through pages under ``docs/``. Rewriting their relative links to site paths lets MkDocs resolve and validate them after inclusion.
"""

from __future__ import annotations

import re
from typing import Any

_REWRITES = (
    (re.compile(r"\]\((?:\.\./)*CHANGELOG\.md"), "](/changelog.md"),
    (re.compile(r"\]\((?:\.\./)+README\.md"), "](/index.md"),
)


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    for pattern, replacement in _REWRITES:
        markdown = pattern.sub(replacement, markdown)
    return markdown
