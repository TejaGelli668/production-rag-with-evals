"""PDF parsing into per-page text."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf

from rag.schema import Page

Parser = Callable[[Path], list[Page]]


def parse_pymupdf(path: Path) -> list[Page]:
    """Plain text extraction, one entry per page (including empty pages, to keep indices)."""
    with pymupdf.open(path) as pdf:
        return [Page(page_num=i, text=page.get_text()) for i, page in enumerate(pdf)]


PARSERS: dict[str, Parser] = {"pymupdf": parse_pymupdf}
