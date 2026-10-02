"""PDF parsing into per-page text, with an on-disk cache per (parser, document)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pymupdf

from rag.data.financebench import PROJECT_ROOT
from rag.schema import Page

Parser = Callable[[Path], list[Page]]
CACHE_DIR = PROJECT_ROOT / "storage" / "parsed"


def parse_pymupdf(path: Path) -> list[Page]:
    """Plain text extraction, one entry per page (including empty pages, to keep indices).

    Table cells come out one per line, so a row's label and its values get separated.
    """
    with pymupdf.open(path) as pdf:
        return [Page(page_num=i, text=page.get_text()) for i, page in enumerate(pdf)]


_MD_NOISE = re.compile(r"\*\*|<br>")


def parse_pymupdf4llm(path: Path) -> list[Page]:
    """Markdown extraction that keeps table rows together: `| Label | 1,577 | 1,373 |`.

    Bold markers and <br> tags are stripped: they cost tokens and carry no meaning here.
    """
    import pymupdf4llm  # deferred: slow import, only needed for this parser

    with pymupdf.open(path) as pdf:
        n_pages = pdf.page_count
    texts = {
        c["metadata"]["page_number"] - 1: _MD_NOISE.sub("", c["text"])
        for c in pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    }
    return [Page(page_num=i, text=texts.get(i, "")) for i in range(n_pages)]


PARSERS: dict[str, Parser] = {"pymupdf": parse_pymupdf, "pymupdf4llm": parse_pymupdf4llm}


def parse_cached(parser: str, path: Path, cache_dir: Path = CACHE_DIR) -> list[Page]:
    """Parse with `parser`, reusing an earlier result for the same file if present."""
    cache = cache_dir / parser / f"{path.stem}.json"
    if cache.exists():
        return [Page.model_validate(p) for p in json.loads(cache.read_text())]
    pages = PARSERS[parser](path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps([p.model_dump() for p in pages]))
    tmp.replace(cache)
    return pages
