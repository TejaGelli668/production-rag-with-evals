"""Map the model's [n] citation markers back to documents and pages."""

from __future__ import annotations

import re

from rag.schema import Citation, RetrievedChunk

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_markers(text: str) -> list[int]:
    """All distinct [n] / [n, m] markers in order of first appearance."""
    seen: dict[int, None] = {}
    for group in _MARKER.findall(text):
        for n in group.split(","):
            seen.setdefault(int(n), None)
    return list(seen)


def resolve_citations(text: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Citations for markers that point at a real source; out-of-range markers are dropped."""
    by_rank = {rc.rank: rc.chunk for rc in retrieved}
    return [
        Citation(
            marker=n,
            doc_name=by_rank[n].doc_name,
            page_start=by_rank[n].page_start,
            page_end=by_rank[n].page_end,
        )
        for n in cited_markers(text)
        if n in by_rank
    ]
