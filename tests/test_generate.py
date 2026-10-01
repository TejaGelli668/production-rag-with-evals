from rag.generate.citations import cited_markers, resolve_citations
from rag.generate.prompts import build_user_prompt
from rag.schema import Chunk, RetrievedChunk


def retrieved(rank: int, doc: str, start: int, end: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        doc_name=doc,
        chunk_index=rank,
        text=f"text {rank}",
        page_start=start,
        page_end=end,
        company="Acme & Co",
        doc_type="10k",
        fiscal_year=2022,
        gics_sector="Industrials",
    )
    return RetrievedChunk(chunk=chunk, score=0.5, rank=rank)


def test_cited_markers_handles_groups_and_dedupes_in_order():
    text = "Capex was $1.5B [2]. Margin rose [1, 3] while revenue fell [2]. See note [10]."
    assert cited_markers(text) == [2, 1, 3, 10]


def test_cited_markers_ignores_non_numeric_brackets():
    assert cited_markers("Use [a] or [1a] or [] but not these") == []


def test_resolve_citations_maps_to_pages_and_drops_unknown_markers():
    sources = [retrieved(1, "A_10K", 4, 4), retrieved(2, "B_10K", 7, 8)]
    citations = resolve_citations("x [2] y [1] z [9]", sources)
    assert [(c.marker, c.doc_name, c.page_start, c.page_end) for c in citations] == [
        (2, "B_10K", 7, 8),
        (1, "A_10K", 4, 4),
    ]


def test_prompt_shows_one_based_pages_and_escapes_attributes():
    prompt = build_user_prompt("Q?", [retrieved(1, "A_10K", 0, 0), retrieved(2, "A_10K", 4, 5)])
    assert 'id="1"' in prompt and 'pages="1"' in prompt
    assert 'pages="5-6"' in prompt
    assert 'company="Acme &amp; Co"' in prompt
    assert prompt.rstrip().endswith("Question: Q?")
