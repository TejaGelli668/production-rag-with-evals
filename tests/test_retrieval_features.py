import numpy as np

from rag.config import RetrieverConfig
from rag.data.financebench import Document
from rag.query import QueryAnalyzer
from rag.retrieve import SearchRetriever
from rag.schema import Chunk, RetrievedChunk
from rag.sparse import BM25Index, rrf_fuse


def doc(company: str) -> Document:
    return Document(
        doc_name=f"{company}_2020_10K",
        company=company,
        gics_sector="X",
        doc_type="10k",
        doc_period=2020,
        doc_link="https://example.com",
    )


ANALYZER = QueryAnalyzer([doc(c) for c in ["3M", "Johnson & Johnson", "Block", "Apple", "Netflix"]])


def test_company_aliases_and_case_sensitive_names():
    assert ANALYZER.companies("Are JnJ's FY2022 financials high growth?") == ("Johnson & Johnson",)
    assert ANALYZER.companies("Block's (formerly known as Square) working capital") == ("Block",)
    assert ANALYZER.companies("a block of shares and an apple") == ()
    assert ANALYZER.companies("Compare 3M and Netflix") == ("3M", "Netflix")


def test_fiscal_year_prefers_explicit_fiscal_years():
    assert ANALYZER.fiscal_year("What is the FY2018 capex for 3M?") == 2018
    assert ANALYZER.fiscal_year("As of FY 2021, how much will it pay in 2024?") == 2021
    assert ANALYZER.fiscal_year("change from FY2015 to FY2016") == 2016
    assert ANALYZER.fiscal_year("biggest drop in Q22023 revenues") == 2023
    assert ANALYZER.fiscal_year("FY'19 margin") == 2019
    assert ANALYZER.fiscal_year("What industry is it in?") is None


def test_filter_attempts_relax_from_strict_to_none():
    attempts = ANALYZER.filters("3M FY2018 capex", "company_year")
    assert attempts == [{"company": ["3M"], "fiscal_year": 2018}, {"company": ["3M"]}, {}]
    assert ANALYZER.filters("NVIDIA revenue in 2022", "company_year") == [{}]
    assert ANALYZER.filters("3M FY2018 capex", "none") == [{}]


def chunk(cid: str, text: str, company: str = "3M", year: int = 2018) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_name=f"{company}_{year}",
        chunk_index=0,
        text=text,
        page_start=0,
        page_end=0,
        company=company,
        doc_type="10k",
        fiscal_year=year,
        gics_sector="X",
    )


def test_bm25_matches_numbers_with_or_without_commas_and_applies_filters():
    index = BM25Index.build(
        [
            chunk("a", "Purchases of property, plant and equipment (1,577)"),
            chunk("b", "Net cash used in investing activities"),
            chunk("c", "Purchases of property and equipment 1577", company="Netflix"),
        ]
    )
    hits = index.search("capex 1577 property purchases", top_k=3)
    assert {h.chunk.chunk_id for h in hits} == {"a", "c"}
    filtered = index.search("property purchases 1577", top_k=3, filters={"company": ["3M"]})
    assert [h.chunk.chunk_id for h in filtered] == ["a"]


def test_rrf_rewards_agreement_between_rankings():
    a, b, c = chunk("a", ""), chunk("b", ""), chunk("c", "")
    dense = [RetrievedChunk(chunk=x, score=0, rank=i) for i, x in enumerate([a, b, c], 1)]
    sparse = [RetrievedChunk(chunk=x, score=0, rank=i) for i, x in enumerate([b, c], 1)]
    fused = rrf_fuse([dense, sparse])
    assert fused[0].chunk.chunk_id == "b"  # 2nd + 1st beats 1st alone and 3rd + 2nd
    assert [r.rank for r in fused] == [1, 2, 3]


class FakeStore:
    """Returns chunks only for filters it 'has' data for, recording each call."""

    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, vector, top_k, filters=None):
        self.calls.append(filters)
        return self.results.get(str(filters), [])[:top_k]


class FakeEmbedder:
    def embed_query(self, text):
        return np.zeros(3)


def test_retriever_relaxes_filters_until_enough_results():
    hits = [RetrievedChunk(chunk=chunk(str(i), "t"), score=1.0, rank=i) for i in range(1, 4)]
    store = FakeStore({str({"company": ["3M"]}): hits})
    cfg = RetrieverConfig(top_k=2, filters="company_year")
    out = SearchRetriever(cfg, FakeEmbedder(), store, ANALYZER).retrieve("3M FY2018 capex")
    assert store.calls == [{"company": ["3M"], "fiscal_year": 2018}, {"company": ["3M"]}]
    assert [r.rank for r in out] == [1, 2]


def test_explicit_filters_override_inferred_ones():
    store = FakeStore({})
    cfg = RetrieverConfig(top_k=2, filters="company_year")
    SearchRetriever(cfg, FakeEmbedder(), store, ANALYZER).retrieve("3M FY2018", {"company": "X"})
    assert store.calls == [{"company": "X"}]


class ScriptedLLM:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, system, user, json_schema=None):
        from rag.schema import LLMResponse

        self.calls += 1
        return LLMResponse(
            text=self.text,
            model="m",
            input_tokens=1,
            output_tokens=1,
            stop_reason="stop",
            latency_s=0.0,
        )


def test_rewriter_dedupes_caps_and_survives_bad_output():
    from rag.rewrite import QueryRewriter

    llm = ScriptedLLM(
        '{"queries": ["3M capex 2018", "3m CAPEX 2018", "3M revenue 2018", "x", "y"]}'
    )
    assert QueryRewriter(llm, max_queries=2).rewrite("Q?") == ["3M capex 2018", "3M revenue 2018"]
    assert QueryRewriter(ScriptedLLM("not json")).rewrite("Q?") == []


class QueryStore:
    """Each query 'finds' a different chunk; records the filters used."""

    def __init__(self):
        self.filters = []

    def search(self, vector, top_k, filters=None):
        self.filters.append(filters)
        cid = str(int(vector[0]))
        return [RetrievedChunk(chunk=chunk(cid, cid), score=1.0, rank=1)]


class QueryEmbedder:
    def embed_query(self, text):
        return np.array([{"orig": 1, "sub a": 2, "sub b": 3}[text]])


def test_rewrites_pool_results_under_the_original_questions_filters():
    from rag.rewrite import QueryRewriter

    rewriter = QueryRewriter(ScriptedLLM('{"queries": ["sub a", "sub b"]}'))
    cfg = RetrieverConfig(top_k=3, candidates=5)
    retriever = SearchRetriever(cfg, QueryEmbedder(), QueryStore(), rewriter=rewriter)
    out = retriever.retrieve("orig", filters={"company": ["3M"]})
    assert {r.chunk.chunk_id for r in out} == {"1", "2", "3"}
    assert retriever.store.filters == [{"company": ["3M"]}] * 3
    assert retriever.last_queries == ["orig", "sub a", "sub b"]


def test_stepwise_prompt_is_selected_by_config():
    from rag.config import GeneratorConfig, PipelineConfig
    from rag.generate.prompts import STEPWISE_SYSTEM_PROMPT
    from rag.pipeline import RAGPipeline
    from rag.retrieve import NoRetriever

    cfg = PipelineConfig(name="x", generator=GeneratorConfig(prompt="stepwise"))
    assert RAGPipeline(cfg, NoRetriever(), ScriptedLLM("")).system_prompt == STEPWISE_SYSTEM_PROMPT
