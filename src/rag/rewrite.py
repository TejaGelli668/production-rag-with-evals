"""LLM query rewriting (E7).

A question like "What is the FY2019 cash conversion cycle?" needs several line items
(inventory, receivables, payables, COGS, revenue) that may sit on different pages.
One embedding of the whole question tends to land near one of them. The rewriter
asks for short search queries, one per thing to look up; their results are pooled
with the original query's and reranked against the original question.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from rag.generate.llm import LLM, LLMError

REWRITE_SYSTEM = """\
You write search queries for a search engine over companies' SEC filings (10-K, 10-Q, \
8-K, earnings releases). Given a question, list the separate things that must be \
looked up to answer it, as short keyword-style queries.

- One query per distinct item: a line item, a disclosure, or a section.
- Name the company and fiscal period in every query, e.g. \
"3M FY2018 purchases of property plant and equipment cash flow statement".
- Use the wording a filing would use (e.g. "accounts payable", "total current \
liabilities", "net revenue"), not the question's wording.
- Do not answer the question.\
"""


class _Rewrites(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queries: list[str] = Field(description="Search queries, most important first")


class QueryRewriter:
    def __init__(self, llm: LLM, max_queries: int = 3):
        self.llm = llm
        self.max_queries = max_queries

    def rewrite(self, question: str) -> list[str]:
        """Up to `max_queries` extra queries; [] if the model's output is unusable."""
        try:
            response = self.llm.complete(
                REWRITE_SYSTEM,
                f"Question: {question}\n\nGive at most {self.max_queries} queries.",
                json_schema=_Rewrites.model_json_schema(),
            )
            queries = _Rewrites.model_validate(json.loads(response.text)).queries
        except (LLMError, json.JSONDecodeError, ValueError):
            return []
        seen = {question.strip().lower()}
        unique = []
        for q in (q.strip() for q in queries):
            if q and q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)
        return unique[: self.max_queries]
