"""Rule-based query analysis: which company and fiscal year a question is about.

Used to turn a question into metadata filters (E5). On FinanceBench `dev`, every
question names its company, and the latest fiscal year mentioned equals the gold
filing's year in 46 of 50 questions. Years written as "FY2021" or "fiscal year
2021" win over bare years, so "As of FY2021, ... retirees in 2024?" targets 2021.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.data.financebench import Document

# Extra names people use, keyed by the corpus's canonical company name.
ALIASES: dict[str, list[str]] = {
    "AES Corporation": ["AES"],
    "Activision Blizzard": ["Activision"],
    "American Express": ["AmEx", "Amex"],
    "Block": ["Square"],
    "CVS Health": ["CVS"],
    "Coca-Cola": ["Coca Cola", "CocaCola", "Coke"],
    "Foot Locker": ["Footlocker"],
    "JPMorgan": ["JP Morgan", "JPMorgan Chase", "JPM"],
    "Johnson & Johnson": ["JnJ", "J&J", "Johnson and Johnson"],
    "MGM Resorts": ["MGM"],
    "McDonalds": ["McDonald's", "McDonald\u2019s", "McDonald"],
    "PG&E Corporation": ["PG&E"],
    "Paypal": ["PayPal"],
    "PepsiCo": ["Pepsi"],
    "Ulta Beauty": ["Ulta"],
}
# Common English words that are also company names: only match them capitalized.
CASE_SENSITIVE = {"Block", "Square", "Oracle", "Apple", "Corning", "Intel"}

_FY_YEAR = re.compile(r"\b(?:FY|fiscal\s+year|fiscal)\s?'?(20\d{2}|\d{2})\b", re.I)
_QUARTER_YEAR = re.compile(r"\bQ[1-4]\s?(?:of\s+)?(?:FY\s?)?(20\d{2})", re.I)
_BARE_YEAR = re.compile(r"\b(20[12]\d)\b")


@dataclass(frozen=True)
class QueryInfo:
    companies: tuple[str, ...]
    fiscal_year: int | None


class QueryAnalyzer:
    def __init__(self, documents: list[Document]):
        names = sorted({d.company for d in documents})
        patterns = []
        for name in names:
            for surface in [name, *ALIASES.get(name, [])]:
                flags = 0 if surface in CASE_SENSITIVE else re.I
                patterns.append((name, re.compile(rf"(?<!\w){re.escape(surface)}(?!\w)", flags)))
        self._patterns = patterns

    def companies(self, question: str) -> tuple[str, ...]:
        found = {name for name, pattern in self._patterns if pattern.search(question)}
        return tuple(sorted(found))

    @staticmethod
    def fiscal_year(question: str) -> int | None:
        def to_year(y: str) -> int:
            return int(y) if len(y) == 4 else 2000 + int(y)

        explicit = [to_year(y) for y in _FY_YEAR.findall(question)]
        explicit += [int(y) for y in _QUARTER_YEAR.findall(question)]
        if explicit:
            return max(explicit)
        bare = [int(y) for y in _BARE_YEAR.findall(question)]
        return max(bare) if bare else None

    def analyze(self, question: str) -> QueryInfo:
        return QueryInfo(self.companies(question), self.fiscal_year(question))

    def filters(self, question: str, mode: str) -> list[dict[str, str | int | list[str]]]:
        """Filter sets to try in order, strictest first; the retriever falls back down the list.

        mode: "none" | "company" | "company_year".
        """
        if mode == "none":
            return [{}]
        info = self.analyze(question)
        if not info.companies:
            return [{}]
        company = {"company": list(info.companies)}
        attempts: list[dict[str, str | int | list[str]]] = []
        if mode == "company_year" and info.fiscal_year:
            attempts.append(company | {"fiscal_year": info.fiscal_year})
        attempts += [company, {}]
        return attempts
