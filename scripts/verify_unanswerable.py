"""Check each hand-drafted unanswerable case against the corpus (metadata and parsed text).

    uv run python scripts/verify_unanswerable.py     # needs `make data` and the PyMuPDF parse cache

Absent companies must not be corpus companies; future periods must be later than the
company's latest filing; nonexistent items and undisclosed details must not appear in that
company's filings. Exits non-zero if any case fails.
"""

import json
import re
import sys
from collections import defaultdict

from rag.data.financebench import load_documents
from rag.evals.cases import load_cases
from rag.ingest.parse import parse_cached

# case number -> ("absent", name) | ("future", company, year) | ("phrase", company, regex)
CHECKS = {
    1: ("absent", "nvidia"),
    2: ("absent", "tesla"),
    3: ("absent", "alphabet"),
    4: ("absent", "meta platforms"),
    5: ("absent", "visa"),
    6: ("absent", "home depot"),
    7: ("absent", "starbucks"),
    8: ("absent", "exxon"),
    9: ("absent", "goldman sachs"),
    10: ("absent", "unitedhealth"),
    11: ("future", "3M", 2027),
    12: ("future", "Netflix", 2028),
    13: ("future", "Boeing", 2026),
    14: ("future", "Apple", 2027),
    15: ("future", "PepsiCo", 2026),
    16: ("future", "Microsoft", 2028),
    17: ("future", "Verizon", 2026),
    18: ("future", "Walmart", 2029),
    19: ("phrase", "Netflix", r"theme park"),
    20: ("phrase", "Costco", r"airline"),
    21: ("phrase", "Microsoft", r"automotive manufactur"),
    22: ("phrase", "Nike", r"hotel"),
    23: ("phrase", "Coca-Cola", r"semiconductor"),
    24: ("phrase", "Adobe", r"crude oil"),
    25: ("phrase", "Verizon", r"theme restaurant"),
    26: ("phrase", "McDonalds", r"cups of coffee"),
    27: ("phrase", "Walmart", r"highest[- ]grossing store"),
    28: ("phrase", "Amazon", r"ohio.{0,80}hourly|hourly.{0,80}ohio"),
    29: ("phrase", "Microsoft", r"xbox series x.{0,120}units|units.{0,120}xbox series x"),
    30: ("phrase", "Best Buy", r"largest (single )?customer"),
}


def main() -> int:
    docs = load_documents()
    companies = {d.company.lower() for d in docs}
    by_company = defaultdict(list)
    for d in docs:
        by_company[d.company].append(d)

    def text(doc) -> str:
        return " ".join(p.text for p in parse_cached("pymupdf", doc.pdf_path)).lower()

    failures = 0
    for case in load_cases("unanswerable"):
        check = CHECKS[int(case.case_id.split("_")[1])]
        if check[0] == "absent":
            ok = not any(check[1] in c for c in companies)
            note = "not a corpus company" if ok else "IS a corpus company"
        elif check[0] == "future":
            latest = max(d.doc_period for d in by_company[check[1]])
            ok = latest < check[2]
            note = f"asks FY{check[2]}, latest filing FY{latest}"
        else:
            found = [d.doc_name for d in by_company[check[1]] if re.search(check[2], text(d))]
            ok = not found
            note = f"{check[2]!r} in {len(found)} of {len(by_company[check[1]])} filings"
        failures += not ok
        print(f"{'✓' if ok else '✗'} {case.case_id} [{case.tags[0]}] {note}")
    print(json.dumps({"cases": len(CHECKS), "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
