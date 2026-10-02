"""Parse the corpus into the on-disk page cache (storage/parsed/<parser>/), in parallel.

Ingestion reuses this cache, so chunking experiments don't re-parse.

    uv run python scripts/preparse.py --parser pymupdf4llm --corpus full --workers 6
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from rag.data.financebench import CorpusMode, load_documents, load_questions, select_documents
from rag.ingest.parse import PARSERS, parse_cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parser", choices=list(PARSERS), default="pymupdf4llm")
    parser.add_argument("--corpus", type=CorpusMode, choices=list(CorpusMode), default="full")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    docs = select_documents(args.corpus, load_documents(), load_questions())
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work = pool.map(partial(parse_cached, args.parser), [d.pdf_path for d in docs])
        for i, _ in enumerate(work, start=1):
            if i % 20 == 0 or i == len(docs):
                print(f"{i}/{len(docs)} docs · {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
