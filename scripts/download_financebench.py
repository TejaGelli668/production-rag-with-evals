"""Download FinanceBench questions, document metadata, and PDFs at a pinned commit.

Usage:
    uv run python scripts/download_financebench.py --mode focused   # 84 PDFs
    uv run python scripts/download_financebench.py --mode full      # 360 PDFs (~700 MB)

Every PDF is checked against its git blob SHA from the upstream tree, so a
re-run skips files that are already intact and re-fetches any that are corrupt.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from rag.data.financebench import (
    DATA_DIR,
    DOCUMENTS_PATH,
    PDF_DIR,
    PINNED_SHA,
    QUESTIONS_PATH,
    REPO,
    CorpusMode,
    load_documents,
    load_questions,
    select_documents,
)

RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{PINNED_SHA}"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{PINNED_SHA}?recursive=1"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_pdf_shas(client: httpx.Client) -> dict[str, str]:
    """Map `pdfs/<name>.pdf` -> git blob SHA for the pinned commit."""
    headers = {}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    resp = client.get(TREE_URL, headers=headers)
    resp.raise_for_status()
    tree = resp.json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing was truncated; cannot verify all files.")
    return {e["path"]: e["sha"] for e in tree["tree"] if e["path"].startswith("pdfs/")}


def download(client: httpx.Client, url: str, dest: Path, expected_sha: str | None) -> str:
    """Download `url` to `dest` atomically. Returns 'skipped' or 'downloaded'."""
    if dest.exists() and expected_sha and git_blob_sha(dest.read_bytes()) == expected_sha:
        return "skipped"
    resp = client.get(url)
    resp.raise_for_status()
    data = resp.content
    if expected_sha and git_blob_sha(data) != expected_sha:
        raise ValueError(f"checksum mismatch for {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", type=CorpusMode, choices=list(CorpusMode), default="focused")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for remote, local in [
            ("data/financebench_open_source.jsonl", QUESTIONS_PATH),
            ("data/financebench_document_information.jsonl", DOCUMENTS_PATH),
        ]:
            download(client, f"{RAW_BASE}/{remote}", local, expected_sha=None)

        docs = select_documents(args.mode, load_documents(), load_questions())
        shas = fetch_pdf_shas(client)
        missing = [d.doc_name for d in docs if f"pdfs/{d.doc_name}.pdf" not in shas]
        if missing:
            print(f"error: no upstream PDF for {missing}", file=sys.stderr)
            return 1

        counts = {"skipped": 0, "downloaded": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    download,
                    client,
                    f"{RAW_BASE}/pdfs/{d.doc_name}.pdf",
                    d.pdf_path,
                    shas[f"pdfs/{d.doc_name}.pdf"],
                ): d.doc_name
                for d in docs
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="PDFs", unit="pdf"):
                try:
                    counts[fut.result()] += 1
                except Exception as e:
                    counts["failed"] += 1
                    tqdm.write(f"failed {futures[fut]}: {e}")

    total_mb = sum(d.pdf_path.stat().st_size for d in docs if d.pdf_path.exists()) / 1e6
    print(
        f"mode={args.mode} docs={len(docs)} downloaded={counts['downloaded']} "
        f"skipped={counts['skipped']} failed={counts['failed']} "
        f"size={total_mb:.0f}MB -> {DATA_DIR}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
