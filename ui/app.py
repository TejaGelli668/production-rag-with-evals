"""Streamlit front end for the RAG API.

    uv run streamlit run ui/app.py        # or: make ui   (needs `make serve` running)

Streams the answer token by token, shows each source with its filing, page and rerank score
(and whether the answer cited it), renders the cited PDF page, and records 👍/👎 feedback.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")
# Page images are loaded by the viewer's browser, which may not resolve the server-side
# hostname (e.g. `api` inside Docker Compose), so they get their own URL.
PUBLIC_API_URL = os.environ.get("RAG_API_PUBLIC_URL", API_URL)
EXAMPLES = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response "
    "based on the cash flow statement.",
    "What is Netflix's year end FY2017 total current liabilities (in USD millions)? Base your "
    "judgment on the balance sheet.",
    "How does Boeing's effective tax rate in FY2022 compare to FY2021?",
    "Are JnJ's FY2022 financials that of a high growth company?",
    "What was NVIDIA's total revenue in fiscal year 2022?",  # not in the corpus: should decline
]

st.set_page_config(page_title="FinanceBench RAG", page_icon="📄", layout="wide")


def sse_events(question: str) -> Iterator[tuple[str, dict]]:
    """POST /ask/stream and yield (event, data) pairs as they arrive."""
    with httpx.stream(
        "POST", f"{API_URL}/ask/stream", json={"question": question}, timeout=600
    ) as r:
        r.raise_for_status()
        event = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event:
                yield event, json.loads(line.removeprefix("data: "))
                event = None


def health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=10).json()
    except httpx.HTTPError:
        return None


def send_feedback(request_id: str, rating: int) -> None:
    try:
        httpx.post(
            f"{API_URL}/feedback", json={"request_id": request_id, "rating": rating}, timeout=10
        )
        st.session_state.feedback[request_id] = rating
    except httpx.HTTPError:
        st.toast("Couldn't send feedback: is the API running?")


def render_sources(sources: list[dict], show_text: bool) -> None:
    for s in sources:
        badge = "✅ cited" if s.get("cited") else ""
        title = (
            f"[{s['rank']}] {s['company']} · {s['doc_name']} · {s['page_label']} · "
            f"score {s['score']:.2f} {badge}"
        )
        with st.expander(title, expanded=bool(s.get("cited"))):
            cols = st.columns([3, 2]) if show_text else [st.container()]
            if show_text:
                cols[0].text(s["text"][:3000])
            cols[-1].image(
                f"{PUBLIC_API_URL}/pages/{s['doc_name']}/{s['page_start']}.png",
                caption=f"{s['doc_name']}, {s['page_label']}",
            )


def render_answer(turn: dict, show_text: bool) -> None:
    answer = turn["answer"]
    if answer["refused"]:
        st.warning(answer["answer"])
    else:
        st.markdown(answer["answer"])
    t, u = answer["timings"], answer["usage"]
    st.caption(
        f"retrieval {t['retrieval_s']:.1f}s · generation {t['generation_s']:.1f}s · "
        f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens · "
        f"{answer['model']} · config `{answer['config']}`"
    )
    rid = answer["request_id"]
    given = st.session_state.feedback.get(rid)
    # st.feedback("thumbs") returns 0 for 👎 and 1 for 👍; the API takes -1 / 1.
    st.feedback(
        "thumbs",
        key=f"fb-{rid}",
        disabled=given is not None,
        on_change=lambda: send_feedback(rid, 1 if st.session_state[f"fb-{rid}"] else -1),
    )
    if given is not None:
        st.caption("Thanks for the feedback.")
    st.markdown("**Sources**")
    render_sources(answer["sources"], show_text)


st.session_state.setdefault("history", [])
st.session_state.setdefault("feedback", {})

with st.sidebar:
    st.header("FinanceBench RAG")
    st.caption(
        "Questions over 360 SEC filings (10-K, 10-Q, 8-K) from 40 companies, "
        "answered with cited pages by a local model."
    )
    info = health()
    if info is None:
        st.error(f"API not reachable at {API_URL}. Start it with `make serve`.")
    else:
        status = "🟢" if info["status"] == "ok" else "🟠"
        st.markdown(f"{status} **{info['config']}** · {info['llm_model']}")
        if info.get("chunks"):
            st.caption(f"{info['chunks']:,} chunks indexed")
        if not info["llm_reachable"]:
            st.warning("The LLM isn't reachable. Is Ollama running?")
    show_text = st.toggle("Show retrieved text", value=False)
    st.subheader("Try a question")
    example = None
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"ex-{i}", use_container_width=True):
            example = q
    if st.button("Clear conversation"):
        st.session_state.history = []

if not st.session_state.history:
    st.title("Ask SEC filings, get cited answers")
    st.markdown(
        "Answers come from 360 10-K, 10-Q and 8-K filings of 40 companies, retrieved with "
        "company and fiscal-year filters and a cross-encoder reranker, then written by a "
        "local model that must cite the page it used, or say it lacks the information. "
        "On FinanceBench's held-out questions it answers **49%** correctly vs 15% for plain "
        "vector search."
    )
    st.markdown("**Try one of these:**")
    for i, q in enumerate(EXAMPLES):
        if st.button(q, key=f"main-ex-{i}"):
            example = q

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn, show_text)

question = st.chat_input("Ask about a company's filings, e.g. 3M's FY2018 capex") or example
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Searching filings…", expanded=False)
        answer_box = st.empty()
        text, final = "", None
        try:
            for event, data in sse_events(question):
                if event == "sources":
                    status.update(label=f"Found {len(data['sources'])} passages · writing answer…")
                elif event == "token":
                    text += data["text"]
                    answer_box.markdown(text + "▌")
                elif event == "answer":
                    final = data
                elif event == "error":
                    st.error(data["detail"])
        except httpx.HTTPError as e:
            st.error(f"Request failed: {e}")
        status.update(label="Done", state="complete")
        answer_box.empty()
        if final:
            turn = {"question": question, "answer": final}
            st.session_state.history.append(turn)
            render_answer(turn, show_text)
