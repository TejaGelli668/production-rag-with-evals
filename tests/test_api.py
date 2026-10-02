import json

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.config import PipelineConfig, Settings
from rag.generate.llm import LLMError
from rag.schema import Answer, Chunk, Citation, LLMResponse, RetrievedChunk


def chunk(rank: int) -> RetrievedChunk:
    c = Chunk(
        chunk_id=f"c{rank}",
        doc_name="3M_2018_10K",
        chunk_index=rank,
        text=f"text {rank}",
        page_start=59,
        page_end=59,
        company="3M",
        doc_type="10k",
        fiscal_year=2018,
        gics_sector="Industrials",
    )
    return RetrievedChunk(chunk=c, score=0.9, rank=rank)


class FakePipeline:
    def __init__(self, fail: bool = False):
        self.cfg = PipelineConfig(name="fake")
        self.retriever = object()  # no store attribute
        self.fail = fail
        self.calls = []

    def _answer(self, question):
        llm = LLMResponse(
            text="Capex was $1,577M [1].",
            model="qwen3:14b",
            input_tokens=100,
            output_tokens=10,
            stop_reason="stop",
            latency_s=1.5,
        )
        return Answer(
            question=question,
            text=llm.text,
            retrieved=[chunk(1), chunk(2)],
            citations=[Citation(marker=1, doc_name="3M_2018_10K", page_start=59, page_end=59)],
            llm=llm,
            retrieval_latency_s=0.2,
            config_name="fake",
            user_prompt="p",
        )

    def ask(self, question, filters=None):
        self.calls.append((question, filters))
        if self.fail:
            raise LLMError("cannot reach Ollama")
        return self._answer(question)

    def ask_stream(self, question, filters=None):
        if self.fail:
            raise LLMError("cannot reach Ollama")
        answer = self._answer(question)
        yield "sources", answer.retrieved
        yield "token", "Capex was "
        yield "token", "$1,577M [1]."
        yield "answer", answer


@pytest.fixture
def client(tmp_path):
    fake = FakePipeline()
    app = create_app(
        lambda: fake, log_dir=tmp_path, settings=Settings(ollama_host="http://127.0.0.1:9")
    )
    with TestClient(app) as c:
        c.fake, c.log_dir = fake, tmp_path
        yield c


def test_ask_returns_answer_sources_and_marks_cited(client):
    r = client.post("/ask", json={"question": "What was 3M's FY2018 capex?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Capex was $1,577M [1]." and not body["refused"]
    assert [(s["rank"], s["cited"], s["page_label"]) for s in body["sources"]] == [
        (1, True, "p. 60"),
        (2, False, "p. 60"),
    ]
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 10}
    logged = json.loads((client.log_dir / "requests.jsonl").read_text())
    assert logged["request_id"] == body["request_id"] and "text" not in logged["sources"][0]


def test_ask_passes_explicit_filters(client):
    client.post("/ask", json={"question": "capex?", "filters": {"company": ["3M"]}})
    assert client.fake.calls[-1] == ("capex?", {"company": ["3M"]})


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_ask_stream_emits_sources_tokens_then_answer(client):
    with client.stream("POST", "/ask/stream", json={"question": "3M capex?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())
    events = [
        (block.split("\n")[0].removeprefix("event: "), json.loads(block.split("\n")[1][6:]))
        for block in raw.strip().split("\n\n")
    ]
    assert [e for e, _ in events] == ["sources", "token", "token", "answer"]
    assert "".join(d["text"] for e, d in events if e == "token") == "Capex was $1,577M [1]."
    assert events[-1][1]["sources"][0]["cited"] is True
    assert events[0][1]["request_id"] == events[-1][1]["request_id"]


def test_llm_failure_is_503_and_stream_error_event(tmp_path):
    app = create_app(lambda: FakePipeline(fail=True), log_dir=tmp_path)
    with TestClient(app) as c:
        assert c.post("/ask", json={"question": "3M capex?"}).status_code == 503
        with c.stream("POST", "/ask/stream", json={"question": "3M capex?"}) as r:
            assert "event: error" in "".join(r.iter_text())


def test_feedback_is_logged(client):
    r = client.post("/feedback", json={"request_id": "abc", "rating": -1, "comment": "wrong"})
    assert r.status_code == 204
    row = json.loads((client.log_dir / "feedback.jsonl").read_text())
    assert (row["request_id"], row["rating"], row["comment"]) == ("abc", -1, "wrong")
    assert client.post("/feedback", json={"request_id": "a", "rating": 5}).status_code == 422


def test_health_reports_unreachable_llm_as_degraded(client):
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["llm_reachable"] is False
    assert body["config"] == "fake"


def test_pages_rejects_unknown_documents(client):
    assert client.get("/pages/..%2F..%2Fetc%2Fpasswd/0.png").status_code == 404
    assert client.get("/pages/NOPE_2020_10K/0.png").status_code == 404


def test_pages_renders_a_known_page(client):
    from rag.data.financebench import PDF_DIR

    if not (PDF_DIR / "3M_2018_10K.pdf").exists():
        pytest.skip("corpus not downloaded")
    r = client.get("/pages/3M_2018_10K/59.png")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get("/pages/3M_2018_10K/9999.png").status_code == 404
