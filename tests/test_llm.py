from types import SimpleNamespace

import pytest

from rag.config import GeneratorConfig, Settings
from rag.generate.llm import AnthropicLLM, LLMError, OllamaLLM


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def anthropic_llm(response) -> tuple[AnthropicLLM, FakeMessages]:
    llm = AnthropicLLM(
        GeneratorConfig(provider="anthropic", model="claude-sonnet-5-5"),
        Settings(anthropic_api_key="test-key"),
    )
    messages = FakeMessages(response)
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return llm, messages


def message(content, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-sonnet-5-5",
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )


def test_anthropic_joins_text_blocks_and_skips_thinking():
    llm, fake = anthropic_llm(
        message(
            [
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Capex was $1,577M [1]."),
            ]
        )
    )
    out = llm.complete("system", "user")
    assert out.text == "Capex was $1,577M [1]."
    assert (out.input_tokens, out.output_tokens) == (120, 30)
    assert fake.kwargs["fallbacks"] == "default"
    assert fake.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert fake.kwargs["output_config"] == {"effort": "medium"}
    assert "temperature" not in fake.kwargs


def test_anthropic_refusal_raises():
    llm, _ = anthropic_llm(message([], stop_reason="refusal"))
    with pytest.raises(LLMError):
        llm.complete("system", "user")


def test_ollama_strips_think_blocks():
    llm = OllamaLLM(GeneratorConfig(), Settings())
    payload = {
        "model": "qwen3:14b",
        "message": {"content": "<think>hmm</think>\n$5,466M [3]."},
        "prompt_eval_count": 10,
        "eval_count": 5,
        "done_reason": "stop",
    }
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    llm.client = SimpleNamespace(post=lambda *a, **k: response, base_url="http://x")
    assert llm.complete("s", "u").text == "$5,466M [3]."
