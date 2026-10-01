"""LLM providers behind one interface: Claude (Anthropic API) and Ollama (local)."""

from __future__ import annotations

import re
import time
from typing import Protocol

import anthropic
import httpx

from rag.config import GeneratorConfig, Settings
from rag.schema import LLMResponse


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, cfg: GeneratorConfig, settings: Settings):
        self.cfg = cfg
        key = settings.anthropic_api_key
        # With no key, the SDK resolves credentials itself (env vars or an `ant auth` profile).
        self.client = (
            anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        # Thinking is adaptive by default on current models; `effort` sets its depth.
        # `fallbacks="default"` re-runs a safety-classifier decline on a fallback model
        # server-side instead of returning the refusal.
        response = self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.cfg.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"model declined the request: {response.stop_details}")
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            latency_s=time.perf_counter() - started,
        )


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLM:
    def __init__(self, cfg: GeneratorConfig, settings: Settings):
        self.cfg = cfg
        self.client = httpx.Client(base_url=settings.ollama_host, timeout=600)

    def complete(self, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        try:
            resp = self.client.post(
                "/api/chat",
                json={
                    "model": self.cfg.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "think": False,  # qwen3 is a reasoning model; keep answers fast and clean
                    "options": {"temperature": 0, "num_ctx": self.cfg.num_ctx},
                },
            )
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise LLMError(f"cannot reach Ollama at {self.client.base_url}; is it running?") from e
        data = resp.json()
        return LLMResponse(
            text=_THINK_BLOCK.sub("", data["message"]["content"]).strip(),
            model=data["model"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            stop_reason=data.get("done_reason"),
            latency_s=time.perf_counter() - started,
        )


def make_llm(cfg: GeneratorConfig, settings: Settings) -> LLM:
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg, settings)
    return OllamaLLM(cfg, settings)
