"""LLM providers behind one interface: Claude (Anthropic API) and Ollama (local)."""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

import anthropic
import httpx

from rag.config import GeneratorConfig, Settings
from rag.schema import LLMResponse

JsonSchema = dict[str, Any]

# Models that accept the server-side refusal fallback (`fallbacks="default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(
        self, system: str, user: str, json_schema: JsonSchema | None = None
    ) -> LLMResponse:
        """Return the model's reply; with `json_schema`, the reply text is JSON matching it."""
        ...


class AnthropicLLM:
    def __init__(self, cfg: GeneratorConfig, settings: Settings):
        self.cfg = cfg
        key = settings.anthropic_api_key
        # With no key, the SDK resolves credentials itself (env vars or an `ant auth` profile).
        self.client = (
            anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()
        )

    def complete(
        self, system: str, user: str, json_schema: JsonSchema | None = None
    ) -> LLMResponse:
        started = time.perf_counter()
        output_config: dict[str, Any] = {}
        if self.cfg.effort:  # thinking is adaptive by default; effort sets its depth
            output_config["effort"] = self.cfg.effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict[str, Any] = {}
        if output_config:
            kwargs["output_config"] = output_config
        if self.cfg.model in _FALLBACK_MODELS:
            # Re-run a safety-classifier decline on a fallback model server-side
            # instead of returning the refusal.
            kwargs |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
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

    def complete(
        self, system: str, user: str, json_schema: JsonSchema | None = None
    ) -> LLMResponse:
        started = time.perf_counter()
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # qwen3 is a reasoning model; keep answers fast and clean
            "options": {
                "temperature": 0,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.max_tokens,
            },
        }
        if json_schema:
            body["format"] = json_schema
        try:
            resp = self.client.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise LLMError(f"cannot reach Ollama at {self.client.base_url}; is it running?") from e
        data = resp.json()
        return LLMResponse(
            text=_THINK_BLOCK.sub("", data["message"]["content"]).strip(),
            model=data["model"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            # Normalize Ollama's "length" to the Anthropic name so truncation is detected uniformly.
            stop_reason="max_tokens"
            if data.get("done_reason") == "length"
            else data.get("done_reason"),
            latency_s=time.perf_counter() - started,
        )


def make_llm(cfg: GeneratorConfig, settings: Settings) -> LLM:
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg, settings)
    return OllamaLLM(cfg, settings)
