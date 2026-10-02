"""LLM providers behind one interface: Claude (Anthropic API) and Ollama (local)."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
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

    def stream(self, system: str, user: str) -> Iterator[str | LLMResponse]:
        """Yield text pieces as they are generated, then the complete LLMResponse."""
        ...


class AnthropicLLM:
    """Claude via the Anthropic API. Unused by default (the project runs on local Ollama)."""

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

    def stream(self, system: str, user: str) -> Iterator[str | LLMResponse]:
        """Not incremental: yields the whole reply at once (this provider is unused by default)."""
        response = self.complete(system, user)
        yield response.text
        yield response


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaLLM:
    def __init__(self, cfg: GeneratorConfig, settings: Settings):
        self.cfg = cfg
        self.client = httpx.Client(base_url=settings.ollama_host, timeout=600)

    def _body(
        self, system: str, user: str, json_schema: JsonSchema | None, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "think": False,  # qwen3 is a reasoning model; keep answers fast and clean
            "options": {
                "temperature": 0,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.max_tokens,
            },
        }
        if json_schema:
            body["format"] = json_schema
        return body

    @staticmethod
    def _response(data: dict[str, Any], text: str, started: float) -> LLMResponse:
        return LLMResponse(
            text=_THINK_BLOCK.sub("", text).strip(),
            model=data["model"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            # Normalize Ollama's "length" to the Anthropic name so truncation is detected uniformly.
            stop_reason="max_tokens"
            if data.get("done_reason") == "length"
            else data.get("done_reason"),
            latency_s=time.perf_counter() - started,
        )

    def complete(
        self, system: str, user: str, json_schema: JsonSchema | None = None
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            resp = self.client.post("/api/chat", json=self._body(system, user, json_schema, False))
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise LLMError(f"cannot reach Ollama at {self.client.base_url}; is it running?") from e
        data = resp.json()
        return self._response(data, data["message"]["content"], started)

    def stream(self, system: str, user: str) -> Iterator[str | LLMResponse]:
        """Yield text pieces as they are generated, then the complete LLMResponse."""
        started = time.perf_counter()
        parts: list[str] = []
        try:
            with self.client.stream(
                "POST", "/api/chat", json=self._body(system, user, None, True)
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if piece := data.get("message", {}).get("content", ""):
                        parts.append(piece)
                        yield piece
                    if data.get("done"):
                        yield self._response(data, "".join(parts), started)
                        return
        except httpx.ConnectError as e:
            raise LLMError(f"cannot reach Ollama at {self.client.base_url}; is it running?") from e
        raise LLMError("Ollama stream ended without a final message")


def make_llm(cfg: GeneratorConfig, settings: Settings) -> LLM:
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg, settings)
    return OllamaLLM(cfg, settings)
