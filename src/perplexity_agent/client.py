"""Web-grounded answers via the Perplexity Agent API.

Endpoint: POST https://api.perplexity.ai/v1/agent (the Python SDK calls its
OpenAI-compatible alias, /v1/responses, through `client.responses.create`).
See https://docs.perplexity.ai/docs/agent-api/quickstart.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from perplexity import APIStatusError, AuthenticationError, Perplexity, RateLimitError

DEFAULT_PRESET = "medium"
_DEFAULT_TOOLS: list[dict[str, Any]] = [{"type": "web_search"}]


class PerplexityConfigError(RuntimeError):
    """Raised when the request can't be configured or the API rejects it (non-429)."""


class PerplexityRateLimitError(RuntimeError):
    """Raised on HTTP 429. Carries the server's Retry-After hint, if any."""

    def __init__(self, message: str, *, retry_after: str | None) -> None:
        super().__init__(message)
        self.retry_after_seconds = float(retry_after) if retry_after else None


@dataclass
class WebAnswer:
    """Result of a web-grounded query, distilled from the raw Agent API response."""

    text: str
    response_id: str
    model: str
    citations: list[str]
    search_results: list[dict[str, Any]]
    raw: Any = field(repr=False)


def _client() -> Perplexity:
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise PerplexityConfigError(
            "PERPLEXITY_API_KEY is not set. Create a key in the API Console "
            "(https://console.perplexity.ai) and export it in your shell, e.g. "
            "`export PERPLEXITY_API_KEY=...` — never pass it as a literal in code."
        )
    return Perplexity(api_key=api_key)


def ask_web(
    query: str,
    *,
    preset: str | None = DEFAULT_PRESET,
    model: str | None = None,
    tools: Iterable[dict[str, Any]] | None = None,
    instructions: str | None = None,
    previous_response_id: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> WebAnswer:
    """Ask a web-grounded question through the Perplexity Agent API.

    Pass `model=` to target a specific model instead of a preset (presets bundle
    a model, tools, and limits already). `tools` defaults to `web_search` only
    when a bare `model` is used without a preset; presets supply their own tools
    unless overridden here.
    """
    if model is not None:
        preset = None

    params: dict[str, Any] = {"input": query}
    if model is not None:
        params["model"] = model
    if preset is not None:
        params["preset"] = preset
    if tools is not None:
        params["tools"] = list(tools)
    elif model is not None:
        params["tools"] = _DEFAULT_TOOLS
    if instructions is not None:
        params["instructions"] = instructions
    if previous_response_id is not None:
        params["previous_response_id"] = previous_response_id
    if response_format is not None:
        params["response_format"] = response_format

    client = _client()
    try:
        response = client.responses.create(**params)
    except AuthenticationError as exc:
        raise PerplexityConfigError(
            "Perplexity rejected the API key (401 Unauthorized). Verify "
            "PERPLEXITY_API_KEY is correct and active in the API Console "
            "(https://console.perplexity.ai); rotate it there if it may have leaked."
        ) from exc
    except RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after")
        hint = f" Honor Retry-After: retry in {retry_after}s." if retry_after else ""
        raise PerplexityRateLimitError(
            f"Perplexity rate-limited the request (429).{hint}", retry_after=retry_after
        ) from exc
    except APIStatusError as exc:
        raise PerplexityConfigError(
            f"Perplexity API error {exc.status_code}: {exc.message}"
        ) from exc

    return _to_web_answer(response)


def _to_web_answer(response: Any) -> WebAnswer:
    citations: list[str] = []
    search_results: list[dict[str, Any]] = []

    for item in response.output:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for content in item.content:
                for annotation in getattr(content, "annotations", None) or []:
                    if annotation.url:
                        citations.append(annotation.url)
        elif item_type == "search_results":
            for result in item.results:
                search_results.append(
                    {"title": result.title, "url": result.url, "snippet": result.snippet}
                )

    return WebAnswer(
        text=response.output_text,
        response_id=response.id,
        model=response.model,
        citations=citations,
        search_results=search_results,
        raw=response,
    )
