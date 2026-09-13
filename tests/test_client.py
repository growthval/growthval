from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from perplexity import AuthenticationError, RateLimitError

from perplexity_agent.client import (
    PerplexityConfigError,
    PerplexityRateLimitError,
    ask_web,
)


def _fake_response(
    *,
    text: str = "Paris is the capital of France.",
    citations: list[str] | None = None,
    search_results: list[dict[str, str]] | None = None,
) -> SimpleNamespace:
    content = SimpleNamespace(
        type="output_text",
        text=text,
        annotations=[
            SimpleNamespace(type="url_citation", url=url, title=None) for url in (citations or [])
        ],
    )
    message = SimpleNamespace(type="message", content=[content])
    output = [message]
    if search_results:
        output.append(
            SimpleNamespace(
                type="search_results",
                results=[
                    SimpleNamespace(title=r["title"], url=r["url"], snippet=r["snippet"])
                    for r in search_results
                ],
            )
        )
    return SimpleNamespace(id="resp_123", model="sonar-pro", output=output, output_text=text)


class _FakeResponses:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


class _FakeClient:
    def __init__(self, responses: _FakeResponses) -> None:
        self.responses = responses


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")


def test_ask_web_returns_text_citations_and_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _fake_response(
        citations=["https://example.com/a"],
        search_results=[{"title": "Example", "url": "https://example.com", "snippet": "..."}],
    )
    fake_responses = _FakeResponses(result=response)
    monkeypatch.setattr(
        "perplexity_agent.client.Perplexity", lambda api_key: _FakeClient(fake_responses)
    )

    answer = ask_web("What is the capital of France?", preset="low")

    assert answer.text == "Paris is the capital of France."
    assert answer.response_id == "resp_123"
    assert answer.model == "sonar-pro"
    assert answer.citations == ["https://example.com/a"]
    assert answer.search_results == [
        {"title": "Example", "url": "https://example.com", "snippet": "..."}
    ]
    assert fake_responses.last_kwargs == {
        "input": "What is the capital of France?",
        "preset": "low",
    }


def test_ask_web_model_without_preset_forces_web_search_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_responses = _FakeResponses(result=_fake_response())
    monkeypatch.setattr(
        "perplexity_agent.client.Perplexity", lambda api_key: _FakeClient(fake_responses)
    )

    ask_web("hello", model="openai/gpt-5.6-sol")

    assert fake_responses.last_kwargs == {
        "input": "hello",
        "model": "openai/gpt-5.6-sol",
        "tools": [{"type": "web_search"}],
    }


def test_ask_web_explicit_tools_override_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_responses = _FakeResponses(result=_fake_response())
    monkeypatch.setattr(
        "perplexity_agent.client.Perplexity", lambda api_key: _FakeClient(fake_responses)
    )

    ask_web("hello", model="openai/gpt-5.6-sol", tools=[{"type": "fetch_url"}])

    assert fake_responses.last_kwargs["tools"] == [{"type": "fetch_url"}]


def test_ask_web_missing_api_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    with pytest.raises(PerplexityConfigError, match="PERPLEXITY_API_KEY is not set"):
        ask_web("hello")


def test_ask_web_authentication_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://api.perplexity.ai/v1/responses")
    http_response = httpx.Response(401, request=request, json={"error": "invalid key"})
    error = AuthenticationError("Unauthorized", response=http_response, body=None)
    fake_responses = _FakeResponses(error=error)
    monkeypatch.setattr(
        "perplexity_agent.client.Perplexity", lambda api_key: _FakeClient(fake_responses)
    )

    with pytest.raises(PerplexityConfigError, match="401"):
        ask_web("hello")


def test_ask_web_rate_limit_error_carries_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("POST", "https://api.perplexity.ai/v1/responses")
    http_response = httpx.Response(
        429, request=request, headers={"retry-after": "5"}, json={"error": "rate limited"}
    )
    error = RateLimitError("Too Many Requests", response=http_response, body=None)
    fake_responses = _FakeResponses(error=error)
    monkeypatch.setattr(
        "perplexity_agent.client.Perplexity", lambda api_key: _FakeClient(fake_responses)
    )

    with pytest.raises(PerplexityRateLimitError) as exc_info:
        ask_web("hello")

    assert exc_info.value.retry_after_seconds == 5.0
