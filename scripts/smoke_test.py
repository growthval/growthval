#!/usr/bin/env python3
"""Minimal real-request smoke test for the Perplexity Agent API integration.

Requires PERPLEXITY_API_KEY to be exported in the environment already —
this script never prompts for or prints the key. Run with:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from perplexity_agent.client import (  # noqa: E402
    PerplexityConfigError,
    PerplexityRateLimitError,
    ask_web,
)


def main() -> int:
    if not os.environ.get("PERPLEXITY_API_KEY"):
        print(
            "PERPLEXITY_API_KEY is not set. Create a key at https://console.perplexity.ai "
            "and export it in your shell, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    try:
        answer = ask_web("What is the capital of France?", preset="low")
    except PerplexityRateLimitError as exc:
        print(f"HTTP 429 (rate limited): {exc}")
        return 1
    except PerplexityConfigError as exc:
        print(f"request failed: {exc}")
        return 1

    print("HTTP 200 OK")
    print(f"response_id={answer.response_id} model={answer.model}")
    print(f"output_text length={len(answer.text)} chars, {len(answer.citations)} citation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
