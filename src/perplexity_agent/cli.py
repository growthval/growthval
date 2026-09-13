"""Command-line entry point: `perplexity-agent "your question"`."""

from __future__ import annotations

import argparse
import sys

from .client import PerplexityConfigError, PerplexityRateLimitError, ask_web


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask a web-grounded question via the Perplexity Agent API."
    )
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument("--preset", default="medium", help="Agent preset (default: medium).")
    parser.add_argument("--model", default=None, help="Explicit model instead of a preset.")
    parser.add_argument(
        "--show-citations", action="store_true", help="Print cited URLs after the answer."
    )
    args = parser.parse_args(argv)

    try:
        answer = ask_web(args.query, preset=args.preset, model=args.model)
    except PerplexityRateLimitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except PerplexityConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(answer.text)
    if args.show_citations and answer.citations:
        print("\nSources:")
        for url in answer.citations:
            print(f"- {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
