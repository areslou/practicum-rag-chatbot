"""Command line: build the index, ask questions, or start the API.

    python -m practicum_rag.cli ingest
    python -m practicum_rag.cli ask "Can I start my hours before the IOM is signed?"
    python -m practicum_rag.cli chat
    python -m practicum_rag.cli serve
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import get_settings


def _print_answer(answer) -> None:
    print(f"\n{answer.answer}\n")
    for i, src in enumerate(answer.sources, 1):
        page = f" p.{src.page}" if src.page else ""
        score = f" (score {src.score})" if src.score is not None else ""
        print(f"  [{i}] {src.file}{page}{score}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="practicum_rag", description="DLSU Practicum Q&A (RAG)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="(Re)build the vector index from the data folders")
    ask_p = sub.add_parser("ask", help="Ask one question")
    ask_p.add_argument("question")
    sub.add_parser("chat", help="Interactive question loop")
    serve_p = sub.add_parser("serve", help="Run the FastAPI server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()

    if args.command == "ingest":
        from .ingest import ingest

        report = ingest(settings)
        print(f"Indexed {report.chunks} chunks from {report.documents} documents "
              f"into collection '{report.collection}'.")
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("practicum_rag.api:app", host=args.host, port=args.port)
        return 0

    from .engine import PracticumBot

    bot = PracticumBot(settings)
    if bot.is_empty():
        print("The index is empty. Run:  python -m practicum_rag.cli ingest", file=sys.stderr)
        return 1

    if args.command == "ask":
        _print_answer(bot.ask(args.question))
        return 0

    print("Ask about the DLSU practicum process. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            _print_answer(bot.ask(question))


if __name__ == "__main__":
    raise SystemExit(main())
