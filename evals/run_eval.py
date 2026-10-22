"""Evaluate the chatbot on a fixed question set.

Metrics
- Retrieval hit rate @k: did any retrieved chunk contain the expected keyword?
- MRR: how high the first correct chunk ranked (1.0 = always first).
- Faithfulness: is the answer supported by the retrieved chunks? (LLM judge; catches hallucination)
- Relevancy: do the answer and chunks actually address the question? (LLM judge)
- Correctness: 1 to 5 score against the reference answer. (LLM judge)
- Refusal rate: for out-of-scope questions, did the bot say it could not find the answer?

Usage:
    python -m evals.run_eval              # full run with Gemini
    python -m evals.run_eval --retrieval  # retrieval metrics only (fast, no generation)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from practicum_rag.config import get_settings
from practicum_rag.engine import NOT_FOUND, PracticumBot

HERE = Path(__file__).resolve().parent


def keyword_rank(chunks: list[str], keyword: str) -> int | None:
    """1-based rank of the first chunk containing the keyword, or None."""
    for rank, text in enumerate(chunks, 1):
        if keyword.lower() in text.lower():
            return rank
    return None


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower()[:40] in answer.lower() or "couldn't find" in answer.lower()


def run(retrieval_only: bool, pause: float) -> dict:
    settings = get_settings()
    bot = PracticumBot(settings)
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))

    judges = {}
    if not retrieval_only and not settings.offline:
        from llama_index.core.evaluation import (
            CorrectnessEvaluator,
            FaithfulnessEvaluator,
            RelevancyEvaluator,
        )

        judges = {
            "faithfulness": FaithfulnessEvaluator(llm=bot.llm),
            "relevancy": RelevancyEvaluator(llm=bot.llm),
            "correctness": CorrectnessEvaluator(llm=bot.llm),
        }

    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "out_of_scope": q.get("out_of_scope", False)}
        retrieved = [n.node.get_content() for n in bot.retrieve(q["question"])]

        if q.get("expected_keyword"):
            rank = keyword_rank(retrieved, q["expected_keyword"])
            row["hit"] = rank is not None
            row["rr"] = 1 / rank if rank else 0.0

        if not retrieval_only:
            ans = bot.ask(q["question"])
            row["answer"] = ans.answer
            contexts = [s.text for s in ans.sources] or retrieved
            if row["out_of_scope"]:
                row["refused"] = is_refusal(ans.answer)
            elif judges:
                f = judges["faithfulness"].evaluate(q["question"], ans.answer, contexts)
                r = judges["relevancy"].evaluate(q["question"], ans.answer, contexts)
                c = judges["correctness"].evaluate(
                    q["question"], ans.answer, contexts, reference=q["reference"]
                )
                row["faithful"] = bool(f.passing)
                row["relevant"] = bool(r.passing)
                row["correctness"] = c.score
            time.sleep(pause)  # stay under free-tier rate limits
        rows.append(row)
        print(f"{q['id']} done")

    def pct(values: list[bool]) -> float | None:
        return round(100 * mean(values), 1) if values else None

    in_scope = [r for r in rows if not r["out_of_scope"]]
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline-mock" if settings.offline else "gemini",
        "llm_model": settings.llm_model or "library default",
        "embed_model": "mock" if settings.offline else settings.embed_model,
        "top_k": settings.top_k,
        "questions": len(rows),
        "retrieval_hit_rate_pct": pct([r["hit"] for r in in_scope if "hit" in r]),
        "mrr": round(mean([r["rr"] for r in in_scope if "rr" in r]), 3) if in_scope else None,
        "faithfulness_pct": pct([r["faithful"] for r in in_scope if "faithful" in r]),
        "relevancy_pct": pct([r["relevant"] for r in in_scope if "relevant" in r]),
        "avg_correctness_1_to_5": (
            round(mean([r["correctness"] for r in in_scope if r.get("correctness")]), 2)
            if any(r.get("correctness") for r in in_scope) else None
        ),
        "out_of_scope_refusal_pct": pct([r["refused"] for r in rows if "refused" in r]),
    }
    return {"summary": summary, "rows": rows}


def write_report(result: dict) -> Path:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = result["summary"]
    lines = [
        "# Evaluation results",
        "",
        f"Run at {s['run_at']} · mode `{s['mode']}` · LLM `{s['llm_model']}` · "
        f"embeddings `{s['embed_model']}` · top_k {s['top_k']} · {s['questions']} questions",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    labels = {
        "retrieval_hit_rate_pct": "Retrieval hit rate (%)",
        "mrr": "Mean reciprocal rank",
        "faithfulness_pct": "Faithfulness (%)",
        "relevancy_pct": "Relevancy (%)",
        "avg_correctness_1_to_5": "Correctness (1 to 5)",
        "out_of_scope_refusal_pct": "Out-of-scope refusal (%)",
    }
    for key, label in labels.items():
        value = s.get(key)
        lines.append(f"| {label} | {'n/a' if value is None else value} |")
    lines += ["", "## Per question", "", "| ID | Hit | Faithful | Relevant | Correctness | Refused |", "|---|---|---|---|---|---|"]
    for r in result["rows"]:
        def fmt(k):
            v = r.get(k)
            return "" if v is None else ("yes" if v is True else "no" if v is False else str(v))
        lines.append(f"| {r['id']} | {fmt('hit')} | {fmt('faithful')} | {fmt('relevant')} | "
                     f"{fmt('correctness')} | {fmt('refused')} |")
    path = out_dir / "latest.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true", help="Only measure retrieval")
    parser.add_argument("--pause", type=float, default=4.0, help="Seconds between questions")
    args = parser.parse_args()
    result = run(retrieval_only=args.retrieval, pause=args.pause)
    path = write_report(result)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
