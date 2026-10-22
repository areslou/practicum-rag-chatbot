"""Evaluate the chatbot on a fixed question set.

Metrics
- Retrieval hit rate @k: did any retrieved chunk contain the expected keyword?
- MRR: how high the first correct chunk ranked (1.0 = always first).
- Faithfulness: is every claim in the answer supported by the retrieved chunks? (catches hallucination)
- Relevancy: does the answer address the question?
- Correctness: 1 to 5 score against the reference answer.
- Refusal rate: for out-of-scope questions, did the bot decline instead of guessing?

The three answer-quality scores come from ONE grading call per question ("LLM as judge"),
which keeps a full run within the Gemini free tier's small daily request limit.

Results are saved after every question, so if a run stops (for example at the daily limit),
running the command again continues where it left off. Use --fresh to start over.

Usage:
    python -m evals.run_eval              # full run
    python -m evals.run_eval --retrieval  # retrieval metrics only (no answer generation)
    python -m evals.run_eval --fresh      # ignore saved progress
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from practicum_rag.config import get_settings
from practicum_rag.engine import NOT_FOUND, PracticumBot
from practicum_rag.models import build_llm

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROGRESS = RESULTS / "progress.json"

JUDGE_PROMPT = """You are grading a question-answering chatbot. Be strict and objective.

Question:
{question}

Reference answer (ground truth):
{reference}

Retrieved context the chatbot was given:
---
{context}
---

Chatbot answer:
{answer}

Grade the answer:
- faithful: true only if EVERY claim in the answer is supported by the retrieved context.
- relevant: true if the answer directly addresses the question.
- correctness: integer 1 to 5 comparing the answer with the reference answer
  (5 = fully correct and complete, 3 = partly correct or missing key details, 1 = wrong).

Reply with only a JSON object, no other text:
{{"faithful": true, "relevant": true, "correctness": 5, "reason": "one short sentence"}}"""


class QuotaExhausted(RuntimeError):
    """Raised when the API asks us to wait so long (daily limit) that retrying is pointless."""


def keyword_rank(chunks: list[str], keyword: str) -> int | None:
    """1-based rank of the first chunk containing the keyword, or None."""
    for rank, text in enumerate(chunks, 1):
        if keyword.lower() in text.lower():
            return rank
    return None


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower()[:40] in answer.lower() or "couldn't find" in answer.lower()


def parse_retry_seconds(message: str) -> float | None:
    """Read 'retry in 1h2m3.5s' or 'retry in 11.2s' from an API error message."""
    match = re.search(r"retry in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", message)
    if not match or not match.group(1):
        return None
    parts = {unit: value for value, unit in re.findall(r"([\d.]+)([hms])", match.group(1))}
    return float(parts.get("h", 0)) * 3600 + float(parts.get("m", 0)) * 60 + float(parts.get("s", 0))


def parse_judgement(text: str) -> dict:
    """Pull the JSON object out of the judge's reply (it may wrap it in ```json fences)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {text[:200]}")
    data = json.loads(match.group(0))
    score = int(round(float(data.get("correctness", 0))))
    return {
        "faithful": bool(data.get("faithful")),
        "relevant": bool(data.get("relevant")),
        "correctness": max(1, min(5, score)),
        "judge_reason": str(data.get("reason", ""))[:300],
    }


class Throttle:
    """Keeps calls to one model under its per-minute limit and waits out short quota errors.

    Each model has its own free-tier quota, so answers and grading get separate throttles.
    """

    def __init__(self, rpm: float, max_retries: int = 5, max_wait: float = 120.0):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._last = 0.0

    def call(self, fn, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # quota or overload errors from the API
                message = str(exc)
                retryable = any(
                    code in message for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
                )
                if not retryable:
                    raise
                delay = parse_retry_seconds(message)
                if "PerDay" in message or (delay is not None and delay > self.max_wait):
                    model = re.search(r"model:\s*([\w.-]+)", message)
                    which = f" for {model.group(1)}" if model else ""
                    raise QuotaExhausted(
                        f"Daily free-tier limit reached{which} "
                        f"(resets in about {(delay or 0) / 3600:.1f} h)."
                    ) from exc
                if attempt == self.max_retries:
                    raise
                delay = (delay + 1) if delay is not None else 15.0 * (attempt + 1)
                print(f"  rate-limited, waiting {delay:.0f}s...")
                time.sleep(delay)


def config_signature(settings) -> dict:
    """Settings that change the answers. If any differ from the saved run, start over.
    (The judge model is recorded per question instead, so it can be switched mid-run.)"""
    return {
        "llm_model": settings.llm_model or "library default",
        "embed_model": "mock" if settings.offline else settings.embed_model,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
    }


def load_progress(signature: dict, fresh: bool) -> dict:
    if fresh or not PROGRESS.exists():
        return {}
    saved = json.loads(PROGRESS.read_text(encoding="utf-8"))
    if saved.get("signature") != signature:
        print("Settings changed since the last run; starting fresh.")
        return {}
    return saved.get("rows", {})


def save_progress(signature: dict, rows: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps({"signature": signature, "rows": rows}, indent=2), encoding="utf-8")


def run(retrieval_only: bool, rpm: float, fresh: bool) -> tuple[dict, str | None]:
    settings = get_settings()
    bot = PracticumBot(settings)
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))

    judge_model = settings.judge_model or settings.llm_model or "library default"
    judge = None
    if not retrieval_only and not settings.offline:
        judge = build_llm(settings, model=settings.judge_model or None)

    signature = config_signature(settings)
    done = {} if retrieval_only else load_progress(signature, fresh)
    answer_throttle = Throttle(rpm)
    judge_throttle = Throttle(rpm)
    stopped: str | None = None

    for q in questions:
        if q["id"] in done:
            print(f"{q['id']} already done, skipping")
            continue
        row = {"id": q["id"], "question": q["question"], "out_of_scope": q.get("out_of_scope", False)}
        retrieved = [n.node.get_content() for n in bot.retrieve(q["question"])]
        if q.get("expected_keyword"):
            rank = keyword_rank(retrieved, q["expected_keyword"])
            row["hit"] = rank is not None
            row["rr"] = 1 / rank if rank else 0.0

        try:
            if not retrieval_only:
                ans = answer_throttle.call(bot.ask, q["question"])
                row["answer"] = ans.answer
                if row["out_of_scope"]:
                    row["refused"] = is_refusal(ans.answer)
                elif judge is not None:
                    context = "\n\n".join(s.text for s in ans.sources) or "\n\n".join(retrieved)
                    prompt = JUDGE_PROMPT.format(
                        question=q["question"], reference=q["reference"],
                        context=context, answer=ans.answer,
                    )
                    reply = judge_throttle.call(judge.complete, prompt)
                    row.update(parse_judgement(reply.text))
                    row["judge_model"] = judge_model
        except QuotaExhausted as exc:
            stopped = str(exc)
            break

        done[q["id"]] = row
        if not retrieval_only:
            save_progress(signature, done)
        print(f"{q['id']} done ({len(done)}/{len(questions)})")

    rows = [done[q["id"]] for q in questions if q["id"] in done]

    def pct(values: list[bool]) -> float | None:
        return round(100 * mean(values), 1) if values else None

    in_scope = [r for r in rows if not r["out_of_scope"]]
    scores = [r["correctness"] for r in in_scope if r.get("correctness")]
    summary = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "offline-mock" if settings.offline else "gemini",
        **signature,
        "judge_model": ", ".join(sorted({r["judge_model"] for r in rows if r.get("judge_model")}))
        or ("n/a" if retrieval_only or settings.offline else judge_model),
        "questions_scored": f"{len(rows)}/{len(questions)}",
        "retrieval_hit_rate_pct": pct([r["hit"] for r in in_scope if "hit" in r]),
        "mrr": round(mean([r["rr"] for r in in_scope if "rr" in r]), 3) if in_scope else None,
        "faithfulness_pct": pct([r["faithful"] for r in in_scope if "faithful" in r]),
        "relevancy_pct": pct([r["relevant"] for r in in_scope if "relevant" in r]),
        "avg_correctness_1_to_5": round(mean(scores), 2) if scores else None,
        "out_of_scope_refusal_pct": pct([r["refused"] for r in rows if "refused" in r]),
    }
    return {"summary": summary, "rows": rows}, stopped


def write_report(result: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    s = result["summary"]
    lines = [
        "# Evaluation results",
        "",
        f"Run at {s['run_at']} · answers `{s['llm_model']}` · judge `{s['judge_model']}` · "
        f"embeddings `{s['embed_model']}` · top_k {s['top_k']} · chunk size {s['chunk_size']} · "
        f"{s['questions_scored']} questions",
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
    lines += [
        "", "## Per question", "",
        "| ID | Hit | Faithful | Relevant | Correctness | Refused | Judge note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        def fmt(k):
            v = r.get(k)
            return "" if v is None else ("yes" if v is True else "no" if v is False else str(v))
        note = fmt("judge_reason").replace("|", "/")
        lines.append(f"| {r['id']} | {fmt('hit')} | {fmt('faithful')} | {fmt('relevant')} | "
                     f"{fmt('correctness')} | {fmt('refused')} | {note} |")
    path = RESULTS / "latest.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", action="store_true", help="Only measure retrieval")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start over")
    parser.add_argument(
        "--rpm", type=float, default=4.5,
        help="Max calls per minute per model (Gemini free tier allows about 5). 0 = no limit.",
    )
    args = parser.parse_args()
    result, stopped = run(retrieval_only=args.retrieval, rpm=args.rpm, fresh=args.fresh)
    path = write_report(result)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nReport written to {path}")
    if stopped:
        print(f"\nStopped early: {stopped}\nProgress is saved. Run the same command again later "
              "(or set a different RAG_JUDGE_MODEL / RAG_LLM_MODEL in .env) to finish.")


if __name__ == "__main__":
    main()
