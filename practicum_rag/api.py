"""REST API and a small chat page.

    GET  /         chat page
    GET  /health   status and number of indexed chunks
    POST /ask      {"question": "..."}  ->  {"question", "answer", "sources": [...]}
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .engine import PracticumBot

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="DLSU Practicum Q&A",
    description="Retrieval-augmented chatbot over DLSU practicum documents.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class SourceOut(BaseModel):
    file: str
    page: str | None = None
    score: float | None = None
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]


@lru_cache(maxsize=1)
def get_bot() -> PracticumBot:
    return PracticumBot(get_settings())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    bot = get_bot()
    count = bot.index.vector_store._collection.count()  # type: ignore[attr-defined]
    return {"status": "ok" if count else "empty_index", "chunks": count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    bot = get_bot()
    if bot.is_empty():
        raise HTTPException(status_code=503, detail="Index is empty. Run the ingest command first.")
    try:
        return bot.ask(req.question).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
