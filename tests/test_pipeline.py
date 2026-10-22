"""Tests run fully offline with mock models (RAG_OFFLINE=1), so no API key is needed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from evals.run_eval import is_refusal, keyword_rank
from practicum_rag.config import ROOT, get_settings
from practicum_rag.engine import NOT_FOUND, PracticumBot
from practicum_rag.ingest import clean_text, ingest, load_documents, normalize_spaced_text


@pytest.fixture()
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_OFFLINE", "1")
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("RAG_DATA_DIRS", str(ROOT / "data" / "sample"))
    monkeypatch.setenv("RAG_COLLECTION", "test")
    return get_settings()


def test_normalize_spaced_text_fixes_letter_spacing():
    raw = "C h e c k  i f  y o u r  c h o s e n  H T E"
    assert normalize_spaced_text(raw) == "Check if your chosen HTE"


def test_normalize_spaced_text_leaves_normal_lines_alone():
    line = "You may now apply as an intern to your chosen HTE."
    assert normalize_spaced_text(line) == line


def test_clean_text_collapses_blank_lines():
    assert clean_text("a\n\n\n\nb   c") == "a\n\nb c"


def test_load_documents_reads_sample(offline_env):
    docs = load_documents(offline_env)
    assert docs, "sample document should load"
    assert any("SAS3000" in d.text for d in docs)


def test_ingest_is_idempotent(offline_env):
    first = ingest(offline_env)
    second = ingest(offline_env)
    bot = PracticumBot(offline_env)
    assert first.chunks == second.chunks
    assert bot.index.vector_store._collection.count() == first.chunks


def test_ask_returns_answer_with_sources(offline_env):
    ingest(offline_env)
    bot = PracticumBot(offline_env)
    result = bot.ask("How long does a new IOM take?")
    assert result.answer
    assert result.sources
    assert all(s.file.endswith(".md") for s in result.sources)


def test_ask_rejects_empty_question(offline_env):
    ingest(offline_env)
    with pytest.raises(ValueError):
        PracticumBot(offline_env).ask("   ")


def test_api_health_and_ask(offline_env):
    ingest(offline_env)
    from practicum_rag import api

    api.get_bot.cache_clear()
    client = TestClient(api.app)

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["chunks"] > 0

    res = client.post("/ask", json={"question": "Which companies should I avoid?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] and body["sources"]

    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.get("/").status_code == 200
    api.get_bot.cache_clear()


def test_eval_helpers():
    assert keyword_rank(["nothing here", "needs SAS3000 first"], "sas3000") == 2
    assert keyword_rank(["a", "b"], "z") is None
    assert is_refusal(NOT_FOUND)
    assert not is_refusal("Finish SAS3000 first.")
