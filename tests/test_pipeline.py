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


def test_folder_with_only_hidden_files_is_skipped(tmp_path, monkeypatch):
    empty = tmp_path / "private"
    empty.mkdir()
    (empty / ".gitkeep").write_text("")
    monkeypatch.setenv("RAG_OFFLINE", "1")
    monkeypatch.setenv("RAG_DATA_DIRS", f"{ROOT / 'data' / 'sample'},{empty}")
    docs = load_documents(get_settings())
    assert docs and all(d.metadata["file_name"].endswith(".md") for d in docs)


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


def test_chunks_follow_sections_and_hide_noisy_metadata(offline_env):
    from llama_index.core.schema import MetadataMode

    from practicum_rag.ingest import split_documents

    nodes = split_documents(load_documents(offline_env), offline_env)
    prereq = [n for n in nodes if "SAS3000" in n.text]
    assert len(prereq) == 1
    assert "prerequisites" in prereq[0].text.lower()
    assert "Step 1" not in prereq[0].text  # stays inside its own section
    for node in nodes:
        embedded = node.get_content(metadata_mode=MetadataMode.EMBED)
        assert "file_path" not in embedded and "file_size" not in embedded
        assert "creation_date" not in node.get_content(metadata_mode=MetadataMode.LLM)


def test_api_reports_busy_model_as_503(offline_env, monkeypatch):
    ingest(offline_env)
    from practicum_rag import api

    api.get_bot.cache_clear()
    bot = api.get_bot()

    def busy(_question):
        raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")

    monkeypatch.setattr(bot, "ask", busy)
    res = TestClient(api.app).post("/ask", json={"question": "How long does an IOM take?"})
    assert res.status_code == 503
    assert "busy" in res.json()["detail"]
    api.get_bot.cache_clear()


def test_throttle_retries_quota_errors(monkeypatch):
    from evals import run_eval

    monkeypatch.setattr(run_eval.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2.5s.")
        return "ok"

    assert run_eval.Throttle(rpm=0).call(flaky) == "ok"
    assert calls["n"] == 3

    def broken():
        raise ValueError("something else")

    with pytest.raises(ValueError):
        run_eval.Throttle(rpm=0).call(broken)
