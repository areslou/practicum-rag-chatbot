"""Builds the LLM and embedding model: Gemini normally, mock models in offline mode."""

from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from .config import Settings

MOCK_EMBED_DIM = 64


class MissingAPIKeyError(RuntimeError):
    pass


def build_embed_model(settings: Settings) -> BaseEmbedding:
    if settings.offline:
        from llama_index.core.embeddings import MockEmbedding

        return MockEmbedding(embed_dim=MOCK_EMBED_DIM)

    _require_key(settings)
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    return GoogleGenAIEmbedding(model_name=settings.embed_model, api_key=settings.google_api_key)


def build_llm(settings: Settings, model: str | None = None) -> LLM:
    if settings.offline:
        from llama_index.core.llms import MockLLM

        return MockLLM(max_tokens=64)

    _require_key(settings)
    from llama_index.llms.google_genai import GoogleGenAI

    kwargs = {"api_key": settings.google_api_key, "temperature": 0.1}
    chosen = model or settings.llm_model
    if chosen:
        kwargs["model"] = chosen
    return GoogleGenAI(**kwargs)


def _require_key(settings: Settings) -> None:
    if not settings.google_api_key:
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add a free key "
            "from https://aistudio.google.com/apikey, or set RAG_OFFLINE=1 to use mock models."
        )
