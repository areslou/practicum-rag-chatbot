"""Load documents, clean them, split them into chunks, embed them, and store them in Chroma."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import chromadb
from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import Settings
from .models import build_embed_model

log = logging.getLogger(__name__)

SUPPORTED_EXTS = [".pdf", ".md", ".txt"]

# A line like "C h e c k  i f  y o u r" (letters split by single spaces, words by two or more).
_SPACED_LINE = re.compile(r"^(?:\S {1,2}){4,}\S?\s*$")


def normalize_spaced_text(text: str) -> str:
    """Repair letter-spaced lines that some PDF exports produce.

    "C h e c k  i f  y o u r" becomes "Check if your". Normal lines are left untouched.
    """
    fixed = []
    for line in text.splitlines():
        if _SPACED_LINE.match(line.strip()):
            words = re.split(r" {2,}", line.strip())
            line = " ".join(w.replace(" ", "") for w in words)
        fixed.append(line)
    return "\n".join(fixed)


def clean_text(text: str) -> str:
    text = normalize_spaced_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(settings: Settings) -> list[Document]:
    docs: list[Document] = []
    for folder in settings.data_dirs:
        if not folder.exists() or not any(folder.rglob("*")):
            continue
        try:
            loaded = SimpleDirectoryReader(
                input_dir=str(folder),
                recursive=True,
                required_exts=SUPPORTED_EXTS,
                exclude_hidden=True,
            ).load_data()
        except ValueError:  # folder only has hidden or unsupported files
            continue
        for doc in loaded:
            cleaned = clean_text(doc.text)
            if cleaned:
                docs.append(Document(text=cleaned, metadata=doc.metadata))
    return docs


@dataclass
class IngestReport:
    documents: int
    chunks: int
    collection: str


def get_vector_store(settings: Settings, reset: bool = False) -> ChromaVectorStore:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.storage_dir))
    if reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:  # collection did not exist yet
            pass
    collection = client.get_or_create_collection(
        settings.collection,
        metadata={"embed_model": "mock" if settings.offline else settings.embed_model},
    )
    return ChromaVectorStore(chroma_collection=collection)


def ingest(settings: Settings) -> IngestReport:
    """Rebuild the index from scratch so re-running never duplicates chunks."""
    docs = load_documents(settings)
    if not docs:
        folders = ", ".join(str(p) for p in settings.data_dirs)
        raise FileNotFoundError(f"No .pdf, .md or .txt files found in: {folders}")

    splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)

    vector_store = get_vector_store(settings, reset=True)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage, embed_model=build_embed_model(settings))

    log.info("Indexed %d chunks from %d documents", len(nodes), len(docs))
    return IngestReport(documents=len(docs), chunks=len(nodes), collection=settings.collection)
