"""Answer questions: retrieve the most relevant chunks, then have the LLM answer only from them."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from llama_index.core import PromptTemplate, VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

from .config import Settings
from .ingest import get_vector_store
from .models import build_embed_model, build_llm

NOT_FOUND = (
    "I couldn't find that in the practicum documents I have. "
    "Please check with your Department Practicum Coordinator (DPC)."
)

QA_PROMPT = PromptTemplate(
    "You are a helpful assistant for De La Salle University students preparing for their "
    "practicum (internship).\n"
    "Answer the question using ONLY the context below. Rules:\n"
    "- If the context does not contain the answer, reply exactly: \"{not_found}\"\n"
    "- Do not invent deadlines, numbers, links, or office names.\n"
    "- Be concise. Use short bullet points for multi-step answers.\n\n"
    "Context:\n---------------------\n{context_str}\n---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
).partial_format(not_found=NOT_FOUND)


@dataclass
class Source:
    file: str
    page: str | None
    score: float | None
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _to_source(node: NodeWithScore, max_chars: int = 400) -> Source:
    meta = node.node.metadata or {}
    text = node.node.get_content().strip().replace("\n", " ")
    return Source(
        file=meta.get("file_name", "unknown"),
        page=meta.get("page_label"),
        score=round(node.score, 4) if node.score is not None else None,
        text=text[:max_chars] + ("..." if len(text) > max_chars else ""),
    )


class PracticumBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        vector_store = get_vector_store(settings)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=build_embed_model(settings)
        )
        self.llm = build_llm(settings)
        self.retriever = self.index.as_retriever(similarity_top_k=settings.top_k)
        postprocessors = []
        if settings.min_score > 0:
            postprocessors.append(SimilarityPostprocessor(similarity_cutoff=settings.min_score))
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=get_response_synthesizer(
                llm=self.llm, text_qa_template=QA_PROMPT, response_mode="compact"
            ),
            node_postprocessors=postprocessors,
        )

    def is_empty(self) -> bool:
        return self.index.vector_store._collection.count() == 0  # type: ignore[attr-defined]

    def retrieve(self, question: str) -> list[NodeWithScore]:
        return self.retriever.retrieve(question)

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        response = self.query_engine.query(question)
        nodes = response.source_nodes
        if not nodes:
            return Answer(question=question, answer=NOT_FOUND)
        return Answer(
            question=question,
            answer=str(response).strip(),
            sources=[_to_source(n) for n in nodes],
        )
