"""
RAG chain — Adaptive RAG with query rewriting and source citation.

Pipeline
--------
1. Query rewriter    — refines the user question for better retrieval
2. FAISS retriever   — fetches the top-k relevant chunks
3. Answer generator  — Groq LLM synthesises a grounded answer with citations
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq
from loguru import logger

from safetyrag.config import settings
from safetyrag.retriever import get_retriever

# ── Prompts ───────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a workplace safety expert. Rewrite the user's question to be more specific "
        "and retrieval-friendly for an OSHA/industrial safety document corpus. "
        "Return ONLY the rewritten question — nothing else.",
    ),
    ("human", "{question}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a knowledgeable workplace safety advisor. \
Answer the employee's question using ONLY the provided context from official \
OSHA and industrial safety guidelines.

Rules:
- Be clear, actionable, and safety-first.
- If the context doesn't contain enough information, say so explicitly.
- Always cite the source document name where relevant (available in metadata).
- Use bullet points for lists of steps or requirements.
""",
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    ),
])

# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_docs(docs: list) -> str:
    """Concatenate document chunks with source metadata."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source} | Page: {page}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _build_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or export it before running."
        )
    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )


# ── Public API ────────────────────────────────────────────────────────────────


def build_rag_chain(retriever: BaseRetriever | None = None):
    """
    Build and return the full adaptive RAG chain.

    Parameters
    ----------
    retriever:
        Pre-built retriever. If None, loads from the persisted FAISS index.

    Returns
    -------
    A LangChain Runnable that accepts ``{"question": str}`` and returns ``str``.
    """
    if retriever is None:
        retriever = get_retriever()

    llm = _build_llm()

    # Stage 1: rewrite the question
    rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()

    # Stage 2: retrieve + answer
    def retrieve_and_answer(inputs: dict) -> str:
        original_q = inputs["question"]
        rewritten_q = rewrite_chain.invoke({"question": original_q})
        logger.debug("Rewritten question: {}", rewritten_q)

        docs = retriever.invoke(rewritten_q)
        logger.debug("Retrieved {} chunks", len(docs))

        context = _format_docs(docs)
        answer = (
            _ANSWER_PROMPT
            | llm
            | StrOutputParser()
        ).invoke({"context": context, "question": original_q})
        return answer

    return RunnableLambda(retrieve_and_answer)


def ask(question: str, retriever: BaseRetriever | None = None) -> str:
    """Convenience wrapper — build chain and answer *question* in one call."""
    chain = build_rag_chain(retriever)
    return chain.invoke({"question": question})
