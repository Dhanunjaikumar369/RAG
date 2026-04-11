"""Retriever factory — wraps FAISS with configurable top-k search."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

from safetyrag.config import settings
from safetyrag.ingest import load_index


def get_retriever(vectorstore: FAISS | None = None) -> BaseRetriever:
    """
    Return a LangChain retriever backed by the FAISS index.

    Parameters
    ----------
    vectorstore:
        Pre-loaded FAISS store. If None, loads from disk automatically.
    """
    if vectorstore is None:
        vectorstore = load_index()

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    )
