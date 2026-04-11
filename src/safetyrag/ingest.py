"""
Document ingestion pipeline.

Steps
-----
1. Load PDF(s) with PyPDFLoader
2. Chunk with RecursiveCharacterTextSplitter
3. Embed with HuggingFace BGE embeddings
4. Store in a persistent FAISS index
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from safetyrag.config import settings


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_and_split(pdf_path: Path) -> list:
    """Load a PDF and return chunked LangChain Documents."""
    logger.info("Loading PDF: {}", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    logger.info("  → {} pages loaded", len(pages))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    logger.info("  → {} chunks after splitting", len(chunks))
    return chunks


def build_index(pdf_paths: list[Path], force: bool = False) -> FAISS:
    """
    Ingest one or more PDFs into a FAISS vector store and persist it to disk.

    Parameters
    ----------
    pdf_paths : list[Path]
        PDFs to ingest.
    force : bool
        If True, rebuild the index even if one already exists.

    Returns
    -------
    FAISS
        The populated vector store.
    """
    store_path = settings.vector_store_path

    if store_path.exists() and not force:
        logger.info("Index already exists at {}. Use force=True to rebuild.", store_path)
        return load_index()

    embeddings = _build_embeddings()
    all_chunks: list = []
    for path in pdf_paths:
        all_chunks.extend(load_and_split(path))

    logger.info("Embedding {} chunks — this may take a minute ...", len(all_chunks))
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    store_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(store_path))
    logger.success("Index saved → {}", store_path)
    return vectorstore


def load_index() -> FAISS:
    """Load an existing FAISS index from disk."""
    store_path = settings.vector_store_path
    if not store_path.exists():
        raise FileNotFoundError(
            f"No vector store found at '{store_path}'. "
            "Run `safetyrag-ingest` first."
        )
    embeddings = _build_embeddings()
    logger.info("Loading index from {}", store_path)
    return FAISS.load_local(
        str(store_path), embeddings, allow_dangerous_deserialization=True
    )
