"""Unit tests for safetyrag.chain — mocks out LLM and retriever."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


def _fake_retriever(chunks: list[Document]):
    retriever = MagicMock()
    retriever.invoke.return_value = chunks
    return retriever


def test_ask_returns_string():
    doc = Document(
        page_content="Always keep fire exits clear of obstructions.",
        metadata={"source": "osha3088.pdf", "page": 5},
    )
    retriever = _fake_retriever([doc])

    with patch("safetyrag.chain._build_llm") as mock_llm_factory:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Keep fire exits unobstructed.")
        # Make the chain work by returning a string from the pipe
        mock_llm_factory.return_value = llm

        from safetyrag.chain import ask

        with patch("safetyrag.chain.build_rag_chain") as mock_build:
            chain = MagicMock()
            chain.invoke.return_value = "Keep fire exits unobstructed at all times."
            mock_build.return_value = chain

            result = ask("What should I do about fire exits?", retriever=retriever)

    assert isinstance(result, str)
    assert len(result) > 0


def test_build_llm_raises_without_key(monkeypatch):
    from safetyrag import config

    monkeypatch.setattr(config.settings, "groq_api_key", "")

    from safetyrag.chain import _build_llm

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _build_llm()
