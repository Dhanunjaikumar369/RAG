"""Unit tests for safetyrag.ingest — no real PDFs or embeddings needed."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_load_and_split_calls_loader(tmp_path):
    """load_and_split should invoke PyPDFLoader and split the pages."""
    fake_page = MagicMock()
    fake_page.page_content = "Fire extinguisher must be checked annually. " * 20

    with (
        patch("safetyrag.ingest.PyPDFLoader") as MockLoader,
        patch("safetyrag.ingest.RecursiveCharacterTextSplitter") as MockSplitter,
    ):
        MockLoader.return_value.load.return_value = [fake_page]
        splitter_instance = MagicMock()
        splitter_instance.split_documents.return_value = [fake_page, fake_page]
        MockSplitter.return_value = splitter_instance

        from safetyrag.ingest import load_and_split

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        chunks = load_and_split(pdf)

    assert len(chunks) == 2
    MockLoader.assert_called_once_with(str(pdf))


def test_load_index_raises_when_missing(tmp_path, monkeypatch):
    """load_index should raise FileNotFoundError if the store path doesn't exist."""
    from safetyrag import config

    monkeypatch.setattr(config.settings, "vector_store_path", tmp_path / "nonexistent")

    from safetyrag.ingest import load_index

    with pytest.raises(FileNotFoundError, match="safetyrag-ingest"):
        load_index()


def test_build_index_skips_if_exists(tmp_path, monkeypatch):
    """build_index should skip rebuilding if the store already exists and force=False."""
    from safetyrag import config

    store_path = tmp_path / "vector_store"
    store_path.mkdir()
    monkeypatch.setattr(config.settings, "vector_store_path", store_path)

    with patch("safetyrag.ingest.load_index") as mock_load:
        mock_load.return_value = MagicMock()
        from safetyrag.ingest import build_index

        result = build_index([Path("dummy.pdf")], force=False)

    mock_load.assert_called_once()
