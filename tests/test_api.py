"""Integration tests for the SafetyRAG FastAPI endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from safetyrag.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_status_no_index(tmp_path, monkeypatch):
    from safetyrag import config
    monkeypatch.setattr(config.settings, "vector_store_path", tmp_path / "no_store")
    resp = client.get("/ingest/status")
    assert resp.status_code == 200
    assert resp.json()["index_exists"] is False


def test_query_no_index_returns_503(tmp_path, monkeypatch):
    from safetyrag import config
    monkeypatch.setattr(config.settings, "vector_store_path", tmp_path / "no_store")
    resp = client.post("/query", json={"question": "What is PPE?"})
    assert resp.status_code == 503


def test_query_with_mocked_chain(tmp_path, monkeypatch):
    from safetyrag import config
    store_path = tmp_path / "store"
    store_path.mkdir()
    monkeypatch.setattr(config.settings, "vector_store_path", store_path)

    with patch("safetyrag.chain.ask", return_value="Always wear PPE."):
        resp = client.post("/query", json={"question": "What is PPE?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["answer"] == "Always wear PPE."


def test_query_missing_body():
    resp = client.post("/query", json={})
    assert resp.status_code == 422
