"""
POST /ingest/upload  — upload a PDF and add it to the FAISS index
POST /ingest/rebuild — rebuild the index from all PDFs in ./docs/
GET  /ingest/status  — check whether an index exists
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from safetyrag.config import settings
from safetyrag.ingest import build_index, load_and_split

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int


class StatusResponse(BaseModel):
    index_exists: bool
    index_path: str


@router.get("/status", response_model=StatusResponse, summary="Check index status")
def index_status() -> StatusResponse:
    return StatusResponse(
        index_exists=settings.vector_store_path.exists(),
        index_path=str(settings.vector_store_path),
    )


@router.post(
    "/upload",
    response_model=IngestResponse,
    summary="Upload a PDF and add it to the safety knowledge base",
)
async def upload_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a PDF workplace-safety document to be ingested into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted.")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / file.filename
        pdf_path.write_bytes(await file.read())
        chunks = load_and_split(pdf_path)

    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the PDF.")

    try:
        from langchain_community.vectorstores import FAISS  # noqa: PLC0415
        from safetyrag.ingest import _build_embeddings  # noqa: PLC0415

        embeddings = _build_embeddings()
        new_store = FAISS.from_documents(chunks, embeddings)

        if settings.vector_store_path.exists():
            existing = FAISS.load_local(
                str(settings.vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            existing.merge_from(new_store)
            existing.save_local(str(settings.vector_store_path))
        else:
            settings.vector_store_path.mkdir(parents=True, exist_ok=True)
            new_store.save_local(str(settings.vector_store_path))

    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        message=f"'{file.filename}' ingested successfully.",
        chunks_indexed=len(chunks),
    )


@router.post(
    "/rebuild",
    response_model=IngestResponse,
    summary="Rebuild the index from all PDFs in ./docs/",
)
def rebuild_index() -> IngestResponse:
    """Re-index every PDF found in the ./docs/ directory."""
    docs_dir = Path("docs")
    pdfs = list(docs_dir.glob("**/*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail="No PDF files found in ./docs/")

    try:
        store = build_index(pdfs, force=True)
    except Exception as exc:
        logger.exception("Rebuild failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_chunks = store.index.ntotal
    return IngestResponse(
        message=f"Index rebuilt from {len(pdfs)} PDF(s).",
        chunks_indexed=total_chunks,
    )
