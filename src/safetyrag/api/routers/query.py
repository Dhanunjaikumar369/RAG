"""
POST /query  — ask a workplace safety question against the indexed documents
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from safetyrag.config import settings

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "question": "What should I do if a fire alarm sounds in my workplace?"
    }}}

    question: str = Field(..., description="Workplace safety question to ask.")


class QueryResponse(BaseModel):
    question: str
    answer: str


@router.post("", response_model=QueryResponse, summary="Ask a workplace safety question")
def ask_question(body: QueryRequest) -> QueryResponse:
    """
    Query the indexed safety documents using adaptive RAG.

    The pipeline:
    1. Rewrites your question for better retrieval.
    2. Retrieves the top-k relevant chunks from OSHA guidelines.
    3. Generates a grounded, citation-aware answer.
    """
    if not settings.vector_store_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not yet built. Call POST /ingest/rebuild first.",
        )

    try:
        from safetyrag.chain import ask  # noqa: PLC0415

        answer = ask(body.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(question=body.question, answer=answer)
