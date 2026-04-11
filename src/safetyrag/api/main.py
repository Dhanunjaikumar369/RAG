"""FastAPI application entry point for SafetyRAG."""

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from safetyrag import __version__
from safetyrag.api.routers import ingest, query
from safetyrag.config import settings

logger.remove()
logger.add(sys.stderr, level=settings.log_level.upper(), colorize=True)

app = FastAPI(
    title="SafetyRAG",
    description=(
        "Ask questions against official **OSHA workplace safety guidelines** "
        "using Retrieval-Augmented Generation (RAG)."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health", tags=["meta"], summary="Health check")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def start() -> None:
    import uvicorn  # noqa: PLC0415
    uvicorn.run("safetyrag.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
