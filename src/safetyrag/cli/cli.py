"""
SafetyRAG CLI

Commands
--------
safetyrag-ingest   — index PDFs from ./docs/ (or a custom folder)
safetyrag-query    — ask a workplace safety question interactively

Examples
--------
# Index default docs folder
safetyrag-ingest

# Index a custom folder, force rebuild
safetyrag-ingest --docs-dir /path/to/pdfs --force

# Ask a question
safetyrag-query "What are the fire evacuation procedures?"

# Interactive REPL mode
safetyrag-query
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from safetyrag.config import settings


@click.command("safetyrag-ingest")
@click.option(
    "--docs-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("docs"),
    show_default=True,
    help="Directory containing PDF files to index.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force rebuild even if an index already exists.",
)
@click.option(
    "--log-level",
    default=settings.log_level,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
)
def ingest(docs_dir: Path, force: bool, log_level: str) -> None:
    """Index all PDFs in DOCS_DIR into the FAISS vector store."""
    import sys  # noqa: PLC0415

    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    from safetyrag.ingest import build_index  # noqa: PLC0415

    pdfs = list(docs_dir.glob("**/*.pdf"))
    if not pdfs:
        click.echo(f"No PDF files found in '{docs_dir}'.", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(pdfs)} PDF(s):")
    for p in pdfs:
        click.echo(f"  • {p.name}")

    store = build_index(pdfs, force=force)
    total = store.index.ntotal
    click.echo(f"\nDone! {total} chunks indexed → {settings.vector_store_path}")


@click.command("safetyrag-query")
@click.argument("question", required=False)
@click.option(
    "--log-level",
    default=settings.log_level,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
)
def query(question: str | None, log_level: str) -> None:
    """
    Ask a workplace safety question.

    If QUESTION is omitted, starts an interactive prompt loop.
    Type 'exit' or 'quit' to stop.
    """
    import sys  # noqa: PLC0415

    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    if not settings.vector_store_path.exists():
        click.echo(
            "Knowledge base not found. Run `safetyrag-ingest` first.", err=True
        )
        raise SystemExit(1)

    from safetyrag.chain import ask  # noqa: PLC0415

    if question:
        click.echo(ask(question))
        return

    # Interactive REPL
    click.echo("SafetyRAG — Workplace Safety Q&A  (type 'exit' to quit)\n")
    while True:
        try:
            q = click.prompt("Question", prompt_suffix=" > ")
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in {"exit", "quit", "q"}:
            break
        answer = ask(q.strip())
        click.echo(f"\nAnswer:\n{answer}\n")
