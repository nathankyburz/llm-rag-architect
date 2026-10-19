"""Command line entry point."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag_architect.adapters.embedding import FastEmbedEmbedder
from rag_architect.adapters.vector_store import QdrantVectorStore
from rag_architect.config import Settings, get_settings
from rag_architect.ingestion.chunking.recursive import RecursiveTokenChunker
from rag_architect.ingestion.loaders.pdf import PyMuPDFLoader
from rag_architect.ingestion.manifest import load_manifest
from rag_architect.ingestion.pipeline import IngestionPipeline, IngestionReport

app = typer.Typer(add_completion=False, help="RAG over Premier League club filings.")
console = Console()


@app.callback()
def main() -> None:
    """Forces subcommand mode: with a single command Typer would drop its name."""


def build_pipeline(settings: Settings) -> IngestionPipeline:
    """
    Composition root: the only place concrete adapters are chosen.

    Each component receives the narrowest settings it needs, so the chunker
    cannot reach the vector store's API key.
    """
    return IngestionPipeline(
        loader=PyMuPDFLoader(),
        chunker=RecursiveTokenChunker(settings.chunking),
        embedder=FastEmbedEmbedder(settings.embedding),
        store=QdrantVectorStore(settings.vector_store),
    )


def render(reports: list[IngestionReport], total_points: int) -> None:
    table = Table(title="Ingestion")
    table.add_column("document")
    table.add_column("pages", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("seconds", justify="right")
    table.add_column("status")

    for report in reports:
        table.add_row(
            report.document_id,
            str(report.pages),
            "-" if report.skipped else str(report.chunks),
            f"{report.seconds:.1f}",
            "unchanged" if report.skipped else "indexed",
        )
    console.print(table)

    indexed = sum(r.chunks for r in reports)
    skipped = sum(1 for r in reports if r.skipped)
    elapsed = sum(r.seconds for r in reports)
    console.print(
        f"{indexed} chunks written, {skipped} document(s) unchanged, "
        f"{elapsed:.1f}s total, {total_points} points in the collection."
    )


@app.command()
def ingest(
    manifest: Annotated[Path, typer.Option(help="Corpus manifest.")] = Path("data/manifest.toml"),
    raw_dir: Annotated[
        Path, typer.Option(help="Directory the manifest paths are relative to.")
    ] = Path("data/raw"),
    force: Annotated[
        bool, typer.Option(help="Re-index documents whose content is unchanged.")
    ] = False,
) -> None:
    """Load, chunk, embed and index every document in the manifest."""
    settings = get_settings()
    specs = load_manifest(manifest, root=raw_dir)
    console.print(
        f"{len(specs)} document(s) listed, collection '{settings.vector_store.collection_name}'."
    )

    pipeline = build_pipeline(settings)
    reports = pipeline.ingest_all(specs, force=force)

    store = QdrantVectorStore(settings.vector_store)
    render(reports, store.count())


if __name__ == "__main__":
    app()
