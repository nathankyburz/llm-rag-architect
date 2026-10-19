"""Ingestion: manifest entry in, indexed chunks out."""

import time
from dataclasses import dataclass

from rag_architect.domain.ports import Chunker, DocumentLoader, Embedder, VectorStore
from rag_architect.ingestion.manifest import DocumentSpec


@dataclass(frozen=True)
class IngestionReport:
    """What happened to one document."""

    document_id: str
    pages: int
    chunks: int
    skipped: bool
    seconds: float


class IngestionPipeline:
    """
    Wires the four ports together. It holds no knowledge of PDFs, tokenizers or
    Qdrant -- only of the ports -- so the whole pipeline runs against in-memory
    fakes in the tests.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest(self, spec: DocumentSpec, *, force: bool = False) -> IngestionReport:
        started = time.perf_counter()

        document = self._loader.load(
            spec.path,
            document_type=spec.document_type,
            title=spec.title,
            club=spec.club,
            fiscal_year=spec.fiscal_year,
        )

        # Extraction is cheap next to embedding, so the hash is computed from the
        # loaded document rather than stored alongside the file.
        if not force and self._store.contains_content_hash(document.content_hash):
            return IngestionReport(
                document_id=document.metadata.document_id,
                pages=len(document.pages),
                chunks=0,
                skipped=True,
                seconds=time.perf_counter() - started,
            )

        chunks = self._chunker.split(document)
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        written = self._store.upsert(chunks, vectors)

        return IngestionReport(
            document_id=document.metadata.document_id,
            pages=len(document.pages),
            chunks=written,
            skipped=False,
            seconds=time.perf_counter() - started,
        )

    def ingest_all(
        self, specs: list[DocumentSpec], *, force: bool = False
    ) -> list[IngestionReport]:
        self._store.ensure_collection(self._embedder.dimension)
        return [self.ingest(spec, force=force) for spec in specs]
