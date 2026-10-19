"""
The whole pipeline, driven by in-memory fakes.

This is what the hexagonal layout buys: no PDF is parsed, no model is
downloaded, no container runs, and the test still exercises the real
orchestration -- including the skip that makes re-ingestion cheap.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_architect.domain.documents import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    Page,
    RawDocument,
)
from rag_architect.domain.retrieval import SearchResult
from rag_architect.ingestion.manifest import DocumentSpec
from rag_architect.ingestion.pipeline import IngestionPipeline


class FakeLoader:
    def __init__(self, pages: int = 3) -> None:
        self.pages = pages
        self.calls: list[dict[str, object]] = []

    def load(self, path, *, document_type, title, club=None, fiscal_year=None) -> RawDocument:
        self.calls.append(
            {
                "path": path,
                "document_type": document_type,
                "title": title,
                "club": club,
                "fiscal_year": fiscal_year,
            }
        )
        return RawDocument(
            metadata=DocumentMetadata(
                document_id=Path(path).stem,
                document_type=document_type,
                title=title,
                club=club,
                fiscal_year=fiscal_year,
            ),
            pages=tuple(Page(number=i, text=f"page {i}") for i in range(1, self.pages + 1)),
        )


class FakeChunker:
    def __init__(self, per_document: int = 4) -> None:
        self.per_document = per_document

    def split(self, document: RawDocument) -> list[Chunk]:
        return [
            Chunk(
                id=Chunk.build_id(document.metadata.document_id, 1, f"chunk {i}"),
                text=f"chunk {i}",
                metadata=document.metadata,
                content_hash=document.content_hash,
                page_start=1,
                page_end=1,
                char_start=i,
                char_end=i + 1,
            )
            for i in range(self.per_document)
        ]


class FakeEmbedder:
    dimension = 8

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension


class FakeStore:
    def __init__(self) -> None:
        self.points: dict[str, Chunk] = {}
        self.dimensions: list[int] = []

    def ensure_collection(self, dimension: int) -> None:
        self.dimensions.append(dimension)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        for chunk in chunks:
            self.points[str(chunk.id)] = chunk
        return len(chunks)

    def search(self, vector, *, top_k, document_type=None) -> list[SearchResult]:
        return []

    def contains_content_hash(self, content_hash: str) -> bool:
        return any(c.content_hash == content_hash for c in self.points.values())


@pytest.fixture
def parts():
    return FakeLoader(), FakeChunker(), FakeEmbedder(), FakeStore()


@pytest.fixture
def spec(tmp_path) -> DocumentSpec:
    return DocumentSpec(
        path=tmp_path / "brighton24.pdf",
        document_type=DocumentType.CLUB_ACCOUNTS,
        title="Brighton FY2024",
        club="Brighton & Hove Albion",
        fiscal_year=2024,
    )


def test_a_document_is_loaded_chunked_embedded_and_stored(parts, spec):
    loader, chunker, embedder, store = parts

    report = IngestionPipeline(loader, chunker, embedder, store).ingest(spec)

    assert report.document_id == "brighton24"
    assert report.pages == 3
    assert report.chunks == 4
    assert report.skipped is False
    assert len(store.points) == 4
    assert embedder.embedded == ["chunk 0", "chunk 1", "chunk 2", "chunk 3"]


def test_manifest_metadata_reaches_the_loader(parts, spec):
    loader, chunker, embedder, store = parts

    IngestionPipeline(loader, chunker, embedder, store).ingest(spec)

    assert loader.calls[0]["club"] == "Brighton & Hove Albion"
    assert loader.calls[0]["fiscal_year"] == 2024
    assert loader.calls[0]["document_type"] is DocumentType.CLUB_ACCOUNTS


def test_an_unchanged_document_is_skipped_without_embedding(parts, spec):
    """The expensive half -- chunking and embedding -- must not run twice."""
    loader, chunker, embedder, store = parts
    pipeline = IngestionPipeline(loader, chunker, embedder, store)

    pipeline.ingest(spec)
    embedder.embedded.clear()
    second = pipeline.ingest(spec)

    assert second.skipped is True
    assert second.chunks == 0
    assert embedder.embedded == []
    assert len(store.points) == 4


def test_force_reindexes_an_unchanged_document(parts, spec):
    loader, chunker, embedder, store = parts
    pipeline = IngestionPipeline(loader, chunker, embedder, store)

    pipeline.ingest(spec)
    second = pipeline.ingest(spec, force=True)

    assert second.skipped is False
    assert second.chunks == 4
    assert len(store.points) == 4  # same ids, replaced rather than duplicated


def test_a_changed_document_is_reindexed(parts, spec):
    """A different content hash must not be mistaken for the stored one."""
    loader, chunker, embedder, store = parts
    pipeline = IngestionPipeline(loader, chunker, embedder, store)

    pipeline.ingest(spec)
    loader.pages = 5  # the file changed, so the text and its hash change
    second = pipeline.ingest(spec)

    assert second.skipped is False
    assert second.pages == 5


def test_ingest_all_creates_the_collection_with_the_embedder_dimension(parts, spec):
    """Dimension comes from the model, never from configuration."""
    loader, chunker, embedder, store = parts

    reports = IngestionPipeline(loader, chunker, embedder, store).ingest_all([spec, spec])

    assert store.dimensions == [8]
    assert [r.skipped for r in reports] == [False, True]
