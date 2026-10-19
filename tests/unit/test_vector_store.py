"""
Qdrant adapter tests, run against the client's in-process mode.

That mode implements the same API without a server, so the suite needs no
Docker. It does warn that payload indexes have no effect locally: filtering
still returns the right points, by scanning rather than by index. Index
behaviour on a real server belongs in an integration test.
"""

import pytest
from qdrant_client import QdrantClient

from rag_architect.adapters.vector_store import QdrantVectorStore
from rag_architect.config import VectorStoreSettings
from rag_architect.domain.documents import Chunk, DocumentMetadata, DocumentType
from rag_architect.domain.ports import VectorStore

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

CLUB = DocumentMetadata(
    document_id="brighton24",
    document_type=DocumentType.CLUB_ACCOUNTS,
    title="Brighton 2024",
    club="Brighton",
    fiscal_year=2024,
)
REPORT = DocumentMetadata(
    document_id="deloitte26",
    document_type=DocumentType.INDUSTRY_REPORT,
    title="Deloitte 2026",
)


def make_chunk(text: str, page: int, metadata: DocumentMetadata = CLUB) -> Chunk:
    return Chunk(
        id=Chunk.build_id(metadata.document_id, page, text),
        text=text,
        metadata=metadata,
        content_hash="a" * 64,
        page_start=page,
        page_end=page,
        char_start=0,
        char_end=len(text),
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    s = QdrantVectorStore(VectorStoreSettings(), client=QdrantClient(":memory:"))
    s.ensure_collection(4)
    return s


class TestCollection:
    def test_satisfies_the_port(self, store):
        check: VectorStore = store
        assert check is store

    def test_ensure_collection_can_be_called_repeatedly(self, store):
        """Ingestion calls it on every run; it must not fail on the second."""
        store.ensure_collection(4)
        store.ensure_collection(4)

        assert store.count() == 0


class TestUpsert:
    def test_returns_the_number_written(self, store):
        chunks = [make_chunk("Turnover", 3), make_chunk("Costs", 5)]

        assert store.upsert(chunks, [[1, 0, 0, 0], [0, 1, 0, 0]]) == 2
        assert store.count() == 2

    def test_re_upserting_the_same_chunk_does_not_duplicate(self, store):
        """
        The point of deterministic chunk ids: re-running ingestion over an
        unchanged document replaces points instead of piling up copies.
        """
        chunk = make_chunk("Turnover 221,293", 3)

        store.upsert([chunk], [[1, 0, 0, 0]])
        store.upsert([chunk], [[1, 0, 0, 0]])
        store.upsert([chunk], [[1, 0, 0, 0]])

        assert store.count() == 1

    def test_empty_upsert_is_a_no_op(self, store):
        assert store.upsert([], []) == 0

    def test_mismatched_lengths_are_rejected(self, store):
        with pytest.raises(ValueError, match="2 chunks for 1 vectors"):
            store.upsert([make_chunk("a", 1), make_chunk("b", 2)], [[1, 0, 0, 0]])


class TestSearch:
    def test_results_come_back_ordered_by_score(self, store):
        store.upsert(
            [make_chunk("exact", 1), make_chunk("orthogonal", 2)],
            [[1, 0, 0, 0], [0, 1, 0, 0]],
        )

        results = store.search([1, 0, 0, 0], top_k=2)

        assert [r.chunk.text for r in results] == ["exact", "orthogonal"]
        assert results[0].score > results[1].score

    def test_top_k_bounds_the_result_count(self, store):
        chunks = [make_chunk(f"chunk {i}", i + 1) for i in range(5)]
        store.upsert(chunks, [[1, 0, 0, 0]] * 5)

        assert len(store.search([1, 0, 0, 0], top_k=3)) == 3

    def test_document_type_filters_the_corpus(self, store):
        """Primary filings and market commentary answer different questions."""
        store.upsert(
            [make_chunk("club figure", 1), make_chunk("market commentary", 1, REPORT)],
            [[1, 0, 0, 0], [1, 0, 0, 0]],
        )

        club = store.search([1, 0, 0, 0], top_k=5, document_type=DocumentType.CLUB_ACCOUNTS)
        report = store.search([1, 0, 0, 0], top_k=5, document_type=DocumentType.INDUSTRY_REPORT)

        assert [r.chunk.text for r in club] == ["club figure"]
        assert [r.chunk.text for r in report] == ["market commentary"]

    def test_an_empty_collection_returns_nothing(self, store):
        assert store.search([1, 0, 0, 0], top_k=5) == []

    def test_the_chunk_survives_the_round_trip(self, store):
        """The payload is the chunk minus its id, so nothing is lost in storage."""
        original = make_chunk("Turnover 221,293", 3)
        store.upsert([original], [[1, 0, 0, 0]])

        recovered = store.search([1, 0, 0, 0], top_k=1)[0].chunk

        assert recovered == original
